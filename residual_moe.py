"""Small invariant residual experts sharing one frozen MACE backbone.

This module is a numerical prototype, not a completed scientific training run.
Even scalar features are selected explicitly from the backbone's irreps; arbitrary
nonlinear layers never act on vector/tensor channels. Coordinate gradients through
the frozen backbone remain enabled so energy and force losses can train adapters.
"""

import numpy as np
import torch
from torch import nn
from ase.calculators.calculator import Calculator, all_changes

from soft_mixture import CompositionMixtureCalculator


def invariant_feature_indices(backbone):
    indices, offset = [], 0
    for product in backbone.products:
        irreps = product.linear.irreps_out
        for (_, irrep), span in zip(irreps, irreps.slices()):
            if irrep.l == 0 and irrep.p == 1:
                indices.extend(range(offset+span.start, offset+span.stop))
        offset += irreps.dim
    if not indices:
        raise ValueError("Backbone has no even scalar descriptors")
    return indices, offset


def residual_head(features, width):
    head = nn.Sequential(nn.Linear(features, width), nn.SiLU(), nn.Linear(width, 1, bias=False))
    nn.init.zeros_(head[-1].weight)
    return head


class SharedResidualMoE(nn.Module):
    def __init__(self, backbone, kind="shared_experts", width=16, expert_count=2):
        super().__init__()
        if kind not in ("shared_experts", "single_head") or width < 1 or expert_count < 2:
            raise ValueError("Unsupported residual adapter configuration")
        self.backbone = backbone
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        self.kind, self.width, self.expert_count = kind, width, expert_count
        indices, self.full_feature_count = invariant_feature_indices(backbone)
        self.register_buffer("scalar_indices", torch.tensor(indices, dtype=torch.long))
        self.normalization = nn.LayerNorm(len(indices), elementwise_affine=False)
        if kind == "shared_experts":
            self.shared = residual_head(len(indices), width)
            self.experts = nn.ModuleList([residual_head(len(indices), width) for _ in range(expert_count)])
            self.element_logits = nn.Parameter(torch.randn(4, expert_count)*.05)
        else:
            self.shared = residual_head(len(indices), width*(expert_count+1))
            self.experts = nn.ModuleList()
        self.to(dtype=next(backbone.parameters()).dtype, device=next(backbone.parameters()).device)

    def adapter_parameters(self):
        return [parameter for name, parameter in self.named_parameters() if not name.startswith("backbone.")]

    def forward(self, data, training=False):
        # Freezing model weights does not permit torch.no_grad(): feature-coordinate
        # derivatives are needed for residual forces and their training gradients.
        base = self.backbone(data, training=training, compute_force=False, compute_stress=False)
        features = base["node_feats"]
        if features.shape[1] != self.full_feature_count:
            raise ValueError("Backbone descriptor layout differs from its irreps")
        scalar = self.normalization(features.index_select(1, self.scalar_indices))
        residual = self.shared(scalar).squeeze(-1)
        weights = None
        if self.kind == "shared_experts":
            numbers = self.backbone.atomic_numbers[data["node_attrs"].argmax(dim=1)]
            mapping = torch.full_like(numbers, -1)
            for index, number in enumerate((1, 6, 7, 8)):
                mapping = torch.where(numbers == number, index, mapping)
            if torch.any(mapping < 0):
                raise ValueError("Residual pilot supports only CHNO")
            weights = torch.softmax(self.element_logits, dim=-1)[mapping]
            expert_output = torch.cat([expert(scalar) for expert in self.experts], dim=1)
            residual = residual+(weights*expert_output).sum(dim=1)
        correction = torch.zeros_like(base["energy"]).index_add(0, data["batch"], residual)
        energy = base["energy"]+correction
        forces = -torch.autograd.grad(energy.sum(), data["positions"], create_graph=training, retain_graph=training)[0]
        return {"energy": energy, "forces": forces, "residual_energy": correction, "expert_weights": weights}

    def save_adapter(self, path, base_model_file):
        torch.save({"schema": "ai4chem-shared-residual-moe-1", "kind": self.kind,
            "width": self.width, "expert_count": self.expert_count, "base_model_file": str(base_model_file),
            "base_model_required": True, "scope": "numerical interface prototype; no chemical improvement established",
            "state_dict": {key: value for key, value in self.state_dict().items() if not key.startswith("backbone.")}}, path)

    @classmethod
    def load_adapter(cls, path, backbone):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["schema"] != "ai4chem-shared-residual-moe-1":
            raise ValueError("Unknown residual adapter schema")
        model = cls(backbone, checkpoint["kind"], checkpoint["width"], checkpoint["expert_count"])
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if unexpected or any(not key.startswith("backbone.") for key in missing):
            raise ValueError("Residual adapter state differs from its declared architecture")
        return model


class ResidualMoECalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, graph_provider, **kwargs):
        super().__init__(**kwargs)
        self.model, self.graph_provider = model, graph_provider

    def get_property(self, name, atoms=None, allow_calculation=True):
        CompositionMixtureCalculator.validate_atoms(atoms if atoms is not None else self.atoms)
        return super().get_property(name, atoms, allow_calculation)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        atoms = atoms if atoms is not None else self.atoms
        CompositionMixtureCalculator.validate_atoms(atoms)
        super().calculate(atoms, properties, system_changes)
        batch = self.graph_provider._atoms_to_batch(atoms).to_dict()
        result = self.model(batch, training=False)
        self.results = {"energy": float(result["energy"].detach().sum()),
                        "forces": result["forces"].detach().cpu().numpy()}
        if not all(np.isfinite(value).all() for value in self.results.values()):
            raise ValueError("Nonfinite residual model prediction")
