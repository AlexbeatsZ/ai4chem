"""Check shared-backbone expert gradients on two existing training frames only."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator

from residual_moe import SharedResidualMoE, ResidualMoECalculator


def main():
    output = ROOT/"results/v0.11.0/residual_interface"
    output.mkdir(parents=True, exist_ok=True)
    if (output/"acceptance.json").exists():
        raise FileExistsError("Existing residual interface evidence must not be overwritten")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(20261125)
    base_path = ROOT/"models/v0.8.0/mh1_global512_e40.model"
    provider = MACECalculator(model_paths=str(base_path), device="cpu", default_dtype="float64")
    model = SharedResidualMoE(provider.models[0])
    frames = read(ROOT/"data/v0.8.0/general4096.extxyz", index=":2")
    frozen_before = [parameter.detach().clone() for parameter in model.backbone.parameters()]
    adapter_before = [parameter.detach().clone() for parameter in model.adapter_parameters()]
    atoms = frames[0].copy(); atoms.calc = provider
    base_energy, base_force = atoms.get_potential_energy(), atoms.get_forces()
    atoms.calc = ResidualMoECalculator(model, provider)
    np.testing.assert_allclose(atoms.get_potential_energy(), base_energy, atol=1e-8, rtol=0)
    np.testing.assert_allclose(atoms.get_forces(), base_force, atol=1e-8, rtol=0)
    optimizer = torch.optim.Adam(model.adapter_parameters(), lr=1e-4)
    losses, durations, gradients = [], [], []
    for step in range(3):
        began = time.perf_counter()
        optimizer.zero_grad()
        terms = []
        for frame in frames:
            batch = provider._atoms_to_batch(frame).to_dict()
            prediction = model(batch, training=True)
            reference_energy = float(frame.info["REF_energy"])
            reference_force = torch.as_tensor(frame.arrays["REF_forces"])
            terms.append(10*((prediction["energy"].sum()-reference_energy)/len(frame))**2
                         +torch.mean((prediction["forces"]-reference_force)**2))
        loss = torch.stack(terms).mean()
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.adapter_parameters()):
            raise ValueError("Nonfinite force-training gradient")
        gradients.append(float(model.element_logits.grad.norm()))
        optimizer.step()
        losses.append(float(loss.detach()))
        durations.append(time.perf_counter()-began)
    if any(not torch.equal(before, after) for before, after in zip(frozen_before, model.backbone.parameters())):
        raise ValueError("Frozen backbone changed")
    changed = sum(not torch.equal(before, after) for before, after in zip(adapter_before, model.adapter_parameters()))
    if not changed or gradients[-1] <= 0:
        raise ValueError("Adapters or element gate did not receive force-training gradients")
    checkpoint = output/"numerical-prototype.pt"
    model.save_adapter(checkpoint, "models/v0.8.0/mh1_global512_e40.model")
    restored = SharedResidualMoE.load_adapter(checkpoint, model.backbone)
    for before, after in zip(model.adapter_parameters(), restored.adapter_parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    atoms.calc = ResidualMoECalculator(restored, provider)
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    positions = atoms.positions.copy()
    delta = 1e-4
    atoms.positions[0, 0] += delta; plus = atoms.get_potential_energy()
    atoms.positions[0, 0] -= 2*delta; minus = atoms.get_potential_energy()
    finite_difference = abs(-(plus-minus)/(2*delta)-forces[0, 0])
    if finite_difference > 2e-5:
        raise ValueError("Residual energy-force finite difference failed")
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    atoms.positions = positions @ rotation.T+[2., -1., .5]
    rotation_energy = abs(atoms.get_potential_energy()-energy)
    rotation_force = float(np.max(np.abs(atoms.get_forces()-forces @ rotation.T)))
    if max(rotation_energy, rotation_force) > 2e-6:
        raise ValueError("Residual model broke rotational/translation symmetry")
    baseline = SharedResidualMoE(model.backbone, kind="single_head")
    result = {"status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Two MACE-training frames, three optimizer updates for interface acceptance only; not a scientific performance study",
        "training_frames": [frame.info.get("reaction") for frame in frames], "optimizer_steps": 3,
        "invariant_features": len(model.scalar_indices), "backbone_features": model.full_feature_count,
        "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
        "trainable_adapter_parameters": sum(p.numel() for p in model.adapter_parameters()),
        "matched_single_head_parameters": sum(p.numel() for p in baseline.adapter_parameters()),
        "losses": losses, "update_seconds": durations, "element_gate_gradient_norm": gradients,
        "adapter_tensors_changed": changed, "frozen_backbone_unchanged": True, "adapter_reload_passed": True,
        "finite_difference_force_error_eV_per_A": finite_difference,
        "rotation_energy_difference_eV": rotation_energy, "rotation_force_max_difference_eV_per_A": rotation_force,
        "new_GPU_training_hours": 0, "holdout_labels_evaluated": False,
        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_bytes": checkpoint.stat().st_size,
        "scientific_training_submitted": False, "chemical_improvement_established": False}
    (output/"acceptance.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
