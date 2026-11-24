"""Lazy ASE adapters for genuinely different molecular-potential families."""

import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "configs/molecular-experts.json"
EXTERNAL_MODELS = {
    "aimnet2-wb97m": {"label": "AIMNet2 wB97M-D3", "family": "AIMNet2"},
    "aimnet2-rxn": {"label": "AIMNet2-rxn", "family": "AIMNet2"},
    "ani2x": {"label": "ANI-2x ensemble", "family": "ANI"},
    "orbmol-conservative": {"label": "OrbMol conservative", "family": "Orb"},
}


def external_calculator(name, device, dtype):
    config = json.loads(MANIFEST.read_text())
    spec = config["models"][name]
    if name.startswith("aimnet2"):
        os.environ["AIMNET_CACHE_DIR"] = str(ROOT / "models/aimnet")
        from aimnet.calculators import AIMNet2Calculator
        from aimnet.calculators.aimnet2ase import AIMNet2ASE
        from aimnet.calculators.model_registry import load_model_registry
        actual = load_model_registry()["models"][spec["registry_name"]]
        if any(actual[key] != spec[key] for key in ("file", "url", "sha256")):
            raise ValueError("Pinned AIMNet2 registry identity changed")
        base = AIMNet2Calculator(spec["registry_name"], device=device,
                                 compile_model=False, cuda_graph=False)
        return AIMNet2ASE(base, charge=0, mult=1)
    if name == "ani2x":
        os.environ["TORCHANI_DATA_DIR"] = str(ROOT / "models/ani")
        from huggingface_hub import hf_hub_download
        hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"],
                        local_dir=ROOT / "models/ani/StateDicts", token=False)
        import torchani
        model = torchani.models.ANI2x(periodic_table_index=True, device=device,
            dtype=getattr(torch, dtype), strategy="pyaev")
        return model.ase()
    if name == "orbmol-conservative":
        from download_file import download_file
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
        weight = ROOT / spec["local"]
        download_file(spec["url"], weight, spec["bytes"])
        model, adapter = pretrained.orb_v3_conservative_omol(
            weights_path=str(weight), device=device, precision=dtype, compile=False)
        return ORBCalculator(model, atoms_adapter=adapter, device=device,
                             edge_method="knn_scipy")
    raise ValueError(f"Undefined external expert {name}")


def external_metadata(name, calc, device):
    spec = json.loads(MANIFEST.read_text())["models"][name]
    model = calc.base_calc.model if name.startswith("aimnet2") else calc.model
    parameters = list(model.parameters())
    actual_devices = sorted({str(p.device) for p in parameters})
    if any(p.device.type != torch.device(device).type for p in parameters):
        raise RuntimeError(f"Unexpected parameter device for {name}: {actual_devices}")
    return {"head": None, "available_heads": [], "family": EXTERNAL_MODELS[name]["family"],
            "provenance": spec, "parameter_devices": actual_devices,
            "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
            "parameter_count": sum(p.numel() for p in parameters),
            "precision_note": "AIMNet2 retains its native mixed precision; requested dtype applies to ANI/Orb/MACE",
            "curves": [], "reaction_metrics": []}
