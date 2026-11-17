"""Batch independent NEB image evaluations through MACE's public data API."""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class MACEBand:
    def __init__(self, images, calculator):
        if len(calculator.models) != 1:
            raise ValueError("The NEB batch backend requires one MACE model")
        self.images = images
        self.calculator = calculator
        self.positions = None
        self.results = None

    def evaluate(self):
        from mace import data
        from mace.tools import torch_geometric, torch_tools

        positions = np.array([image.positions for image in self.images])
        if self.positions is not None and np.array_equal(positions, self.positions):
            return self.results
        calc = self.calculator
        with torch_tools.default_dtype(calc.default_dtype):
            graphs = [data.AtomicData.from_config(
                data.config_from_atoms(image.copy(), head_name=calc.head),
                z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)
                for image in self.images]
            batch = torch_geometric.Batch.from_data_list(graphs).to(calc.device)
            output = calc.models[0](batch.to_dict(), training=False, compute_force=True, compute_stress=False)
        energies = output["energy"].detach().cpu().numpy() * calc.energy_units_to_eV
        forces = output["forces"].detach().cpu().numpy() * calc.energy_units_to_eV / calc.length_units_to_A
        if not np.isfinite(energies).all() or not np.isfinite(forces).all():
            raise ValueError("Non-finite batched MACE output")
        offsets = np.r_[0, np.cumsum([len(image) for image in self.images])]
        self.results = [{"energy": float(energies[index]),
                         "forces": forces[offsets[index]:offsets[index + 1]].copy()}
                        for index in range(len(self.images))]
        self.positions = positions.copy()
        return self.results

    def attach(self):
        for index, image in enumerate(self.images):
            image.calc = BandImageCalculator(self, index)


class BandImageCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, band, index):
        super().__init__()
        self.band = band
        self.index = index

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        result = self.band.evaluate()[self.index]
        self.results = {"energy": result["energy"], "forces": result["forces"].copy()}


def check_batch_parity(images, calculator):
    reference = []
    for image in images:
        image.calc = calculator
        reference.append((image.get_potential_energy(), image.get_forces().copy()))
    band = MACEBand(images, calculator)
    results = band.evaluate()
    energy_error = max(abs(result["energy"] - energy) for result, (energy, _) in zip(results, reference))
    force_error = max(float(np.max(np.abs(result["forces"] - forces)))
                      for result, (_, forces) in zip(results, reference))
    if energy_error > 1e-7 or force_error > 1e-8:
        raise ValueError(f"Batched MACE differs from ASE calculator: E={energy_error}, F={force_error}")
    return {"maximum_energy_difference_eV": float(energy_error),
            "maximum_force_difference_eV_A": float(force_error), "images": len(images)}
