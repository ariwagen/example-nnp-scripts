import numpy as np

from ase.build import molecule
from ase.io import write
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from sella import Sella  # pip install sella


def _fmax(atoms) -> float:
    forces = atoms.get_forces()
    return float(np.linalg.norm(forces, axis=1).max())


def main():
    device = "cpu"  # or device="cuda"
    orbff = pretrained.orb_v3_conservative_omol(
        device=device,
        precision="float32-high",  # or "float32-highest" / "float64
    )
    calc = ORBCalculator(orbff, device=device)

    atoms = molecule("C6H6")
    atoms.calc = calc
    atoms.info = {"charge": 0, "spin": 0}

    optimizer = Sella(atoms, order=0, internal=True)

    optimizer.run(fmax=0.05, steps=300)
    fmax = _fmax(atoms)
    e_final = atoms.get_potential_energy()

    write("output.xyz", atoms)

    print(f"Final energy: {e_final:.6f} eV | max|F| = {fmax:.4f} eV/Å")
    assert fmax < 0.05, f"Did not converge: fmax={fmax:.4f} eV/Å"


if __name__ == "__main__":
    main()
