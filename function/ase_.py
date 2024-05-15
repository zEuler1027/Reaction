from ase import Atoms
from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from torch import Tensor
from ase.atoms import Atoms


def data_to_ase(data: Tensor) -> Atoms:
    ele = data[:, -1].long().cpu().numpy()
    pos = data[:, :3].cpu().numpy()
    atoms = Atoms(positions=pos, numbers=ele)
    return atoms


def ase_to_pymatgen(data: Atoms) -> Molecule:
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_molecule(data)
    return atoms


def data_to_pymatgen(data: Tensor) -> Molecule:
    atoms = data_to_ase(data)
    atoms = ase_to_pymatgen(atoms)
    return atoms
    
