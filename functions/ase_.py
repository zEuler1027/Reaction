from ase import Atoms
from pymatgen.core.structure import Molecule
from pymatgen.io.ase import AseAtomsAdaptor


def data_to_ase(data) -> Atoms:
    ele = data[:, -1].long().cpu().numpy()
    pos = data[:, :3].cpu().numpy()
    atoms = Atoms(positions=pos, numbers=ele)
    return atoms


def ase_to_pymatgen(data) -> Molecule:
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_molecule(data)
    return atoms


def data_to_pymatgen(data) -> Molecule:
    atoms = data_to_ase(data)
    atoms = ase_to_pymatgen(atoms)
    return atoms
    
