import py3Dmol
from typing import List
from pymatgen.core.structure import Molecule


def draw_reaction(reaction: List[Molecule]) -> py3Dmol.view:
    natoms = int(sum([mol_.num_sites for mol_ in reaction]))
    mol = f'{natoms}\n\n'
    
    for ii, mol_ in enumerate(reaction):
        coords = mol_.cart_coords
        coords[:, 0] += ii * 10
        mol_ = Molecule(
            species=mol_.atomic_numbers,
            coords=coords,
        )
        mol += '\n'.join(mol_.to(fmt="xyz").split("\n")[2:]) + '\n'
        
    viewer = py3Dmol.view(2024, 1576)
    viewer.addModel(mol, "xyz")
    viewer.setStyle({'stick': {'radius': 0.20}, "sphere": {"radius": 0.35}})
    viewer.zoomTo()
    return viewer
