"""Utility functions for graphs."""
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import radius_graph

def get_edges_index(
    combined_mask: Tensor,
    pos: Optional[Tensor] = None,
    edge_cutoff: Optional[float] = None,
    remove_self_edge: bool = False,
) -> Tensor:
    r"""

    Args:
        combined_mask (Tensor): Combined mask for all fragments.
            Edges are built for nodes with the same indexes in the mask.
        pos (Optional[Tensor]): 3D coordinations of nodes. Defaults to None.
        edge_cutoff (Optional[float]): cutoff for building edges within a fragment.
            Defaults to None.
        remove_self_edge (bool): whether to remove self-connecting edge (i.e., ii).
            Defaults to False.

    Returns:
        Tensor: [2, n_edges], i for node index.
    """
    # TODO: cache batches for each example in self._edges_dict[n_nodes]
    adj = combined_mask[:, None] == combined_mask[None, :]
    if edge_cutoff is not None:
        assert pos is not None, "Position must be provided for building edges."
        edge_index = radius_graph(
            pos,
            r=edge_cutoff, 
            batch=combined_mask, 
            loop=False, 
            max_num_neighbors=60
        )
        return edge_index
    if remove_self_edge:
        adj = adj.fill_diagonal_(False)
    edges = torch.stack(torch.where(adj), dim=0)
    return edges


def get_subgraph_mask(edge_index: Tensor, n_frag_switch: Tensor) -> Tensor:
    r"""Filter out edges that have inter-fragment connections.
    Example:
    edge_index: [
        [0, 0, 1, 1, 2, 2],
        [1, 2, 0, 2, 0, 1],
        ]
    n_frag_switch: [0, 0, 1]
    -> [1, 0, 1, 0, 0, 0]

    Args:
        edge_index (Tensor): e_ij
        n_frag_switch (Tensor): fragment that a node belongs to

    Returns:
        Tensor: [n_edge], 1 for inner- and 0 for inter-fragment edge
    """
    subgraph_mask = torch.zeros(edge_index.size(1)).long()
    in_same_frag = n_frag_switch[edge_index[0]] == n_frag_switch[edge_index[1]]
    subgraph_mask[torch.where(in_same_frag)] = 1
    return subgraph_mask.to(edge_index.device)


def get_atoms_mask_rtp(natm_list: List[Tensor]) -> Tensor:
    r"""Get the type of fragments to which each node belongs
    Example: [Tensor(1, 1), Tensor(2, 1)] -> [0, 0, 1, 1 ,1]

    Args:
        natm_list (List[Tensor]): [Tensor([number of atoms per small fragment])]

    Returns:
        Tensor: [n_nodes], type of fragment each node belongs to
    """
    shapes = [natm.shape[0] for natm in natm_list]
    assert np.std(shapes) == 0, "Tensor must be the same length for <natom_list>"
    n_frag_switch = torch.repeat_interleave(
        torch.arange(len(natm_list), device=natm_list[0].device),
        torch.tensor(
            [torch.sum(natm).item() for natm in natm_list],
            device=natm_list[0].device,
        ),
    )
    return n_frag_switch.to(natm_list[0].device)


def get_mask_for_frag(natm: Tensor) -> Tensor:
    r"""Get fragment index for each node
    Example: Tensor([2, 0, 3]) -> [0, 0, 2, 2, 2]

    Args:
        natm (Tensor): number of nodes per small fragment

    Returns:
        Tensor: [n_node], the natural index of fragment a node belongs to
    """
    return torch.repeat_interleave(
        torch.arange(natm.size(0), device=natm.device), natm
    ).to(natm.device)


def get_inner_edge_index(subgraph_mask: Tensor):
    return torch.stack(torch.where(subgraph_mask), dim=0)


def get_self_edge_index(fragments_nodes: List[Tensor]) -> Tensor:
    ''' Get self-connecting edges for all fragments
    -input: fragments_nodes,  List[(batch_size,) * 3]
    -output:  edge_index_self, 2 * E
    '''
    natom = fragments_nodes[0].sum(dim=-1)
    index = torch.arange(natom.item()).repeat(3)
    adj = index[:, None] == index[None, :]
    # remove self-connecting edge
    adj = adj.fill_diagonal_(False) 
    edge_index_self = torch.stack(torch.where(adj), dim=0)
    return edge_index_self.to(fragments_nodes[0].device)


def get_custom_edge_index(masks: List[Tensor], 
                          fragments_nodes: List[Tensor], 
                          remove_self_edge: bool = True) -> Tensor:
    masks = torch.cat([masks[0], masks[1] + 1000, masks[2] + 2000])
    adj = masks[:, None] == masks[None, :]

    natom = fragments_nodes[0].sum(dim=-1)
    index = torch.arange(natom.item(), device=masks[0].device).repeat(3)
    self_adj = index[:, None] == index[None, :]

    # remove self-connecting edge
    if remove_self_edge:
        adj = adj.fill_diagonal_(False)
        self_adj = self_adj.fill_diagonal_(False)
        
    edge_index = torch.stack(torch.where(adj), dim=0)
    edge_index_self = torch.stack(torch.where(self_adj), dim=0)
    edge_index = torch.cat([edge_index, edge_index_self], dim=-1)

    return edge_index.to(fragments_nodes[0].device)
    