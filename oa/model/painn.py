import torch 
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.utils import add_self_loops, degree

import torch.nn as nn
import torch.nn.functional as Func
from torch.nn import Embedding, Sequential, Linear, ModuleList, Module
import numpy as np
import math


class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class CosineCutoff(torch.nn.Module):

    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        #self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        self.cutoff = cutoff

    def forward(self, distances):
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs
    

class BesselBasis(torch.nn.Module):
    """
    Sine for radial basis expansion with coulomb decay. (0th order Bessel from DimeNet)
    """

    def __init__(self, cutoff=5.0, n_rbf=20):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselBasis, self).__init__()
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        inputs = torch.norm(inputs, p=2, dim=1)
        a = self.freqs
        ax = torch.outer(inputs,a)
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[:,None]

        return y
    

class PaiNN(torch.nn.Module):
    def __init__(self, num_feats, out_channels, in_hidden_channels, cutoff=5.0, n_rbf=20, num_interactions=3, **kargs):
        super(PaiNN, self).__init__() 
        '''PyG implementation of PaiNN network of Schütt et. al. Supports two arrays  
           stored at the nodes of shape (num_nodes,num_feats,1) and (num_nodes, num_feats,3). For this 
           representation to be compatible with PyG, the arrays are flattened and concatenated. 
           Important to note is that the out_channels must match number of features'''
        
        self.out_channels = out_channels
        self.embedding = nn.Linear(in_hidden_channels, num_feats)
        self.cut_off = cutoff
        self.num_interactions = num_interactions
        self.n_rbf = n_rbf
        self.linear = Linear(num_feats, in_hidden_channels)
        self.silu = Func.silu

        self.list_message = nn.ModuleList(
            [
                MessagePassPaiNN(num_feats, out_channels, cutoff, n_rbf)
                for _ in range(self.num_interactions)
            ]
        )
        self.list_update = nn.ModuleList(
            [
                UpdatePaiNN(num_feats, out_channels)
                for _ in range(self.num_interactions)
            ]
        )

        self.output = PaiNNOutput(out_channels)


    def forward(self, s, pos, edge_index, **kargs):
        j, i = edge_index
        edge_attr = pos[j] - pos[i]
        s = self.embedding(s)
        v = torch.zeros(pos.shape[0], self.out_channels, 3, device=pos.device)

        for i in range(self.num_interactions):
            
            s_temp,v_temp = self.list_message[i](s, v, edge_index, edge_attr)
            s, v = s_temp+s, v_temp+v
            s_temp,v_temp = self.list_update[i](s, v) 
            s, v = s_temp+s, v_temp+v       

        v = v.transpose(1, 2)
        v = self.output(s, v)

        s = self.silu(s)
        s = self.linear(s)
        
        v = pos + v
        return s, v, None
    

class MessagePassPaiNN(MessagePassing):
    def __init__(self, num_feats, out_channels, cut_off=5.0, n_rbf=20):
        super(MessagePassPaiNN, self).__init__(aggr='add') 
        
        self.lin1 = Linear(num_feats, out_channels) 
        self.lin2 = Linear(out_channels, 3*out_channels) 
        self.lin_rbf = Linear(n_rbf, 3*out_channels) 
        self.silu = Func.silu
        
        #self.prepare = Prepare_Message_Vector(num_nodes)
        self.RBF = BesselBasis(cut_off, n_rbf)
        self.f_cut = CosineCutoff(cut_off)
    
    def forward(self, s,v, edge_index, edge_attr):
        
        s = s.flatten(-1)
        v = v.flatten(-2)
        
        flat_shape_v = v.shape[-1]
        flat_shape_s = s.shape[-1]
    
        x =torch.cat([s, v], dim =-1)
        
        
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr
                            ,flat_shape_s=flat_shape_s, flat_shape_v=flat_shape_v)
            
        return x    
    
    def message(self, x_j, edge_attr, flat_shape_s, flat_shape_v):
        
        
        # Split Input into s_j and v_j
        s_j, v_j = torch.split(x_j, [flat_shape_s, flat_shape_v], dim=-1)
        
        # r_ij channel
        rbf = self.RBF(edge_attr)
        ch1 = self.lin_rbf(rbf)
        cut = self.f_cut(edge_attr.norm(dim=-1))
        W = torch.einsum('ij,i->ij',ch1, cut) # ch1 * f_cut
        
        # s_j channel
        phi = self.lin1(s_j)
        phi = self.silu(phi)
        phi = self.lin2(phi)
        
        # Split 
        left, dsm, right = torch.tensor_split(phi*W,3,dim=-1)
        
        # v_j channel
        normalized = Func.normalize(edge_attr, p=2, dim=1)
        
        v_j = v_j.reshape(-1, int(flat_shape_v/3), 3)
        hadamard_right = torch.einsum('ij,ik->ijk',right, normalized)
        hadamard_left = torch.einsum('ijk,ij->ijk',v_j,left)
        dvm = hadamard_left + hadamard_right 
        
        # Prepare vector for update
        x_j = torch.cat((dsm,dvm.flatten(-2)), dim=-1)
       
        return x_j
    
    def update(self, out_aggr,flat_shape_s, flat_shape_v):
        
        s_j, v_j = torch.split(out_aggr, [flat_shape_s, flat_shape_v], dim=-1)
        
        return s_j, v_j.reshape(-1, int(flat_shape_v/3), 3)
    

class UpdatePaiNN(torch.nn.Module):
    def __init__(self, num_feats, out_channels):
        super(UpdatePaiNN, self).__init__() 
        
        self.lin_up = Linear(2*num_feats, out_channels) 
        self.denseU = Linear(num_feats,out_channels, bias = False) 
        self.denseV = Linear(num_feats,out_channels, bias = False) 
        self.lin2 = Linear(out_channels, 3*out_channels) 
        self.silu = Func.silu
        
        
    def forward(self, s,v):
        
        # split and take linear combinations
        #s, v = torch.split(out_aggr, [flat_shape_s, flat_shape_v], dim=-1)
        
        s = s.flatten(-1)
        v = v.flatten(-2)
        
        flat_shape_v = v.shape[-1]
        flat_shape_s = s.shape[-1]
        
        v_u = v.reshape(-1, int(flat_shape_v/3), 3)
        v_ut = torch.transpose(v_u,1,2)
        U = torch.transpose(self.denseU(v_ut),1,2)
        V = torch.transpose(self.denseV(v_ut),1,2)
        
        
        # form the dot product
        UV =  torch.einsum('ijk,ijk->ij',U,V) 
        
        # s_j channel
        nV = torch.norm(V, dim=-1)

        s_u = torch.cat([s, nV], dim=-1)
        s_u = self.lin_up(s_u) 
        s_u = Func.silu(s_u)
        s_u = self.lin2(s_u)
        #s_u = Func.silu(s_u)
        
        # final split
        top, middle, bottom = torch.tensor_split(s_u,3,dim=-1)
        
        # outputs
        dvu = torch.einsum('ijk,ij->ijk',v_u,top) 
        dsu = middle*UV + bottom 
        
        #update = torch.cat((dsu,dvu.flatten(-2)), dim=-1)
        
        return dsu, dvu.reshape(-1, int(flat_shape_v/3), 3)


class PaiNNOutput(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return vec.squeeze()


# Borrowed from TorchMD-Net
class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False
        )
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )

        self.act = Func.silu

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        x = self.act(x)
        return x, v


class OAPaiNN(torch.nn.Module):
    def __init__(self, num_feats, out_channels, in_hidden_channels, cutoff=5.0, n_rbf=20, num_interactions=3, **kargs):
        super(PaiNN, self).__init__() 
        '''PyG implementation of PaiNN network of Schütt et. al. Supports two arrays  
            stored at the nodes of shape (num_nodes,num_feats,1) and (num_nodes, num_feats,3). For this 
            representation to be compatible with PyG, the arrays are flattened and concatenated. 
            Important to note is that the out_channels must match number of features'''
        
        self.out_channels = out_channels
        self.embedding = nn.Linear(in_hidden_channels, num_feats)
        self.cut_off = cutoff
        self.num_interactions = num_interactions
        self.n_rbf = n_rbf
        self.linear = Linear(num_feats, in_hidden_channels)
        self.silu = Func.silu

        self.list_message = nn.ModuleList(
            [
                MessagePassPaiNN(num_feats, out_channels, cutoff, n_rbf)
                for _ in range(self.num_interactions)
            ]
        )
        self.list_update = nn.ModuleList(
            [
                UpdatePaiNN(num_feats, out_channels)
                for _ in range(self.num_interactions)
            ]
        )

        self.output = PaiNNOutput(out_channels)

    def forward(self, s, pos, edge_index, **kargs):
        j, i = edge_index
        edge_attr = pos[j] - pos[i]
        s = self.embedding(s)
        v = torch.zeros(pos.shape[0], self.out_channels, 3, device=pos.device)

        for i in range(self.num_interactions):
            
            s_temp,v_temp = self.list_message[i](s, v, edge_index, edge_attr)
            s, v = s_temp+s, v_temp+v
            s_temp,v_temp = self.list_update[i](s, v) 
            s, v = s_temp+s, v_temp+v       

        v = v.transpose(1, 2)
        v = self.output(s, v)

        s = self.silu(s)
        s = self.linear(s)
        
        v = pos + v
        return s, v, None
    
    
class OAPaiNNCondition(torch.nn.Module):
    pass
