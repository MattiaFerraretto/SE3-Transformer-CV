import torch
from torch import nn
from se3.fibers import Fiber

from dgl.nn.pytorch.glob import MaxPooling

import dgl
from dgl.geometry import farthest_point_sampler
import dgl.function as fn

class GSum(nn.Module):
    """SE(3)-equvariant graph residual sum function."""
    def __init__(self, f_x: Fiber, f_y: Fiber):
        """SE(3)-equvariant graph residual sum function.

        Args:
            f_x: Fiber() object for fiber of summands
            f_y: Fiber() object for fiber of summands
        """
        super().__init__()
        self.f_x = f_x
        self.f_y = f_y
        self.f_out = Fiber.combine_max(f_x, f_y)

    def __repr__(self):
        return f"GSum(structure={self.f_out})"

    def forward(self, x, y):
        out = {}
        for k in self.f_out.degrees:
            k = str(k)
            if (k in x) and (k in y):
                if x[k].shape[1] > y[k].shape[1]:
                    diff = x[k].shape[1] - y[k].shape[1]
                    zeros = torch.zeros(x[k].shape[0], diff, x[k].shape[2]).to(y[k].device)
                    y[k] = torch.cat([y[k], zeros], 1)
                elif x[k].shape[1] < y[k].shape[1]:
                    diff = y[k].shape[1] - x[k].shape[1]
                    zeros = torch.zeros(x[k].shape[0], diff, x[k].shape[2]).to(y[k].device)
                    x[k] = torch.cat([x[k], zeros], 1)

                out[k] = x[k] + y[k]
            elif k in x:
                out[k] = x[k]
            elif k in y:
                out[k] = y[k]
        return out


class GCat(nn.Module):
    """Concat only degrees which are in f_x"""
    def __init__(self, f_x: Fiber, f_y: Fiber):
        super().__init__()
        self.f_x = f_x
        self.f_y = f_y
        f_out = {}
        for k in f_x.degrees:
            f_out[k] = f_x.dict[k]
            if k in f_y.degrees:
                f_out[k] += f_y.dict[k]
        self.f_out = Fiber(dictionary=f_out)

    def __repr__(self):
        return f"GCat(structure={self.f_out})"

    def forward(self, x, y):
        out = {}
        for k in self.f_out.degrees:
            k = str(k)
            if k in y:
                out[k] = torch.cat([x[k], y[k]], 1)
            else:
                out[k] = x[k]
        return out

class BN(nn.Module):
    """SE(3)-equvariant batch/layer normalization"""
    def __init__(self, m):
        """SE(3)-equvariant batch/layer normalization

        Args:
            m: int for number of output channels
        """
        super().__init__()
        self.bn = nn.LayerNorm(m)

    def forward(self, x):
        return self.bn(x)
    
class GMaxPooling(nn.Module):
    """Graph Max Pooling module."""
    def __init__(self):
        super().__init__()
        self.pool = MaxPooling()

    def forward(self, features, G, **kwargs):
        h = features['0'][...,-1]
        return self.pool(G, h)

class Pooling3D(nn.Module):

    def __init__(self, in_features: int, pooling_ratio: float, aggr: str='mean'):
        super().__init__()
        self.in_features = in_features
        self.pooling_ratio = pooling_ratio
        self.aggr = aggr


    def forward(self, G: dgl.graph, features: str, batch_size: int=1):
        device = G.device

        n_points = G.ndata['x'].size(0) // batch_size
        self.downsampled_points = round(n_points * (1 - self.pooling_ratio))

        pos =  G.ndata['x'].view(-1, n_points, self.in_features)
        
        fp_idx = farthest_point_sampler(
            pos,
            self.downsampled_points 
        )

        starting_idx = (torch.arange(pos.size(0), device=device) * n_points).view(-1, 1)
        fp_idx = (starting_idx + fp_idx).flatten()

        subgraphs = dgl.node_subgraph(G, fp_idx)

        G.update_all(
            fn.copy_u(features, 'm'),               # Copy source node feature to the message field
            getattr(fn, self.aggr)('m', features)   # Aggregate messages by taking an aggregation function in dgl.fn module
        )
        
        src, dst = subgraphs.edges()

        subgraphs.ndata['x'] = G.ndata['x'][fp_idx]
        subgraphs.ndata[features] = G.ndata[features][fp_idx]

     
        pos = torch.squeeze(G.ndata['x'], dim=1)
        subgraphs.edata['d'] = pos[dst] - pos[src]
        subgraphs.edata['w'] = torch.sqrt(torch.sum(subgraphs.edata['d']**2, dim=-1, keepdim=True))

        
        G_level_structure = dgl.graph(G.edges(), device=device)
        G_level_structure.ndata['x'] = G.ndata['x']
        G_level_structure.edata['d'] = G.edata['d']
        G_level_structure.edata['w'] = G.edata['w']

        return subgraphs, G_level_structure, fp_idx
    
class Upsampling3D(nn.Module):
    '''
    The Upsampling3D class implements IDW (Inverse Distance Weighting) to upsample the given point cloud (as a DGL graph) to the original resolution
    with support for multi-channel features.
    '''

    def __init__(self, in_features: int, power: int):
        super().__init__()
        self.in_features = in_features
        self.power = power

    def forward(self, source_graph: dgl.DGLGraph, features: str, target_graph: dgl.DGLGraph, fp_idx: torch.Tensor):
        device = source_graph.device

        _, in_channels, in_features = source_graph.ndata[features].shape
        nodes_feature = torch.zeros(target_graph.ndata['x'].size(0), in_channels, in_features, device=device)

        # Assign known features at the provided indices
        nodes_feature[fp_idx, :] = source_graph.ndata[features]

        # Identify nodes where features must be estimated
        nodes = target_graph.nodes().to(device)
        nodes_idx = nodes[~torch.isin(nodes, fp_idx)]
        
        # Get the neighborhoods of the nodes of interest
        srcs, dsts = target_graph.in_edges(nodes_idx)

        srcs = srcs.to(device)
        dsts = dsts.to(device)
        
        # Compute weights, taking care of avoiding zero division
        weights = 1 / torch.pow(target_graph.edata['w'][srcs] + 1e-10, self.power).to(device)
        weights = torch.repeat_interleave(weights, in_channels, dim=-1).view(-1, in_channels, 1)

        # Masking unknown features
        feature_mask = torch.any(nodes_feature[srcs] != 0, dim=-1, keepdim=True)
        
        # Apply the feature mask to ignore weights for unknown features
        weights *= feature_mask.float()

        # Initialize accumulators
        weighted_features = torch.zeros_like(nodes_feature)
        weights_sum = torch.zeros(target_graph.num_nodes(), in_channels, 1, device=device)

        # Accumulate weighted features and weights for each destination node
        weighted_features.index_add_(0, dsts, nodes_feature[srcs] * weights)
        weights_sum.index_add_(0, dsts, weights)

        # Normalize by the sum of weights, taking care of potential zero division
        nodes_feature[nodes_idx] = weighted_features[nodes_idx] / weights_sum[nodes_idx].clamp(min=1e-10)


        # non optimized code
        # for node_id in nodes_idx:
        #     neighborhood = G_level_structure.in_edges(node_id)[0]
        #     neighborhood = neighborhood[torch.any(nodes_feature[neighborhood] != 0, dim=1)]

        #     weights = 1 / torch.pow(G_level_structure.edata['w'][neighborhood], self.power)
        #     node_features = nodes_feature[neighborhood] * weights
        #     nodes_feature[node_id] = node_features.sum(dim=0) / weights.sum()


        # Update the node features in the graph with the estimated values
        target_graph.ndata[features] = nodes_feature
        
        return target_graph
