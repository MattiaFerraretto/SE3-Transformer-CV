import torch
from torch import nn
from se3.fibers import Fiber
from se3.modules import get_basis_and_r, GSE3Res, GNormBias

import dgl
from se3.utils.wrappers import Pooling3D, Upsampling3D

class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""

    def __init__(self, num_layers: int, num_channels: int, num_degrees: int = 4, div: float = 4,
                 n_heads: int = 1, si_m='1x1', si_e='att', x_ij='add'):
        """
        Args:
            num_layers: number of attention layers
            num_channels: number of channels per degree
            num_degrees: number of degrees (aka types) in hidden layer, count start from type-0
            div: (int >= 1) keys, queries and values will have (num_channels/div) channels
            n_heads: (int >= 1) for multi-headed attention
            si_m: ['1x1', 'att'] type of self-interaction in hidden layers
            si_e: ['1x1', 'att'] type of self-interaction in final layer
            x_ij: ['add', 'cat'] use relative position as edge feature
        """
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = 1
        self.div = div
        self.n_heads = n_heads
        self.si_m, self.si_e = si_m, si_e
        self.x_ij = x_ij

        self.fibers = {'in': Fiber(dictionary={1: 1}),
                       'mid': Fiber(self.num_degrees, self.num_channels),
                       'out': Fiber(dictionary={1: 1})}

        self.Gblock = self._build_gcn(self.fibers)

    def _build_gcn(self, fibers):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, div=self.div, n_heads=self.n_heads,
                                  learnable_skip=True, skip='cat', selfint=self.si_m, x_ij=self.x_ij))
            Gblock.append(GNormBias(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(
            GSE3Res(fibers['mid'], fibers['out'], edge_dim=self.edge_dim, div=1, n_heads=min(self.n_heads, 2),
                    learnable_skip=True, skip='cat', selfint=self.si_e, x_ij=self.x_ij))
        return nn.ModuleList(Gblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)
        h_enc = {'1': G.ndata['v']}
        #print(h_enc)
        for layer in self.Gblock:
            h_enc = layer(h_enc, G=G, r=r, basis=basis)

        return h_enc['1']

class SE3ConvBlock(nn.Module):

    def __init__(
            self,
            f_in: Fiber,
            f_out: Fiber,
            num_layers: int,
            div: float = 1,
            n_heads: int = 1,
            selfint ='att',
            x_ij='add'
    ):
        """
        Args:
            num_layers: number of attention layers
            num_channels: number of channels per degree
            num_degrees: number of degrees (aka types) in hidden layer, count start from type-0
            div: (int >= 1) keys, queries and values will have (num_channels/div) channels
            n_heads: (int >= 1) for multi-headed attention
            selfint: ['1x1', 'att'] type of self-interaction in hidden layers
            x_ij: ['add', 'cat'] use relative position as edge feature
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.num_layers = num_layers
        self.edge_dim = 1
        self.div = div
        self.n_heads = n_heads
        self.selfint = selfint
        self.x_ij = x_ij

        self.Gblock = self._build_gcn(self.f_in, self.f_out)
        

    def _build_gcn(self, f_in, f_out):
        # Equivariant layers
        Gblock = []
        fin = f_in
        for _ in range(self.num_layers):
            Gblock.append(GSE3Res(fin, f_out, edge_dim=self.edge_dim, div=self.div, n_heads=self.n_heads,
                                  learnable_skip=True, skip='cat', selfint=self.selfint, x_ij=self.x_ij))
            Gblock.append(GNormBias(f_out))
            fin = f_out
 
        return nn.ModuleList(Gblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.f_out.max_degree)
        h_enc = {'1': G.ndata['v']}
        #print(h_enc)
        for layer in self.Gblock:
            h_enc = layer(h_enc, G=G, r=r, basis=basis)

        return h_enc['1']
    

class SE3UPointnet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = SE3ConvBlock(
            f_in= Fiber(dictionary={1: 1}),
            f_out= Fiber(4, 2),
            num_layers=2,
            n_heads=1,
            selfint='att'
        )
        self.pooling1 = Pooling3D(
            in_features=3,
            pooling_ratio=0.2,
            aggr='max'
        )


        self.conv2 = SE3ConvBlock(
            f_in= Fiber(dictionary={1: 2}),
            f_out= Fiber(4, 4),
            num_layers=2,
            n_heads=1,
            selfint='att'
        )
        self.pooling2 = Pooling3D(
            in_features=3,
            pooling_ratio=0.2,
            aggr='max'
        )
        

        self.conv3 = SE3ConvBlock(
            f_in= Fiber(dictionary={1: 4}),
            f_out= Fiber(4, 8),
            num_layers=2,
            n_heads=1,
            selfint='att'
        )
        self.pooling3 = Pooling3D(
            in_features=3,
            pooling_ratio=0.2,
            aggr='max'
        )
      
        self.conv4 = SE3ConvBlock(
            f_in= Fiber(dictionary={1: 8}),
            f_out= Fiber(4, 16),
            num_layers=2,
            n_heads=1,
            selfint='att'
        )
        self.pooling4 = Pooling3D(
            in_features=3,
            pooling_ratio=0.2,
            aggr='max'
        )


        self.conv5 = SE3ConvBlock(
            f_in= Fiber(dictionary={1: 16}),
            f_out= Fiber(4, 32),
            num_layers=2,
            n_heads=1,
            selfint='att'
        )
        self.pooling5 = Pooling3D(
            in_features=3,
            pooling_ratio=0.2,
            aggr='max'
        )
       
       

        self.upsampler = Upsampling3D(
            in_features=3,
            power=2
        )

    def forward(self, G: dgl.graph, features: str, batch_size: int=1):
        G.ndata[features] = self.conv1(G)
        G_pooled1, G_level_structure1, fp_idx1 = self.pooling1(G, features, batch_size)

        G_pooled1.ndata[features] = self.conv2(G_pooled1)
        G_pooled2, G_level_structure2, fp_idx2 = self.pooling2(G_pooled1, features, batch_size)

        G_pooled2.ndata[features] = self.conv3(G_pooled2)
        G_pooled3, G_level_structure3, fp_idx3 = self.pooling3(G_pooled2, features, batch_size)

        G_pooled3.ndata[features] = self.conv4(G_pooled3)
        G_pooled4, G_level_structure4, fp_idx4 = self.pooling4(G_pooled3, features, batch_size)

        

        G_upsampled4 = self.upsampler(G_pooled4, features, G_level_structure4, fp_idx4)
        G_upsampled3 = self.upsampler(G_upsampled4, features, G_level_structure3, fp_idx3)
        G_upsampled2 = self.upsampler(G_upsampled3, features, G_level_structure2, fp_idx2)
        G_upsampled1 = self.upsampler(G_upsampled2, features, G_level_structure1, fp_idx1)

        

        return G_upsampled1



