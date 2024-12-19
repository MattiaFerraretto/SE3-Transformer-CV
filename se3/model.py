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


class SE3Unet(nn.Module):

    def __init__(
            self,
            n_layers: int,
            si_m: str,
            si_e: str,
            in_features: int,
            hidden_channels: int,
            out_features: int,
            pooling_ratio: float,
            aggr: str
        ):
        super().__init__()
        self.n_layers = n_layers
        self.si_m = si_m
        self.si_e = si_e
        self.in_features = in_features
        self.hidden_channels = hidden_channels
        self.out_features = out_features
        self.pooling_ratio = pooling_ratio
        self.aggr = aggr

        self.down_branch, self.bottleneck, self.up_branch, self.mlp = self._build_unet(n_layers)


    def _build_unet(self, n_layers):
        down_branch = []
        in_channels = 1
        out_channels = self.hidden_channels

        for _ in range(n_layers):
            down_branch.append(
                SE3ConvBlock(
                    f_in=Fiber(dictionary={1: in_channels}),
                    f_out=Fiber(3, out_channels),
                    num_layers=2,
                    n_heads=1,
                    selfint=self.si_m
                )
            )
            down_branch.append(
                Pooling3D(
                    in_features=self.in_features,
                    pooling_ratio=self.pooling_ratio,
                    aggr=self.aggr
                )
            )

            in_channels = out_channels
            #out_channels = in_channels*2
        
        #out_channels = in_channels
        bottleneck =  SE3ConvBlock(
            f_in=Fiber(dictionary={1: in_channels}),
            f_out=Fiber(3, out_channels),
            num_layers=2,
            n_heads=1,
            selfint=self.si_m
        )

        up_branch = []
        #out_channels = out_channels//2
        
        for _ in range(n_layers-1):
            up_branch.append(
                Upsampling3D(
                    in_features=self.in_features,
                    power=2
                )
            )
            up_branch.append(
                SE3ConvBlock(
                    f_in=Fiber(dictionary={1: in_channels}),
                    f_out=Fiber(3, out_channels),
                    num_layers=2,
                    n_heads=1,
                    selfint=self.si_m
                )
            )
            
            #in_channels = out_channels
            #out_channels = out_channels//2

        up_branch.append(
            Upsampling3D(
                in_features=self.in_features,
                power=2
            )
        )
        up_branch.append(
            SE3ConvBlock(
                f_in=Fiber(dictionary={1: in_channels}),
                f_out=Fiber(3, 1),
                num_layers=2,
                n_heads=1,
                selfint=self.si_e
            )
        )

        mlp = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=False),
            nn.ReLU(),
            #nn.Linear(in_features=self.out_features, out_features=self.out_features, bias=False),
            #nn.ReLU(),
        )
        
        return nn.ModuleList(down_branch), bottleneck, nn.ModuleList(up_branch), mlp
    
    def forward(self, G: dgl.graph, features: str, batch_size: int=1):
        res = []

        for i in range(0, self.n_layers*2, 2):
            G.ndata[features] = self.down_branch[i](G)

            G_pooled, G_level_structure, fp_idx = self.down_branch[i+1](G, features, batch_size)
            res.append((G, G_level_structure, fp_idx))

            G = G_pooled

        G.ndata[features] = self.bottleneck(G)

        res_idx = -1
        for i in range(0, self.n_layers*2, 2):
            G_res, G_level_structure, fp_idx = res[res_idx]

            G_upsmapled = self.up_branch[i](G, features, G_level_structure, fp_idx)

            G_upsmapled.ndata[features] = G_res.ndata[features] + G_upsmapled.ndata[features]

            G_upsmapled.ndata[features] = self.up_branch[i+1](G_upsmapled)

            G = G_upsmapled
            res_idx = res_idx - 1

        x = G.ndata[features].view(-1, G.ndata[features].size(0) // batch_size, self.in_features)
        y_hat = self.mlp(x)

        return y_hat

class SE3UnetV2(nn.Module):

    def __init__(
            self,
            n_layers: int,
            si_m: str,
            si_e: str,
            in_features: int,
            hidden_channels: int,
            out_features: int,
            pooling_ratio: float,
            aggr: str
        ):
        super().__init__()
        self.n_layers = n_layers
        self.si_m = si_m
        self.si_e = si_e
        self.in_features = in_features
        self.hidden_channels = hidden_channels
        self.out_features = out_features
        self.pooling_ratio = pooling_ratio
        self.aggr = aggr

        self.down_branch, self.bottleneck, self.up_branch, self.mlp = self._build_unet(n_layers)


    def _build_unet(self, n_layers):
        down_branch = []

        for _ in range(n_layers):
            down_branch.append(
                SE3Transformer(
                    num_layers=2,
                    num_channels=self.hidden_channels,
                    num_degrees=3,
                    div=1,
                    n_heads=1,
                    si_m=self.si_m,
                    si_e=self.si_m,
                    x_ij= 'add'
                )
            )
            down_branch.append(
                Pooling3D(
                    in_features=self.in_features,
                    pooling_ratio=self.pooling_ratio,
                    aggr=self.aggr
                )
            )

        
        bottleneck = SE3Transformer(
            num_layers=2,
            num_channels=self.hidden_channels,
            num_degrees=3,
            div=1,
            n_heads=1,
            si_m=self.si_m,
            si_e=self.si_m,
            x_ij= 'add'
        )

        up_branch = []
        
        for _ in range(n_layers-1):
            up_branch.append(
                Upsampling3D(
                    in_features=self.in_features,
                    power=2
                )
            )
            up_branch.append(
                SE3Transformer(
                    num_layers=2,
                    num_channels=self.hidden_channels,
                    num_degrees=3,
                    div=1,
                    n_heads=1,
                    si_m=self.si_m,
                    si_e=self.si_m,
                    x_ij= 'add'
                )
            )
            
        up_branch.append(
            Upsampling3D(
                in_features=self.in_features,
                power=2
            )
        )
        up_branch.append(
            SE3Transformer(
                num_layers=2,
                num_channels=self.hidden_channels,
                num_degrees=3,
                div=1,
                n_heads=1,
                si_m=self.si_m,
                si_e=self.si_e,
                x_ij= 'add'
            )
        )

        mlp = nn.Sequential(
            #nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=False),
            nn.Linear(in_features=self.in_features, out_features=128, bias=True),
            #nn.Sigmoid()
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.out_features, bias=True),
            #nn.Linear(in_features=self.out_features, out_features=self.out_features, bias=False),
            #nn.ReLU(),
        )
        
        return nn.ModuleList(down_branch), bottleneck, nn.ModuleList(up_branch), mlp
    
    def forward(self, G: dgl.graph, features: str, batch_size: int=1):
        res = []

        for i in range(0, self.n_layers*2, 2):
            G.ndata[features] = self.down_branch[i](G)

            G_pooled, G_level_structure, fp_idx = self.down_branch[i+1](G, features, batch_size)
            res.append((G, G_level_structure, fp_idx))

            G = G_pooled

        G.ndata[features] = self.bottleneck(G)

        res_idx = -1
        for i in range(0, self.n_layers*2, 2):
            G_res, G_level_structure, fp_idx = res[res_idx]

            G_upsmapled = self.up_branch[i](G, features, G_level_structure, fp_idx)

            G_upsmapled.ndata[features] = G_res.ndata[features] + G_upsmapled.ndata[features]

            G_upsmapled.ndata[features] = self.up_branch[i+1](G_upsmapled)

            G = G_upsmapled
            res_idx = res_idx - 1

        x = G.ndata[features].view(-1, G.ndata[features].size(0) // batch_size, self.in_features)
        return x
        y_hat = self.mlp(x)

        return y_hat