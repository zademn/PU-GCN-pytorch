from typing import Tuple
import torch
from torch.nn import Sequential, Linear, ReLU
from .feature_extractor import (
    InceptionFeatureExtractor,
    InceptionPointTransformer,
    InceptionPointTransformerExtractor,
)
from .upsample import NodeShuffle
from pugcn_lib.torch_nn import MLP
from gcn_lib.sparse.torch_vertex import DynConv, GraphConv
from gcn_lib.sparse.torch_edge import DilatedKnnGraph, Dilated
from torch_geometric.nn import radius_graph, PointTransformerConv, global_max_pool

# --------------------------------------------------
# ---------------- PUGCN  --------------------------
# --------------------------------------------------


class PUGCN(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        dilations: Tuple[int],
        r: int,
        n_idgcn_blocks: int,
        n_dgcn_blocks: int,
        **idgcn_kwargs,
    ):
        """
        Parameters:
        ----------
        channels: int
            number of channels. Used in InceptionFeatureExtractor, NodeShuffle and the reconstructor

        k: int
            number of neighbours for constructing the knn graph

        dilations: Tuple[int]
            the dilations that the DenseGCNs from each InceptionDenseGCN block will have.

        n_idgcn_blocks: int
            number of Inception DenseGCN blocks in the feature extractor

        n_dgcn_blocks: int
            number of  DenseGCN blocks in each InceptionDenseGCN block in the feature extractor

        **idgcn_kwargs: args
            Key arguments for `InceptionDenseGCN`:
            use_bottleneck: bool, default = False
            use_pooling: bool, default = False
            use_residual: bool, default = False
            conv: str, default = "edge"
            pool_type: str, default "max"
            dynamic: bool, default = False.
        """

        super(PUGCN, self).__init__()

        # Config
        # Layers
        self.feature_extractor = InceptionFeatureExtractor(
            channels=channels,
            k=k,
            dilations=dilations,
            n_idgcn_blocks=n_idgcn_blocks,
            n_dgcn_blocks=n_dgcn_blocks,
            **idgcn_kwargs,
        )
        self.upsampler = NodeShuffle(in_channels=channels, k=k, r=r)

        self.reconstructor = MLP([channels, channels, 3])

    def forward(self, x, batch=None):
        """
        Parameters:
        ----------
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, C]

        batch: Optional[LongTensor]
            batch tensor [N, ] as described in PyG docs.
            For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]

        """
        x = self.feature_extractor(x, batch=batch)
        x = self.upsampler(x, batch=batch)
        x = self.reconstructor(x)

        return x


# --------------------------------------------------
# ---------------- Transformers --------------------
# --------------------------------------------------


class PUInceptionTransformer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        dilations: Tuple[int],
        r: int,
        n_ipt_blocks: int,
        use_refiner: Tuple[bool, torch.nn.Module] = False,
        **ipt_kwargs,
    ):
        """
        Parameters:
        ----------
        channels: int
            number of channels. Used in InceptionFeatureExtractor, NodeShuffle and the reconstructor

        k: int
            number of neighbours for constructing the knn graph

        dilations: Tuple[int]
            the dilations that the DenseGCNs from each InceptionDenseGCN block will have.

        n_ipt_blocks: int
            Number of InceptionPointTransformer blocks

        refine: bool | torch.nn.Module
          True - Add a RefinerTransfromer
          torch.nn.Module - add your own refiner

        **ipt_kwargs: args
            Key arguments for `InceptionPointTransformer`:
            use_bottleneck: bool, default = False
            use_pooling: bool, default = False
            use_residual: bool, default = False
            conv: str, default = "edge"
            pool_type: str, default "max"
            dynamic: bool, default = False.
        """

        super(PUInceptionTransformer, self).__init__()

        # Config
        self.k = k
        self.r = r
        # Layers
        self.feature_extractor = InceptionPointTransformerExtractor(
            channels=channels,
            k=k,
            dilations=dilations,
            n_ipt=n_ipt_blocks,
            **ipt_kwargs,
        )
        self.upsampler = NodeShuffle(
            in_channels=channels, out_channels=channels, k=k, r=r
        )
        self.reconstructor = MLP([channels, channels, 3])

        if isinstance(use_refiner, torch.nn.Module):
            self.refiner = use_refiner
        elif use_refiner is True:
            self.refiner = RefinerTransformer(
                in_channels=channels, out_channels=3, k=k, dilation=1
            )
        else:
            self.refiner = None

    def forward(self, x, batch=None):
        """
        Parameters:
        ----------
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, C]

        batch: Optional[LongTensor]
            batch tensor [N, ] as described in PyG docs.
            For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]

        """
        x = self.feature_extractor(x, batch=batch)
        x = self.upsampler(x, batch=batch)
        q = self.reconstructor(x)

        if self.refiner is not None:
            # Compute batch here or take the batch from the upsampler.
            if batch is not None:
                batch_ = batch.repeat_interleave(self.r)
            else:
                batch_ = None
            res = self.refiner(x, pos=q, batch=batch_)
        else:
            res = q

        return res


# --------------------------------------------------
# ---------------- Basic ---------------------------
# --------------------------------------------------


class PUGNN(torch.nn.Module):
    def __init__(
        self,
        k: int,
        r: int,
        extractor_channels: list,
        reconstructor_channels: list,
        dilation: list,
        dynamic: bool = False,
        **conv_kwargs,
    ):
        super(PUGNN, self).__init__()

        # Config
        self.dynamic = dynamic
        # Layers
        assert len(extractor_channels) > 0
        assert len(reconstructor_channels) > 0

        if dynamic:
            self.knn = DilatedKnnGraph(k=k, d=dilation)

        self.feature_extractor = torch.nn.ModuleList(
            [
                DynConv(
                    in_channels=t1,
                    out_channels=t2,
                    kernel_size=k,
                    dilation=dilation,
                    **conv_kwargs,
                )
                for t1, t2, d in zip([3, *extractor_channels[1:]], extractor_channels)
            ]
        )

        self.upsampler = NodeShuffle(in_channels=extractor_channels[-1], k=k, r=r)
        self.reconstructor = MLP([*reconstructor_channels, 3])

    def forward(self, x, batch=None):
        if not self.dynamic:
            edge_index = self.knn(x, batch=batch)  # Compute initial graph
        else:
            edge_index = None  # If edge_index=None it will be computed at each layer
        for layer in self.feature_extractor:
            x = layer(x, edge_index=edge_index, batch=batch)
        x = self.upsampler(x, batch=batch)
        x = self.reconstructor(x)

        return x


class PUGNNRadius(torch.nn.Module):
    def __init__(
        self,
        k: int,
        r: int,
        extractor_channels: list,
        reconstructor_channels: list,
        radius,
        dynamic: bool = False,
        **conv_kwargs,
    ):
        super(PUGNNRadius, self).__init__()

        # Config
        self.radius = radius
        self.dynamic = dynamic
        # Layers
        assert len(extractor_channels) > 0
        assert len(reconstructor_channels) > 0
        self.feature_extractor = torch.nn.ModuleList(
            [
                GraphConv(in_channels=t1, out_channels=t2, **conv_kwargs)
                for t1, t2 in zip([3, *extractor_channels[1:]], extractor_channels)
            ]
        )
        self.upsampler = NodeShuffle(in_channels=extractor_channels[-1], k=k, r=r)
        self.reconstructor = MLP([*reconstructor_channels, 3])

    def forward(self, x, batch=None):

        if not self.dynamic:
            edge_index = radius_graph(x, self.radius)
        for layer in self.feature_extractor:
            if self.dynamic:
                edge_index = radius_graph(x, self.radius)
            x = layer(x, edge_index=edge_index)
        x = self.upsampler(x, batch=batch)
        x = self.reconstructor(x)

        return x


# --------------------------------------------------
# ---------------- Refined -------------------------
# --------------------------------------------------


class RefinerTransformer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        dilation: int,
        **transformer_kwargs,
    ):
        super(RefinerTransformer, self).__init__()

        self.knn = DilatedKnnGraph(k=k, dilation=dilation)
        self.transformer = PointTransformerConv(
            in_channels=in_channels, out_channels=in_channels, **transformer_kwargs
        )
        self.pool = global_max_pool
        self.mlp = MLP([2 * in_channels, out_channels])

    def forward(self, x, pos, batch=None):
        edge_index = self.knn(x, batch=batch)
        h1 = self.transformer(x=x, pos=pos, edge_index=edge_index)

        if batch is None:  # This is fixed in PyG 2.0.4
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        h2 = self.pool(x, batch=batch)[batch]
        h = torch.cat([h1, h2], axis=-1)
        res = self.mlp(h)
        return res


class PUGCNRefinedTransformer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        dilations: Tuple[int],
        r: int,
        n_idgcn_blocks: int,
        n_dgcn_blocks: int,
        **idgcn_kwargs,
    ):
        """
        Parameters:
        ----------
        channels: int
            number of channels. Used in InceptionFeatureExtractor, NodeShuffle and the reconstructor

        k: int
            number of neighbours for constructing the knn graph

        dilations: Tuple[int]
            the dilations that the DenseGCNs from each InceptionDenseGCN block will have.

        n_idgcn_blocks: int
            number of Inception DenseGCN blocks in the feature extractor

        n_dgcn_blocks: int
            number of  DenseGCN blocks in each InceptionDenseGCN block in the feature extractor

        **idgcn_kwargs: args
            Key arguments for `InceptionDenseGCN`:
            use_bottleneck: bool, default = False
            use_pooling: bool, default = False
            use_residual: bool, default = False
            conv: str, default = "edge"
            pool_type: str, default "max"
            dynamic: bool, default = False.
        """

        super(PUGCNRefinedTransformer, self).__init__()

        # Config
        # Layers
        self.feature_extractor = InceptionFeatureExtractor(
            channels=channels,
            k=k,
            dilations=dilations,
            n_idgcn_blocks=n_idgcn_blocks,
            n_dgcn_blocks=n_dgcn_blocks,
            **idgcn_kwargs,
        )
        self.upsampler = NodeShuffle(
            in_channels=channels, out_channels=channels, k=k, r=r
        )

        self.reconstructor = MLP([channels, channels, 3])
        self.refiner = RefinerTransformer(
            in_channels=channels, out_channels=3, k=k, dilation=1
        )

    def forward(self, x, batch=None):
        """
        Parameters:
        ----------
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, C]

        batch: Optional[LongTensor]
            batch tensor [N, ] as described in PyG docs.
            For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]

        """
        x = self.feature_extractor(x, batch=batch)
        x = self.upsampler(x, batch=batch)
        p = self.reconstructor(x)
        res = self.refiner(x=x, pos=p, batch=batch)

        return res
