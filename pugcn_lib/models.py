from typing import Tuple, Union
import torch
from torch.nn import Sequential, Linear, ReLU
from .feature_extractor import (
    InceptionFeatureExtractor,
    InceptionPointTransformer,
    InceptionPointTransformerExtractor,
)
from .upsample import GeneralUpsampler, PointShuffle
from pugcn_lib.torch_nn import MLP
from gcn_lib.sparse.torch_vertex import DynConv, GraphConv, EdgConv
from gcn_lib.sparse.torch_edge import DilatedKnnGraph, Dilated
from torch_geometric.nn import (
    radius_graph,
    PointTransformerConv,
    global_max_pool,
    global_mean_pool,
    global_add_pool,
)
from torch_geometric.utils import to_dense_batch


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
        use_radius_graph: bool = False,
        use_refiner: Union[bool, torch.nn.Module] = False,
        upsampler: str = "nodeshuffle",
        **idgcn_kwargs,
    ):
        """

        PUGCN module that uses InceptionDenseGCN as backbone

        Parameters:
        ----------
        channels: int
            number of channels. Used in InceptionFeatureExtractor, NodeShuffle and the reconstructor

        k: int
            number of neighbours for constructing the knn graph

        dilations: Tuple[int]
            dilations that the DenseGCNs from each InceptionDenseGCN block will have.

        r: int
            upsampling ratio

        n_idgcn_blocks: int
            number of Inception DenseGCN blocks in the feature extractor

        n_dgcn_blocks: int
            number of  DenseGCN blocks in each InceptionDenseGCN block in the feature extractor

        use_radius_graph: bool = False,
            True - will use radius_graph instead of knn graph to compute edge_index.
                k will become max_num_neighbours and dilations will become the radius.

        use_refiner: bool | torch.nn.Module
          True - Add a RefinerTransfromer
          False - No refiner used
          torch.nn.Module - add your own refiner

        upsampler: str, default = "nodeshuffle"
            The upsampler to use. One of ["nodeshuffle", "duplicate"]

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
        self.r = r
        self.k = k
        # Layers
        self.feature_extractor = InceptionFeatureExtractor(
            channels=channels,
            k=k,
            dilations=dilations,
            n_idgcn_blocks=n_idgcn_blocks,
            n_dgcn_blocks=n_dgcn_blocks,
            use_radius_graph=use_radius_graph,
            **idgcn_kwargs,
        )
        self.upsampler = GeneralUpsampler(
            in_channels=channels,
            out_channels=channels,
            k=k,
            r=r,
            upsampler=upsampler,
            conv=idgcn_kwargs["conv"],
        )

        self.reconstructor = torch.nn.Sequential(
            torch.nn.Linear(channels, channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(channels, 3),
        )

        # Init refiner
        if isinstance(use_refiner, torch.nn.Module):
            self.refiner = use_refiner
        elif use_refiner is True:
            self.refiner = Refiner(
                in_channels=channels,
                out_channels=3,
                k=r * k,
                dilations=[1],
                conv="point_transformer",
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
        # edge_index haas k nodes, not k * max(dilations)
        pos = x
        x, edge_index = self.feature_extractor(
            x, pos=pos, batch=batch, return_index=True
        )
        x = self.upsampler(x, edge_index=edge_index, pos=pos, batch=batch)
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

        return x

    @torch.no_grad()
    def predict_unrefined(self, x, batch=None):
        x, edge_index = self.feature_extractor(x, batch=batch, return_index=True)
        x = self.upsampler(x, edge_index=edge_index, batch=batch)
        q = self.reconstructor(x)
        return q


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
        use_refiner: Union[bool, torch.nn.Module] = False,
        upsampler: str = "nodeshuffle",
        **ipt_kwargs,
    ):
        """
        PUGCN module that uses InceptionPointTransformer as backbone

        Parameters:
        ----------
        channels: int
            number of channels. Used in InceptionFeatureExtractor, NodeShuffle and the reconstructor

        k: int
            number of neighbours for constructing the knn graph

        dilations: Tuple[int]
            Dilatations for InceptionPointTransformer

        n_ipt_blocks: int
            Number of InceptionPointTransformer blocks

        use_refiner: bool | torch.nn.Module
          True - Add a RefinerTransfromer
          torch.nn.Module - add your own refiner

        upsampler: str, default = "nodeshuffle"
            The upsampler to use. One of ["nodeshuffle", "duplicate"]

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
        self.upsampler = GeneralUpsampler(
            in_channels=channels,
            out_channels=channels,
            r=r,
            k=k,
            upsampler=upsampler,
        )

        self.reconstructor = torch.nn.Sequential(
            torch.nn.Linear(channels, channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(channels, 3),
        )

        # Init refiner
        if isinstance(use_refiner, torch.nn.Module):
            self.refiner = use_refiner
        elif use_refiner is True:
            self.refiner = Refiner(
                in_channels=channels, out_channels=3, k=k, dilations=dilations
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
        x, edge_index = self.feature_extractor(x, batch=batch, return_index=True)
        x = self.upsampler(x, edge_index=edge_index, batch=batch)
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

    @torch.no_grad()
    def predict_unrefined(self, x, batch=None):
        x, edge_index = self.feature_extractor(x, batch=batch, return_index=True)
        x = self.upsampler(x, edge_index=edge_index, batch=batch)
        q = self.reconstructor(x)
        return q


# --------------------------------------------------
# ---------------- JustUpsample --------------------
# --------------------------------------------------


class JustUpsample(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        dilations: list[int],
        r: int,
        upsampler: str = "nodeshuffle",
        conv="edge",
        hierarchical: bool = False,
        use_global: bool = True,
        use_bottleneck: bool = True,
        use_refiner: Union[bool, torch.nn.Module] = False,
    ):
        """

        PUGCN module that uses InceptionDenseGCN as backbone

        Parameters:
        ----------
        channels: int
            number of channels

        k: int
            number of neighbours for constructing the knn graph

        dilations: list[int]
            the dilations that the DenseGCNs from each InceptionDenseGCN block will have.

        r: int
            upsampling ratio

        conv: str, default = "edge"

        upsampler: str, default = "nodeshuffle"
            The upsampler to use. One of ["nodeshuffle", "duplicate"]

        use_refiner: bool | torch.nn.Module
          True - Add a RefinerTransfromer
          False - No refiner used
          torch.nn.Module - add your own refiner

        use_bottleneck: bool, default = True
            True - Use a bottleneck MLP layer to reduce the dimensionality of the features

        use_global: bool, default = True
            True - Use the global layer

        """
        super().__init__()

        # Config
        self.r = r
        self.dilations = dilations
        self.use_global = use_global
        self.use_bottleneck = use_bottleneck
        self.k = k
        self.channels = channels
        # Layers
        self.knn = DilatedKnnGraph(k=k * max(dilations), dilation=1)
        self.knn1 = Dilated(k=k, dilation=max(dilations))
        self.pre_gcn = GraphConv(
            in_channels=3,
            out_channels=channels,
            conv="edge",
        )

        div = len(dilations) + use_global
        # Set bottleneck
        if use_bottleneck:
            # We need to set the input shape of the DenseGCN as the output of MLP
            assert (
                channels % div == 0
            ), "The number of in_channels must be divisible by the number of DenseGCN layers + pooling (if used)"
            channels_upsampler = channels // div
            # assert(growth_rate % n_blocks == 0, "the number of blocks")
            self.bottleneck = MLP(
                [channels, channels_upsampler],
                act="leaky_relu",
                act_kwargs={"negative_slope": 0.2},
            )
        else:
            channels_upsampler = channels

        self.knns = torch.nn.ModuleList(
            [Dilated(k=k, dilation=d, hierarchical=hierarchical) for d in dilations]
        )
        self.upsamplers = torch.nn.ModuleList(
            [
                GeneralUpsampler(
                    in_channels=channels_upsampler,
                    out_channels=channels,
                    k=k,
                    r=r,
                    conv=conv,
                )
                for _ in dilations
            ]
        )
        # Global shuffle
        if use_global:
            self.ps = PointShuffle(r=r)
            self.lin_global1 = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=channels_upsampler + 3, out_features=channels * r
                )
            )
            self.lin_global2 = torch.nn.Sequential(
                torch.nn.Linear(in_features=channels, out_features=channels)
            )

        self.reconstructor = torch.nn.Sequential(
            torch.nn.Linear(channels, channels),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(channels, 3),
        )

        if isinstance(use_refiner, torch.nn.Module):
            self.refiner = use_refiner
        elif use_refiner is True:
            self.refiner = Refiner(
                in_channels=channels, out_channels=3, k=k, dilations=dilations
            )
        else:
            self.refiner = None

    def forward(self, x, batch=None):
        # keep initial positions
        pos = x
        # Compute initial edge_index
        edge_index = self.knn(x, batch=batch)  # [N * k * d, 2]
        # Get initial neighbourhood to apply pre feature extraction
        edge_index_1 = self.knn1(
            edge_index=edge_index,
            batch=batch,
            k_constructed=self.k * max(self.dilations),
        )  # [N * k, 2]
        x = self.pre_gcn(x, edge_index=edge_index_1)  # [N, 3] ->  [N, C]

        if self.use_bottleneck:
            x = self.bottleneck(x)
        res = torch.zeros((x.shape[0] * self.r, self.channels)).to(x.device)
        for knn, upsampler in zip(self.knns, self.upsamplers):
            ei = knn(
                edge_index=edge_index,
                batch=batch,
                k_constructed=self.k * max(self.dilations),
            )
            h = upsampler(x, edge_index=ei, pos=pos, batch=batch)
            res += h
        # Global shuffle
        if self.use_global:
            h = torch.cat([x, pos], axis=-1)
            h = self.lin_global1(h)
            h = self.ps(h, batch=batch)
            h = self.lin_global2(h)
            res += h

        x = res / (len(self.dilations) + self.use_global)
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
# ---------------- Refiners ------------------------
# --------------------------------------------------


class Refiner(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        dilations: int,
        hierarchical: bool = False,
        add_points: bool = True,
        conv="point_transformer",
        **conv_kwargs,
    ):
        """Refiner that uses PointTransformer for local Points and a global_max_pool for global features

        Parameters
        ----------
        in_channels : int
            Input channels for the PointTransformer

        out_channels : int
            Output channels for the PointTransformer

        k: int
            number of neighbours for constructing the knn graph

        dilations: int
            Dilatations for the point transfrormer

        add_points : bool, default=True
            True -  add initial points after refinement

        """

        super().__init__()
        # Config
        self.add_points = add_points
        self.dilations = dilations
        self.k = k
        # Layers

        self.knn = DilatedKnnGraph(k=k * max(dilations), dilation=1)
        self.knns = torch.nn.ModuleList(
            [Dilated(k=k, dilation=d, hierarchical=hierarchical) for d in dilations]
        )

        self.layers = torch.nn.ModuleList(
            [
                GraphConv(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    **conv_kwargs,
                )
                for _ in dilations
            ]
        )

        # This might be expensive af.
        # self.transformer_global = PointTransformerConv(
        #     in_channels=in_channels, out_channels=in_channels, **transformer_kwargs
        # )
        # self.pool = global_max_pool
        self.lin_global = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channels + 3, out_features=in_channels),
            torch.nn.LeakyReLU(0.2),
        )

        self.lin = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channels, out_features=in_channels),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(in_features=in_channels, out_features=3),
        )

    def forward(self, x, pos, batch=None):
        # Local features
        edge_index = self.knn(pos, batch=batch)
        h = None
        for knn, layer in zip(self.knns, self.layers):
            edge_index_ = knn(edge_index, k_constructed=self.k * max(self.dilations))
            h_ = layer(x=x, pos=pos, edge_index=edge_index_)
            if h is None:
                h = h_
            else:
                h = h + h_

        # Global features
        # t = torch.arange(256)
        # edge_index2 = torch.combinations(t, with_replacement=True).T.to(x.device)
        # h2 = self.transformer_global(x=x, pos=pos, edge_index=edge_index2)

        # Global features
        # if batch is None:  # This is fixed in PyG 2.0.4
        #     batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        # h2 = self.pool(x, batch=batch)[batch]
        h2 = torch.cat([x, pos], dim=-1)  # [N, C + 3]
        h2 = self.lin_global(h2)

        h = h + h2
        res = self.lin(h)  # {N, 3}

        if self.add_points:
            res = res + pos
        return res
