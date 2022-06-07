import torch
from typing import Optional, List, Tuple
from torch_cluster import knn_graph
from gcn_lib.sparse.torch_vertex import DenseGraphBlock, EdgConv, DynConv, GraphConv
from gcn_lib.sparse.torch_edge import DilatedKnnGraph, Dilated
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from .torch_nn import MLP
from torch_geometric.nn import PointTransformerConv, radius_graph


class DenseGCN(torch.nn.Module):
    def __init__(
        self,
        n_blocks: int,
        in_channels: int,
        out_channels: int,
        growth_rate: int,
        conv: str = "edge",
        **kwargs,
    ):
        """
        Parameters:
        ----------
        n_blocks: int
            The number of DenseGraphBlock blocks the DenseGCN will have

        in_channels: int
            The number of input channels
        out_channels: int
            The number of output channels

        growth_rate: int
            The output channels for each DenseGraphBlock and
            with how much the in_channels of the next block will grow
            After applying all the DenseGraphBlocks the output will have in_channels + n_blocks * growth_rate

        conv: str = "edge"
            Convolution type in DenseGraphBlock
            One of ["edge", "mr", "gat", "gcn", "gin", "sage", "rsage"]

        **kwargs: key arguments
            Key arguments for DenseGraphBlock

        """
        super(DenseGCN, self).__init__()

        # Config
        self.n_blocks = n_blocks
        self.growth_rate = growth_rate
        self.in_channels = in_channels
        self.out_channels = out_channels

        # GCN blocks
        self.blocks = torch.nn.ModuleList(
            [
                DenseGraphBlock(
                    in_channels=self.in_channels + i * self.growth_rate,
                    out_channels=self.growth_rate,
                    conv=conv,
                    **kwargs,
                )
                for i in range(self.n_blocks)
            ]
        )

        self.lin_down = torch.nn.Linear(
            in_features=in_channels + growth_rate * n_blocks, out_features=out_channels
        )

    def forward(self, x, edge_index, pos=None):
        # Apply DenseGCN blocks
        for block in self.blocks:
            x, _ = block(x, edge_index, pos)
        # Downscale from in_channels + growth_rate * n_blocks to out_channels
        res = self.lin_down(x)
        return res


# --------------------------------------------------
# ---------------- PUGCN --------------------
# --------------------------------------------------


class InceptionDenseGCN(torch.nn.Module):
    def __init__(
        self,
        n_blocks: int,
        in_channels: int,
        k: int,
        dilations: Tuple[int],
        use_bottleneck: bool = False,
        use_pooling: bool = False,
        use_residual: bool = False,
        conv: str = "edge",
        pool_type: str = "max",
        hierarchical: bool = False,
        dynamic: bool = False,
        **kwargs,
    ):
        """The inception GCN is formed from parallel DenseGCNs that use different dilations

        Parameters:
        ----------
        in_channels: int
            Input channels + It's used to compute growth rate for DenseGCN.

        k: int
            num neighbours

        dilations: Tuple[int]
            a list containing the dilation for each DenseGCN layer.

        n_blocks: int
            number of blocks each DenseGCN will have

        use_bottleneck: bool, default = False
            True - Applies a bottleneck 1 layer MLP with dimensions
            [in_channels, growth_rate / (n_blocks + use_pooling)].
            Also switches to concatenating instead of pooling the outputs of the DenseGCNs

        use_pooling: bool, default = False
            True - applies a `global_max_pool` and in parallel to the DenseGCN

        use_residual: bool, default = False
            True - adds the inputs to the result

        conv: str, default = "edge"
            Convolution type in DenseGraphBlock
            One of ["edge", "mr", "gat", "gcn", "gin", "sage", "rsage"]

        pool_type: str, default = "max"
            global pooling type if `use_pooling == True`.
            One of ["max", "mean", "add"]

        dynamic: bool, default = False
            True - The knn-graph will be computed for each dilation
            False - The knn-graph is provided at inference time => Must pass edge_index as a parameter.

        **kwargs: key arguments
            DenseGCN kwargs.

        Remark:
            - The if `use_bottleneck` is True then `in_channels % t == 0` where `t = len(dilations) + use_pooling`

        """
        super(InceptionDenseGCN, self).__init__()

        # Config
        self.k = k
        self.dilations = dilations  # hardcoded for now
        self.n_blocks = n_blocks
        self.use_bottleneck = use_bottleneck
        self.use_pooling = use_pooling
        self.use_residual = use_residual
        self.dynamic = dynamic
        # Layers

        div = len(dilations)  # Number of dense_gcn
        # Pool layer
        if use_pooling:
            if pool_type == "max":
                self.pool = global_max_pool
            elif pool_type == "mean":
                self.pool = global_mean_pool
            elif pool_type == "add":
                self.pool = global_add_pool
            else:
                raise ValueError('`pool_type` must be one of ["max", "mean", "add"]')
            div += 1
        # MLP bottleneck
        if use_bottleneck:
            # We need to set the input shape of the DenseGCN as the output of MLP
            assert (
                in_channels % div == 0
            ), "The number of in_channels must be divisible by the number of DenseGCN layers + pooling (if used)"
            channels_dgcn = in_channels // div
            # assert(growth_rate % n_blocks == 0, "the number of blocks")
            self.bottleneck = MLP(
                [in_channels, channels_dgcn],
                act="leaky_relu",
                act_kwargs={"negative_slope": 0.2},
            )

        else:
            channels_dgcn = in_channels  # Else use the in channels

        # in_channels = growth_rate because we apply a reduce at the end and dimensions must match
        self.dense_gcns = torch.nn.ModuleList(
            [
                DenseGCN(
                    n_blocks=n_blocks,
                    in_channels=channels_dgcn,
                    out_channels=channels_dgcn,
                    growth_rate=channels_dgcn,
                    conv=conv,
                    **kwargs,
                )
                for _ in dilations
            ]
        )
        # If we want to compute the graph at the start of each layer.
        if dynamic:
            # knn = "matrix" is better than knn = "not matrix"
            self.knns = torch.nn.ModuleList(
                [DilatedKnnGraph(k=k, dilation=d, hierarchical=hierarchical) for d in dilations]
            )
        else:
            self.knns = torch.nn.ModuleList(
                [Dilated(k=k, dilation=d, hierarchical=hierarchical) for d in dilations]
            )

    def forward(self, x, edge_index=None, pos=None, batch=None):

        """
        Parameters:
        ----------
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, C]

        edge_index: Tensor, default = None
            Edge index of shape [N * k * d, 2]. Must be provided if the graph is not computed dynamically

        batch: LongTensor, default = None
            batch tensor [N, ] as described in PyG docs.
            For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]
        """

        k = self.k
        dilations = self.dilations
        inputs = x
        if self.use_bottleneck:
            x = self.bottleneck(x)

        res = None
        for i, (d, dense_gcn, knn) in enumerate(
            zip(dilations, self.dense_gcns, self.knns)
        ):
            if self.dynamic:
                edge_index_d = knn(x, batch=batch)  # [N * k, 2]
            else:
                if edge_index is None:
                    raise ValueError(
                        "edge_index is required if the graphs is not dynamically computed"
                    )
                elif isinstance(edge_index, (tuple, list)):
                    edge_index_d = edge_index[i]
                else:
                    edge_index_d = knn(edge_index, batch)  # [N * k, 2]
            h = dense_gcn(x, edge_index_d, pos)

            # If we use the bottleneck concat, else stack and max over.
            if res is None:
                res = h
            else:
                if self.use_bottleneck:
                    res = torch.cat([res, h], axis=-1)  # [N, C]

                else:
                    res = torch.stack([res, h])  # [2, N, C]
                    res = torch.max(res, axis=0).values  # [N, C]

        # global pooling
        if self.use_pooling:
            if batch is None:  # This is fixed in 2.0.4
                t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                res_pool = self.pool(x, batch=t)[t]
            else:
                res_pool = self.pool(x, batch=batch)[batch]
            # Add the res_pool to the residual layers
            if self.use_bottleneck:
                res = torch.cat([res, res_pool], axis=-1)  # [N, C]
            else:
                res = torch.stack([res, res_pool])  # [2, N, C]
                res = torch.max(res, axis=0).values  # [N, C]

        # Add residual connection
        if self.use_residual:
            res = res + inputs

        return res


class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        dilations: Tuple[int],
        n_idgcn_blocks: int,
        n_dgcn_blocks: int,
        use_radius_graph: bool = True,
        **idgcn_kwargs,
    ):

        """
        Feature extractor that  transforms [N, 3] into [N, C] and then
        applies a few blocks of InceptionDenseGCN to extract features -> [N, C]

        Parameters:
        ----------
        channels: int
            number of channels. Will be used as the output of PreFeatureExtractor
            and as the growth rate of InceptionDenseGCN

        k: int
            number of neighbours for constructing the knn graph

        dilations: Tuple[int]
            the dilations that the DenseGCNs from each InceptionDenseGCN block will have.

        n_idgcn_blocks: int
            number of Inception DenseGCN blocks

        n_dgcn_blocks: int
            number of  DenseGCN blocks in each InceptionDenseGCN block

        **idgcn_kwargs: args
            Key arguments for `InceptionDenseGCN`:
            use_bottleneck: bool, default = False
            use_pooling: bool, default = False
            use_residual: bool, default = False
            conv: str, default = "edge"
            pool_type: str, default "max"
            dynamic: bool, default = False.
        """
        super(InceptionFeatureExtractor, self).__init__()

        # Config
        self.k = k
        self.n_idgcn = n_idgcn_blocks
        self.dilations = dilations
        self.use_radius_graph = use_radius_graph

        # Layers
        self.knn = DilatedKnnGraph(k=k * max(dilations), dilation=1)
        self.knn1 = Dilated(k=k, dilation=1)
        self.pre_gcn = DynConv(
            in_channels=3,
            out_channels=channels,
            kernel_size=k,
            dilation=1,
            conv="edge",
        )
        self.layers = torch.nn.ModuleList(
            [
                InceptionDenseGCN(
                    n_blocks=n_dgcn_blocks,
                    k=k,
                    dilations=dilations,
                    in_channels=channels,
                    **idgcn_kwargs,
                )
                for i in range(self.n_idgcn)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        pos=None,
        batch: Optional[torch.Tensor] = None,
        return_index: bool = False,
    ):
        """
        Parameters:
        ----------
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, 3]

        batch: Optional[LongTensor]
            batch tensor [N, ] as described in PyG docs.
            For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]

        return_index: bool, default = False
            True - Return the edge_index of the knn (k neighbours, NOT k * d )
        """

        # Compute initial edge_index
        if self.use_radius_graph:
            edge_index = [
                radius_graph(x, r=d, max_num_neighbors=self.k) for d in self.dilations
            ]
            edge_index1 = edge_index[0]
        else:
            edge_index = self.knn(x, batch=batch)  # [2, N * k * d]
            # Get d=1 neighbourhood to apply pre feature extraction
            edge_index1 = self.knn1(
                edge_index=edge_index,
                batch=batch,
                k_constructed=self.k * max(self.dilations),
            )  # [2, N * k]
        x = self.pre_gcn(x, edge_index=edge_index, batch=batch)  # [N, 3] ->  [N, C]

        # Apply the InceptionDenseGCNs
        res = torch.zeros_like(x)  # [N, C]
        for layer in self.layers:
            x = layer(
                x, edge_index=edge_index, pos=pos, batch=batch
            )  # if dynamic=True edge_index will be ignored
            res = res + x  # add residuals
        res = res / self.n_idgcn  # [N, C]

        if return_index:
            return res, edge_index1
        return res


# --------------------------------------------------
# ---------------- Transformers --------------------
# --------------------------------------------------


class InceptionPointTransformer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        k: int,
        dilations: Tuple[int],
        use_bottleneck: bool = False,
        use_pooling: bool = False,
        use_residual: bool = False,
        pool_type: str = "max",
        hierarchical: bool = False,
        dynamic: bool = False,
        **transformer_kwargs,
    ):
        """The inception GCN is formed from parallel DenseGCNs that use different dilations

        Parameters:
        ----------
        in_channels: int
            Input channels. It's used to compute growth rate for DenseGCN.

        k: int
            num neighbours

        dilations: Tuple[int]
            a list containing the dilation for each DenseGCN layer.


        use_bottleneck: bool, default = False
            True - Applies a bottleneck 1 layer MLP with dimensions
            [in_channels, growth_rate / (n_blocks + use_pooling)].
            Also switches to concatenating instead of pooling the outputs of the DenseGCNs

        use_pooling: bool, default = False
            True - applies a `global_max_pool` and in parallel to the DenseGCN

        use_residual: bool, default = False
            True - adds the inputs to the result

        conv: str, default = "edge"
            Convolution type in DenseGraphBlock
            One of ["edge", "mr", "gat", "gcn", "gin", "sage", "rsage"]

        pool_type: str, default = "max"
            global pooling type if `use_pooling == True`.
            One of ["max", "mean", "add"]

        dynamic: bool, default = False
            True - The knn-graph will be computed for each dilation
            False - The knn-graph is provided at inference time. Must pass edge_index as a parameter.

        **transformer_kwargs: key arguments
            Point transformer kwargs.

        Remark:
            - The if `use_bottleneck` is True then `in_channels % t == 0` where `t = len(dilations) + use_pooling`

        """
        super(InceptionPointTransformer, self).__init__()

        # Config
        self.k = k
        self.dilations = dilations
        self.use_bottleneck = use_bottleneck
        self.use_pooling = use_pooling
        self.use_residual = use_residual
        self.dynamic = dynamic
        # Layers

        div = len(dilations)  # Number of dense_gcn
        # Pool layer
        if use_pooling:
            if pool_type == "max":
                self.pool = global_max_pool
            elif pool_type == "mean":
                self.pool = global_mean_pool
            elif pool_type == "add":
                self.pool = global_add_pool
            else:
                raise ValueError('`pool_type` must be one of ["max", "mean", "add"]')
            div += 1
        # MLP bottleneck
        if use_bottleneck:
            # We need to set the input shape of the DenseGCN as the output of MLP
            assert (
                in_channels % div == 0
            ), "The number of in_channels must be divisible by the number of DenseGCN layers + pooling (if used)"
            channels_ = in_channels // div
            # assert(growth_rate % n_blocks == 0, "the number of blocks")
            self.bottleneck = MLP(
                [in_channels, channels_],
                act="leaky_relu",
                act_kwargs={"negative_slope": 0.2},
            )

        else:
            channels_ = in_channels  # Else use the in channels

        self.transformers = torch.nn.ModuleList(
            [
                PointTransformerConv(
                    in_channels=channels_,
                    out_channels=channels_,
                    **transformer_kwargs,
                )
                for _ in dilations
            ]
        )

        # IF we want to compute the graph at the start of each layer.
        if dynamic:
            # knn = "matrix" is better than knn = "not matrix"
            self.knns = torch.nn.ModuleList(
                [DilatedKnnGraph(k=k, dilation=d, hierarchical=hierarchical) for d in dilations]
            )
        else:
            self.knns = torch.nn.ModuleList(
                [Dilated(k=k, dilation=d, hierarchical=hierarchical) for d in dilations]
            )

    def forward(self, x, pos, edge_index=None, batch=None):

        """
        Parameters:
        ----------
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, C]

        pos: Tensor
            Point positions

        edge_index: Tensor, default = None
            Edge index of shape [N * k * d, 2]. Must be provided if the graph is not computed dynamically

        batch: LongTensor, default = None
            batch tensor [N, ] as described in PyG docs.
            For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]
        """

        k = self.k
        dilations = self.dilations
        inputs = x
        if self.use_bottleneck:
            x = self.bottleneck(x)

        res = None
        for d, point_transformer, knn in zip(dilations, self.transformers, self.knns):
            if self.dynamic:
                edge_index_d = knn(x, batch=batch)  # [N * k, 2]
            else:
                if edge_index is None:
                    raise ValueError(
                        "edge_index is required if the graphs is not dynamically computed"
                    )
                edge_index_d = knn(edge_index, batch)  # [N * k, 2]
            h = point_transformer(x, pos=pos, edge_index=edge_index_d)

            # If we use the bottleneck concat, else stack and max over.
            if res is None:
                res = h
            else:
                if self.use_bottleneck:
                    res = torch.cat([res, h], axis=-1)  # [N, C]

                else:
                    res = torch.stack([res, h])  # [2, N, C]
                    res = torch.max(res, axis=0).values  # [N, C]

        # global pooling
        if self.use_pooling:
            if batch is None:  # This is fixed in PyG 2.0.4
                t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                res_pool = self.pool(x, batch=t)[t]
            else:
                res_pool = self.pool(x, batch=batch)[batch]
            # Add the res_pool to the residual layers
            if self.use_bottleneck:
                res = torch.cat([res, res_pool], axis=-1)  # [N, C]
            else:
                res = torch.stack([res, res_pool])  # [2, N, C]
                res = torch.max(res, axis=0).values  # [N, C]

        # Add residual connection
        if self.use_residual:
            res = res + inputs

        return res


class InceptionPointTransformerExtractor(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        dilations: Tuple[int],
        n_ipt: int,
        **ipt_kwargs,
    ):
        """
        Feature extractor that  transforms [N, 3] into [N, C] and then
        applies a few blocks of InceptionDenseGCN to extract features -> [N, C]

        Parameters:
        ----------
        channels: int
            number of channels. Will be used as the output of PreFeatureExtractor
            and as the growth rate of InceptionDenseGCN

        k: int
            number of neighbours for constructing the knn graph

        dilations: Tuple[int]
            the dilations that the DenseGCNs from each InceptionDenseGCN block will have.

        n_ipt: int
            number of Inception Transformer blocks

        **ipt_kwargs: args
            Key arguments for `InceptionTransformer`:
            use_bottleneck: bool, default = False
            use_pooling: bool, default = False
            use_residual: bool, default = False
            conv: str, default = "edge"
            pool_type: str, default "max"
            dynamic: bool, default = False.
            **transformer_kwargs
        """
        super().__init__()

        # Config
        self.dilations = dilations
        self.n_ipt = n_ipt

        # Layers
        self.knn = DilatedKnnGraph(k=k * max(dilations), dilation=1)
        self.knn1 = Dilated(k=k, dilation=1)
        self.pre_gcn = GraphConv(
            in_channels=3,
            out_channels=channels,
            conv="edge",
        )
        self.layers = torch.nn.ModuleList(
            [
                InceptionPointTransformer(
                    in_channels=channels, k=k, dilations=dilations, **ipt_kwargs
                )
                for i in range(self.n_ipt)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_index: bool = False,
    ):
        """
        Parameters:
        ----------
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, 3]

        batch: Optional[LongTensor]
            batch tensor [N, ] as described in PyG docs.
            For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]

        return_index: bool, default = False
            True - Return the edge_index of the knn (k neighbours, NOT k * d )
        """

        # keep initial positions
        pos = x
        # Compute initial edge_index
        edge_index = self.knn(x, batch=batch)  # [N * k * d, 2]
        # Get d=1 neighbourhood to apply pre feature extraction
        edge_index_1 = self.knn1(edge_index=edge_index, batch=batch)  # [N * k, 2]
        x = self.pre_gcn(x, edge_index=edge_index_1, batch=batch)  # [N, 3] ->  [N, C]

        # Apply the InceptionDenseGCNs
        res = torch.zeros_like(x)  # [N, C]
        for layer in self.layers:
            x = layer(
                x, pos=pos, edge_index=edge_index, batch=batch
            )  # if dynamic=True edge_index will be ignored
            res = res + x  # add residuals
        res = res / self.n_ipt  # [N, C]

        if return_index:
            return res, edge_index_1
        return res
