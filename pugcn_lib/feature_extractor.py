import torch
from typing import Optional, List, Tuple
from torch_cluster import knn_graph
from gcn_lib.sparse.torch_vertex import DenseGraphBlock, EdgConv
from gcn_lib.sparse.torch_edge import DilatedKnnGraph
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from .torch_nn import MLP


class DenseGCN(torch.nn.Module):
    def __init__(
        self, n_blocks: int, in_channels: int, growth_rate: int, conv="edge", **kwargs
    ):
        """
        Parameters:
        ----------
        n_blocks: int
            The number of DenseGraphBlock blocks the DenseGCN will have

        in_channels: int
            The number of input channels

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

        # GCN blocks
        self.blocks = torch.nn.ModuleList(
            [
                DenseGraphBlock(
                    in_channels=self.in_channels + i * self.growth_rate,
                    out_channels=self.growth_rate,
                    conv=conv,
                    heads=1,
                    **kwargs
                )
                for i in range(self.n_blocks)
            ]
        )

    def forward(self, x, edge_index):
        # Apply DenseGCN blocks
        for block in self.blocks:
            x, _ = block(x, edge_index)

        # Since DenseGraphBlock concatneates the output we need to rehsape to prepare for global pooling
        # I think this will break if in_channels != growth_rate
        x = x.reshape(
            x.shape[0], self.growth_rate, self.n_blocks + 1
        )  # [N, C, n_blocks+1]
        res = torch.max(x, axis=-1).values  # [N, C]
        return res


class InceptionDenseGCN(torch.nn.Module):
    def __init__(
        self,
        n_blocks: int,
        in_channels: int,
        k: int,
        dilations: Tuple[int],
        use_bottleneck: bool = False,
        use_pooling=False,
        use_residual=False,
        conv="edge",
        pool_type="max",
        **kwargs
    ):
        """The inception GCN is formed from 2 DenseGCNs, one that samples neighbours with d = 1 and one with d = 2

        Parameters:
        ----------
        in_channels: int
            Input channels. It's used to compute growth rate for DenseGCN.

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
                    growth_rate=channels_dgcn,
                    conv=conv,
                    **kwargs
                )
                for _ in dilations
            ]
        )
        self.knns = torch.nn.ModuleList(
            [DilatedKnnGraph(k=k, dilation=d, knn="not matrix") for d in dilations]
        )

    def forward(self, x, batch=None):

        """
        Parameters:
        ----------
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, C]

        batch: LongTensor, default = None
            batch tensor [N, ] as described in PyG docs.
            For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]
        """

        k = self.k
        dilations = self.dilations
        # Construct graph
        # edge_index = knn_graph(x, k * d)  # [N * k * d, 2]
        # Select indexes based on dilation
        # edge_index1 = (
        #     edge_index.T.reshape(x.shape[0], k * d, 2)[:, :k, :]
        #     .reshape(x.shape[0] * k, 2)
        #     .T
        # )  # [N * k,  2]
        inputs = x
        if self.use_bottleneck:
            x = self.bottleneck(x)

        res = None
        for d, dense_gcn, knn in zip(dilations, self.dense_gcns, self.knns):
            # edge_index = knn_graph(x, k * d, batch=batch)[:, ::d]  # [N * k, 2]
            edge_index = knn(x, batch)
            h = dense_gcn(x, edge_index)

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


class PreFeatureExtractor(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int):
        """
        Constructs the graph and applies one EdgConv layer

        Parameters:
        ----------
            k: int
                num neighbours for constructing the knn graph

            in_channels: int
                number of input channels. For point clouds it's usually 3 for xyz coords

            out_channels: int
                number of output channels
        """
        super(PreFeatureExtractor, self).__init__()
        # Config
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Layers
        self.gcn = EdgConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, batch=None):
        """
        Args:
            x: Tensor
                Node feature matrix of all point clouds concatenated [N, 3]
            batch: Optional[LongTensor]
                batch tensor [N, ] as described in PyG docs.
                For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]
        """
        edge_index = knn_graph(x, self.k, batch=batch)  # [N * k, 2]
        res = self.gcn(x, edge_index)  # [N, C]
        return res


class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        dilations: Tuple[int],
        n_idgcn_blocks: int,
        n_dgcn_blocks: int,
        **idgcn_kwargs
    ):

        """
        Feature extractor that has a PreFeatureExtractor that transforms [N, 3] into [N, C] and then
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
            use_bottleneck: bool = False
            use_pooling = False
            use_residual = False
            conv = "edge"
            pool_type = "max"
        """
        super(InceptionFeatureExtractor, self).__init__()

        # Config
        self.n_idgcn = n_idgcn_blocks

        # Layers
        self.pre = PreFeatureExtractor(in_channels=3, out_channels=channels, k=1)
        self.layers = torch.nn.ModuleList(
            [
                InceptionDenseGCN(
                    n_blocks=n_dgcn_blocks,
                    k=k,
                    dilations=dilations,
                    in_channels=channels,
                    **idgcn_kwargs
                )
                for i in range(self.n_idgcn)
            ]
        )

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """
        Parameters:
        ----------
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, 3]

        batch: Optional[LongTensor]
            batch tensor [N, ] as described in PyG docs.
            For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]
        """

        x = self.pre(x, batch=batch)  # [N, 3] ->  [N, C]
        res = torch.zeros_like(x)  # [N, C]
        for layer in self.layers:
            x = layer(x, batch)
            res = res + x
        res = res / self.n_idgcn  # [N, C]
        return res
