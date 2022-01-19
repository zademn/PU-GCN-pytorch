import torch
from torch_cluster import knn_graph
from gcn_lib.sparse.torch_vertex import DenseGraphBlock, EdgConv


class DenseGCN(torch.nn.Module):
    def __init__(self, n_blocks: int, in_channels: int, growth_rate: int):
        """
        n_blocks: int
            The number of DenseGraphBlock blocks the DenseGCN will have
        in_channels: int
            The number of input channels
        growth_rate: int
            The output channels for each DenseGraphBlock and with how much the in_channels of the next block will grow
            After applying all the DenseGraphBlocks the output will have in_channels + n_blocks * growth_rate
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
                    conv="edge",
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
    def __init__(self, n_blocks: int, growth_rate: int, k: int):

        """
        The inception GCN is formed from 2 DenseGCNs, one that samples neighbours with d = 1 and one with d = 2

        growth_rate: int
            Input channels **and** growth rate for DenseGCN
        k: int
            num neighbours
        n_blocks: int
            number of blocks each DenseGCN will have
        """
        super(InceptionDenseGCN, self).__init__()

        # Config
        self.k = k
        self.d = 2  # hardcoded for now
        self.n_blocks = n_blocks
        # Layers
        self.dense_gcn1 = DenseGCN(
            n_blocks=n_blocks, in_channels=growth_rate, growth_rate=growth_rate
        )
        self.dense_gcn2 = DenseGCN(
            n_blocks=n_blocks, in_channels=growth_rate, growth_rate=growth_rate
        )

    def forward(self, x):

        k = self.k
        d = self.d
        # Construct graph
        # edge_index = knn_graph(x, k * d)  # [N * k * d, 2]
        # Select indexes based on dilation
        # edge_index1 = (
        #     edge_index.T.reshape(x.shape[0], k * d, 2)[:, :k, :]
        #     .reshape(x.shape[0] * k, 2)
        #     .T
        # )  # [N * k,  2]
        edge_index1 = knn_graph(x, k)  # [N * k, 2]
        edge_index2 = knn_graph(x, k * d)[:, ::d]  # [N * k, 2]

        h1 = self.dense_gcn1(x, edge_index1)  # [N, C]
        h2 = self.dense_gcn2(x, edge_index2)  # [N, C]

        # Where to add global pooling?

        # Concat and pool over
        res = torch.stack([h1, h2])  # [2, N, C]
        res = torch.max(res, axis=0).values  # [N, C]
        # Add residual connection
        res = res + x

        return res


class PreFeatureExtractor(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int):
        """
        Constructs the graph and applies one EdgConv layer

        k: int
            num neighbours
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

    def forward(self, x):
        edge_index = knn_graph(x, self.k)  # [N * k, 2]
        res = self.gcn(x, edge_index)  # [N, C]
        return res


class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self, opts):

        """
        Feature extractor that has a PreFeatureExtractor that transforms [N, 3] into [N, C] and then
        applies a few blocks of InceptionDenseGCN to extract features -> [N, C]

        opts: config for Inception feature extractor.
        """
        super(InceptionFeatureExtractor, self).__init__()

        # Config
        self.n_idgcn = opts.n_idgcn_blocks

        # Layers
        self.pre = PreFeatureExtractor(in_channels=3, out_channels=opts.channels, k=1)
        self.layers = torch.nn.ModuleList(
            [
                InceptionDenseGCN(
                    n_blocks=opts.n_dgcn_blocks,
                    k=opts.num_neighbours,
                    growth_rate=opts.channels,
                )
                for i in range(self.n_idgcn)
            ]
        )

    def forward(self, x):

        x = self.pre(x)  # [N, 3] ->  [N, C]
        res = torch.zeros_like(x)  # [N, C]
        for layer in self.layers:
            x = layer(x)
            res = res + x
        res = res / self.n_idgcn  # [N, C]
        return res
