import torch
from gcn_lib.sparse import EdgConv
from torch_cluster import knn_graph


class PointSuffle(torch.nn.Module):
    def __init__(self, r):
        """
        Shuffles [N, r * C] -> [r * N, C]
        r: int
            scale
        """
        super(PointSuffle, self).__init__()
        # Config
        self.r = r

    def forward(self, x):
        r = self.r
        x = x.reshape((x.shape[0], 1, x.shape[1] // r, r))  # [N, r*C] -> [N, 1, C, r]
        x = x.permute(0, 3, 2, 1)  # [N, 1, C, r] -> [N, r, C, 1]
        x = x.reshape([x.shape[0] * r, x.shape[2]])  # [N, r, C, 1] -> [r * N, C]

        return x


class NodeShuffle(torch.nn.Module):
    def __init__(self, in_channels: int, k: int, r: int):
        """
        Transforms input: [N, C] -> [r * N, C]
        k: int
            number of neighbours to sample
        r: int
            upsampling ratio
        """
        super(NodeShuffle, self).__init__()

        # Config
        self.k = k
        self.r = r
        self.in_channels = in_channels
        # Layers
        self.gcn = EdgConv(in_channels=in_channels, out_channels=in_channels * r)
        self.ps = PointSuffle(r=r)

    def forward(self, x):

        edge_index = knn_graph(x, self.k)
        x = self.gcn(x, edge_index)  # [N, C] -> [N, r * C]
        x = self.ps(x)  # [N, r * C] -> [r * N, C]
        return x
