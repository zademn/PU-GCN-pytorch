import torch
from gcn_lib.sparse import EdgConv
from torch_cluster import knn_graph


class PointSuffle(torch.nn.Module):
    def __init__(self, r):
        """
        Shuffles [N, r * C] -> [r * N, C]

        Args:
            r: int
                scale
        """
        super(PointSuffle, self).__init__()
        # Config
        self.r = r

    def forward(self, x, batch=None):
        """
        Args:
            x: Tensor
                Node feature matrix of all point clouds concatenated [N, r * C]
            batch: Optional[LongTensor]
                batch tensor [N, ] as described in PyG docs. For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]
        """
        r = self.r

        n, c = x.shape[0], x.shape[1]

        if batch is not None:
            b = torch.max(batch) + 1
            n_ = n // b
            x = x.reshape(b, n_, c)  # [N, C] -> [B, N_, C] where N = B * N_
            x = x.reshape((b, n, 1, c // r, r))  # [B, N_, r*C] -> [B, N_, 1, C, r]
            x = x.permute(0, 1, 4, 3, 2)  # [B, N_, 1, C, r] -> [B, N_, r, C, 1]
            x = x.reshape((b, n_ * r, c))  # [B, N_, r, C, 1] -> [B, r * N, C]
            x = x.reshape((n, c))  #  [B, r * N_, C] -> [r * N, C]

        else:
            x = x.reshape(
                (x.shape[0], 1, x.shape[1] // r, r)
            )  # [N, r*C] -> [N, 1, C, r]
            x = x.permute(0, 3, 2, 1)  # [N, 1, C, r] -> [N, r, C, 1]
            x = x.reshape([x.shape[0] * r, x.shape[2]])  # [N, r, C, 1] -> [r * N, C]

        return x


class NodeShuffle(torch.nn.Module):
    def __init__(self, in_channels: int, k: int, r: int):
        """
        Transforms input: [N, C] -> [r * N, C]

        Args:
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

    def forward(self, x, batch=None):
        """
        Args:
            x: Tensor
                Node feature matrix of all point clouds concatenated [N, C]
            batch: Optional[LongTensor]
                batch tensor [N, ] as described in PyG docs. For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]
        """

        edge_index = knn_graph(x, self.k, batch=batch)
        x = self.gcn(x, edge_index)  # [N, C] -> [N, r * C]
        x = self.ps(x)  # [N, r * C] -> [r * N, C]
        return x

