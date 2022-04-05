import torch
from gcn_lib.sparse import EdgConv
from torch_cluster import knn_graph
from gcn_lib.sparse.torch_vertex import DynConv
from .torch_nn import MLP


class PointSuffle(torch.nn.Module):
    def __init__(self, r: int):
        """
        Shuffles [N, r * C] -> [r * N, C]

        Parameters:
        ----------
        r: int
            scale
        """
        super(PointSuffle, self).__init__()
        # Config
        self.r = r

    def forward(self, x, batch=None):
        """
        Parameters:
        ----------
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, r * C]

        batch: Optional[LongTensor]
            batch tensor [N, ] as described in PyG docs.
            For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]
        """
        r = self.r
        n, rc = x.shape[0], x.shape[1]
        c = rc // r

        # TODO should I use view?
        if batch is not None:
            b = torch.max(batch) + 1
            n_ = n // b  # N_ = number of points per batch
            x = x.reshape(b, n_, rc)  # [N, r * C] -> [B, N_, r * C]
            x = x.reshape((b, n_, 1, c, r))  # [B, N_, r*C] -> [B, N_, 1, C, r]
            x = x.permute(0, 1, 4, 3, 2)  # [B, N_, 1, C, r] -> [B, N_, r, C, 1]
            x = x.reshape((b, n_ * r, c))  # [B, N_, r, C, 1] -> [B, r * N_, C]
            x = x.reshape((b * n_ * r, c))  # [B, r * N_, C] -> [r * N, C]

        else:
            x = x.reshape((n, 1, c, r))  # [N, r*C] -> [N, 1, C, r]
            x = x.permute(0, 3, 2, 1)  # [N, 1, C, r] -> [N, r, C, 1]
            x = x.reshape([n * r, c])  # [N, r, C, 1] -> [r * N, C]

        return x


class NodeShuffle(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        r: int,
        return_batch: bool = False,
    ):
        """
        Transforms input: [N, C] -> [r * N, C']

        Parameters:
        ----------
        in_channels: int
            number of input channels C

        out_channels: int
            number of output channels C'
        k: int
            number of neighbours to sample

        r: int
            upsampling ratio

        return_batch: bool, default, = False
            True - will return the upsampled batch
        """
        super(NodeShuffle, self).__init__()

        # Config
        self.k = k
        self.r = r
        self.in_channels = in_channels
        self.return_batch = return_batch
        # Layers
        # self.gcn = EdgConv(in_channels=in_channels, out_channels=in_channels * r)
        self.gcn = DynConv(
            kernel_size=k,
            dilation=1,
            in_channels=in_channels,
            out_channels=in_channels * r,
            conv="edge",
            knn="matrix",
        )
        self.ps = PointSuffle(r=r)
        self.mlp = MLP([in_channels, out_channels])

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

        # edge_index = knn_graph(x, self.k, batch=batch)
        # x = self.gcn(x, edge_index)  # [N, C] -> [N, r * C]
        x = self.gcn(x, batch=batch)
        x = self.ps(x, batch=batch)  # [N, r * C] -> [r * N, C]

        if batch is not None and self.return_batch:
            batch_ = batch.repeat_interleave(self.r)  # repeat each number r times
            return x, batch_
        return x
