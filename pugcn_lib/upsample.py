import torch
from gcn_lib.sparse import EdgConv
from torch_cluster import knn_graph
from gcn_lib.sparse.torch_vertex import DynConv, GraphConv
from .torch_nn import MLP
from einops import rearrange, repeat
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import EdgeConv

# from .torch_geometric_nn import ShuffleConv

class PointShuffleOld(torch.nn.Module):
    def __init__(self, r: int):
        """
        Shuffles [N, r * C] -> [r * N, C]

        Parameters:
        ----------
        r: int
            scale
        """
        super().__init__()
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
            x = x.reshape(n, 1, c, r)  # [N, r*C] -> [N, 1, C, r]
            x = x.permute(0, 3, 2, 1)  # [N, 1, C, r] -> [N, r, C, 1]
            x = x.reshape([n * r, c])  # [N, r, C, 1] -> [r * N, C]

        return x


class PointShuffle(torch.nn.Module):
    def __init__(self, r):
        """
        Shuffles [N, r * C] -> [r * N, C]

        Args:
            r: int
                scale
        """
        super().__init__()
        # Config
        self.r = r

    def forward(self, x, batch=None):
        """
        Args:
            x: Tensor
                Node feature matrix of all point clouds concatenated [N, r * C]
            batch: Optional[LongTensor]
                batch tensor [N, ] as described in PyG docs.
                For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]
        """
        r = self.r

        n, c = x.shape[0], x.shape[1]

        if batch is not None:
            # split into batches.
            x, _ = to_dense_batch(
                x, batch=batch
            )  # [N, C] -> [B, N_, C] where N = B * N_
            # Split the channels dim c = (c2, r). Then combine (n r).
            x = rearrange(x, "b n (c2 r) -> b (n r) c2", c2=c // r)
            # combine back
            x = rearrange(x, "b n c -> (b n) c")

        else:
            x = rearrange(x, "n (c2 r) -> (n r) c2", c2=c // r)

        return x


class NodeShuffle(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        r: int,
        conv="edge",
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


        """
        super(NodeShuffle, self).__init__()

        # Config
        self.k = k
        self.r = r
        self.in_channels = in_channels
        # Layers
        self.gcn = GraphConv(
            in_channels=in_channels, out_channels=in_channels * r, conv=conv
        )  # self.gcn = DynConv(
        #     kernel_size=k,
        #     dilation=1,
        #     in_channels=in_channels,
        #     out_channels=in_channels * r,
        #     conv="edge",
        #     knn="matrix",
        # )
        self.ps = PointShuffle(r=r)
        self.lin = torch.nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(
        self, x, edge_index=None, pos=None, batch=None, return_batch: bool = False
    ):
        """
        Parameters:
        ----------
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, C]

        edge_index: Tensor, default = None
            Edge index of shape [N * k, 2]. If it's not provided the graph is computed dynamically.

        batch: Optional[LongTensor]
            batch tensor [N, ] as described in PyG docs.
            For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]

        return_batch: bool, default, = False
            True - will return the upsampled batch vector.
        """

        if edge_index is None:
            edge_index = knn_graph(x, self.k, batch=batch)
        x = self.gcn(x, edge_index, pos=pos)  # [N, C] -> [N, r * C]
        # x = self.gcn(x, batch=batch)
        x = self.ps(x, batch=batch)  # [N, r * C] -> [r * N, C]
        x = self.lin(x)  # [r * N, C] -> [r * N, C']

        if return_batch:
            if batch is not None:
                batch_ = batch.repeat_interleave(self.r)  # repeat each number r times
            else:
                batch_ = torch.zeros(len(x))
            return x, batch_
        return x


class DuplicatePoints(torch.nn.Module):
    def __init__(self, r: int):
        super(DuplicatePoints, self).__init__()

        self.r = r

    def forward(self, x, batch=None):
        r = self.r

        if batch is not None:
            # split into batches.
            x, _ = to_dense_batch(x, batch=batch)
            x = repeat(x, "b n c -> b (n r) c", r=r)
            x = rearrange(x, "b n c -> (b n) c")
        else:
            x = repeat(x, "n c -> (n r) c", r=r)  #
        return x


class DuplicateUpsampler(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, k: int, r: int, conv="edge"
    ):
        """
        Transforms input: [N, C] -> [r * N, C'] by duplicating points `r` times and applying a neural network over them.

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


        """
        super(DuplicateUpsampler, self).__init__()

        # Config
        self.k = k
        self.r = r
        # Layers
        self.gcn = GraphConv(
            in_channels=in_channels, out_channels=in_channels * r, conv=conv
        )
        self.upsampler = DuplicatePoints(r=r)
        self.lin = torch.nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, edge_index=None, batch=None, return_batch: bool = False):
        """
        Parameters:
        ----------
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, C]

        edge_index: Tensor, default = None
            Edge index of shape [N * k, 2]. If it's not provided the graph is computed dynamically.

        batch: Optional[LongTensor]
            batch tensor [N, ] as described in PyG docs.
            For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]

        return_batch: bool, default, = False
            True - will return the upsampled batch vector.
        """

        if edge_index is None:
            edge_index = knn_graph(x, self.k, batch=batch)
        x = self.upsampler(x, batch=batch)  # [N, r * C] -> [r * N, C]
        x = self.lin(x)  # [r * N, C] -> [r * N, C']

        if return_batch:
            if batch is not None:
                batch_ = batch.repeat_interleave(self.r)  # repeat each number r times
            else:
                batch_ = torch.zeros(len(x))
            return x, batch_
        return x


class VariationalShuffle(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int, r: int):
        super().__init__()

        # Config
        self.k = k
        self.r = r
        self.z_dim = in_channels

        # Layers
        self.encoder = GraphConv(
            in_channels=in_channels, out_channels=in_channels, conv="edge"
        )

        self.nn_mean = GraphConv(
            in_channels=in_channels, out_channels=self.z_dim * r, conv="edge"
        )
        self.nn_log_var = GraphConv(
            in_channels=in_channels, out_channels=self.z_dim * r, conv="edge"
        )
        self.ps = PointShuffle(r)
        self.decoder = torch.nn.Linear(
            in_features=self.z_dim, out_features=out_channels
        )

    def encode(self, x, edge_index, batch=None):
        x = self.encoder(x, edge_index)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        z_mean = self.nn_mean(x, edge_index)
        z_mean = self.ps(z_mean, batch=batch)
        z_log_var = self.nn_log_var(x, edge_index)
        z_log_var = self.ps(z_log_var, batch=batch)
        z = self.reparametrize(z_mean, z_log_var)
        self.last_z_mean = z_mean
        self.last_z_log_var = z_log_var
        return z, z_mean, z_log_var

    def reparametrize(self, z_mean, z_log_var):
        # Generate noise from a normal distribution
        noise = torch.randn(z_mean.shape[0], z_mean.shape[1]).to(z_mean.device)
        # Reparametrize
        z_std = torch.exp(z_log_var * 0.5)
        z = z_mean + noise * z_std
        return z

    def forward(
        self, x, pos=None, edge_index=None, batch=None, return_batch: bool = False
    ):
        if edge_index is None:
            edge_index = knn_graph(x, self.k, batch=batch)
        z, _, _ = self.encode(x, edge_index)
        out = self.decoder(z)
        return out


class GeneralUpsampler(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        r: int,
        upsampler: str = "nodeshuffle",
        **upsampler_kwargs
    ):
        super().__init__()
        if upsampler == "nodeshuffle":
            self.upsampler = NodeShuffle(
                in_channels=in_channels,
                out_channels=out_channels,
                k=k,
                r=r,
                **upsampler_kwargs,
            )
        elif upsampler == "duplicate":
            self.upsampler = DuplicateUpsampler(
                in_channels=in_channels, out_channels=out_channels, k=k, r=r
            )
        elif upsampler == "variational":
            self.upsampler = VariationalShuffle(
                in_channels=in_channels, out_channels=out_channels, k=k, r=r
            )
        else:
            raise ValueError("Upsampler doesn't exist")

    def forward(
        self, x, edge_index=None, pos=None, batch=None, return_batch: bool = False
    ):
        return self.upsampler(
            x, edge_index=edge_index, pos=pos, batch=batch, return_batch=return_batch
        )
