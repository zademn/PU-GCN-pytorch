import torch
from torch import nn
from einops import rearrange, reduce
from torch_geometric.nn import knn_graph, MLP
import inspect
from torch_geometric.nn import (
    PointTransformerConv,
    EdgeConv,
    GATConv,
    GCNConv,
    SAGEConv,
    GATv2Conv,
    PointNetConv,
)


def get_dilated_k(edge_index, k: int, d: int):
    _, counts = torch.unique(edge_index[1], return_counts=True)
    ei = torch.tensor([], dtype=edge_index.dtype, device=edge_index.device)
    start = 0
    # This takes a while
    for count in counts:
        ei = torch.concatenate(
            [ei, edge_index[:, start : start + count : d][:, :k]], axis=1
        )
        start += count
    return ei


def get_dilated_k_fast(
    edge_index, k: int, d: int, k_constructed: int = None, hierarchical: bool = False
):
    if k_constructed is None:
        _, counts = torch.unique(edge_index[1], return_counts=True)
        k_constructed = counts[0].detach().item()  # assume we always find k neighbours.
    edge_index = rearrange(
        edge_index,
        "e (n2 k_constructed) -> e n2 k_constructed",
        k_constructed=k_constructed,
    )
    if hierarchical:
        edge_index = edge_index[:, :, k * (d - 1) : k * d]
        edge_index = rearrange(edge_index, "e d1 d2 -> e (d1 d2)")
        return edge_index

    edge_index = edge_index[:, :, ::d]  # Res dilated
    edge_index = edge_index[:, :, :k]
    edge_index = rearrange(edge_index, "e d1 d2 -> e (d1 d2)")
    return edge_index


def knn_graph_dilated(x, k: int, d: int, batch=None, fast: bool = True, **kwargs):
    edge_index = knn_graph(x, k=k * d, batch=batch, **kwargs)
    if fast:
        return get_dilated_k_fast(edge_index, k=k, d=d, k_constructed=k * d)
    else:
        return get_dilated_k(edge_index, k=k, d=d)


class GraphConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv="edge",
        act="relu",
        norm=None,
        bias=True,
        heads=2,
    ):
        super().__init__()
        if isinstance(conv, str):
            if conv.lower() == "edge":
                nn = MLP(
                    [2 * in_channels, out_channels],
                    act=act,
                    norm=norm,
                    bias=bias,
                    last_lin=True,
                )
                self.gconv = EdgeConv(nn)
            elif conv.lower() == "gat":
                self.gconv = GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                    concat=False,
                    bias=bias,
                )
            elif conv.lower() == "gatv2":
                self.gconv = GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                    concat=False,
                    bias=bias,
                )
            elif conv.lower() == "gcn":
                self.gconv = GCNConv(
                    in_channels, out_channels, improved=True, bias=bias
                )

            elif conv.lower() == "sage":
                self.gconv = SAGEConv(in_channels, out_channels, bias=bias, aggr="mean")

            # Convs that need position
            elif conv == "point_transformer":
                # attn_nn = nn.Linear(
                #     in_features=out_channels, out_features=out_channels
                # )
                self.gconv = PointTransformerConv(
                    in_channels=in_channels, out_channels=out_channels
                )

            elif conv == "pointnet":
                local_nn = nn.Linear(
                    in_features=in_channels + 3, out_features=out_channels
                )
                global_nn = nn.Linear(
                    in_features=out_channels, out_features=out_channels
                )
                self.gconv = PointNetConv(local_nn=local_nn, global_nn=global_nn)

            else:
                raise NotImplementedError(f"conv {conv} is not implemented")
        elif isinstance(conv, nn.Module):
            self.gconv = conv
        else:
            raise ValueError("conv must be either `str` or `nn.Module`")

        if "pos" in inspect.signature(self.gconv.forward).parameters:
            self._needs_pos = True
        else:
            self._needs_pos = False

    def forward(self, x, edge_index, pos=None):
        if self._needs_pos:
            return self.gconv(x, pos=pos, edge_index=edge_index)
        else:
            return self.gconv(x, edge_index=edge_index)


class DenseGCN(nn.Module):
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

        # NN blocks
        self.lin_x = nn.Linear(in_channels, growth_rate)
        self.blocks = nn.ModuleList(
            [
                GraphConv(
                    in_channels=(i + 1) * self.growth_rate,
                    out_channels=self.growth_rate,
                    conv=conv,
                    **kwargs,
                )
                for i in range(self.n_blocks)
            ]
        )
        self.lin_down = nn.Linear(
            in_features=in_channels + growth_rate * n_blocks, out_features=out_channels
        )

    def forward(self, x, edge_index, pos=None):
        # Apply DenseGCN blocks
        # (n_batches * n_nodes, n_features_in) -> (n_batches * n_nodes, growth_rate)
        x = self.lin_x(x)
        # (n_batches * n_nodes, growth_rate) -> (n_batches * n_nodes, growth_rate * n_blocks)
        for block in self.blocks:
            x_ = block(x, edge_index=edge_index, pos=pos)
            x = torch.cat([x, x_], dim=-1)
        # Downscale from in_channels + growth_rate * n_blocks to out_channels
        res = rearrange(x, "n (f n_blocks) -> n f n_blocks", n_blocks=self.n_blocks + 1)
        res = reduce(res, "n f n_blocks -> n f", "max")
        # res = self.lin_down(x)
        return res
