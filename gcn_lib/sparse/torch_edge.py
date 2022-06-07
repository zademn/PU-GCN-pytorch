import torch
from torch import nn
from torch_cluster import knn_graph
from torch_geometric.nn import radius_graph
from einops import rearrange


def get_dilated_k(
    edge_index, k: int, d: int, k_constructed: int = None, hierarchical: bool = False
):
    if k_constructed is None:
        u, counts = torch.unique(edge_index[1], return_counts=True)
        k_constructed = counts[0]  # assume we always find k neighbours.
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


class Dilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    """

    def __init__(
        self, k=9, dilation=1, stochastic=False, epsilon=0.0, hierarchical=False
    ):
        super(Dilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self.hierarchical = hierarchical

    def forward(self, edge_index, batch=None, k_constructed=None):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[: self.k]
                edge_index = edge_index.view(2, -1, num)
                edge_index = edge_index[:, :, randnum]
                return edge_index.view(2, -1)
            else:
                edge_index = get_dilated_k(
                    edge_index,
                    self.k,
                    self.dilation,
                    k_constructed=k_constructed,
                    hierarchical=self.hierarchical,
                )
        else:
            edge_index = get_dilated_k(
                edge_index,
                self.k,
                self.dilation,
                k_constructed=k_constructed,
                hierarchical=self.hierarchical,
            )
        return edge_index


class DilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(
        self,
        k=9,
        dilation=1,
        stochastic=False,
        epsilon=0.0,
        knn="matrix",
        hierarchical=False,
    ):
        super(DilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = Dilated(
            k, dilation, stochastic, epsilon, hierarchical=hierarchical
        )
        if knn == "matrix":
            self.knn = knn_graph_matrix
        else:
            self.knn = knn_graph

    def forward(self, x, batch, k_constructed=None):
        edge_index = self.knn(x, self.k * self.dilation, batch)
        return self._dilated(edge_index, batch, k_constructed)


class RadiusKnnGraph(nn.Module):
    def __init__(self, radius: float = 0.25, max_num_neighbors: int = 32):
        super(RadiusKnnGraph, self).__init__()
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors

    def forward(self, x, batch=None):
        edge_index = radius_graph(
            x, self.radius, max_num_neighbors=self.max_num_neighbors, batch=batch
        )
        return edge_index


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)


def knn_matrix(x, k=16, batch=None):
    """Get KNN based on the pairwise distance.
    Args:
        pairwise distance: (num_points, num_points)
        k: int
    Returns:
        nearest neighbors: (num_points*k ,1) (num_points, k)
    """
    with torch.no_grad():
        if batch is None:
            batch_size = 1
        else:
            batch_size = batch[-1] + 1
        x = x.view(batch_size, -1, x.shape[-1])

        neg_adj = -pairwise_distance(x.detach())

        _, nn_idx = torch.topk(neg_adj, k=k)

        n_points = x.shape[1]
        start_idx = torch.arange(
            0, n_points * batch_size, n_points, device=x.device
        ).view(batch_size, 1, 1)
        nn_idx += start_idx

        nn_idx = nn_idx.view(1, -1)
        center_idx = (
            torch.arange(0, n_points * batch_size, device=x.device)
            .expand(k, -1)
            .transpose(1, 0)
            .contiguous()
            .view(1, -1)
        )
    return nn_idx, center_idx


def knn_graph_matrix(x, k=16, batch=None):
    """Construct edge feature for each point
    Args:
        x: (num_points, num_dims)
        batch: (num_points, )
        k: int
    Returns:
        edge_index: (2, num_points*k)
    """
    nn_idx, center_idx = knn_matrix(x, k, batch)
    return torch.cat((nn_idx, center_idx), dim=0)
