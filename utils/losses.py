from hausdorff import hausdorff_distance
from torch_geometric.utils import to_dense_batch
from torch import Tensor, LongTensor
import torch
from ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import (
    chamfer_3DDist as ChamferLoss,
)
from einops import rearrange, reduce
from torch_geometric.nn import knn_graph


def hausdorff_loss(p: Tensor, q: Tensor, batches: LongTensor = None):
    """Compute hausdorff loss between 2 point clouds or
    batches of point clouds

    Parameters
    ----------
    p : Tensor of shape [N, C]
        First point cloud

    q : Tensor of shape [N, C]
        Second point cloud

    batches : LongTensor, default=None
        Tensor that signals for each point the batch number

    Returns
    -------
    float
        The hausdorff loss
    """
    if batches is None:
        return hausdorff_distance(p.cpu().numpy(), q.cpu().numpy())
    else:
        p_batch, q_batch = batches
        b = p_batch.max() + 1
        loss = 0
        p, _ = to_dense_batch(p, batch=p_batch)
        q, _ = to_dense_batch(q, batch=q_batch)
        for pi, qi in zip(p, q):
            loss += hausdorff_distance(pi.cpu().numpy(), qi.cpu().numpy())
        return loss / b


def chamfer_dist(output, gt, return_raw=False):
    # https://github.com/wutong16/Density_aware_Chamfer_Distance

    # cham_loss = dist_chamfer_3D.chamfer_3DDist()
    cham_loss = ChamferLoss()
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = dist1.mean(1) + dist2.mean(1)

    res = [cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res


def density_chamfer_dist(
    x, gt, alpha=200, n_lambda=0.5, return_raw=False, non_reg=False
):
    # https://github.com/wutong16/Density_aware_Chamfer_Distance

    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = chamfer_dist(x, gt, return_raw=True)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    loss = (loss1 + loss2) / 2

    res = (loss, cd_p, cd_t)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res


def chamfer_dist_repulsion(p1, p2, k: int = 4, return_proportion: bool = False):
    dist1, dist2, idxs1, idxs2 = ChamferLoss()(p1, p2)
    b, n, _ = p1.shape

    # Rearrange
    p1_ = rearrange(p1, "b n c -> (b n) c")
    p2_ = rearrange(p2, "b n c -> (b n) c")
    idxs1_ = rearrange(idxs1, "b n -> (b n)").type(torch.long)
    idxs2_ = rearrange(idxs2, "b n -> (b n)").type(torch.long)

    # Calculate graph
    batch = torch.repeat_interleave(torch.arange(b), n).type(torch.long).to(p1_.device)
    edge_index1 = knn_graph(p1_, k=k, batch=batch)
    edge_index2 = knn_graph(p2_, k=k, batch=batch)

    inter_dist1 = torch.abs(p1_[edge_index1[0]] - p1_[edge_index1[1]])
    inter_dist1 = reduce(inter_dist1, "(n r) c -> n c", "mean", r=k)
    inter_dist2 = torch.abs(p2_[edge_index2[0]] - p2_[edge_index2[1]])
    inter_dist2 = reduce(inter_dist2, "(n r) c -> n c", "mean", r=k)

    loss1 = torch.abs(inter_dist1 - inter_dist2[idxs1_])
    loss2 = torch.abs(inter_dist2 - inter_dist1[idxs2_])
    loss_rep = (loss1.mean() + loss2.mean()) / 2
    loss_cd = (dist1.mean() + dist2.mean()) / 2
    loss = (loss_rep + loss_cd) / 2

    if return_proportion:
        return loss, loss_rep / loss_cd
    return loss
