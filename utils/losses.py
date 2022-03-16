from hausdorff import hausdorff_distance
from torch_geometric.utils import to_dense_batch
from torch import Tensor, LongTensor


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
