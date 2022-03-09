from hausdorff import hausdorff_distance
from torch_geometric.utils import to_dense_batch


def hausdorff_loss(p, q, batches=None):
    """
    Using https://github.com/mavillan/py-hausdorff
    Args:
        p: [N, C]
        q: [N, C]
        batches: None or Tuple of batch array for p and q
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
