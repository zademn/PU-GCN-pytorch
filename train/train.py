import sys
import os

sys.path.append("../")
import torch
from config import DataConfig, TrainConfig, ModelConfig
from tqdm import tqdm
from pugcn_lib import PUGCN
from torch_geometric.loader import DataLoader as PyGLoader
from ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import (
    chamfer_3DDist as ChamferLoss,
)
from utils.data import PCDDataset
from datetime import datetime


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, trainloader, loss_fn, optimizer):

    total_loss = 0.0
    for d in (t := tqdm(trainloader)):
        # Extract source and target point clouds and batches
        p, q = d.pos_s.to(device), d.pos_t.to(device)
        if trainloader.follow_batch:
            p_batch, q_batch = d.pos_s_batch.to(device), d.pos_t_batch.to(device)
        else:
            p_batch, q_batch = None, None

        # get batch and target dimesnions since chamfer loss will need rehsaping
        b, nq = p_batch.max().item() + 1, q.shape[0]

        # Train step
        optimizer.zero_grad()

        pred = model(p, batch=p_batch)
        d1, d2, _, _ = loss_fn(pred.reshape(b, nq // b, 3), q.reshape(b, nq // b, 3))
        loss = d1.mean() + d2.mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        t.set_description(f"loss = {loss.item() :.4f}")
    return total_loss


@torch.no_grad()
def evaluate(model, valloader, loss_fn):

    total_loss = 0.0
    for d in (t := tqdm(valloader)):
        # Extract source and target point clouds and batches
        p, q = d.pos_s.to(device), d.pos_t.to(device)
        if trainloader.follow_batch:
            p_batch, q_batch = d.pos_s_batch.to(device), d.pos_t_batch.to(device)
        else:
            p_batch, q_batch = None, None

        # get batch and target dimesnions since chamfer loss will need rehsaping
        b, nq = p_batch.max().item() + 1, q.shape[0]

        pred = model(p, batch=p_batch)
        d1, d2, _, _ = loss_fn(pred.reshape(b, nq // b, 3), q.reshape(b, nq // b, 3))
        loss = d1.mean() + d2.mean()

        total_loss += loss.item()
        t.set_description(f"loss = {loss.item() :.4f}")
    return total_loss


if __name__ == "__main__":
    print(f"Training on {device}")

    torch.cuda.set_per_process_memory_fraction(0.8)

    # Create folder to save checkpoints based on current date
    if "trained-models" not in os.listdir("."):
        os.mkdir("trained-models")
    dir_name = datetime.now().strftime("%d_%m_%Y_%H_%M")
    os.mkdir(os.path.join("trained-models", dir_name))
    print(f"Saving checkpoints at {dir_name}")
    # Load configs
    data_config = DataConfig(
        path=os.path.join(
            "..",
            "..",
            "data",
            "PU1K",
            "train",
            "pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5",
        )
    )
    train_config = TrainConfig()
    model_config = ModelConfig()
    # config = AllConfig(data_config=data_config)

    # Setup dataset
    trainset = PCDDataset(
        data_config.path,
        data_config.num_point,
        up_ratio=model_config.up_ratio,
        skip_rate=data_config.skip_rate,
    )
    trainloader = PyGLoader(
        trainset,
        batch_size=train_config.batch_size,
        follow_batch=["pos_s", "pos_t"],
    )

    # Setup model

    pugcn = PUGCN(
        channels=model_config.channels,
        k=model_config.num_neighbours,
        r=model_config.up_ratio,
        n_idgcn_blocks=model_config.n_idgcn_blocks,
        n_dgcn_blocks=model_config.n_dgcn_blocks,
        use_bottleneck=True,
        use_pooling=True,
        use_residual=True,
    ).to(device)

    loss_fn = loss_fn = ChamferLoss()
    optimizer = torch.optim.Adam(
        params=pugcn.parameters(),
        lr=train_config.lr,
    )

    for epoch in tqdm(range(train_config.epochs)):
        total_loss = train(pugcn, trainloader, loss_fn, optimizer)
        print(f"epoch: {epoch} \t train loss: {total_loss}")

        torch.save(
            {
                "model_config": model_config.__dict__,
                "epoch": epoch,
                "model_state_dict": pugcn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join("trained-models", dir_name, f"ckpt_epoch-{epoch}"),
        )
