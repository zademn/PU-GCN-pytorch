from dataclasses import dataclass, field
import os


@dataclass
class ModelConfig:
    """Configuration for creating the model"""

    num_point: int = 256  # number of points per sample
    up_ratio: int = 4  # upsampling ratio
    dilation: int = 2  # dilation in DenseGCN
    num_neighbours: int = 20  # num neighbours in DenseGCN
    n_idgcn_blocks: int = 2  # number of inception dense blocks
    channels: int = 32  # number of channels for gcn
    n_dgcn_blocks: int = 3  # number of DenseGCNBlocks in the DenseGCN


@dataclass
class TrainConfig:
    """Configuration for the training step"""

    batch_size: int = 8
    epochs: int = 10
    optimizer = "adam"
    lr = 0.001
    betas = (0.9, 0.999)


@dataclass
class DataConfig:
    """Configuration for loading the data"""

    path: str = os.path.join(
        "data",
        "PU1K",
        "train",
        "pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5",
    )  # path to data directory
    skip_rate: int = 5
    use_randominput: bool = True
    num_point: int = 256


@dataclass
class AllConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
