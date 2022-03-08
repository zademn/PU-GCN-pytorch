from typing import Tuple
import torch
from torch.nn import Sequential, Linear, ReLU
from .feature_extractor import InceptionFeatureExtractor
from .upsample import NodeShuffle
from pugcn_lib.torch_nn import MLP


class PUGCN(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        k: int,
        dilations: Tuple[int],
        r: int,
        n_idgcn_blocks: int,
        n_dgcn_blocks: int,
        **idgcn_kwargs
    ):
        """
        Args:
            channels: int
                number of channels. Used in InceptionFeatureExtractor, NodeShuffle and the reconstructor
            k: int
                number of neighbours for constructing the knn graph
            dilations: Tuple[int]
                the dilations that the DenseGCNs from each InceptionDenseGCN block will have.
            n_idgcn_blocks: int
                number of Inception DenseGCN blocks in the feature extractor
            n_dgcn_blocks: int
                number of  DenseGCN blocks in each InceptionDenseGCN block in the feature extractor
            **idgcn_kwargs: args
                Key arguments for `InceptionDenseGCN`: use_bottleneck: bool = False, use_pooling = False, use_residual = False
        """

        super(PUGCN, self).__init__()

        # Config
        # Layers
        self.feature_extractor = InceptionFeatureExtractor(
            channels=channels,
            k=k,
            dilations=dilations,
            n_idgcn_blocks=n_idgcn_blocks,
            n_dgcn_blocks=n_dgcn_blocks,
            **idgcn_kwargs
        )
        self.upsampler = NodeShuffle(in_channels=channels, k=k, r=r)

        self.reconstructor = MLP([channels, channels, 3])

    def forward(self, x, batch=None):
        """
        x: Tensor
            Node feature matrix of all point clouds concatenated [N, C]
        batch: Optional[LongTensor]
            batch tensor [N, ] as described in PyG docs. For example if we have 2 graphs with 2 nodes each we will have [0, 0, 1, 1]
        """
        x = self.feature_extractor(x, batch)
        x = self.upsampler(x, batch)
        x = self.reconstructor(x)

        return x
