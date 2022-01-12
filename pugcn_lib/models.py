import torch
from torch.nn import Sequential, Linear, ReLU
from .feature_extractor import InceptionFeatureExtractor
from .upsample import NodeShuffle


class PUGCN(torch.nn.Module):
    def __init__(self, opts):
        super(PUGCN, self).__init__()

        # Config
        # Layers
        self.feature_extractor = InceptionFeatureExtractor(opts)
        self.upsampler = NodeShuffle(
            in_channels=opts.channels, k=opts.num_neighbours, r=opts.up_ratio
        )
        self.reconstructor = Sequential(
            Linear(opts.channels, opts.channels),
            ReLU(),
            Linear(opts.channels, 3),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.upsampler(x)
        x = self.reconstructor(x)

        return x
