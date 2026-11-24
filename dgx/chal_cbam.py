"""CBAM for CHAL. The public CHAL repository imports `nets/module/CBAM.py` but does not include it (20 Oct 2026).
This restores it from the two attention parts that ARE in the repository (nets/util.py), combined exactly as the
repository's own Res_CBAM_block combines them: channel attention, then spatial attention, each multiplied onto
the features (Woo et al., ECCV 2018). Copied into /opt/CHAL/nets/module/ by dgx/Dockerfile.istd.
"""
import torch.nn as nn

from ..util import ChannelAttention, SpatialAttention


class CBAM(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        return self.sa(x) * x
