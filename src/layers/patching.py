"""
Patching.
"""

import torch.nn as nn
from einops import rearrange

class Patching(nn.Module):
    def __init__(self, patch_lenght, patch_stride):
        super().__init__()
        self.patch_lenght = patch_lenght
        self.patch_stride = patch_stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.patch_stride))

    def forward(self, x):
        x = rearrange(x, "B T C -> B C T") # Channel independence
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_lenght, step=self.patch_stride) # [B, C, num_patches, patch_len]
        return x