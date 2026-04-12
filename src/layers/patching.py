"""Patching Module.

Provides a patching layer that segments multivariate time series into
overlapping or non-overlapping patches for patch-based transformer models.

Example:
        >>> patching = Patching(patch_length=16, patch_stride=8)
        >>> x = torch.randn(8, 512, 7)   # [B, T, C]
        >>> out = patching(x)            # [B, C, num_patches, patch_length]
"""

import torch
import torch.nn as nn
from einops import rearrange


class Patching(nn.Module):
    """Segments a multivariate time series into fixed-length patches.

    Applies replication padding to the time axis before unfolding the
    sequence into patches of length ``patch_length`` with stride
    ``patch_stride``. Operates channel-independently.

    Attributes:
        patch_length (int): Length of each patch.
        patch_stride (int): Stride between consecutive patches.
        padding_patch_layer (nn.ReplicationPad1d): Replication padding
            applied along the time axis before unfolding.
    """

    def __init__(self, patch_lenght: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_lenght = patch_lenght
        self.patch_stride = patch_stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.patch_stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segments the input time series into patches.

        Args:
            x (torch.Tensor): Multivariate time series of shape
                ``(B, T, C)`` where ``B`` is the batch size, ``T`` is
                the sequence length, and ``C`` is the number of channels.

        Returns:
            torch.Tensor: Patch tensor of shape
                ``(B, C, num_patches, patch_length)``.
        """
        x = rearrange(x, "B T C -> B C T") # Channel independence
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_lenght, step=self.patch_stride) # [B, C, num_patches, patch_len]
        return x