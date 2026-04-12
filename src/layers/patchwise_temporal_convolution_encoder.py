"""Patchwise Temporal Convolution Encoder Module.
 
Provides a causal CNN encoder that processes time-series patches
channel-independently using dilated causal convolutions with residual
connections, followed by adaptive pooling and a linear projection.

    - ``B`` -- batch size
    - ``T`` -- sequence lenght
    - ``C`` -- number of channels (features)

References:
        .. admonition:: Paper
 
            Chen, Y. et al.: LLMs are few-shot multivariate time series classifiers (2025)
            Chen, Yakun; Li, Zihao; Yang, Chao; Wang, Xianzhi; Xu, Guandong: LLMs are few-shot multivariate
            time series classifiers, in: Data Mining and Knowledge Discovery, Vol. 39, pp. 66, 2025
    
        .. admonition:: Source Code
 
            https://github.com/junekchen/llm-fewshot-mtsc/blob/main/CasualCNN.py
Example:
        >>> x = torch.randn(8, 3, 16, 64) # (B, C, num_patches, patch_length)
        >>> encoder = CausalCNNEncoder(
        ...     in_channels=3,
        ...     channels=32,
        ...     depth=4,
        ...     reduced_size=128,
        ...     out_channels=256,
        ...     kernel_size=3,
        ... )
        >>> embeddings = encoder(x)
        >>> print(embeddings.shape)
        torch.Size([128, 256])
"""

import torch
from einops import rearrange


class Chomp1d(torch.nn.Module):
    """Removes trailing elements from a 1D convolution output.
 
    Used after padded causal convolutions to restore the original
    sequence length and ensure causality — no future information leaks
    into the current timestep.
 
    Attributes:
        chomp_size (int): Number of trailing elements to remove from
            the time axis.
    """
    
    def __init__(self, chomp_size: int) -> None:
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Removes trailing elements from the time axis.
 
        Args:
            x (torch.Tensor): Input tensor of shape ``(B, C, T)``.
 
        Returns:
            torch.Tensor: Tensor of shape ``(B, C, T - chomp_size)``.
        """
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """Squeezes the third dimension of a three-dimensional tensor.
 
    Used after adaptive pooling to remove the singleton time dimension,
    converting ``(B, C, 1)`` to ``(B, C)``.
    """
 
    def __init__(self) -> None:
        super(SqueezeChannels, self).__init__()
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Squeezes the third dimension of the input tensor.
 
        Args:
            x (torch.Tensor): Input tensor of shape ``(B, C, 1)``.
 
        Returns:
            torch.Tensor: Squeezed tensor of shape ``(B, C)``.
        """
        return x.squeeze(2)
 

class CausalConvolutionBlock(torch.nn.Module):
    """A residual block of two dilated causal convolutions.
 
    Each block consists of two weight-normalised causal Conv1d layers
    with LeakyReLU activations, followed by a residual connection. If
    ``in_channels != out_channels``, a 1×1 convolution is used to
    match dimensions. An optional final activation can be disabled for
    the last block in a network.
 
    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of the non-residual convolutions.
        dilation (int): Dilation factor of the non-residual convolutions.
        final (bool, optional): If ``True``, disables the final activation
            function after the residual addition. Defaults to ``False``.
        causal (nn.Sequential): Sequential block of two causal convolutions,
            chomping layers, and activations.
        upordownsample (nn.Conv1d or None): 1×1 convolution for residual
            channel matching, or ``None`` if channels already match.
        relu (nn.LeakyReLU or None): Final activation applied after the
            residual addition, or ``None`` if ``final=True``.
    """
 
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        final: bool = False
    ) -> None:
        super(CausalConvolutionBlock, self).__init__()
 
        # Left padding to ensure causality
        padding = (kernel_size - 1) * dilation
 
        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()
 
        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()
 
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )
 
        # 1x1 convolution for residual channel matching if needed
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None
 
        self.relu = torch.nn.LeakyReLU() if final else None
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies causal convolutions and residual connection.
 
        Args:
            x (torch.Tensor): Input tensor of shape ``(B, in_channels, T)``.
 
        Returns:
            torch.Tensor: Output tensor of shape ``(B, out_channels, T)``.
        """
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)
 

class CausalCNN(torch.nn.Module):
    """A stack of dilated causal convolution blocks with exponentially growing dilation.
 
    Builds a causal CNN by stacking :class:`CausalConvolutionBlock` layers
    where the dilation doubles at each depth level, allowing the network to
    capture long-range temporal dependencies efficiently.
 
    Attributes:
        in_channels (int): Number of input channels.
        channels (int): Number of channels in intermediate blocks.
        depth (int): Number of intermediate causal convolution blocks.
            The total number of blocks is ``depth + 1``.
        out_channels (int): Number of output channels of the final block.
        kernel_size (int): Kernel size for all non-residual convolutions.
        network (nn.Sequential): Sequential stack of
            :class:`CausalConvolutionBlock` layers.
    """
 
    def __init__(
        self,
        in_channels: int,
        channels: int,
        depth: int,
        out_channels: int,
        kernel_size: int
    ) -> None:
        super(CausalCNN, self).__init__()
 
        layers = []
        dilation_size = 1
 
        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Double dilation at each step
 
        # Final block projects to out_channels
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]
 
        self.network = torch.nn.Sequential(*layers)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input through the causal CNN stack.
 
        Args:
            x (torch.Tensor): Input tensor of shape ``(B, in_channels, T)``.
 
        Returns:
            torch.Tensor: Output tensor of shape ``(B, out_channels, T)``.
        """
        return self.network(x)
 

class CausalCNNEncoder(torch.nn.Module):
    """Encodes time-series patches into fixed-size representations.
 
    Applies a :class:`CausalCNN` to each patch independently, reduces the
    time axis to a single value via adaptive max pooling, squeezes the
    singleton dimension, and projects to the output embedding size via a
    linear layer.
 
    Attributes:
        in_channels (int): Number of input channels.
        channels (int): Number of channels in intermediate causal CNN blocks.
        depth (int): Depth of the causal CNN.
        reduced_size (int): Fixed length to which the causal CNN output
            is reduced before the linear projection.
        out_channels (int): Number of output embedding dimensions.
        kernel_size (int): Kernel size of the non-residual convolutions.
        network (nn.Sequential): Sequential pipeline of causal CNN,
            adaptive max pooling, channel squeezing, and linear projection.
    """
 
    def __init__(
        self,
        in_channels: int,
        channels: int,
        depth: int,
        reduced_size: int,
        out_channels: int,
        kernel_size: int
    ) -> None:
        super().__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()
        linear = torch.nn.Linear(reduced_size, out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of time-series patches into fixed-size embeddings.
 
        Reshapes the input to process each patch independently across the
        batch and channel dimensions, then applies the encoder network.
 
        Args:
            x (torch.Tensor): Patch tensor of shape
                ``(B, C, num_patches, patch_length)`` where ``B`` is the
                batch size, ``C`` is the number of channels, and
                ``num_patches`` is the number of patches.
 
        Returns:
            torch.Tensor: Encoded patch embeddings of shape
                ``(B * num_patches, out_channels)``.
        """
        x = rearrange(x, "B C N P -> (B N) C P")
        return self.network(x)
 