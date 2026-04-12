"""Time Series Embeddings.

Maps the temporal or feature dimensions of a time series sequence into a
higher-dimensional representation space (e.g., the hidden size of an LLM).
 
Tensor dimension conventions used throughout this module:
 
    - B: batch size
    - T: sequence length (time dimension)
    - C: number of channels / variables
    - N: number of extracted patches
    - P: patch length
    - H: embedding dimension (LLM hidden size)
 
References:

    .. admonition:: Papers

        Jin, Ming; Wang, Shiyu; Ma, Lintao; Chu, Zhixuan; Zhang, James Y.; Shi, Xiaoming;
        Chen, Pin-Yu; Liang, Yuxuan; Li, Yuan-Fang; Pan, Shirui; Wen, Qingsong:
        Time-LLM: Time Series Forecasting by Reprogramming Large Language Models,
        in: The Twelfth International Conference on Learning Representations, 2024

        Nie, Yuqi; Nguyen, Nam H.; Sinthong, Phanwadee; Kalagnanam, Jayant:
        A Time Series is Worth 64 Words: Long-term Forecasting with Transformers,
        in: The Eleventh International Conference on Learning Representations, 2023

    .. admonition:: Source Codes

        https://github.com/KimMeen/Time-LLM/blob/main/layers/Embed.py
        
        https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/Embed.py
"""

import torch
import torch.nn as nn
from einops import rearrange
import math


class PatchEmbedding(nn.Module):
    """Embeds a multivariate time series into patch-level LLM-compatible representations.
 
    Splits each channel's time series into overlapping patches, optionally pads
    the sequence, and projects each patch into the LLM hidden space using either
    a linear layer or a token (convolutional) embedding. Sinusoidal positional
    encodings and temporal (calendar) embeddings can be added on top.
 
    Args:
        use_linear (bool): If ``True``, uses a single ``nn.Linear`` layer to
            project flattened patches of shape ``[patch_length * num_channels]``
            into ``llm_hidden_size``. Mutually exclusive with ``use_token``.
        embed_type (str | None): Embedding variant for the optional temporal
            embedding. Use ``'fixed'`` for sinusoidal-initialised ``nn.Embedding``
            weights, ``'learned'`` for standard ``nn.Embedding``, or ``'timeF'``
            for a linear projection of continuous time features. Ignored when
            ``use_temporal`` is ``False`` or ``use_token`` is ``False``.
        freq (str | None): Sampling frequency token that controls which calendar
            features are built. Supported values for ``TemporalEmbedding``:
            ``'t'`` (minute), ``'h'`` (hour, default). For
            ``TimeFeatureEmbedding`` the mapping is
            ``{'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}``.
            Ignored when ``use_temporal`` is ``False`` or ``use_token`` is ``False``.
        use_token (bool): If ``True``, uses a causal ``TokenEmbedding`` (1-D
            circular convolution) instead of a linear projection. Mutually
            exclusive with ``use_linear``.
        use_positional (bool): If ``True`` and ``use_token`` is ``True``, adds
            sinusoidal positional encodings to the patch embeddings.
        use_temporal (bool): If ``True`` and ``use_token`` is ``True``, adds
            calendar-aware temporal embeddings to the patch embeddings.
        llm_hidden_size (int): Target embedding dimensionality ``H``; should
            match the hidden size of the downstream LLM.
        patch_lenght (int): Number of time steps ``P`` in each patch.
        patch_stride (int): Step size between consecutive patch start positions.
            Also used as the right-side replication padding length so that the
            last patch is always fully populated.
        dropout (float): Dropout probability applied to the final patch
            embeddings.
        num_channels (int): Number of input channels / variables ``C``. Used
            to compute the flattened patch dimension ``P * C``.
 
    Raises:
        ValueError: If both ``use_linear`` and ``use_token`` are ``True``.

    Example:
        >>> embedder = PatchEmbedding(
        ...     use_linear=True,
        ...     embed_type=None,
        ...     freq=None,
        ...     use_token=False,
        ...     use_positional=False,
        ...     use_temporal=False,
        ...     llm_hidden_size=768,
        ...     patch_lenght=16,
        ...     patch_stride=8,
        ...     dropout=0.1,
        ...     num_channels=4,
        ... )
        >>> x = torch.randn(2, 128, 4)  # [B, T, C]
        >>> embedder(x).shape
        torch.Size([2, 16, 768])
    """

    def __init__(
        self, 
        use_linear: bool, 
        embed_type: str | None,
        freq: str | None,
        use_token: bool, 
        use_positional: bool, 
        use_temporal: bool, 
        llm_hidden_size: int, 
        patch_lenght: int,  
        patch_stride: int, 
        dropout: float,
        num_channels: int
    ):
        super().__init__()
        self.patch_lenght = patch_lenght
        self.patch_stride = patch_stride
        self.llm_hidden_size = llm_hidden_size
        self.num_channels = num_channels
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.patch_stride))
        self.patch_embedding_dropout = nn.Dropout(dropout)
        
        if use_linear and use_token:
            raise ValueError("Only one of use_linear or use_token can be True")
        
        if use_linear:
            self.embedding = nn.Linear(self.patch_lenght * self.num_channels, self.llm_hidden_size) # F: [patch lenght, hidden size]
        
        elif use_token:
            self.embedding = TokenEmbedding(self.patch_lenght * self.num_channels, self.llm_hidden_size) # F: [patch lenght, hidden size]
            if use_positional:
                self.positional_embedding = PositionalEmbedding(self.llm_hidden_size)
            if use_temporal:
                if embed_type != 'timeF':
                    self.temporal_embedding = TemporalEmbedding(self.llm_hidden_size, embed_type, freq)
                else:
                    self.temporal_embedding = TimeFeatureEmbedding(self.llm_hidden_size, embed_type, freq)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Converts a raw multivariate time series into patch embeddings.

        Args:
            x (torch.Tensor): Raw time series of shape ``[B, T, C]``.
 
        Returns:
            torch.Tensor: Patch embeddings of shape ``[B, N, H]``, where
            ``N = floor((T + patch_stride - patch_length) / patch_stride) + 1``
            after padding.
        """
        x = rearrange(x, "B T C -> B C T")
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_lenght, step=self.patch_stride)  # [B, C, N, P]
        x = rearrange(x, 'B C N P -> B N (C P)')
        x = self.embedding(x)  # [B, N, H]
        if hasattr(self, "positional_embedding"):
            x = x + self.positional_embedding(x)
        x = self.patch_embedding_dropout(x)
        return x


class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding.
 
    Pre-computes a table of sine/cosine positional encodings up to
    ``max_length`` positions and reads off the first ``T`` rows at forward
    time. The encoding is stored as a non-trainable buffer.
 
    Args:
        llm_hidden_size (int): Embedding dimensionality ``H``. Must be even so
            that sine and cosine values can be interleaved across all
            dimensions.
        max_lenght (int): Maximum sequence length supported. Defaults to
            ``5000``.
 
    Example::
 
        >>> pe = PositionalEmbedding(llm_hidden_size=64)
        >>> x = torch.zeros(2, 10, 64)  # [B, T, H]
        >>> pe(x).shape
        torch.Size([1, 10, 64])
    """
 
    def __init__(self, llm_hidden_size: int, max_lenght: int = 5000):
        super(PositionalEmbedding, self).__init__()
 
        pe = torch.zeros(max_lenght, llm_hidden_size).float()
        pe.require_grad = False
 
        position = torch.arange(0, max_lenght).float().unsqueeze(1)
        div_term = (torch.arange(0, llm_hidden_size, 2).float()
                    * -(math.log(10000.0) / llm_hidden_size)).exp()
 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
 
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns positional encodings for the first ``T`` positions.
 
        Args:
            x (torch.Tensor): Input tensor of shape ``[B, T, H]``. Only
                ``x.size(1)`` (the sequence length) is used; values are
                ignored.
 
        Returns:
            torch.Tensor: Positional encodings of shape ``[1, T, H]``,
            broadcastable over the batch dimension.
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """Projects patch tokens into the embedding space with a 1-D convolution.
 
    Uses a circular-padded ``Conv1d`` with kernel size 3 to mix neighbouring
    time steps within each token, providing a richer inductive bias than a
    plain linear layer. Weights are initialised with Kaiming normal
    (fan-in, leaky ReLU).
 
    Args:
        c_in (int): Number of input features per time step (i.e., the flattened
            patch dimension ``P * C``).
        llm_hidden_size (int): Output embedding dimensionality ``H``.
 
    Example::
 
        >>> te = TokenEmbedding(c_in=64, llm_hidden_size=256)
        >>> x = torch.randn(4, 16, 64)  # [B, N, P*C]
        >>> te(x).shape
        torch.Size([4, 16, 256])
    """
 
    def __init__(self, c_in: int, llm_hidden_size: int):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=llm_hidden_size,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds input tokens via circular convolution.
 
        Permutes the input to ``[B, C_in, N]`` for ``Conv1d``, applies the
        convolution, then transposes back to ``[B, N, H]``.
 
        Args:
            x (torch.Tensor): Patch tokens of shape ``[B, N, P*C]``.
 
        Returns:
            torch.Tensor: Embedded tokens of shape ``[B, N, H]``.
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """Non-trainable sinusoidal ``nn.Embedding`` initialised like positional encodings.
 
    Builds an embedding table with sine/cosine values (identical formula to
    :class:`PositionalEmbedding`) and freezes the weights so they are never
    updated during training. Intended as a drop-in replacement for
    ``nn.Embedding`` inside :class:`TemporalEmbedding`.
 
    Args:
        c_in (int): Vocabulary size, i.e., the number of discrete time
            categories (e.g., 24 for hours, 32 for days).
        llm_hidden_size (int): Embedding dimensionality ``H``.
 
    Example::
 
        >>> fe = FixedEmbedding(c_in=24, llm_hidden_size=128)
        >>> idx = torch.arange(24)
        >>> fe(idx).shape
        torch.Size([24, 128])
    """
 
    def __init__(self, c_in: int, llm_hidden_size: int):
        super().__init__()
 
        w = torch.zeros(c_in, llm_hidden_size).float()
        w.require_grad = False
 
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, llm_hidden_size, 2).float()
                    * -(math.log(10000.0) / llm_hidden_size)).exp()
 
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
 
        self.emb = nn.Embedding(c_in, llm_hidden_size)
        self.emb.weight = nn.Parameter(w, requires_grad=False)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Looks up fixed sinusoidal embeddings for the given indices.
 
        Args:
            x (torch.Tensor): Integer index tensor of arbitrary shape.
 
        Returns:
            torch.Tensor: Detached embedding tensor with an extra trailing
            dimension of size ``H``.
        """
        return self.emb(x).detach()
 
 
class TemporalEmbedding(nn.Module):
    """Calendar-aware embedding built from discrete time-of-day / date indices.
 
    Encodes up to five calendar fields — month, day-of-month, weekday, hour,
    and (optionally) minute — each via its own ``Embedding`` table, and sums
    them into a single dense vector. Supports either fixed sinusoidal weights
    (:class:`FixedEmbedding`) or standard trainable weights (``nn.Embedding``).
 
    The expected input layout along the last axis is::
 
        x[..., 0] = month   (1–12, table size 13)
        x[..., 1] = day     (1–31, table size 32)
        x[..., 2] = weekday (0–6,  table size  7)
        x[..., 3] = hour    (0–23, table size 24)
        x[..., 4] = minute  (0–3,  table size  4)  # only when freq == 't'
 
    Args:
        llm_hidden_size (int): Output embedding dimensionality ``H``.
        embed_type (str): Selects the embedding variant. Use ``'fixed'`` for
            frozen sinusoidal weights via :class:`FixedEmbedding`, or any
            other string (e.g. ``'learned'``) for standard ``nn.Embedding``.
            Defaults to ``'fixed'``.
        freq (str): Sampling frequency. When set to ``'t'``, a minute-level
            embedding table is created in addition to the hourly and coarser
            ones. Defaults to ``'h'``.
 
    Example::
 
        >>> te = TemporalEmbedding(llm_hidden_size=64, embed_type='fixed', freq='h')
        >>> # x contains [month, day, weekday, hour] indices, shape [B, T, 4]
        >>> x = torch.randint(0, 12, (2, 10, 4))
        >>> te(x).shape
        torch.Size([2, 10, 64])
    """
 
    def __init__(self, llm_hidden_size: int, embed_type: str = 'fixed', freq: str = 'h'):
        super().__init__()
 
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
 
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, llm_hidden_size)
        self.hour_embed = Embed(hour_size, llm_hidden_size)
        self.weekday_embed = Embed(weekday_size, llm_hidden_size)
        self.day_embed = Embed(day_size, llm_hidden_size)
        self.month_embed = Embed(month_size, llm_hidden_size)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the sum of all active calendar embeddings.
 
        Casts input to ``long``, looks up each calendar field in its own
        embedding table, and returns the element-wise sum.
 
        Args:
            x (torch.Tensor): Integer calendar-feature tensor of shape
                ``[B, T, F]``, where ``F >= 4`` (``F = 5`` when
                ``freq == 't'``). The column order must be
                ``[month, day, weekday, hour, (minute)]``.
 
        Returns:
            torch.Tensor: Summed temporal embedding of shape ``[B, T, H]``.
        """
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
 
        return hour_x + weekday_x + day_x + month_x + minute_x
 
 
class TimeFeatureEmbedding(nn.Module):
    """Projects continuous time features into the embedding space with a linear layer.
 
    Unlike :class:`TemporalEmbedding`, which looks up discrete calendar
    indices, this module accepts the real-valued time features produced by
    ``tsai`` / ``GluonTS`` ``time_features_from_frequency_str`` helpers and
    projects them linearly into ``llm_hidden_size``.
 
    The number of input features ``d_inp`` is determined by the ``freq``
    argument via the lookup table::
 
        {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
 
    Args:
        d_model (int): Output embedding dimensionality ``H``.
        embed_type (str): Included for API consistency with
            :class:`TemporalEmbedding`; the value is not used internally.
            Defaults to ``'timeF'``.
        freq (str): Sampling frequency string that controls the number of
            continuous time features. Defaults to ``'h'`` (4 features).
 
    Example::
 
        >>> tfe = TimeFeatureEmbedding(d_model=128, freq='h')
        >>> x = torch.randn(2, 10, 4)  # [B, T, d_inp=4 for freq='h']
        >>> tfe(x).shape
        torch.Size([2, 10, 128])
    """
 
    def __init__(self, d_model: int, embed_type: str = 'timeF', freq: str = 'h'):
        super().__init__()
 
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linearly projects continuous time features to the embedding space.
 
        Args:
            x (torch.Tensor): Continuous time-feature tensor of shape
                ``[B, T, d_inp]``, where ``d_inp`` matches the value
                associated with ``freq`` in the frequency map.
 
        Returns:
            torch.Tensor: Projected embeddings of shape ``[B, T, H]``.
        """
        return self.embed(x)
 




