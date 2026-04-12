"""Reprogramming Layer Module.

Aligns time-series patch embeddings with natural-language representations
to enable time-series reasoning in the LLM backbone.

Tensor dimension conventions used throughout this module:

    - ``B`` -- batch size
    - ``N`` -- number of patches
    - ``H`` -- LLM hidden dimensionality
    - ``S`` -- number of source token embeddings
    - ``h`` -- number of attention heads
    - ``d`` -- per-head feature dimension (``H // h``)

Example::

    >>> layer = ReprogrammingLayer(llm_hidden_size=768, num_attention_heads=12)
    >>> patches = torch.randn(8, 16, 768)     # [B, N, H]
    >>> tok_emb = torch.randn(1000, 768)      # [S, H]
    >>> out = layer(patches, tok_emb, tok_emb)
    >>> out.shape
    torch.Size([8, 16, 768])

References:

    .. admonition:: Paper

        Jin, Ming; Wang, Shiyu; Ma, Lintao; Chu, Zhixuan; Zhang, James Y.; Shi, Xiaoming;
        Chen, Pin-Yu; Liang, Yuxuan; Li, Yuan-Fang; Pan, Shirui; Wen, Qingsong:
        Time-LLM: Time Series Forecasting by Reprogramming Large Language Models,
        in: The Twelfth International Conference on Learning Representations, 2024

    .. admonition:: Source Code

        https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py
"""

from math import sqrt

from einops import rearrange
import torch
import torch.nn as nn


class ReprogrammingLayer(nn.Module):
    """Reprogramming layer using cross-attention to align patch embeddings with an LLM's token space.

    Projects time-series patch embeddings (queries) into the vocabulary space
    of a pre-trained LLM by attending over a set of source token embeddings
    (keys/values). This allows the LLM to process time-series patches as if
    they were natural-language tokens, without fine-tuning the LLM itself.

    Args:
        llm_hidden_size (int): Hidden dimensionality of the target LLM
            (e.g. 768 for GPT-2, 4096 for LLaMA-7B). Must be divisible
            by ``num_attention_heads``.
        num_attention_heads (int): Number of parallel attention heads. Each
            head operates on a subspace of size
            ``llm_hidden_size // num_attention_heads``.
        attention_dropout (float): Dropout probability applied to the attention
            weight matrix before the weighted sum over values. Defaults to
            ``0.1``.
    """

    def __init__(
        self,
        llm_hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.llm_hidden_size = llm_hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = llm_hidden_size // num_attention_heads
        self.query_projection = nn.Linear(llm_hidden_size, llm_hidden_size)
        self.key_projection   = nn.Linear(llm_hidden_size, llm_hidden_size)
        self.value_projection = nn.Linear(llm_hidden_size, llm_hidden_size)
        self.out_projection   = nn.Linear(llm_hidden_size, llm_hidden_size)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        target_embedding: torch.Tensor,
        source_embedding: torch.Tensor,
        value_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Reprojects time-series patch embeddings into the LLM token space via cross-attention.

        Each patch embedding attends over all source token embeddings, producing
        a convex combination of their value projections. The result lies in the
        same space as the LLM's input embeddings and can be forwarded directly
        through the frozen LLM.

        Args:
            target_embedding (torch.Tensor): Patch embeddings acting as queries,
                of shape ``[B, N, H]``.
            source_embedding (torch.Tensor): Mapped LLM token embeddings acting
                as keys, of shape ``[S, H]``. Shared across the batch (no
                leading batch dimension).
            value_embedding (torch.Tensor): Embeddings used to form the attended
                values, of shape ``[S, H]``. In the original Time-LLM
                formulation this is identical to ``source_embedding``, but kept
                separate to allow independent key/value sources.

        Returns:
            torch.Tensor: Reprogrammed patch representations of shape
            ``[B, N, H]``, matching ``target_embedding``.
        """
        B, N, H = target_embedding.shape
        S, _ = source_embedding.shape

        # Multi-head projections
        Q = rearrange(self.query_projection(target_embedding), "B N (h d) -> B h N d", h=self.num_attention_heads)
        K = rearrange(self.key_projection(source_embedding), "S (h d) -> h S d", h=self.num_attention_heads)
        V = rearrange(self.value_projection(value_embedding), "S (h d) -> h S d", h=self.num_attention_heads)

        # Scaled dot-product attention
        scale = 1.0 / sqrt(self.head_dim)
        attn_scores = torch.einsum("B h N d, h S d -> B h N S", Q, K) * scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.einsum("B h N S, h S d -> B h N d", attn_probs, V)
        out = rearrange(out, "B h N d -> B N (h d)")
        out = self.out_projection(out)
        return out

