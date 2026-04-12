"""Module for Token Prototype Embedding.

This module provides a mechanism to project a large vocabulary embedding
into a smaller embedding space using a linear projection.

Example:
    >>> vocab_size, small_vocab_size, embedding_dim = 1000, 100, 512
    >>> word_embedding = torch.randn(vocab_size, embedding_dim)
    >>> model = TokenPrototypeEmbedding(vocab_size, small_vocab_size, word_embedding)
    >>> projected_embedding = model()
    >>> print(projected_embedding.shape)
    torch.Size([1000, 100])
    
References:

    .. admonition:: Paper

        Jin, Ming; Wang, Shiyu; Ma, Lintao; Chu, Zhixuan; Zhang, James Y.; Shi, Xiaoming;
        Chen, Pin-Yu; Liang, Yuxuan; Li, Yuan-Fang; Pan, Shirui; Wen, Qingsong:
        Time-LLM: Time Series Forecasting by Reprogramming Large Language Models,
        in: The Twelfth International Conference on Learning Representations, 2024

    .. admonition:: Source Code

        https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py
"""

import torch
import torch.nn as nn
from einops import rearrange


class TokenPrototypeEmbedding(nn.Module):
    """Token Prototype Embedding Module.

    Projects a large vocabulary embedding into a smaller embedding space.

    Attributes:
        vocab_size: Size of the original vocabulary.
        small_vocab_size: Size of the target (smaller) vocabulary.
        linear_projection: Linear layer for projection.
        word_embedding: Original word embedding tensor.
    """

    def __init__(self, vocab_size: int, small_vocab_size: int, word_embedding: torch.Tensor):
        super().__init__()
        self.vocab_size = vocab_size
        self.small_vocab_size = small_vocab_size
        self.linear_projection = nn.Linear(vocab_size, small_vocab_size)
        self.word_embedding = word_embedding

    def forward(self) -> torch.Tensor:
        """Projects the word embedding into the smaller vocabulary space.

        Returns:
            torch.Tensor: Projected embedding tensor of shape (vocab_size, small_vocab_size).
        """
        word_embedding = rearrange(self.word_embedding, "N H -> H N")
        source_embedding = self.linear_projection(word_embedding)
        source_embedding = rearrange(source_embedding, "H N -> N H")
        return source_embedding
