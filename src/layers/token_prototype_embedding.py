"""
Source Embedding.
"""

import torch
import torch.nn as nn
from einops import rearrange

class SourceEmbedding(nn.Module):
    def __init__(self, vocab_size: int, small_vocab_size: int, word_embedding: torch.Tensor):
        super().__init__()

        self.vocab_size = vocab_size
        self.small_vocab_size = small_vocab_size
        self.linear_projection = nn.Linear(vocab_size, small_vocab_size)
        self.word_embedding = word_embedding

    def forward(self) -> torch.Tensor:
        word_embedding = rearrange(self.word_embedding, "N H -> H N")
        source_embedding = self.linear_projection(word_embedding)
        source_embedding = rearrange(source_embedding, "H N -> N H")
        return source_embedding
