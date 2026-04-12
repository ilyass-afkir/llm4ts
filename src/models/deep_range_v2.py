"""DeepRangeV2 Module.

This module implements the DeepRangeV2 architecture, an evolution of DeepRange
that replaces the causal CNN encoder with a patch embedding layer that supports
positional encoding and token-level embeddings. As in DeepRange, patch
embeddings are reprogrammed into the LLM token space via cross-attention against
a compressed vocabulary prototype before being passed into a LoRA-adapted LLM.

Example:
    >>> cfg = OmegaConf.load("configs/default.yaml")
    >>> model = DeepRangeV2(cfg).to(cfg.training.device)
    >>> x = torch.randn(8, cfg.training.num_channels, cfg.training.sequence_length)
    >>> x = x.to(cfg.training.device)
    >>> logits = model(x)
    >>> print(logits.shape)  
    (8, num_classes)
"""

import torch
from omegaconf import DictConfig
from torch import nn

from src.layers.classification_heads import ClassificationHeadDeepRange
from src.layers.reprogramming_layer import ReprogrammingLayer
from src.layers.time_series_embeddings import PatchEmbedding
from src.layers.token_prototype_embedding import TokenPrototypeEmbedding
from src.utils.load_llm import LLMLoader


class DeepRangeV2(nn.Module):
    """Time-series classification model using patch embeddings, reprogramming, and LoRA LLM.

    DeepRangeV2 replaces the causal CNN encoder from DeepRange with a richer
    patch embedding layer that supports positional and token-level encodings.
    Patch embeddings are cross-attended against a compressed vocabulary
    prototype before being fed into the LoRA-adapted LLM, aligning the
    time-series representations with the LLM token space.

    Attributes:
        cfg (DictConfig): Full Hydra configuration object. Expected
            top-level keys are ``model``, ``llm``, and ``training``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.patch_length = self.cfg.model.patch_length
        self.patch_stride = self.cfg.model.patch_stride
        self.sequence_length = self.cfg.training.sequence_length
        self.num_patches = (self.sequence_length - self.patch_length) // self.patch_stride + 2

        self.llm_loader = LLMLoader(cfg)
        self.llm, self.tokenizer = self.llm_loader.load_llm_and_tokenizer()
        self.llm = self.llm_loader.prepare_lora(self.llm)
        self.llm = self.llm_loader.define_trainable_params(self.llm)
        _ = self.llm_loader.summarize_configuration(self.llm)

        self.patch_embedding = PatchEmbedding(
            use_linear=False,
            embed_type=None,
            freq=None,
            use_token=True,
            use_positional=True,
            use_temporal=False,
            llm_hidden_size=self.cfg.llm.hidden_size,
            patch_lenght=self.patch_length,
            patch_stride=self.patch_stride,
            dropout=self.cfg.model.patch_embedding_dropout,
            num_channels=self.cfg.training.num_channels,
        )

        self.word_embedding = self.llm.get_input_embeddings().weight
        self.token_prototype_embedding = TokenPrototypeEmbedding(
            vocab_size=self.cfg.llm.vocab_size,
            small_vocab_size=self.cfg.model.small_vocab_size,
            word_embedding=self.word_embedding,
        )

        self.classification_head = ClassificationHeadDeepRange(
            llm_hidden_size=self.cfg.llm.hidden_size,
            num_patches=self.num_patches,
            num_classes=self.cfg.training.num_classes,
            dropout=self.cfg.model.dropout,
            activation=self.cfg.model.activation,
        )

        self.reprogramming_layer = ReprogrammingLayer(
            llm_hidden_size=self.cfg.llm.hidden_size,
            num_attention_heads=self.cfg.llm.num_attention_heads,
            attention_dropout=self.cfg.model.attention_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass for the configured task.

        Currently supports ``"classification"`` as the only task. Dispatches
        to :meth:`classify` and returns its output directly.

        Args:
            x (torch.Tensor): Input time-series tensor of shape
                ``(batch_size, num_channels, sequence_length)``.

        Returns:
            torch.Tensor: Task-specific output tensor. For classification,
                a logit matrix of shape ``(batch_size, num_classes)``.
        """
        if self.cfg.model.task_name == "classification":
            logits = self.classify(x)
            return logits

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Performs time-series classification through the full DeepRangeV2 pipeline.

        Executes the following steps in order:

        1. Embed time-series patches via :attr:`layers.patch_embedding`.
        2. Compute source prototype embeddings via :attr:`layers.token_prototype_embedding`.
        3. Reprogram patch embeddings into the LLM token space via
           :attr:`layers.reprogramming_layer`.
        4. Pass the reprogrammed sequence into the LLM backbone and extract
           the final layer hidden states.
        5. Apply :attr:`layers.classification_head` to produce class logits.

        Args:
            x (torch.Tensor): Input time-series tensor of shape
                ``(batch_size, num_channels, sequence_length)``.

        Returns:
            torch.Tensor: Class logits of shape ``(batch_size, num_classes)``.
        """
        patch_embedding = self.patch_embedding(x)
        token_prototype_embedding = self.token_prototype_embedding()
        reprogrammed_embedding = self.reprogramming_layer(
            patch_embedding, token_prototype_embedding, token_prototype_embedding
        )
        llm_output = self.llm(inputs_embeds=reprogrammed_embedding).hidden_states[-1]
        logits = self.classification_head(llm_output)
        return logits