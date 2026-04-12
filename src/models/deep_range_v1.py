"""DeepRange Module.

This module implements the DeepRange architecture, a time-series classification
model that combines a causal CNN encoder with a reprogramming layer and a
LoRA-adapted LLM backbone. Raw time-series patches are encoded by a causal CNN,
reprogrammed into the LLM token space via cross-attention against a compressed
vocabulary prototype, and then passed into the LLM for classification.


Example:
    >>> model = DeepRange(cfg).to(cfg.training.device)
    >>> x = torch.randn(8, cfg.training.num_channels, cfg.training.sequence_length)
    >>> x = x.to(cfg.training.device)
    >>> logits = model(x)
    >>> print(logits.shape)  
    (8, num_classes)
"""

import torch
from einops import rearrange
from omegaconf import DictConfig
from torch import nn

from src.layers.classification_heads import ClassificationHeadDeepRange
from src.layers.patchwise_temporal_convolution_encoder import CausalCNNEncoder
from src.layers.reprogramming_layer import ReprogrammingLayer
from src.layers.patching import Patching
from src.layers.token_prototype_embedding import TokenPrototypeEmbedding
from src.utils.load_llm import LLMLoader


class DeepRange(nn.Module):
    """Time-series classification model combining a causal CNN, reprogramming, and LoRA LLM.

    DeepRange extends LLMFew by adding a token prototype embedding and a
    reprogramming layer between the causal CNN encoder and the LLM backbone.
    Patch embeddings are cross-attended against a compressed vocabulary
    prototype before being fed into the LoRA-adapted LLM, aligning the
    time-series representations with the LLM token space.

    Attributes:
        cfg (DictConfig): Full Hydra/OmegaConf configuration object. 
            Expected top-level keys are ``model``, ``llm``, and ``training``.

    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.patch_length = self.cfg.model.patch_length
        self.patch_stride = self.cfg.model.patch_stride
        self.sequence_length = self.cfg.training.sequence_length
        self.num_patches = (self.sequence_length - self.patch_length) // self.patch_stride + 2
        self.enc_dropout = nn.Dropout(self.cfg.model.enc_dropout)

        self.llm_loader = LLMLoader(cfg)
        self.llm, self.tokenizer = self.llm_loader.load_llm_and_tokenizer()
        self.llm = self.llm_loader.prepare_lora(self.llm)
        self.llm = self.llm_loader.define_trainable_params(self.llm)
        _ = self.llm_loader.summarize_configuration(self.llm)

        self.patching = Patching(
            patch_lenght=self.patch_length,
            patch_stride=self.patch_stride,
        )

        self.casual_cnn_encoder = CausalCNNEncoder(
            in_channels=self.cfg.training.num_channels,
            channels=self.cfg.model.enc_channels,
            depth=self.cfg.model.enc_depth,
            reduced_size=self.cfg.model.enc_reduced_size,
            out_channels=self.cfg.llm.hidden_size,
            kernel_size=self.cfg.model.enc_kernel_size,
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
        """Performs time-series classification through the full DeepRange pipeline.

        Executes the following steps in order:

        1. Segment the input into patches via :attr:`layers.patching`.
        2. Encode each patch into the LLM hidden dimension via
           :attr:`layers.casual_cnn_encoder`.
        3. Rearrange the flat ``(B*N, H)`` output back to ``(B, N, H)``.
        4. Compute source prototype embeddings via
           :attr:`layers.token_prototype_embedding`.
        5. Reprogram patch embeddings into the LLM token space via
           :attr:`layers.reprogramming_layer`.
        6. Pass the reprogrammed sequence into the LLM backbone and extract
           the final layer hidden states.
        7. Apply :attr:`layers.classification_head` to produce class logits.

        Args:
            x (torch.Tensor): Input time-series tensor of shape
                ``(batch_size, num_channels, sequence_length)``.

        Returns:
            torch.Tensor: Class logits of shape ``(batch_size, num_classes)``.
        """
        B, _, _ = x.shape
        x = self.patching(x)
        x = self.casual_cnn_encoder(x)
        x = rearrange(x, "(B N) H -> B N H", B=B)
        token_prototype_embedding = self.token_prototype_embedding()
        reprogrammed_embedding = self.reprogramming_layer(
            x, token_prototype_embedding, token_prototype_embedding
        )
        llm_output = self.llm(inputs_embeds=reprogrammed_embedding).hidden_states[-1]
        logits = self.classification_head(llm_output)
        return logits