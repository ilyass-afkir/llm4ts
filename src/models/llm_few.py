"""LLMFew Module.

This module implements the LLMFew architecture, a time-series classification
model that combines a causal CNN encoder with a LoRA-finetuned LLM backbone.
Unlike TimeLLM, there is no reprogramming or prompt injection — instead, raw
time-series patches are encoded by a lightweight causal CNN before being passed
directly into the LLM as token embeddings.

Example:
    >>> model = LLMFew(cfg).to(cfg.training.device)
    >>> x = torch.randn(8, cfg.training.num_channels, cfg.training.sequence_length)
    >>> x = x.to(cfg.training.device)
    >>> logits = model(x)
    >>> print(logits.shape)
    (8, num_classes)

References:
        .. admonition:: Paper
 
            Chen, Y. et al.: LLMs are few-shot multivariate time series classifiers (2025)
            Chen, Yakun; Li, Zihao; Yang, Chao; Wang, Xianzhi; Xu, Guandong: LLMs are few-shot multivariate
            time series classifiers, in: Data Mining and Knowledge Discovery, Vol. 39, pp. 66, 2025
    
        .. admonition:: Source Code
 
            https://github.com/junekchen/llm-fewshot-mtsc/blob/main/LLMFew.py

"""

import torch
from einops import rearrange
from omegaconf import DictConfig
from torch import nn

from src.layers.classification_heads import ClassificationHeadDeepRange
from src.layers.patching import Patching
from src.layers.patchwise_temporal_convolution_encoder import CausalCNNEncoder
from src.utils.load_llm import LLMLoader


class LLMFew(nn.Module):
    """Time-series classification model using a causal CNN encoder and LoRA LLM.

    LLMFew segments the input time-series into patches, encodes them with a
    causal CNN into the LLM hidden dimension, and feeds the resulting sequence
    directly into a LoRA-adapted LLM backbone. The final hidden states are
    passed to a deep classification head to produce class logits.

    Attributes:
        cfg (DictConfig): Hydra configuration object. Expected
                top-level keys are ``model``, ``llm``, and ``training``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.patch_length = self.cfg.model.patch_length
        self.patch_stride = self.cfg.model.patch_stride
        self.sequence_length = self.cfg.training.sequence_length
        self.num_patches = (self.sequence_length - self.patch_length) // self.patch_stride + 2
        self.enc_dropout = nn.Dropout(self.cfg.model.enc_dropout)

        self.patching = Patching(
            patch_lenght=self.patch_length,
            patch_stride=self.patch_stride,
        )

        self.llm_loader = LLMLoader(cfg)
        self.llm, self.tokenizer = self.llm_loader.load_llm_and_tokenizer()
        self.llm = self.llm_loader.prepare_lora(self.llm)
        self.llm = self.llm_loader.define_trainable_params(self.llm)
        _ = self.llm_loader.summarize_configuration(self.llm)

        self.classification_head = ClassificationHeadDeepRange(
            llm_hidden_size=self.cfg.llm.hidden_size,
            num_patches=self.num_patches,
            num_classes=self.cfg.training.num_classes,
            dropout=self.cfg.model.dropout,
            activation=self.cfg.model.activation,
        )

        self.casual_cnn_encoder = CausalCNNEncoder(
            in_channels=self.cfg.training.num_channels,
            channels=self.cfg.model.enc_channels,
            depth=self.cfg.model.enc_depth,
            reduced_size=self.cfg.model.enc_reduced_size,
            out_channels=self.cfg.llm.hidden_size,
            kernel_size=self.cfg.model.enc_kernel_size,
        )

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Performs time-series classification through the causal CNN and LLM.

        Executes the following steps in order:

        1. Segment the input into patches via :attr:`layers.patching`.
        2. Encode each patch into the LLM hidden dimension via
           :attr:`layers.casual_cnn_encoder`.
        3. Apply encoder dropout via :attr:`enc_dropout`.
        4. Rearrange the flat ``(B*N, H)`` output back to ``(B, N, H)``.
        5. Pass the patch sequence into the LLM backbone and extract the
           final layer hidden states.
        6. Apply :attr:`layers.classification_head` to produce class logits.

        Args:
            x (torch.Tensor): Input time-series tensor of shape
                ``(batch_size, num_channels, sequence_length)``.

        Returns:
            torch.Tensor: Class logits of shape ``(batch_size, num_classes)``.
        """
        B, _, _ = x.shape
        x = self.patching(x)
        x = self.casual_cnn_encoder(x)
        x = self.enc_dropout(x)
        x = rearrange(x, "(B N) H -> B N H", B=B)
        llm_output = self.llm(inputs_embeds=x).hidden_states[-1]
        logits = self.classification_head(llm_output)
        return logits

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
