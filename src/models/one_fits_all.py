"""One Fits All Model.

This module implements the OneFitsAll architecture, a streamlined time-series
classification model that feeds patch-embedded time-series directly into a
frozen (or partially fine-tuned) LLM backbone without any reprogramming or
prompt injection. It is a lightweight alternative to TimeLLM, relying solely
on the LLM's representational capacity to classify time-series data.

Example:
    >>> model = OneFitsAll(cfg).to(cfg.training.device)
    >>> x = torch.randn(8, cfg.training.num_channels, cfg.training.sequence_length)
    >>> x = x.to(cfg.training.device)
    >>> logits = model(x)
    >>> print(logits.shape)
    (8, num_classes)

References:

    .. admonition:: Paper

        Zhou, T. et al.: Time series analysis by pretrained LM (2023)
        Zhou, Tian; Niu, Peisong; Wang, Xue; Sun, Liang; Jin, Rong: One fits all: power general time series
        analysis by pretrained LM, in: Proceedings of the 37th International Conference on Neural Information
        Processing Systems, 2023

    .. admonition:: Source Code

        https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Classification/src/models/gpt4ts.py
"""

import torch
from omegaconf import DictConfig
from torch import nn

from src.layers.classification_heads import ClassificationHeadDeepRange
from src.layers.time_series_embeddings import PatchEmbedding
from src.utils.load_llm import LLMLoader


class OneFitsAll(nn.Module):
    """Time-series classification model that feeds patches directly into an LLM.

    OneFitsAll simplifies the TimeLLM pipeline by removing the reprogramming
    and prompt embedding stages. Raw time-series patches are linearly projected
    into the LLM hidden size and passed straight into the LLM backbone, whose
    final hidden states are consumed by a classification head.

    Attributes:
        cfg (DictConfig): Full Hydra/OmegaConf configuration object. Expected
            top-level keys are ``model``, ``llm``, and ``training``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.patch_length = self.cfg.model.patch_length
        self.patch_stride = self.cfg.model.patch_stride
        self.patch_embedding_dropout = self.cfg.model.patch_embedding_dropout
        self.sequence_length = self.cfg.training.sequence_length
        self.num_patches = (self.sequence_length - self.patch_length) // self.patch_stride + 2
        self.device = self.cfg.training.device

        self.patch_embedding = PatchEmbedding(
            use_linear=True,
            embed_type=None,
            freq=None,
            use_token=False,
            use_positional=False,
            use_temporal=False,
            llm_hidden_size=self.cfg.llm.hidden_size,
            patch_lenght=self.patch_length,
            patch_stride=self.patch_stride,
            dropout=self.cfg.model.patch_embedding_dropout,
            num_channels=self.cfg.training.num_channels,
        )

        self.llm_loader = LLMLoader(cfg)
        self.llm, self.tokenizer = self.llm_loader.load_llm_and_tokenizer()
        self.llm = self.llm_loader.define_trainable_params(self.llm)
        _ = self.llm_loader.summarize_configuration(self.llm)

        self.classification_head = ClassificationHeadDeepRange(
            llm_hidden_size=self.cfg.llm.hidden_size,
            num_patches=self.num_patches,
            num_classes=self.cfg.training.num_classes,
            dropout=self.cfg.model.dropout,
            activation=self.cfg.model.activation,
        )

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Performs time-series classification through the LLM backbone.

        Executes the following steps in order:

        1. Embed time-series patches via :attr:`layers.patch_embedding`.
        2. Pass the embedded patches directly into the LLM backbone.
        3. Extract the final layer hidden states and apply
           :attr:`classification_head`.

        Args:
            x (torch.Tensor): Input time-series tensor of shape
                ``(batch_size, num_channels, sequence_length)``.

        Returns:
            torch.Tensor: Class logits of shape ``(batch_size, num_classes)``.
        """
        x = self.patch_embedding(x)
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
    

