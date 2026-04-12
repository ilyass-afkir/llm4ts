"""TimeLLM Module.
 
This module defines the TimeLLM architecture, a time-series foundation model
that reprograms patch-level time-series embeddings into the input space of a
frozen (or partially fine-tuned) Large Language Model (LLM) for downstream
classification tasks.
 
Example:
    >>> model = TimeLLM(cfg).to(cfg.training.device)
    >>> x = torch.randn(8, cfg.training.num_channels, cfg.training.sequence_length)
    >>> x = x.to(cfg.training.device)
    >>> logits = model(x)
    >>> print(logits.shape) 
    (8, num_classes)

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
from torch import nn
from omegaconf import DictConfig

from src.layers.time_series_embeddings import PatchEmbedding
from src.layers.prompt_embedding import PromptEmbedding
from src.layers.classification_heads import ClassificationHeadDeepRange
from src.layers.reprogramming_layer import ReprogrammingLayer
from src.layers.token_prototype_embedding import TokenPrototypeEmbedding
from src.utils.load_llm import LLMLoader


class TimeLLM(nn.Module):
    """Time-series classification model built on top of a frozen LLM backbone.
 
    TimeLLM reprograms patch-level time-series representations into the token
    embedding space of a pre-trained LLM using a cross-attention reprogramming
    layer. A learnable text prompt is prepended to the reprogrammed sequence
    before being fed into the LLM, whose final hidden states are consumed by a
    classification head.
 
    Attributes:
        cfg (DictConfig): Hydra configuration object. Expected
            top-level keys are ``model``, ``llm``, and ``training``.
    """
 
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.patch_length = self.cfg.model.patch_length
        self.patch_stride = self.cfg.model.patch_stride
        self.sequence_lenght = self.cfg.training.sequence_length
        self.num_patches = (self.sequence_lenght - self.patch_length) // self.patch_stride + 2
 
        self.llm_loader = LLMLoader(cfg)
        self.llm, self.tokenizer = self.llm_loader.load_llm_and_tokenizer()
        self.llm = self.llm_loader.define_trainable_params(self.llm)
        _ = self.llm_loader.summarize_configuration(self.llm)
 
        self.patch_embedding = PatchEmbedding(
            use_linear=False,
            embed_type=None,
            freq=None,
            use_token=True,
            use_positional=False,
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
 
        self.prompt_embedding = PromptEmbedding(
            llm=self.llm,
            tokenizer=self.tokenizer,
            task_description=self.cfg.model.task_description,
            prompt_max_lenght=self.cfg.model.prompt_max_lenght,
            label=self.cfg.training.label,
            device=self.cfg.training.device,
            num_classes=self.cfg.training.num_classes,
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
        """Performs time-series classification through the full LLM pipeline.
 
        Executes the following steps in order:
 
        1. Embed the task prompt via :attr:`layers.prompt_embedding`.
        2. Embed time-series patches via :attr:`layers.patch_embedding`.
        3. Compute token prototype embeddings via :attr:`layers.token_prototype_embedding`.
        4. Reprogram patch embeddings into the LLM token space via
           :attr:`layers.reprogramming_layer`.
        5. Prepend the prompt embedding and pass the combined sequence through
           the LLM backbone.
        6. Slice out the patch-region hidden states and apply
           :attr:`layers.classification_head`.
 
        The prompt embedding tensor is moved to CPU and deleted after the LLM
        forward pass to conserve GPU memory.
 
        Args:
            x (torch.Tensor): Input time-series tensor of shape
                ``(batch_size, num_channels, sequence_length)``.
 
        Returns:
            torch.Tensor: Class logits of shape ``(batch_size, num_classes)``.
        """
        prompt_embedding = self.prompt_embedding(x)
        patch_embedding = self.patch_embedding(x)
        token_prototype_embedding = self.token_prototype_embedding()

        reprogrammed_embedding = self.reprogramming_layer(
            patch_embedding, token_prototype_embedding, token_prototype_embedding
        )
 
        llm_input = torch.cat([prompt_embedding, reprogrammed_embedding], dim=1)
        llm_output = self.llm(inputs_embeds=llm_input).hidden_states[-1]
        llm_output = llm_output[:, prompt_embedding.shape[1]:, :]
 
        assert llm_output.shape == reprogrammed_embedding.shape
 
        logits = self.classification_head(llm_output)
        return logits