"""PromptEmbedding Module.

This module provides a class to generate prompt embeddings for
time-series classification tasks using a pretrained LLM.

    - ``B`` -- batch size
    - ``T`` -- sequence lenght
    - ``C`` -- number of channels (features)

Example:
    >>> x = torch.randn(8, 64, 3) # (B, T, C)
    >>> prompt_module = PromptEmbedding(
    ...     llm=llm,
    ...     tokenizer=tokenizer,
    ...     task_description="Classify sensor signals",
    ...     prompt_max_lenght=128,
    ...     label=None,
    ...     device="cpu",
    ...     num_classes=5,
    ... )
    >>>
    >>> embeddings = prompt_module(x)
    >>> print(embeddings.shape)
    torch.Size([8, 128, 768])  # (batch, prompt_length, embedding_dim)

References:

    .. admonition:: Paper

        Jin, Ming; Wang, Shiyu; Ma, Lintao; Chu, Zhixuan; Zhang, James Y.; Shi, Xiaoming; Chen, 
        Pin-Yu; Liang, Yuxuan; Li, Yuan-Fang; Pan, Shirui; Wen, Qingsong: Time-LLM: Time Series 
        Forecasting by Reprogramming Large Language Models, in: The Twelfth International Conference 
        on Learning Representations, 2024

    .. admonition:: Source Code

        https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py
    
"""

import torch
import torch.nn as nn


class PromptEmbedding(nn.Module):
    """Generates prompt embeddings.

    This module creates textual prompts describing the classification task and
    converts them into embeddings using a pretrained LLM's input embeddings.

    Attributes:
        llm (nn.Module): Pretrained language model providing embeddings.
        tokenizer: Tokenizer compatible with the LLM.
        task_description (str): Text description of the classification task.
        prompt_max_lenght (int): Maximum token length for prompts.
        label (list[str]): Labels for the classification task (optional).
        device (str | torch.device): Device to place tensors on (e.g., "cpu" or "cuda").
        num_classes (int): Number of target classes.
    """

    def __init__(self, 
        llm: nn.Module,
        tokenizer, 
        task_description: str,
        prompt_max_lenght: int,
        label: list[str],
        device: str | torch.device,
        num_classes: int
    ) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.task_description = task_description
        self.prompt_max_lenght = prompt_max_lenght
        self.label = label
        self.device = device
        self.num_classes = num_classes
        
    def scenario_classification_prompt(self, x: torch.Tensor) -> list[str]:
        """Generates textual classification prompts for a batch of time-series data.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where
                B is the batch size,
                T is the number of timesteps,
                C is the number of channels.

        Returns:
            list[str]: List of prompt strings for each sample in the batch.
        """
        B, T, C = x.shape
        prompts = []

        for b in range(B):
            prompt = (
                f"<|prompt|>Classification: {self.task_description} | "
                f"Input: {C} channels × {T} timesteps | "
                f"Classes: {self.num_classes} | "
                f"Task: Classify this time series into the correct class<|end|>"
            )

            prompts.append(prompt)
        
        return prompts
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the prompt embeddings for a batch of time-series inputs.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where
                B is the batch size,
                T is the number of timesteps,
                C is the number of channels.

        Returns:
            torch.Tensor: Embedded prompts with shape corresponding to the
                LLM input embeddings, typically (B, prompt_length, embedding_dim).
        """
        prompt = self.scenario_classification_prompt(x)
        
        tokens = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.prompt_max_lenght
        ).to(self.device)
        
        input_ids = tokens["input_ids"]
        prompt_embedding = self.llm.get_input_embeddings()(input_ids) 
        
        return prompt_embedding




