"""
Description here!
"""

from typing import List
import torch.nn as nn


class PromptEmbedding(nn.Module):
    def __init__(self, 
        llm, 
        tokenizer, 
        task_description, 
        prompt_max_lenght, 
        label,
        device,
        num_classes
    ):
        super().__init__()
        
        self.llm = llm
        self.tokenizer = tokenizer
        self.task_description = task_description
        self.prompt_max_lenght = prompt_max_lenght
        self.label = label
        self.device = device
        self.num_classes = num_classes
        
    def _scenario_classification_prompt(self, x) -> List[str]:
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
    
    def forward(self, x):

        prompt = self._scenario_classification_prompt(x)
        
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




