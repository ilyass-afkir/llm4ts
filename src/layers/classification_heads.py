"""Classification Heads Module.

A collection of classification head architectures designed to map LLM or
encoder output representations to class logits for time-series classification tasks.
Each head accepts patch embeddings of shape ``(B, N, H)`` and returns logits
of shape ``(B, C)``, where:

- ``B`` -- batch size
- ``N`` -- number of patches
- ``H`` -- LLM hidden dimensionality (e.g. 768 for GPT-2)
- ``C`` -- number of target classes
"""

from einops import rearrange
import torch.nn as nn


class ClassificationHeadOFA(nn.Module):
    """Classification head for the One-Fits-All (OFA) model.
 
    Flattens patch embeddings, applies layer normalization, and projects
    to class logits via a single linear layer. An activation function is
    applied to the raw patch embeddings before flattening.
 
    References:
        .. admonition:: Paper

            Zhou, Tian; Niu, Peisong; Wang, Xue; Sun, Liang; Jin, Rong: One fits all: power general time series
            analysis by pretrained LM, in: Proceedings of the 37th International Conference on Neural Information
            Processing Systems, 2023

        .. admonition:: Source Code

            https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/blob/main/Classification/src/models/gpt4ts.py
    
    Attributes:
        llm_hidden_size (int): Hidden dimensionality of the LLM output.
        num_patches (int): Number of patch embeddings per sample.
        num_classes (int): Number of target classes.
        activation (str): Name of the activation function.
        layer_norm (nn.LayerNorm): Layer normalization over flattened features.
        linear_projection (nn.Linear): Projects flattened features to logits.
    """
    def __init__(
        self,
        llm_hidden_size: int,
        num_patches: int,
        num_classes: int,
        activation: str,
    ):
        """Initializes ClassificationHeadOFA.
 
        Args:
            llm_hidden_size (int): Hidden dimensionality of the LLM output.
            num_patches (int): Number of patch embeddings per sample.
            num_classes (int): Number of target classes.
            activation (str): Activation function to apply before flattening.
                Supported values: ``'gelu'``, ``'relu'``, ``'leaky_relu'``.    
        """
        super().__init__()

        self.llm_hidden_size = llm_hidden_size
        self.num_patches = num_patches
        self.num_classes = num_classes

        self.activation = activation
        self.layer_norm = nn.LayerNorm(self.llm_hidden_size * self.num_patches)
        self.linear_projection = nn.Linear(self.llm_hidden_size * self.num_patches, num_classes)
       
    def forward(self, x):
        """Computes class logits from patch embeddings.
 
        Args:
            x (torch.Tensor): Patch embeddings of shape ``(B, N, H)``.
 
        Returns:
            torch.Tensor: Class logits of shape ``(B, C)``.
 
        Raises:
            ValueError: If ``activation`` is not one of ``'gelu'``, ``'relu'``,
                or ``'leaky_relu'``.
        """
        if self.activation == 'gelu':
            x = nn.GELU(x)
        elif self.activation == 'relu':
            x = nn.ReLU(x)
        elif self.activation == 'leaky_relu':
            x = nn.LeakyReLU(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        x = rearrange(x, "B N H -> B (N H)")

        x = self.layer_norm(x)     
        logits = self.linear_projection(x)
        return logits


class ClassificationHeadLLMFew(nn.Module):
    def __init__(
        self,
        llm_hidden_size: int,
        num_patches: int,
        num_classes: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        self.llm_hidden_size = llm_hidden_size
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.dropout = dropout
        self.activation = activation
        
        self.layer_norm = nn.LayerNorm(self.llm_hidden_size * self.num_patches)
        self.linear_projection = nn.Sequential(
            nn.Linear(self.llm_hidden_size * self.num_patches, self.num_classes),
            nn.Dropout(self.dropout)
        )
       
    def forward(self, x, llm_output):

        if self.activation == 'gelu':
            x = nn.GELU(llm_output + x)
        elif self.activation == 'relu':
            x = nn.ReLU(llm_output + x)
        elif self.activation == 'leaky_relu':
            x = nn.LeakyReLU(llm_output + x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        x = rearrange(x,"B N H -> B (N H)")
        x = self.layer_norm(x)
        logits = self.linear_projection(x)

        return logits


class ClassificationHeadTimeLLM(nn.Module):
    def __init__(
        self,
        llm_hidden_size: int,
        num_patches: int,
        num_classes: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        self.llm_hidden_size = llm_hidden_size
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.layer_norm = nn.LayerNorm(self.llm_hidden_size * self.num_patches)
        self.linear_projection = nn.Linear(self.llm_hidden_size * self.num_patches, self.num_classes)
    
    def forward(self, x):

        x = rearrange(x,"B N H -> B (N H)")

        x = self.layer_norm(x)
        self.dropout(x)
        
        if self.activation == 'gelu':
            x = nn.GELU(x)
        elif self.activation == 'relu':
            x = nn.ReLU(x)
        elif self.activation == 'leaky_relu':
            x = nn.LeakyReLU(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        logits = self.linear_projection(x)
        
        return logits

class ClassificationHeadDeepRange(nn.Module):
    """Classification head for the DeepRange model.
 
    Flattens patch embeddings, applies layer normalization, a configurable
    activation function, dropout, and a linear projection to class logits.
 
    Attributes:
        layer_norm (nn.LayerNorm): Layer normalization over flattened features.
        dropout (nn.Dropout): Dropout layer.
        activation (nn.Module): Instantiated activation function module.
        linear (nn.Linear): Projects flattened features to class logits.
    """
    def __init__(
        self,
        llm_hidden_size: int,
        num_patches: int,
        num_classes: int,
        dropout: float,
        activation: str
    ):
        """Initializes ClassificationHeadDeepRange.
 
        Args:
            llm_hidden_size (int): Hidden dimensionality of the LLM output.
            num_patches (int): Number of patch embeddings per sample.
            num_classes (int): Number of target classes.
            dropout (float): Dropout probability.
            activation (str): Activation function name.
                Supported values: ``'gelu'``, ``'relu'``, ``'leaky_relu'``.
 
        Raises:
            ValueError: If ``activation`` is not a supported value.
        """
        super().__init__()

        hidden_in = llm_hidden_size * num_patches
        self.layer_norm = nn.LayerNorm(hidden_in)

        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "leaky_relu":
            act = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
        self.activation = act
        self.linear = nn.Linear(hidden_in, num_classes)

    def forward(self, x):
        """Computes class logits from patch embeddings.
 
        Args:
            x (torch.Tensor): Patch embeddings of shape ``(B, N, H)`` where
                ``B`` is the batch size, ``N`` is the number of patches, and
                ``H`` is ``llm_hidden_size``.
 
        Returns:
            torch.Tensor: Class logits of shape ``(B, num_classes)``.
        """
        x = rearrange(x, "B N H -> B (N H)")
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.linear(x)
        return logits


class ClassificationHeadLetsC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_cnn_blocks: int,
        cnn_channels: int,
        kernel_size: int,
        mlp_hidden: int,
        dropout: float,
        pooling_type: str,
        use_batch_norm: bool,
    ):
        super().__init__()
        
        self.pooling_type = pooling_type
        cnn_layers = []
        in_channels = input_dim
        
        for _ in range(num_cnn_blocks):
            # Conv block
            cnn_layers.append(nn.Conv1d(
                in_channels, 
                cnn_channels, 
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=not use_batch_norm
            ))
            
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm1d(cnn_channels))
            
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            
            in_channels = cnn_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Global Pooling (instead of flatten!)
        if pooling_type == "avg":
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == "max":
            self.global_pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unknown pooling: {pooling_type}")
        
        # MLP classifier - much smaller than with flatten!
        self.classifier = nn.Sequential(
            nn.Linear(cnn_channels, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )
    
    def forward(self, x):

        x = rearrange(x, "B N H -> B H N")
        
        # CNN feature extraction
        x = self.cnn(x)  # (B, cnn_channels, L)
        
        # Global pooling (KEY: no flatten!)
        x = self.global_pool(x)  # (B, cnn_channels, 1)
        x = rearrange(x, "B C 1 -> B C" )

        # Classification
        logits = self.classifier(x)
        
        return logits
