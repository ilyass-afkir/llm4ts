"""Classification Heads Module.
 
A collection of classification head architectures designed to map LLM or
encoder output representations to class logits for time-series classification tasks.
Each head accepts patch embeddings of shape ``(B, N, H)`` and returns logits
of shape ``(B, K)``, where:
 
    - ``B`` -- batch size
    - ``N`` -- number of patches
    - ``H`` -- LLM hidden dimensionality (e.g. 768 for GPT-2)
    - ``K`` -- number of target classes

Example:
        >>> x = torch.randn(8, 32, 768) # Patch embedding
        >>> head = ClassificationHeadDeepRange(
        ...     llm_hidden_size=768,
        ...     num_patches=32,
        ...     num_classes=6,
        ...     dropout=0.1,
        ...     activation="gelu"
        ... )
        >>> logits = head(x)
        >>> print(logits.shape)
        torch.Size([8, 6])
"""

from einops import rearrange
import torch
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
        activation (str): Name of the activation function. Supported values: ``'gelu'``, ``'relu'``, ``'leaky_relu'``.
    """

    def __init__(
        self,
        llm_hidden_size: int,
        num_patches: int,
        num_classes: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.llm_hidden_size = llm_hidden_size
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.activation = activation
        self.layer_norm = nn.LayerNorm(self.llm_hidden_size * self.num_patches)
        self.linear_projection = nn.Linear(self.llm_hidden_size * self.num_patches, num_classes)
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes class logits from patch embeddings.
 
        Args:
            x (torch.Tensor): Patch embeddings of shape ``(B, N, H)``.
 
        Returns:
            torch.Tensor: Class logits of shape ``(B, K)``.
 
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


class ClassificationHeadTimeLLM(nn.Module):
    """Classification head for the Time-LLM model.
 
    Flattens patch embeddings, applies layer normalization, dropout, and an
    activation function, then projects to class logits via a linear layer.

    References:
        .. admonition:: Paper
 
            Jin, M. et al.: Reprogramming Large Language Models (2024)
            Jin, Ming; Wang, Shiyu; Ma, Lintao; Chu, Zhixuan; Zhang, James Y.; Shi, Xiaoming; Chen, Pin-Yu; Liang,
            Yuxuan; Li, Yuan-Fang; Pan, Shirui; Wen, Qingsong: Time-LLM: Time Series Forecasting by Reprogramming
            Large Language Models, in: The Twelfth International Conference on Learning Representations,
            2024
 
        .. admonition:: Source Code
 
            https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py
 
    Attributes:
        llm_hidden_size (int): Hidden dimensionality of the LLM output.
        num_patches (int): Number of patch embeddings per sample.
        num_classes (int): Number of target classes.
        dropout (nn.Dropout): Dropout layer.
        activation (str): Name of the activation function. Supported values: ``'gelu'``, ``'relu'``, ``'leaky_relu'``.
    """

    def __init__(
        self,
        llm_hidden_size: int,
        num_patches: int,
        num_classes: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.llm_hidden_size = llm_hidden_size
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(self.llm_hidden_size * self.num_patches)
        self.linear_projection = nn.Linear(self.llm_hidden_size * self.num_patches, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes class logits from patch embeddings.
 
        Args:
            x (torch.Tensor): Patch embeddings of shape ``(B, N, H)``.
 
        Returns:
            torch.Tensor: Class logits of shape ``(B, K)``.
 
        Raises:
            ValueError: If ``activation`` is not one of ``'gelu'``, ``'relu'``,
                or ``'leaky_relu'``.
        """
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

    References:
        .. admonition:: Paper
 
            Chen, Y. et al.: LLMs are few-shot multivariate time series classifiers (2025)
            Chen, Yakun; Li, Zihao; Yang, Chao; Wang, Xianzhi; Xu, Guandong: LLMs are few-shot multivariate
            time series classifiers, in: Data Mining and Knowledge Discovery, Vol. 39, pp. 66, 2025
    
        .. admonition:: Source Code
 
            https://github.com/junekchen/llm-fewshot-mtsc/blob/main/LLMFew.py
 
    Attributes:
        llm_hidden_size (int): Hidden dimensionality of the LLM output.
        num_patches (int): Number of patch embeddings per sample.
        num_classes (int): Number of target classes.
        dropout (nn.Dropout): Dropout layer.
        activation (nn.Module): Instantiated activation function module. Supported values: ``'gelu'``, ``'relu'``, ``'leaky_relu'``.
    """

    def __init__(
        self,
        llm_hidden_size: int,
        num_patches: int,
        num_classes: int,
        dropout: float,
        activation: str
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes class logits from patch embeddings.
 
        Args:
            x (torch.Tensor): Patch embeddings of shape ``(B, N, H)`` where
                ``B`` is the batch size, ``N`` is the number of patches, and
                ``H`` is ``llm_hidden_size``.
 
        Returns:
            torch.Tensor: Class logits of shape ``(B, K)``.
        """
        x = rearrange(x, "B N H -> B (N H)")
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.linear(x)
        return logits


class ClassificationHeadLetsC(nn.Module):
    """CNN-based classification head for the LetsC model.
 
    Applies a stack of 1D convolutional blocks to patch embeddings for local
    feature extraction, followed by global pooling and an MLP classifier.
    This avoids flattening the sequence dimension, making it robust to varying
    sequence lengths.

    References:
        .. admonition:: Paper
 
            Kaur, Rachneet, et al. "LETS-C: Leveraging Text Embedding for Time Series Classification." 
            Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics 
            (Volume 1: Long Papers). 2025.
 
    Attributes:
        input_dim (int): Input feature dimensionality (``H`` from patch embeddings).
        num_classes (int): Number of target classes.
        num_cnn_blocks (int): Number of Conv1d blocks to stack.
        cnn_channels (int): Number of output channels for each Conv1d block.
        kernel_size (int): Kernel size for Conv1d layers.
        mlp_hidden (int): Hidden dimensionality of the MLP classifier.
        dropout (float): Dropout probability applied in CNN blocks and MLP.
        pooling_type (str): Global pooling strategy.
            Supported values: ``'avg'`` (adaptive average), ``'max'`` (adaptive max).
        use_batch_norm (bool): If ``True``, inserts ``BatchNorm1d`` after
            each Conv1d layer and disables conv bias.
    """
    
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
    ) -> None:
        super().__init__()
        self.pooling_type = pooling_type
        cnn_layers = []
        in_channels = input_dim
        
        for _ in range(num_cnn_blocks):
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
        
        if pooling_type == "avg":
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == "max":
            self.global_pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unknown pooling: {pooling_type}")
        
        self.classifier = nn.Sequential(
            nn.Linear(cnn_channels, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes class logits via CNN feature extraction and global pooling.
 
        Args:
            x (torch.Tensor): Patch embeddings of shape ``(B, N, H)`` where
                ``B`` is the batch size, ``N`` is the number of patches, and
                ``H`` is the input feature dimensionality.
 
        Returns:
            torch.Tensor: Class logits of shape ``(B, K)``.
        """
        x = rearrange(x, "B N H -> B H N")
        x = self.cnn(x) 
        x = self.global_pool(x)
        x = rearrange(x, "B K 1 -> B K" )
        logits = self.classifier(x)
        return logits
