"""Time Series Embeddings.

Maps the temporal or feature dimensions of a time series sequence into a higher-dimensional 
representation space e.g. the hidden size of the LLM.

References:
    Jin, Ming; Wang, Shiyu; Ma, Lintao; Chu, Zhixuan; Zhang, James Y.; Shi, Xiaoming; 
    Chen, Pin-Yu; Liang,Yuxuan; Li, Yuan-Fang; Pan, Shirui; Wen, Qingsong: Time-LLM: 
    Time Series Forecasting by Reprogramming Large Language Models, in: The Twelfth 
    International Conference on Learning Representations, 2024

    Nie, Yuqi; Nguyen, Nam H.; Sinthong, Phanwadee; Kalagnanam, Jayant: A Time Series 
    is Worth 64 Words: Long-term Forecasting with Transformers, in: The Eleventh 
    International Conference on Learning Representations, 2023

Source codes: 
    https://github.com/KimMeen/Time-LLM/blob/main/layers/Embed.py
    https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/Embed.py
"""

import torch
import torch.nn as nn
from einops import rearrange
import math

class PatchEmbedding(nn.Module):

    def __init__(
        self, 
        use_linear: bool, 
        embed_type: str | None,
        freq: str | None,
        use_token: bool, 
        use_positional: bool, 
        use_temporal: bool, 
        llm_hidden_size: int, 
        patch_lenght: int,  
        patch_stride: int, 
        dropout: float,
        num_channels: int
    ):
        super().__init__()
        
        self.patch_lenght = patch_lenght
        self.patch_stride = patch_stride
        self.llm_hidden_size = llm_hidden_size
        self.num_channels = num_channels
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.patch_stride))
        self.patch_embedding_dropout = nn.Dropout(dropout)
        
        if use_linear and use_token:
            raise ValueError("Only one of use_linear or use_token can be True")
        
        if use_linear:
            self.embedding = nn.Linear(self.patch_lenght * self.num_channels, self.llm_hidden_size) # F: [patch lenght, hidden size]
        
        elif use_token:
            self.embedding = TokenEmbedding(self.patch_lenght * self.num_channels, self.llm_hidden_size) # F: [patch lenght, hidden size]
            if use_positional:
                self.positional_embedding = PositionalEmbedding(self.llm_hidden_size)
            if use_temporal:
                if embed_type != 'timeF':
                    self.temporal_embedding = TemporalEmbedding(self.llm_hidden_size, embed_type, freq)
                else:
                    self.temporal_embedding = TimeFeatureEmbedding(self.llm_hidden_size, embed_type, freq)
    
    def forward(self, x):
        x = rearrange(x, "B T C -> B C T") # Channel independence
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_lenght, step=self.patch_stride) # [B, C, num_patches, patch_len]
        x = rearrange(x, 'B C N P -> B N (C P)') # F: "B C N P -> (B C) N P"
        x = self.embedding(x) # [B, num_patches, llm_hidden_size]
        if hasattr(self, "positional_embedding"):
            x = x + self.positional_embedding(x)
        x = self.patch_embedding_dropout(x)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, llm_hidden_size, max_lenght=5000):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_lenght, llm_hidden_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_lenght).float().unsqueeze(1)
        div_term = (torch.arange(0, llm_hidden_size, 2).float()
                    * -(math.log(10000.0) / llm_hidden_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, llm_hidden_size):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=llm_hidden_size,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, llm_hidden_size):
        super().__init__()

        w = torch.zeros(c_in, llm_hidden_size).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, llm_hidden_size, 2).float()
                    * -(math.log(10000.0) / llm_hidden_size)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, llm_hidden_size)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, llm_hidden_size, embed_type='fixed', freq='h'):
        super().__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, llm_hidden_size)
        self.hour_embed = Embed(hour_size, llm_hidden_size)
        self.weekday_embed = Embed(weekday_size, llm_hidden_size)
        self.day_embed = Embed(day_size, llm_hidden_size)
        self.month_embed = Embed(month_size, llm_hidden_size)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super().__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)




