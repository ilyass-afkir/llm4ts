"""
LLMFew.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import torch
from einops import rearrange
from torch import nn
from omegaconf import DictConfig
from hydra import initialize, compose
from src.layers.patching import Patching
from layers.patchwise_temporal_convolution_encoder import CausalCNNEncoder
from src.utils.load_llm import LLMLoader
from layers.classification_heads import ClassificationHeadLetsC, ClassificationHeadDeepRange

class LLMFew(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        
        self.patch_length = self.cfg.model.patch_length
        self.patch_stride = self.cfg.model.patch_stride
        self.sequence_length = self.cfg.training.sequence_length
        self.num_patches = (self.sequence_length - self.patch_length) // self.patch_stride + 2
        self.enc_dropout = nn.Dropout(self.cfg.model.enc_dropout)
      
        self.patching = Patching(
            patch_lenght=self.patch_length,
            patch_stride=self.patch_stride
        )

        self.llm_loader = LLMLoader(cfg)
        self.llm, self.tokenizer = self.llm_loader.load_llm_and_tokenizer()
        self.llm = self.llm_loader.prepare_lora(self.llm)
        self.llm = self.llm_loader.define_trainable_params(self.llm)
        _ = self.llm_loader.summarize_configuration(self.llm)
       
        #self.classification_head = ClassificationHeadLetsC(
            #input_dim=self.cfg.llm.hidden_size ,
            #num_classes=self.cfg.training.num_classes,
            #num_cnn_blocks=self.cfg.model.num_cnn_blocks,
            #cnn_channels=self.cfg.model.cnn_channels,
            #kernel_size=self.cfg.model.kernel_size,
            #mlp_hidden=self.cfg.model.mlp_hidden,
            #dropout=self.cfg.model.dropout,
            #pooling_type=self.cfg.model.pooling_type,
            #use_batch_norm=self.cfg.model.use_batch_norm,
        #)

        self.classification_head = ClassificationHeadDeepRange(
            llm_hidden_size=self.cfg.llm.hidden_size,
            num_patches=self.num_patches,
            num_classes=self.cfg.training.num_classes,
            dropout=self.cfg.model.dropout,
            activation=self.cfg.model.activation
        )

        self.casual_cnn_encoder = CausalCNNEncoder(
            in_channels=self.cfg.training.num_channels, 
            channels=self.cfg.model.enc_channels, 
            depth=self.cfg.model.enc_depth,
            reduced_size=self.cfg.model.enc_reduced_size, 
            out_channels=self.cfg.llm.hidden_size,
            kernel_size=self.cfg.model.enc_kernel_size
        )

    def classify(self, x):
        B, _, _ = x.shape
        x = self.patching(x)
        x = self.casual_cnn_encoder(x)
        x = self.enc_dropout(x)
        x = rearrange(x, "(B N) H -> B N H", B=B)
        llm_output = self.llm(inputs_embeds=x).hidden_states[-1]
        logits = self.classification_head(llm_output)
        return logits
    
    def forward(self, x):
        if self.cfg.model.task_name == "classification":
            logits = self.classify(x) 
            return logits
    
if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="main_config")  # or your actual YAML name, e.g. "train.yaml"

    print("Loaded Hydra config successfully.")
    print(cfg.model)

    # dummy batch
    batch_size = 4
    seq_len = cfg.training.sequence_length
    num_features = cfg.training.num_channels
    num_classes = cfg.training.num_classes

    x = torch.randn(batch_size, seq_len, num_features).to(dtype=torch.bfloat16, device="cuda:0")
    y = torch.randint(0, num_classes, (batch_size,)).to(dtype=torch.long, device="cuda:0")

    # model forward
    model = LLMFew(cfg)
    model.to("cuda:0", dtype=torch.bfloat16)
    logits = model(x)

    # dummy loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.float(), y)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n📊 Model Parameter Summary")
    print("-" * 60)
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Non-trainable:         {total_params - trainable_params:,}")
    print("-" * 60)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Class labels: {y.tolist()}")
    print(f"Loss: {loss.item():.4f}")

