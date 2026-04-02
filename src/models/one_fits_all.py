"""
One Fits All Model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import torch

from torch import nn
from omegaconf import DictConfig
from hydra import initialize, compose

from layers.time_series_embeddings import PatchBasedDataEmbedding
from layers.classification_heads import ClassificationHeadLetsC, ClassificationHeadDeepRange
from src.utils.load_llm import LLMLoader


class OneFitsAll(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        
        self.patch_length = self.cfg.model.patch_length
        self.patch_stride = self.cfg.model.patch_stride
        self.patch_embedding_dropout = self.cfg.model.patch_embedding_dropout
        self.sequence_length = self.cfg.training.sequence_length
        self.num_patches = (self.sequence_length - self.patch_length) // self.patch_stride + 2
        self.device = self.cfg.training.device
        
        self.patch_embedding  = PatchBasedDataEmbedding(
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
            num_channels=self.cfg.training.num_channels
        )

        self.llm_loader = LLMLoader(cfg)
        self.llm, self.tokenizer = self.llm_loader.load_llm_and_tokenizer()
        self.llm = self.llm_loader.define_trainable_params(self.llm)
        _ = self.llm_loader.summarize_configuration(self.llm)
       
        #self.classification_head = ClassificationHeadLetsC(
           # input_dim=self.cfg.llm.hidden_size ,
            #num_classes=self.cfg.training.num_classes,
            #num_cnn_blocks=self.cfg.model.num_cnn_blocks,
           # cnn_channels=self.cfg.model.cnn_channels,
            #kernel_size=self.cfg.model.kernel_size,
            #mlp_hidden=self.cfg.model.mlp_hidden,
           # dropout=self.cfg.model.dropout,
           # pooling_type=self.cfg.model.pooling_type,
            #use_batch_norm=self.cfg.model.use_batch_norm,
        #)

        self.classification_head = ClassificationHeadDeepRange(
            llm_hidden_size=self.cfg.llm.hidden_size,
            num_patches=self.num_patches,
            num_classes=self.cfg.training.num_classes,
            dropout=self.cfg.model.dropout,
            activation=self.cfg.model.activation
        )

    def classify(self, x):
        x = self.patch_embedding(x)
        llm_output = self.llm(inputs_embeds=x).hidden_states[-1]
        logits = self.classification_head(llm_output)
        return logits
    
    def forward(self, x):
        if self.cfg.model.task_name == "classification":
            logits = self.classify(x) 
            return logits
    
if __name__ == "__main__":
    import gc

    torch.cuda.empty_cache()
    gc.collect()
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="main_config")

    print("Loaded Hydra config successfully.")
    print(cfg.model)

    # Dummy batch
    batch_size = 4
    window_size = cfg.training.sequence_length
    num_features = cfg.training.num_channels
    num_classes = cfg.training.num_classes

    x = torch.randn(batch_size, window_size, num_features).to(dtype=torch.bfloat16, device="cuda:0")
    y = torch.randint(0, num_classes, (batch_size,)).to(dtype=torch.long, device="cuda:0")

    # Model forward
    model = OneFitsAll(cfg)
    model.to("cuda:0", dtype=torch.bfloat16)
    logits = model(x)

    # Dummy loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.float(), y)
    
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params

    # Memory footprint (weights only)
    param_mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3)
    grad_mem_gb = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024 ** 3)
    
    print("\n📊 Model Parameter Summary")
    print("-" * 60)     
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Non-trainable:         {non_trainable:,}")
    print(f"Parameter memory:      {param_mem_gb:.2f} GB (weights only)")
    print(f"Trainable memory:      {grad_mem_gb:.2f} GB")
    print("-" * 60)

    # GPU usage if CUDA
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"CUDA memory allocated: {allocated:.2f} GB")
        print(f"CUDA memory reserved:  {reserved:.2f} GB")
        print("-" * 60)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Class labels: {y.tolist()}")
    print(f"Loss: {loss.item():.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

