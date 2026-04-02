"""
Description here.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import torch
import gc

from torch import nn
from omegaconf import DictConfig
from hydra import initialize, compose
import torch
from torch import nn    
from einops import rearrange
from omegaconf import DictConfig

from layers.time_series_embeddings import PatchBasedDataEmbedding
from src.layers.prompt_embedding import PromptEmbedding
from layers.classification_heads import ClassificationHeadLetsC, ClassificationHeadDeepRange
from src.layers.reprogramming_layer import ReprogrammingLayer
from layers.token_prototype_embedding import SourceEmbedding
from src.utils.load_llm import LLMLoader


class TimeLLM(nn.Module):
    def __init__(self, cfg: DictConfig):
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

        self.patch_embedding = PatchBasedDataEmbedding(
            use_linear=False,
            embed_type=None,
            freq=None,
            use_token=True,
            use_positional=False,
            use_temporal=False,
            llm_hidden_size= self.cfg.llm.hidden_size,
            patch_lenght=self.patch_length,
            patch_stride=self.patch_stride,
            dropout=self.cfg.model.patch_embedding_dropout,
            num_channels=self.cfg.training.num_channels
        )
        
        self.word_embedding = self.llm.get_input_embeddings().weight
        self.source_embedding = SourceEmbedding(
            vocab_size=self.cfg.llm.vocab_size,
            small_vocab_size=self.cfg.model.small_vocab_size,
            word_embedding=self.word_embedding
        )

        self.prompt_embedding = PromptEmbedding(
            llm=self.llm,
            tokenizer=self.tokenizer,
            task_description=self.cfg.model.task_description,
            prompt_max_lenght=self.cfg.model.prompt_max_lenght,
            label=self.cfg.training.label,
            device=self.cfg.training.device,
            num_classes=self.cfg.training.num_classes
        )
        
        #self.classification_head = ClassificationHeadLetsC(
            #input_dim=self.cfg.llm.hidden_size ,
           # num_classes=self.cfg.training.num_classes,
            #num_cnn_blocks=self.cfg.model.num_cnn_blocks,
           # cnn_channels=self.cfg.model.cnn_channels,
           # kernel_size=self.cfg.model.kernel_size,
            #mlp_hidden=self.cfg.model.mlp_hidden,
            #dropout=self.cfg.model.dropout,
            #pooling_type=self.cfg.model.pooling_type,
           # use_batch_norm=self.cfg.model.use_batch_norm,
       # )
        self.classification_head = ClassificationHeadDeepRange(
            llm_hidden_size=self.cfg.llm.hidden_size,
            num_patches=self.num_patches,
            num_classes=self.cfg.training.num_classes,
            dropout=self.cfg.model.dropout,
            activation=self.cfg.model.activation
        )
        
        # Reprogramming layer
        self.reprogramming_layer = ReprogrammingLayer(
            llm_hidden_size=self.cfg.llm.hidden_size, 
            num_attention_heads=self.cfg.llm.num_attention_heads,
            attention_dropout=self.cfg.model.attention_dropout
        )

    def forward(self, x):
        if self.cfg.model.task_name == "classification":
            logits = self.classify(x) 
            return logits
     
    def classify(self, x):

        # Embedding
        prompt_embedding = self.prompt_embedding(x)
        patch_embedding = self.patch_embedding(x)
        source_embedding = self.source_embedding()
        reprogrammed_embedding = self.reprogramming_layer(patch_embedding, source_embedding, source_embedding)

        llm_input = torch.cat([prompt_embedding, reprogrammed_embedding], dim=1)
        llm_output = self.llm(inputs_embeds=llm_input).hidden_states[-1]
        llm_output = llm_output[:, prompt_embedding.shape[1]:, :]
        assert llm_output.shape == reprogrammed_embedding.shape

        logits = self.classification_head(llm_output)

        prompt_embedding.to("cpu")
        del prompt_embedding
        torch.cuda.empty_cache()
        gc.collect()
   
        return logits
    
 
if __name__ == "__main__":
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="main_config")  # or your actual YAML name, e.g. "train.yaml"

    print("Loaded Hydra config successfully.")
    print(cfg.model)

    # Dummy batch
    batch_size = 4
    window_size = cfg.training.sequence_length
    num_features = cfg.training.num_channels
    num_classes = cfg.training.num_classes

    x = torch.randn(batch_size, window_size, num_features).to(dtype=torch.bfloat16, device="cuda:0")
    y = torch.randint(0, num_classes, (batch_size,)).to(dtype=torch.long, device="cuda:0")

    # model forward
    model = TimeLLM(cfg)
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
    print(f"\nInput shape: {x.shape}")
    print(f"Output (logits) shape: {logits.shape}")
    print(f"Class labels: {y.tolist()}")
    print(f"Loss: {loss.item():.4f}")