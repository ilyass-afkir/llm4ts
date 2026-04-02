"""
Description here.
"""

from torch import nn
from omegaconf import DictConfig
from einops import rearrange

from src.layers.patching import Patching
from layers.patchwise_temporal_convolution_encoder import CausalCNNEncoder
from layers.classification_heads import ClassificationHeadDeepRange
from src.layers.reprogramming_layer import ReprogrammingLayer
from layers.token_prototype_embedding import SourceEmbedding
from src.utils.load_llm import LLMLoader


class DeepRange(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        
        self.patch_length = self.cfg.model.patch_length
        self.patch_stride = self.cfg.model.patch_stride
        self.sequence_length = self.cfg.training.sequence_length
        self.num_patches = (self.sequence_length - self.patch_length) // self.patch_stride + 2
        self.enc_dropout = nn.Dropout(self.cfg.model.enc_dropout)

        self.llm_loader = LLMLoader(cfg)
        self.llm, self.tokenizer = self.llm_loader.load_llm_and_tokenizer()
        self.llm = self.llm_loader.prepare_lora(self.llm)
        self.llm = self.llm_loader.define_trainable_params(self.llm)
        _ = self.llm_loader.summarize_configuration(self.llm)

        self.patching = Patching(
            patch_lenght=self.patch_length,
            patch_stride=self.patch_stride
        )

        self.casual_cnn_encoder = CausalCNNEncoder(
            in_channels=self.cfg.training.num_channels, 
            channels=self.cfg.model.enc_channels, 
            depth=self.cfg.model.enc_depth,
            reduced_size=self.cfg.model.enc_reduced_size, 
            out_channels=self.cfg.llm.hidden_size,
            kernel_size=self.cfg.model.enc_kernel_size
        )

        self.word_embedding = self.llm.get_input_embeddings().weight
        self.source_embedding = SourceEmbedding(
            vocab_size=self.cfg.llm.vocab_size,
            small_vocab_size=self.cfg.model.small_vocab_size,
            word_embedding=self.word_embedding
        )

        self.classification_head = ClassificationHeadDeepRange(
            llm_hidden_size=self.cfg.llm.hidden_size,
            num_patches=self.num_patches,
            num_classes=self.cfg.training.num_classes,
            dropout=self.cfg.model.dropout,
            activation=self.cfg.model.activation
        )
        
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
        B, _, _ = x.shape
        x = self.patching(x)
        x = self.casual_cnn_encoder(x)
        x = rearrange(x, "(B N) H -> B N H", B=B)
        source_embedding = self.source_embedding()
        reprogrammed_embedding = self.reprogramming_layer(x, source_embedding, source_embedding)
        llm_output = self.llm(inputs_embeds=reprogrammed_embedding).hidden_states[-1]
        logits = self.classification_head(llm_output)
        
        return logits
    