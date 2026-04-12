"""LLM Loading and Configuration Module.
 
Provides utilities for loading pre-trained LLMs and tokenizers with 
optional quantization, layer truncation, LoRA fine-tuning, and trainable 
parameter configuration.

Example:
    >>> loader = LLMLoader(cfg)
    >>> llm, tokenizer = loader.load_llm_and_tokenizer()
    >>> llm = loader.truncate_llm_layers(llm)
    >>> llm = loader.prepare_lora(llm)
    >>> llm = loader.define_trainable_params(llm)
    >>> _ = loader.summarize_configuration(llm)
"""

import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class LLMLoader:
    """Loads and configures a pre-trained LLM for time-series reasoning.
 
    Handles tokenizer and model loading with optional 4-bit or 8-bit
    quantization, layer truncation, LoRA adapter injection, and
    trainable parameter selection. Configuration summaries are saved
    to disk as JSON.
 
    Attributes:
        cfg (DictConfig): Hydra configuration object.
            Must contain ``llm`` subconfig sub-configs with the fields
            ``name``, ``dir``, ``quantization``, ``num_hidden_layers``,
            ``output_attentions``, ``output_hidden_states``, ``lora_config``, and
            ``training.save_results_dir_path``. And ``model`` sub-configs with the field 
            ``trainable_llm_params``.
    """
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.name = self.cfg.llm.name
        self.model_dir = self.cfg.llm.model_dir
        self.quantization = self.cfg.llm.quantization
        self.num_hidden_layers = self.cfg.llm.num_hidden_layers
        self.output_attentions = self.cfg.llm.output_attentions
        self.output_hidden_states = self.cfg.llm.output_hidden_states
        self.trainable_llm_params = self.cfg.model.trainable_llm_params
        self.lora_config = self.cfg.model.lora_config
        self.results_dir_training = Path(self.cfg.training.results_dir_training)
       
    def load_llm_and_tokenizer(self) ->  tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Loads the tokenizer and LLM with optional quantization.
 
        Loads the tokenizer from ``dir`` and the causal LLM with the
        configured quantization. If the tokenizer has no pad token,
        the EOS token is used as a fallback, or a new ``[PAD]`` token
        is added if EOS is also absent.
 
        Returns:
            tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple of
                ``(llm, tokenizer)`` ready for downstream use.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        
        quantization_config = None
        if self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        llm = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            device_map="auto",
            quantization_config=quantization_config,
            output_attentions=self.output_attentions, 
            output_hidden_states=self.output_hidden_states,
            local_files_only=True,
            dtype=torch.bfloat16
        )
        
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                llm.config.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                llm.resize_token_embeddings(len(tokenizer))
                llm.config.pad_token_id = tokenizer.pad_token_id

        #layers_info = "\n".join(f"{name}: {type(module)}" for name, module in llm.named_modules())
        logging.info(
            f"{self.name} loaded on device: {next(llm.parameters()).device}\n"
            f"Quantization: {self.quantization}\n"
            f"Memory footprint: {llm.get_memory_footprint()/1e9:.2f} GB\n"
            #f"Layers:\n{layers_info}"
        )
        
        return llm, tokenizer

    def truncate_llm_layers(self, llm) -> nn.Module:
        """Truncates the LLM to the configured number of hidden layers.
 
        Supports GPT-2 style (``llm.h``), GPT-2 with transformer wrapper
        (``llm.transformer.h``), and LLaMA/Mistral style
        (``llm.model.layers``) architectures. If truncation fails,
        a warning is logged and all layers are kept.
 
        Args:
            llm (nn.Module): Loaded causal LLM to truncate.
 
        Returns:
            nn.Module: LLM with at most ``num_hidden_layers`` transformer layers.
        """
        try:
            if hasattr(llm, 'h'):
                if len(llm.h) > self.num_hidden_layers:
                    llm.h = llm.h[:self.num_hidden_layers]
                    logger.info(f"{self.name} loaded with {len(llm.h)} layers")
            
            elif hasattr(llm, 'transformer') and hasattr(llm.transformer, 'h'):
                if len(llm.transformer.h) > self.num_hidden_layers:
                    llm.transformer.h = llm.transformer.h[:self.num_hidden_layers]
                    logger.info(f"{self.name} loaded with {len(llm.transformer.h)} layers")
            
            elif hasattr(llm, 'model') and hasattr(llm.model, 'layers'):
                if len(llm.model.layers) > self.num_hidden_layers:
                    llm.model.layers = nn.ModuleList(llm.model.layers[:self.num_hidden_layers])
                    logger.info(f"{self.name} loaded with {len(llm.model.layers)} layers")
            else:
                raise AttributeError("Could not find layers attribute")
                 
        except Exception as e:
            logger.warning(f"Truncation of {self.name} layers failed. Loading all layers. {e}")
        
        return llm
    
    def prepare_lora(self, llm) -> nn.Module:
        """Injects LoRA adapters into the LLM using the configured LoRA settings.
 
        Wraps the LLM with PEFT's ``get_peft_model`` using a
        :class:`peft.LoraConfig` built from ``lora_config``.
 
        Args:
            llm (nn.Module): LLM to inject LoRA adapters into.
 
        Returns:
            nn.Module: PEFT-wrapped LLM with LoRA adapters applied.
        """
        lora_config = LoraConfig(
            r = self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            target_modules=self.lora_config["target_modules"],
            lora_dropout=self.lora_config["lora_dropout"],
            bias=self.lora_config["bias"]
        )

        llm = get_peft_model(llm, lora_config)
        
        return llm
    
    def define_trainable_params(self, llm) -> nn.Module:
        """Sets trainable parameters based on name pattern matching.
 
        Iterates over all LLM parameters and enables gradients only for
        those whose name matches any pattern in ``trainable_llm_params``.
        Non-floating-point parameters are always frozen.
 
        Args:
            llm (nn.Module): LLM whose parameters to configure.
 
        Returns:
            nn.Module: LLM with ``requires_grad`` set per ``trainable_llm_params``.
        """
        for name, param in llm.named_parameters():
            if any(pattern in name for pattern in self.trainable_llm_params):
                if param.dtype.is_floating_point:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
        
        total_trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in llm.parameters())

        logger.info(
            f"Total trainable LLM params: {total_trainable:,} / {total_params:,} "
            f"({100 * total_trainable / total_params:.2f}%)"
        )

        return llm
    
    def summarize_configuration(self, llm) -> dict[str, str | int | float]:
        """Saves a JSON summary of the LLM configuration and parameter counts.
 
        Computes total, trainable, and non-trainable parameter counts,
        and saves a summary alongside the LoRA and training configuration
        to ``save_results_dir_path/llm_summary.json``.
 
        Args:
            llm (nn.Module): Configured LLM to summarize.
 
        Returns:
            dict[str, str | int | float]: Summary dictionary containing model name, quantization,
                parameter counts, trainable percentage, device, dtype,
                and memory footprint.
        """

        total_params = sum(p.numel() for p in llm.parameters())
        trainable_params = sum(p.numel() for p in llm.parameters() if p.requires_grad)
        trainable_percent = (trainable_params / total_params) * 100 if total_params > 0 else 0

        summary = {
            "llm_name": self.name,
            "quantization": self.quantization,
            "num_hidden_layers": self.num_hidden_layers,
            "output_attentions": self.output_attentions,
            "output_hidden_states": self.output_hidden_states,
            "trainable_param_patterns": OmegaConf.to_container(self.trainable_llm_params, resolve=True),
            "lora_config": OmegaConf.to_container(self.lora_config, resolve=True),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "trainable_percent": round(trainable_percent, 4),
            "device": str(next(llm.parameters()).device),
            "dtype": str(next(llm.parameters()).dtype),
            "memory_footprint_GB": round(llm.get_memory_footprint() / 1e9, 2)
        }

        save_path = self.results_dir / "llm_summary.json"
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        return summary

