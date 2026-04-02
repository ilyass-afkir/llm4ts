"""
OH OK
"""

import json
import logging
from collections import OrderedDict, defaultdict
from typing import List, Dict, Any
from pathlib import Path

import torch.nn as nn
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ModelSummarizer:

    def __init__(self, cfg: DictConfig, save_results_dir_path: Path):
        self.save_results_dir_path = save_results_dir_path
        self.cfg = cfg
    
    def summarize_model_parameters(self, model: nn.Module, data_name: str | None) -> Dict[str, Any]:

        def count_params(module: nn.Module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            percent = (trainable / total * 100) if total > 0 else 0
            return total, trainable, round(percent, 4)

        breakdown = OrderedDict()
        for name, module in model.named_children():
            total, trainable, trainable_percent = count_params(module)
            breakdown[name] = {
                "total": total,
                "trainable": trainable,
                "trainable_percent": trainable_percent
            }

        # Compute overall totals
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = total_params - trainable_params
        trainable_percent = (trainable_params / total_params * 100) if total_params > 0 else 0

        # Add percent_of_total_trainable per module
        for v in breakdown.values():
            v["percent_of_total_trainable"] = round(
                (v["trainable"] / trainable_params * 100) if trainable_params > 0 else 0, 4
            )

        summary_json = {
            "model_name": self.cfg.model.name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable,
            "trainable_percent": round(trainable_percent, 4),
            "breakdown": breakdown
        }

        # Save JSON
        save_path = self.save_results_dir_path / f"model_parameters_{data_name}.json"
        with open(save_path, "w") as f:
            json.dump(summary_json, f, indent=4)
        logger.info(f"Saved model summary to {save_path}")

        return summary_json
    
    def average_model_summaries(self) -> Dict[str, Any]:

        json_files = list(self.save_results_dir_path.glob("model_parameters_*.json"))
        
        logger.info(f"Found {len(json_files)} files to average")
        
        # Load all JSON files
        summaries = []
        for file_path in json_files:
            with open(file_path, 'r') as f:
                summaries.append(json.load(f))
        
        n_files = len(summaries)
        
        # Average top-level numeric fields
        avg_summary = {
            "model_name": summaries[0]["model_name"],
            "num_files_averaged": n_files,
            "total_parameters": sum(s["total_parameters"] for s in summaries) / n_files,
            "trainable_parameters": sum(s["trainable_parameters"] for s in summaries) / n_files,
            "non_trainable_parameters": sum(s["non_trainable_parameters"] for s in summaries) / n_files,
            "trainable_percent": sum(s["trainable_percent"] for s in summaries) / n_files,
        }
        
        # Average breakdown per module
        all_modules = set()
        for summary in summaries:
            all_modules.update(summary["breakdown"].keys())
        
        breakdown_avg = {}
        for module_name in all_modules:
            # Collect values for this module across all files
            module_data = defaultdict(list)
            for summary in summaries:
                if module_name in summary["breakdown"]:
                    mod = summary["breakdown"][module_name]
                    module_data["total"].append(mod["total"])
                    module_data["trainable"].append(mod["trainable"])
                    module_data["trainable_percent"].append(mod["trainable_percent"])
                    module_data["percent_of_total_trainable"].append(mod["percent_of_total_trainable"])
            
            # Compute averages
            n_samples = len(module_data["total"])
            breakdown_avg[module_name] = {
                "total": sum(module_data["total"]) / n_samples,
                "trainable": sum(module_data["trainable"]) / n_samples,
                "trainable_percent": round(sum(module_data["trainable_percent"]) / n_samples, 4),
                "percent_of_total_trainable": round(sum(module_data["percent_of_total_trainable"]) / n_samples, 4),
                "present_in_n_files": n_samples
            }
        
        avg_summary["breakdown"] = breakdown_avg
        
        # Round top-level values
        avg_summary["total_parameters"] = round(avg_summary["total_parameters"], 2)
        avg_summary["trainable_parameters"] = round(avg_summary["trainable_parameters"], 2)
        avg_summary["non_trainable_parameters"] = round(avg_summary["non_trainable_parameters"], 2)
        avg_summary["trainable_percent"] = round(avg_summary["trainable_percent"], 4)
        
        output_path = self.save_results_dir_path / "model_parameters_averaged.json"
        with open(output_path, 'w') as f:
            json.dump(avg_summary, f, indent=4)
        
        logger.info(f"Saved averaged summary to {output_path}")
        
        return avg_summary
    


