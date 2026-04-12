"""Utilities for summarizing and aggregating PyTorch model parameter statistics.

This module provides :class:`ModelSummarizer`, which inspects a
:class:`~torch.nn.Module`, counts total and trainable parameters per
top-level child module, serializes the results to JSON, and can later
average those JSON files across multiple dataset runs to produce a single
consolidated report.

Typical usage inside a training loop::

    from src.utils.model_summarizer import ModelSummarizer

    summarizer = ModelSummarizer(cfg, save_dir)

    # After instantiating the model for each dataset:
    summary = summarizer.summarize_model_parameters(model, dataset_name)

    # After all datasets have been processed:
    avg = summarizer.average_model_summaries()

Example:
    >>> summarizer = ModelSummarizer(cfg, Path("results/"))
    >>>
    >>> model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    >>> summary = summarizer.summarize_model_parameters(model, "ArticularyWordRecognition")
    >>> summary["trainable_parameters"]
    9610
    >>>
    >>> avg = summarizer.average_model_summaries()
    >>> avg["num_files_averaged"]
    1
"""

import json
import logging
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch.nn as nn
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ModelSummarizer:
    """Computes, saves, and aggregates parameter summaries for PyTorch models.

    For each model inspected via :meth:`summarize_model_parameters` a JSON
    file is written to ``save_results_dir_path``. After all runs are complete,
    :meth:`average_model_summaries` reads every previously written file and
    produces a single averaged report — useful when the same architecture is
    trained on multiple datasets that may alter the number of classes or
    input channels.

    Attributes:
        cfg (DictConfig): Hydra/OmegaConf configuration object. The field
            ``cfg.model.name`` is embedded in every summary JSON.
        save_results_dir_path (Path): Directory in which JSON summaries
            are written. The directory must already exist; this class
            does not create it.

    Example:
        >>> from pathlib import Path
        >>> from omegaconf import OmegaConf
        >>> import torch.nn as nn
        >>> cfg = OmegaConf.create({"model": {"name": "my_model"}})
        >>> summarizer = ModelSummarizer(cfg, Path("results/"))
        >>> model = nn.Linear(32, 10)
        >>> summary = summarizer.summarize_model_parameters(model, "dataset_A")
        >>> summary["total_parameters"]
        330
    """

    def __init__(self, cfg: DictConfig, save_results_dir_path: Path) -> None:
        self.save_results_dir_path = save_results_dir_path
        self.cfg = cfg

    def summarize_model_parameters(
        self, model: nn.Module, data_name: str | None
    ) -> dict[str, object]:
        """Counts parameters per top-level child module and writes a JSON summary.

        Iterates over every direct child returned by
        :meth:`~torch.nn.Module.named_children`, counts total and trainable
        parameters for each, and also computes overall model-level totals.
        The result is serialized to
        ``<save_results_dir_path>/model_parameters_<data_name>.json``.

        The returned (and saved) dictionary has the following structure::

            {
                "model_name": str,
                "total_parameters": int,
                "trainable_parameters": int,
                "non_trainable_parameters": int,
                "trainable_percent": float,
                "breakdown": {
                    "<child_name>": {
                        "total": int,
                        "trainable": int,
                        "trainable_percent": float,
                        "percent_of_total_trainable": float
                    },
                    ...
                }
            }

        Args:
            model (nn.Module): The PyTorch model to inspect. All parameters
                reachable via :meth:`~torch.nn.Module.parameters` are counted,
                including those in nested sub-modules.
            data_name (str | None): Identifier appended to the output filename
                (e.g. a dataset name). If ``None``, the file is named
                ``model_parameters_None.json``.

        Returns:
            dict[str, Any]: The summary dictionary as described above.
        """

        def count_params(module: nn.Module) -> tuple[int, int, float]:
            """Counts total, trainable, and trainable-percent for a module.

            Args:
                module (nn.Module): Module whose parameters are counted.

            Returns:
                tuple[int, int, float]: A 3-tuple
                ``(total, trainable, trainable_percent)`` where
                ``trainable_percent`` is rounded to four decimal places.
            """
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            percent = (trainable / total * 100) if total > 0 else 0
            return total, trainable, round(percent, 4)

        breakdown: OrderedDict[str, dict[str, object]] = OrderedDict()
        for name, module in model.named_children():
            total, trainable, trainable_percent = count_params(module)
            breakdown[name] = {
                "total": total,
                "trainable": trainable,
                "trainable_percent": trainable_percent,
            }

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = total_params - trainable_params
        trainable_percent = (trainable_params / total_params * 100) if total_params > 0 else 0

        for v in breakdown.values():
            v["percent_of_total_trainable"] = round(
                (v["trainable"] / trainable_params * 100) if trainable_params > 0 else 0, 4
            )

        summary_json: dict[str, object] = {
            "model_name": self.cfg.model.name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable,
            "trainable_percent": round(trainable_percent, 4),
            "breakdown": breakdown,
        }

        save_path = self.save_results_dir_path / f"model_parameters_{data_name}.json"
        with open(save_path, "w") as f:
            json.dump(summary_json, f, indent=4)
        logger.info(f"Saved model summary to {save_path}")

        return summary_json

    def average_model_summaries(self) -> dict[str, object]:
        """Averages all per-dataset parameter summaries into one report.

        Globs every file matching ``model_parameters_*.json`` in
        ``save_results_dir_path``, averages the top-level numeric fields and
        each module's breakdown fields across all files, and writes the result
        to ``model_parameters_averaged.json`` in the same directory.

        Modules that do not appear in every file (e.g. because the model
        architecture differs slightly across datasets) are still included; the
        ``"present_in_n_files"`` field records how many files contributed to
        that module's average.

        The returned (and saved) dictionary has the following structure::

            {
                "model_name": str,
                "num_files_averaged": int,
                "total_parameters": float,
                "trainable_parameters": float,
                "non_trainable_parameters": float,
                "trainable_percent": float,
                "breakdown": {
                    "<child_name>": {
                        "total": float,
                        "trainable": float,
                        "trainable_percent": float,
                        "percent_of_total_trainable": float,
                        "present_in_n_files": int
                    },
                    ...
                }
            }

        Returns:
            dict[str, Any]: The averaged summary dictionary as described above.

        Raises:
            IndexError: If no ``model_parameters_*.json`` files are found in
                ``save_results_dir_path`` (``summaries`` would be empty).
        """
        json_files = list(self.save_results_dir_path.glob("model_parameters_*.json"))
        logger.info(f"Found {len(json_files)} files to average")

        summaries: list[dict[str, object]] = []
        for file_path in json_files:
            with open(file_path, "r") as f:
                summaries.append(json.load(f))

        n_files = len(summaries)

        avg_summary: dict[str, object] = {
            "model_name": summaries[0]["model_name"],
            "num_files_averaged": n_files,
            "total_parameters": sum(s["total_parameters"] for s in summaries) / n_files,
            "trainable_parameters": sum(s["trainable_parameters"] for s in summaries) / n_files,
            "non_trainable_parameters": sum(s["non_trainable_parameters"] for s in summaries) / n_files,
            "trainable_percent": sum(s["trainable_percent"] for s in summaries) / n_files,
        }

        all_modules: set[str] = set()
        for summary in summaries:
            all_modules.update(summary["breakdown"].keys())

        breakdown_avg: dict[str, dict[str, object]] = {}
        for module_name in all_modules:
            module_data: dict[str, list[float]] = defaultdict(list)
            for summary in summaries:
                if module_name in summary["breakdown"]:
                    mod = summary["breakdown"][module_name]
                    module_data["total"].append(mod["total"])
                    module_data["trainable"].append(mod["trainable"])
                    module_data["trainable_percent"].append(mod["trainable_percent"])
                    module_data["percent_of_total_trainable"].append(mod["percent_of_total_trainable"])

            n_samples = len(module_data["total"])
            breakdown_avg[module_name] = {
                "total": sum(module_data["total"]) / n_samples,
                "trainable": sum(module_data["trainable"]) / n_samples,
                "trainable_percent": round(sum(module_data["trainable_percent"]) / n_samples, 4),
                "percent_of_total_trainable": round(
                    sum(module_data["percent_of_total_trainable"]) / n_samples, 4
                ),
                "present_in_n_files": n_samples,
            }

        avg_summary["breakdown"] = breakdown_avg

        avg_summary["total_parameters"] = round(avg_summary["total_parameters"], 2)
        avg_summary["trainable_parameters"] = round(avg_summary["trainable_parameters"], 2)
        avg_summary["non_trainable_parameters"] = round(avg_summary["non_trainable_parameters"], 2)
        avg_summary["trainable_percent"] = round(avg_summary["trainable_percent"], 4)

        output_path = self.save_results_dir_path / "model_parameters_averaged.json"
        with open(output_path, "w") as f:
            json.dump(avg_summary, f, indent=4)
        logger.info(f"Saved averaged summary to {output_path}")

        return avg_summary
    


