"""Comprehensive model evaluator for test set evaluation.

This module provides three evaluator classes for assessing time-series
classification models on test data:

- :class:`UEAEvaluator` – lightweight evaluator for UEA benchmark datasets,
  accumulating per-dataset metrics and saving a JSON report.
- :class:`TruckEvaluator` – full evaluator for the truck dataset, computing
  a rich set of classification metrics and producing publication-quality
  confusion matrix and per-class metric plots.
- :class:`TruckEvaluatorBenchmark` – subclass of :class:`TruckEvaluator` for
  evaluating sklearn-compatible benchmark models (e.g. ``predict()`` API)
  directly on pre-loaded arrays rather than a DataLoader.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_preparation.modules.constants import (
    HIGHWAY_NEW_TO_OLD,
    LABEL_DECODERS,
    WEATHER_NEW_TO_OLD,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class UEAEvaluator:
    """Lightweight evaluator for UEA benchmark time-series classification datasets.

    Iterates over multiple datasets, computes accuracy, balanced accuracy, and
    weighted F1 for each, accumulates the results, and can save a consolidated
    JSON report with per-dataset and average metrics.

    Attributes:
        cfg (DictConfig): Full Hydra/OmegaConf configuration object. 
            Expected keys include ``model.name``, ``training.device``, and 
            ``training.save_results_dir_path``.
        
    Example:
        >>> evaluator = UEAEvaluator(cfg)
        >>> for dataset_name, (model, loader) in datasets.items():
        ...     evaluator.evaluate_model(model, loader, dataset_name)
        >>>
        >>> evaluator.save_evaluation_report()
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialises the UEAEvaluator.

        Args:
            cfg (DictConfig): Hydra/OmegaConf configuration object. Expected
                keys include ``model.name``, ``training.device``, and
                ``training.save_results_dir_path``.
        """
        self.cfg = cfg
        self.model_name = self.cfg.model.name
        self.device = self.cfg.training.device
        self.save_results_dir_path = Path(self.cfg.training.save_results_dir_path)
        self.save_results_dir_path.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, dict] = {}

    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        dataset_name: str,
    ) -> dict[str, float]:
        """Evaluates a model on a single UEA dataset.

        Runs inference over the full test set, computes accuracy, balanced
        accuracy, and weighted F1, stores the result under ``dataset_name``,
        and logs a summary line.

        Args:
            model (nn.Module): Trained PyTorch model to evaluate.
            test_loader (DataLoader): DataLoader yielding ``(data, targets)``
                batches for the test set.
            dataset_name (str): Identifier for the dataset, used as the key in
                :attr:`results`.

        Returns:
            Dict[str, float]: Metrics for this dataset with keys
                ``"accuracy"``, ``"b_accuracy"``, ``"f1_score"``, and
                ``"num_samples"``.
        """
        model.to(self.device)
        model.eval()

        preds_list = []
        targets_list = []

        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device, dtype=torch.bfloat16)
                targets = targets.to(self.device)

                outputs = model(data)
                preds = outputs.argmax(dim=-1)

                preds_list.append(preds.cpu())
                targets_list.append(targets.cpu())

        preds = torch.cat(preds_list).numpy()
        targets = torch.cat(targets_list).numpy()

        acc = accuracy_score(targets, preds)
        b_acc = balanced_accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="weighted")

        self.results[dataset_name] = {
            "accuracy": float(acc),
            "b_accuracy": float(b_acc),
            "f1_score": float(f1),
            "num_samples": len(targets),
        }

        logger.info(
            f"{dataset_name}: Accuracy={acc:.4f}, "
            f"Balanced Accuracy={b_acc:.4f}, F1={f1:.4f}"
        )

        return self.results[dataset_name]

    def compute_average_metrics(self) -> dict[str, float]:
        """Computes mean and standard deviation of metrics across all evaluated datasets.

        Aggregates the per-dataset results stored in :attr:`results` and returns
        a summary dictionary. Must be called after at least one
        :meth:`evaluate_model` call.

        Returns:
            Dict[str, float]: Aggregated metrics with keys ``"avg_accuracy"``,
                ``"avg_f1_score"``, ``"avg_b_accuracy"``, ``"std_accuracy"``,
                ``"std_f1_score"``, ``"std_b_accuracy"``, and
                ``"num_datasets"``.
        """
        accuracies = [r["accuracy"] for r in self.results.values()]
        f1_scores = [r["f1_score"] for r in self.results.values()]
        b_accuracies = [r["b_accuracy"] for r in self.results.values()]

        avg_metrics = {
            "avg_accuracy": float(np.mean(accuracies)),
            "avg_f1_score": float(np.mean(f1_scores)),
            "avg_b_accuracy": float(np.mean(b_accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "std_f1_score": float(np.std(f1_scores)),
            "std_b_accuracy": float(np.std(b_accuracies)),
            "num_datasets": len(self.results),
        }

        return avg_metrics

    def save_evaluation_report(self) -> Path:
        """Saves a consolidated JSON evaluation report to disk.

        Calls :meth:`compute_average_metrics`, then writes a JSON file
        containing the model name, average metrics, per-dataset results, and
        dataset count to ``save_results_dir_path / "evaluation_report.json"``.

        Returns:
            Path: Absolute path to the saved JSON report file.
        """
        save_path = self.save_results_dir_path / "evaluation_report.json"

        avg_metrics = self.compute_average_metrics()

        output = {
            "model_name": self.model_name,
            "average_metrics": avg_metrics,
            "per_dataset_results": self.results,
            "num_datasets": len(self.results),
        }

        with open(save_path, "w") as f:
            json.dump(output, f, indent=4)

        print(f"Evaluation report saved to {save_path}")
        return save_path


class TruckEvaluator:
    """Full evaluator for the truck time-series classification dataset.

    Loads the best saved model checkpoint, runs inference over a DataLoader,
    computes a comprehensive set of classification metrics (weighted, macro,
    micro, per-class), and saves a JSON results file together with
    publication-quality confusion matrix and per-class metric plots.

    Attributes:
        cfg (DictConfig): Full Hydra/OmegaConf configuration object. Expected
            keys include ``training.device``, ``training.num_classes``,
            and ``training.label``.
        model (nn.Module): Model to evaluate, moved to :attr:`device`.
        test_loader (DataLoader): DataLoader for the test set.
        save_results_dir_path (Path): Directory where results and plots are saved.

    Example:
        >>> evaluator = TruckEvaluator(cfg, model, test_loader, save_dir=Path("results/"))
        >>> evaluator.load_best_model()
        >>> results = evaluator.evaluate()
        >>> print(results["overall_metrics"]["f1_weighted"])
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        test_loader: DataLoader,
        save_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.device = self.cfg.training.device
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.save_results_dir_path = save_dir
        self.num_classes = self.cfg.training.num_classes
        self.label_col = self.cfg.training.label
        self.label_decoder = LABEL_DECODERS[self.label_col]
        self.class_names = self.get_class_names()

    def load_best_model(self) -> None:
        """Loads the best model checkpoint from disk into :attr:`model`.

        Looks for ``best_model.pt`` inside :attr:`save_results_dir_path`.
        Supports checkpoints saved as a full state-dict or as a dict with a
        ``"model_state_dict"`` key. Logs the validation F1 and epoch number
        if present in the checkpoint.

        Raises:
            RuntimeError: If the checkpoint file does not contain a valid
                model state dict.
        """
        checkpoint_path = self.save_results_dir_path / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            raise RuntimeError(
                "Loaded checkpoint is not valid or missing model_state_dict"
            )

        if "best_val_f1" in checkpoint:
            logger.info(f"Training Val F1: {checkpoint['best_val_f1']:.4f}")
        if "epoch" in checkpoint:
            logger.info(f"Trained Epochs: {checkpoint['epoch']}")
        logger.info("Model loaded successfully!")

    def get_class_names(self) -> list[str]:
        """Resolves ordered human-readable class names from the label decoder.

        Handles label remapping for ``"highway_label"`` and
        ``"weather_label"`` columns via their respective new-to-old index
        maps, and falls back to direct decoder lookup for all other labels.

        Returns:
            List[str]: Ordered list of class name strings, one per class index.
        """
        if self.label_col == "highway_label":
            return [
                self.label_decoder[HIGHWAY_NEW_TO_OLD[i]]
                for i in range(self.num_classes)
            ]
        elif self.label_col == "weather_label":
            return [
                self.label_decoder[WEATHER_NEW_TO_OLD[i]]
                for i in range(self.num_classes)
            ]
        else:
            return [self.label_decoder[i] for i in range(self.num_classes)]

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Runs full test set evaluation and saves metrics and plots.

        Iterates over :attr:`test_loader`, collects predictions and targets,
        then calls :meth:`_compute_and_save_metrics`,
        :meth:`_plot_confusion_matrix`, and :meth:`_plot_per_class_metrics`.

        Returns:
            Dict: Results dictionary as returned by
                :meth:`_compute_and_save_metrics`, containing
                ``"overall_metrics"``, ``"per_class_metrics"``,
                ``"confusion_matrix"``, ``"num_samples"``, and
                ``"num_classes"``.
        """
        self.model.eval()

        all_preds = []
        all_targets = []

        pbar = tqdm(self.test_loader, desc="Evaluating Test Set")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device, dtype=torch.bfloat16)
            batch_y = batch_y.to(self.device, non_blocking=True)

            outputs = self.model(batch_x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        results = self.compute_and_save_metrics(all_preds, all_targets)
        self.plot_confusion_matrix(results["confusion_matrix"])
        self.plot_per_class_metrics(results)

        return results

    def compute_and_save_metrics(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
    ) -> dict:
        """Computes a full suite of classification metrics and saves them to JSON.

        Computes weighted, macro, and micro averages for precision, recall, and
        F1, along with per-class breakdowns, Matthews correlation coefficient,
        Cohen's kappa, and a confusion matrix. Saves everything to
        ``test_results.json`` inside :attr:`save_results_dir_path`.

        Args:
            preds (np.ndarray): Predicted class indices of shape
                ``(num_samples,)``.
            targets (np.ndarray): Ground-truth class indices of shape
                ``(num_samples,)``.

        Returns:
            Dict: Nested results dictionary with keys ``"overall_metrics"``,
                ``"per_class_metrics"``, ``"confusion_matrix"``,
                ``"num_samples"``, and ``"num_classes"``.
        """
        accuracy = accuracy_score(targets, preds)
        balanced_accuracy = balanced_accuracy_score(targets, preds)
        precision_weighted = precision_score(targets, preds, average="weighted", zero_division=0)
        recall_weighted = recall_score(targets, preds, average="weighted", zero_division=0)
        f1_weighted = f1_score(targets, preds, average="weighted", zero_division=0)

        precision_macro = precision_score(targets, preds, average="macro", zero_division=0)
        recall_macro = recall_score(targets, preds, average="macro", zero_division=0)
        f1_macro = f1_score(targets, preds, average="macro", zero_division=0)

        precision_micro = precision_score(targets, preds, average="micro", zero_division=0)
        recall_micro = recall_score(targets, preds, average="micro", zero_division=0)
        f1_micro = f1_score(targets, preds, average="micro", zero_division=0)

        precision_per_class = precision_score(targets, preds, average=None, zero_division=0)
        recall_per_class = recall_score(targets, preds, average=None, zero_division=0)
        f1_per_class = f1_score(targets, preds, average=None, zero_division=0)

        matthews = matthews_corrcoef(targets, preds)
        kappa = cohen_kappa_score(targets, preds)

        cm = confusion_matrix(targets, preds)
        class_totals = cm.sum(axis=1)
        per_class_accuracy = np.divide(
            cm.diagonal(),
            class_totals,
            where=class_totals != 0,
            out=np.zeros_like(cm.diagonal(), dtype=float),
        )

        results = {
            "overall_metrics": {
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_accuracy),
                "precision_weighted": float(precision_weighted),
                "recall_weighted": float(recall_weighted),
                "f1_weighted": float(f1_weighted),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro),
                "precision_micro": float(precision_micro),
                "recall_micro": float(recall_micro),
                "f1_micro": float(f1_micro),
                "matthews_corrcoef": float(matthews),
                "cohen_kappa": float(kappa),
            },
            "per_class_metrics": {
                self.class_names[i]: {
                    "accuracy": float(per_class_accuracy[i]),
                    "precision": float(precision_per_class[i]),
                    "recall": float(recall_per_class[i]),
                    "f1": float(f1_per_class[i]),
                    "support": int(cm[i].sum()),
                }
                for i in range(self.num_classes)
            },
            "confusion_matrix": cm.tolist(),
            "num_samples": len(targets),
            "num_classes": self.num_classes,
        }

        save_path = self.save_results_dir_path / "test_results.json"
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def plot_confusion_matrix(self, cm: list[list[int]]) -> None:
        """Renders and saves a styled confusion matrix heatmap.

        Produces a square heatmap with a gold gradient colormap, black grid
        lines, and cell-level count annotations. Saves the figure as both
        ``confusion_matrix.pdf`` and ``confusion_matrix.png`` inside
        :attr:`save_results_dir_path`.

        Args:
            cm (List[List[int]]): Confusion matrix as a nested list of integer
                counts, as returned by ``confusion_matrix(...).tolist()``.
        """
        cm = np.array(cm)
        font_path_normal = "llm-erange/src/utils/times.ttf"
        font_path_bold = "llm-erange/src/utils/times_bold.ttf"
        fm.fontManager.addfont(font_path_normal)
        fm.fontManager.addfont(font_path_bold)
        prop_normal = fm.FontProperties(fname=font_path_normal)
        plt.rcParams["font.family"] = prop_normal.get_name()
        plt.rcParams["font.size"] = 12

        n_classes = len(self.class_names)
        cell_size = 1.0
        fig_size = n_classes * cell_size + 3

        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)

        colors = ["#FFFFFF", "#FFE06C", "#FDCA00"]
        cmap = LinearSegmentedColormap.from_list("gold", colors, N=100)

        im = ax.imshow(cm, cmap=cmap, aspect="equal", interpolation="nearest")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=1.0)
        cbar.ax.tick_params(labelsize=11, width=0.5, length=5, color="black")
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(0.5)

        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(self.class_names, fontsize=12, weight="normal", color="black")
        ax.set_yticklabels(self.class_names, fontsize=12, weight="normal", color="black")

        ax.set_xlabel("Predicted class", fontsize=12, fontweight="normal", labelpad=12)
        ax.set_ylabel("True class", fontsize=12, fontweight="normal", labelpad=12)

        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(
                    j, i, f"{int(cm[i, j])}",
                    ha="center", va="center",
                    color="black", fontsize=12, weight="normal",
                )

        for i in range(n_classes + 1):
            ax.axhline(i - 0.5, color="black", linewidth=0.5, zorder=5)
            ax.axvline(i - 0.5, color="black", linewidth=0.5, zorder=5)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("black")

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        plt.tight_layout()

        pdf_path = self.save_results_dir_path / "confusion_matrix.pdf"
        png_path = self.save_results_dir_path / "confusion_matrix.png"
        plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", dpi=300)
        plt.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=300)

    def plot_per_class_metrics(self, results: dict) -> None:
        """Renders and saves a 2x2 grid of per-class metric bar charts.

        Produces bar charts for per-class accuracy, precision, recall, and F1,
        each annotated with the metric value and sample count. Bars exceeding
        the macro average are highlighted in dark grey; others in light grey.
        A dashed gold line marks the macro average. Saves the figure as both
        ``per_class_metrics.pdf`` and ``per_class_metrics.png`` inside
        :attr:`save_results_dir_path`.

        Args:
            results (Dict): Results dictionary as returned by
                :meth:`_compute_and_save_metrics`.
        """
        font_path_normal = "llm-erange/src/utils/times.ttf"
        font_path_bold = "llm-erange/src/utils/times_bold.ttf"
        fm.fontManager.addfont(font_path_normal)
        fm.fontManager.addfont(font_path_bold)
        prop_normal = fm.FontProperties(fname=font_path_normal)
        plt.rcParams["font.family"] = prop_normal.get_name()
        plt.rcParams["font.size"] = 12

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        per_class = results["per_class_metrics"]
        sample_counts = {c: per_class[c]["support"] for c in per_class.keys()}
        classes = list(per_class.keys())

        accuracies = [per_class[c]["accuracy"] for c in classes]
        precisions = [per_class[c]["precision"] for c in classes]
        recalls = [per_class[c]["recall"] for c in classes]
        f1_scores = [per_class[c]["f1"] for c in classes]

        x = np.arange(len(classes))

        for ax in axes:
            ax.set_facecolor("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.5)
            ax.spines["bottom"].set_linewidth(0.5)
            ax.spines["left"].set_color("#000000")
            ax.spines["bottom"].set_color("#000000")
            ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.6, color="#FFFFFF", axis="y")
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=12, colors="#000000")
            ax.set_ylim([0, 1.05])

        axes[2].set_xlabel("Class label", fontsize=12, labelpad=12, color="#000000", weight="normal")
        axes[3].set_xlabel("Class label", fontsize=12, labelpad=12, color="#000000", weight="normal")

        # Plot 1: Accuracy
        avg_acc = np.mean([per_class[c]["accuracy"] for c in classes])
        colors_acc = ["#4d4943" if val >= avg_acc else "#f5f5f5" for val in accuracies]
        bars0 = axes[0].bar(x, accuracies, color=colors_acc, edgecolor="black", linewidth=0.5)
        for bar, val, c in zip(bars0, accuracies, classes):
            axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                         ha="center", va="bottom", fontsize=12, color="#000000")
            axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.06, f"({sample_counts[c]})",
                         ha="center", va="bottom", fontsize=10, color="#000000")
        axes[0].axhline(y=avg_acc, color="#FDCA00", linestyle="--", linewidth=1.5,
                        label=f"Macro average: {avg_acc:.3f}")
        axes[0].legend(loc="upper right", fontsize=10, framealpha=0.6)
        axes[0].set_ylabel("Per-class Accuracy", fontsize=12, labelpad=12, color="#000000", fontweight="normal")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(classes, fontsize=12, color="#000000")

        # Plot 2: Precision
        avg_prec = results["overall_metrics"]["precision_macro"]
        colors_prec = ["#4d4943" if val >= avg_prec else "#f5f5f5" for val in precisions]
        bars1 = axes[1].bar(x, precisions, color=colors_prec, edgecolor="black", linewidth=0.5)
        for bar, val, c in zip(bars1, precisions, classes):
            axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                         ha="center", va="bottom", fontsize=12, color="#000000")
            axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.06, f"({sample_counts[c]})",
                         ha="center", va="bottom", fontsize=10, color="#000000")
        axes[1].axhline(y=avg_prec, color="#FDCA00", linestyle="--", linewidth=1.5,
                        label=f"Macro average: {avg_prec:.3f}")
        axes[1].legend(loc="upper right", fontsize=10, framealpha=0.6)
        axes[1].set_ylabel("Per-class Precision", fontsize=12, labelpad=12, color="#000000", fontweight="normal")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes, fontsize=12, color="#000000")

        # Plot 3: Recall
        avg_rec = results["overall_metrics"]["recall_macro"]
        colors_rec = ["#4d4943" if val >= avg_rec else "#f5f5f5" for val in recalls]
        bars2 = axes[2].bar(x, recalls, color=colors_rec, edgecolor="black", linewidth=0.5)
        for bar, val, c in zip(bars2, recalls, classes):
            axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                         ha="center", va="bottom", fontsize=12, color="#000000")
            axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.06, f"({sample_counts[c]})",
                         ha="center", va="bottom", fontsize=10, color="#000000")
        axes[2].axhline(y=avg_rec, color="#FDCA00", linestyle="--", linewidth=1.5,
                        label=f"Macro Recall: {avg_rec:.3f}")
        axes[2].legend(loc="upper right", fontsize=10, framealpha=0.6)
        axes[2].set_ylabel("Per-class Recall", fontsize=12, labelpad=12, color="#000000", fontweight="normal")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(classes, fontsize=12, color="#000000")

        # Plot 4: F1
        avg_f1 = results["overall_metrics"]["f1_macro"]
        colors_f1 = ["#4d4943" if val >= avg_f1 else "#f5f5f5" for val in f1_scores]
        bars3 = axes[3].bar(x, f1_scores, color=colors_f1, edgecolor="black", linewidth=0.5)
        for bar, val, c in zip(bars3, f1_scores, classes):
            axes[3].text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                         ha="center", va="bottom", fontsize=12, color="#000000")
            axes[3].text(bar.get_x() + bar.get_width() / 2, val + 0.06, f"({sample_counts[c]})",
                         ha="center", va="bottom", fontsize=10, color="#000000")
        axes[3].axhline(y=avg_f1, color="#FDCA00", linestyle="--", linewidth=1.5,
                        label=f"Macro average: {avg_f1:.3f}")
        axes[3].legend(loc="upper right", fontsize=10, framealpha=0.6)
        axes[3].set_ylabel("Per-class F1-score", fontsize=12, labelpad=12, color="#000000", weight="normal")
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(classes, fontsize=12, color="#000000")

        fig.patch.set_facecolor("white")
        plt.tight_layout(pad=2.0)

        png_path = self.save_results_dir_path / "per_class_metrics.png"
        pdf_path = self.save_results_dir_path / "per_class_metrics.pdf"
        plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", dpi=300)
        plt.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=300)

        logger.info("saved")
        plt.close(fig)


class TruckEvaluatorBenchmark(TruckEvaluator):
    """Evaluator for sklearn-compatible benchmark models on the truck dataset.

    Extends :class:`TruckEvaluator` by overriding :meth:`__init__` and
    :meth:`evaluate` to accept pre-loaded arrays and a ``predict()``-style
    model interface instead of a PyTorch DataLoader.

    Attributes:
        cfg (DictConfig): Hydra configuration object. Expected
            keys include ``training.num_classes`` and ``training.label``.
        model: Sklearn-compatible model exposing a ``predict(X)`` method.
        test_data: Input features in whatever format ``model.predict()``
            expects (e.g. a numpy array or pandas DataFrame).
        targets: Ground-truth class indices, convertible to
            ``np.ndarray`` of shape ``(num_samples,)``.
        save_dir (Path): Directory where JSON results and plots are saved.

    Example:
        >>> evaluator = TruckEvaluatorBenchmark(
        ...     cfg=cfg,
        ...     model=sklearn_model,
        ...     test_data=X_test,
        ...     targets=y_test,
        ...     save_dir=Path("results/benchmark/"),
        ... )
        >>> results = evaluator.evaluate()
        >>> print(results["overall_metrics"]["f1_weighted"])
    """

    def __init__(
        self,
        cfg: DictConfig,
        model,
        test_data,
        targets,
        save_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.test_data = test_data
        self.targets = np.array(targets)
        self.save_results_dir_path = save_dir
        self.num_classes = self.cfg.training.num_classes
        self.label_col = self.cfg.training.label
        self.label_decoder = LABEL_DECODERS[self.label_col]
        self.class_names = self.get_class_names()

    def evaluate(self) -> dict:
        """Runs evaluation using the benchmark model's ``predict()`` interface.

        Calls ``model.predict(test_data)`` directly, then delegates to
        :meth:`_compute_and_save_metrics`, :meth:`_plot_confusion_matrix`,
        and :meth:`_plot_per_class_metrics` inherited from
        :class:`TruckEvaluator`.

        Returns:
            Dict: Results dictionary as returned by
                :meth:`_compute_and_save_metrics`, containing
                ``"overall_metrics"``, ``"per_class_metrics"``,
                ``"confusion_matrix"``, ``"num_samples"``, and
                ``"num_classes"``.
        """
        preds = np.array(self.model.predict(self.test_data))

        results = self.compute_and_save_metrics(preds, self.targets)
        self.plot_confusion_matrix(results["confusion_matrix"])
        self.plot_per_class_metrics(results)

        return results

    