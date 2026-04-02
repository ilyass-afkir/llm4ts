"""
Comprehensive model evaluator for test set evaluation.
"""

import logging
import json
from typing import Dict
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager as fm
import numpy as np
from sklearn.metrics import (
    matthews_corrcoef, 
    cohen_kappa_score, 
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    
)

from src.data_preparation.modules.constants import LABEL_DECODERS, HIGHWAY_NEW_TO_OLD, WEATHER_NEW_TO_OLD

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class UEAEvaluator:
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.model_name = self.cfg.model.name
        self.device = self.cfg.training.device
        self.save_results_dir_path = Path(self.cfg.training.save_results_dir_path)
        self.save_results_dir_path.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def evaluate_model(self, model, test_loader, dataset_name):
        
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

        logger.info(f"{dataset_name}: Accuracy={acc:.4f}, Balanced Accuracy={b_acc:.4f}, F1={f1:.4f}")
        
        return self.results[dataset_name]
    
    def compute_average_metrics(self) -> Dict[str, float]:
        
        accuracies = [r['accuracy'] for r in self.results.values()]
        f1_scores = [r['f1_score'] for r in self.results.values()]
        b_accuracies = [r['b_accuracy'] for r in self.results.values()]
        
        avg_metrics = {
            'avg_accuracy': float(np.mean(accuracies)),
            'avg_f1_score': float(np.mean(f1_scores)),
            "avg_b_accuracy": float(np.mean(b_accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'std_f1_score': float(np.std(f1_scores)),
            "std_b_accuracy": float(np.std(b_accuracies)),
            'num_datasets': len(self.results)
        }
        
        return avg_metrics
    
    def save_evaluation_report(self):
   
        save_path = self.save_results_dir_path / "evaluation_report.json"
        
        avg_metrics = self.compute_average_metrics()
        
        output = {
            'model_name': self.model_name,
            'average_metrics': avg_metrics,
            'per_dataset_results': self.results,
            'num_datasets': len(self.results)
        }
        
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=4)
        
        print(f"Evaluation report saved to {save_path}")
        return save_path
    

class TruckEvaluator:
    def __init__(self, cfg: DictConfig, model: nn.Module, test_loader: DataLoader, save_dir: bool):
        self.cfg = cfg
        self.device = self.cfg.training.device
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.save_results_dir_path = save_dir
        self.num_classes = self.cfg.training.num_classes
        self.label_col = self.cfg.training.label
        self.label_decoder = LABEL_DECODERS[self.label_col]
        self.class_names = self._get_class_names()

    def load_best_model(self):
        checkpoint_path = self.save_results_dir_path / 'best_model.pt'
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Ensure model_state_dict is present
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            raise RuntimeError("Loaded checkpoint is not valid or missing model_state_dict")
        
        # Log info if available
        if 'best_val_f1' in checkpoint:
            logger.info(f"Training Val F1: {checkpoint['best_val_f1']:.4f}")
        if 'epoch' in checkpoint:
            logger.info(f"Trained Epochs: {checkpoint['epoch']}")
        logger.info("Model loaded successfully!")
    
    def _get_class_names(self):
        if self.label_col == "highway_label":
            class_names = []
            for i in range(self.num_classes):
                old_label = HIGHWAY_NEW_TO_OLD[i]
                class_name = self.label_decoder[old_label]
                class_names.append(class_name)
            return class_names
        elif self.label_col == "weather_label":
            class_names = []
            for i in range(self.num_classes):
                old_label = WEATHER_NEW_TO_OLD[i]
                class_name = self.label_decoder[old_label]
                class_names.append(class_name)
            return class_names
        else:
            class_names = [self.label_decoder[i] for i in range(self.num_classes)]
            return class_names

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.test_loader, desc='Evaluating Test Set')
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device, dtype=torch.bfloat16)
            batch_y = batch_y.to(self.device, non_blocking=True)
           
            outputs = self.model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
    
        results = self._compute_and_save_metrics(all_preds, all_targets)
        self._plot_confusion_matrix(results['confusion_matrix'])
        self._plot_per_class_metrics(results)
 
        return results
    
    def _compute_and_save_metrics(self, preds, targets):
        # Overall metrics (weighted average)
        accuracy = accuracy_score(targets, preds)
        balanced_accuracy = balanced_accuracy_score(targets, preds)
        precision_weighted = precision_score(targets, preds, average='weighted', zero_division=0)
        recall_weighted = recall_score(targets, preds, average='weighted', zero_division=0)
        f1_weighted = f1_score(targets, preds, average='weighted', zero_division=0)
        
        # Macro average
        precision_macro = precision_score(targets, preds, average='macro', zero_division=0)
        recall_macro = recall_score(targets, preds, average='macro', zero_division=0)
        f1_macro = f1_score(targets, preds, average='macro', zero_division=0)

        # Micro average
        precision_micro = precision_score(targets, preds, average='micro', zero_division=0)
        recall_micro = recall_score(targets, preds, average='micro', zero_division=0)
        f1_micro = f1_score(targets, preds, average='micro', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(targets, preds, average=None, zero_division=0)
        recall_per_class = recall_score(targets, preds, average=None, zero_division=0)
        f1_per_class = f1_score(targets, preds, average=None, zero_division=0)

        # Other
        matthews = matthews_corrcoef(targets, preds)
        kappa = cohen_kappa_score(targets, preds)
        
        # Confusion matrix
        cm = confusion_matrix(targets, preds)
        
        # Per-class accuracy from confusion matrix (handle zero division)
        class_totals = cm.sum(axis=1)
        per_class_accuracy = np.divide(cm.diagonal(), class_totals, 
                                    where=class_totals!=0, 
                                    out=np.zeros_like(cm.diagonal(), dtype=float))
        
        results = {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'balanced_accuracy': float(balanced_accuracy),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_micro': float(precision_micro),
                'recall_micro': float(recall_micro),
                'f1_micro': float(f1_micro),
                'matthews_corrcoef': float(matthews),
                'cohen_kappa': float(kappa),
            },
            'per_class_metrics': {
                self.class_names[i]: {
                    'accuracy': float(per_class_accuracy[i]),
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1': float(f1_per_class[i]),
                    'support': int(cm[i].sum())
                }
                for i in range(self.num_classes)
            },
            'confusion_matrix': cm.tolist(),
            'num_samples': len(targets),
            'num_classes': self.num_classes
        }
        
        save_path = self.save_results_dir_path / 'test_results.json'
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

    def _plot_confusion_matrix(self, cm):

        cm = np.array(cm)
        font_path_normal = "llm-erange/src/utils/times.ttf"
        font_path_bold   = "llm-erange/src/utils/times_bold.ttf"
        fm.fontManager.addfont(font_path_normal)
        fm.fontManager.addfont(font_path_bold)
        prop_normal = fm.FontProperties(fname=font_path_normal)
        plt.rcParams["font.family"] = prop_normal.get_name()
        plt.rcParams["font.size"] = 12
        
        n_classes = len(self.class_names)
        cell_size = 1.0  
        fig_size = n_classes * cell_size + 3 
        
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)
        
        # Gold gradient colormap
        colors = ["#FFFFFF", "#FFE06C", '#FDCA00']
        cmap = LinearSegmentedColormap.from_list('gold', colors, N=100)
        
        # Heatmap - SQUARE aspect
        im = ax.imshow(cm, cmap=cmap, aspect='equal', interpolation='nearest')
        
        # Colorbar with black border - same height as matrix
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=1.0)
        cbar.ax.tick_params(labelsize=11, width=0.5, length=5, color='black')
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(0.5)
        
        # Axis ticks & labels (normal weight)
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(self.class_names, fontsize=12, weight='normal', color='black')
        ax.set_yticklabels(self.class_names, fontsize=12, weight='normal', color='black')
        
        # Axis labels - normal weight
        ax.set_xlabel('Predicted class', fontsize=12, fontweight='normal', labelpad=12)
        ax.set_ylabel('True class', fontsize=12, fontweight='normal', labelpad=12)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', rotation_mode='anchor')
        
        # Cell annotations with better sized font
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(
                    j, i, f'{int(cm[i, j])}',
                    ha='center', va='center',
                    color='black',
                    fontsize=12,
                    weight='normal'
                )
        
        # Grid lines - all black with thickness 0.5
        for i in range(n_classes + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, zorder=5)
            ax.axvline(i - 0.5, color='black', linewidth=0.5, zorder=5)
        
        # Clean spines - matching grid thickness
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor('black')
        
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save
        pdf_path = self.save_results_dir_path / 'confusion_matrix.pdf'
        png_path = self.save_results_dir_path / 'confusion_matrix.png'
        
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.savefig(png_path, bbox_inches='tight', facecolor='white', dpi=300)
        
    def _plot_per_class_metrics(self, results):

        font_path_normal = "llm-erange/src/utils/times.ttf"
        font_path_bold   = "llm-erange/src/utils/times_bold.ttf"
        fm.fontManager.addfont(font_path_normal)
        fm.fontManager.addfont(font_path_bold)
        prop_normal = fm.FontProperties(fname=font_path_normal)
        plt.rcParams["font.family"] = prop_normal.get_name()
        plt.rcParams["font.size"] = 12

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        per_class = results['per_class_metrics']
        sample_counts = {c: per_class[c]['support'] for c in per_class.keys()}
        classes = list(per_class.keys())
        
        # Extract metrics
        accuracies = [per_class[c]['accuracy'] for c in classes]
        precisions = [per_class[c]['precision'] for c in classes]
        recalls = [per_class[c]['recall'] for c in classes]
        f1_scores = [per_class[c]['f1'] for c in classes]
        
        x = np.arange(len(classes))
        
        # Base styling for all subplots
        for ax in axes:
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_color("#000000")
            ax.spines['bottom'].set_color("#000000")
            ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.6, color="#FFFFFF", axis='y')
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=12, colors="#000000")
            ax.set_ylim([0, 1.05])
        
        axes[2].set_xlabel("Class label", fontsize=12, labelpad=12, color='#000000', weight="normal")
        axes[3].set_xlabel('Class label', fontsize=12, labelpad=12, color='#000000', weight="normal")
        
        # Plot 1: Accuracy
        avg_acc = np.mean([per_class[c]['accuracy'] for c in classes])
        colors_acc = ['#4d4943' if val >= avg_acc else '#f5f5f5' for val in accuracies]
        bars0 = axes[0].bar(x, accuracies, color=colors_acc, edgecolor='black', linewidth=0.5, alpha=1)
        for i, (bar, val, c) in enumerate(zip(bars0, accuracies, classes)):
            axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
                        ha='center', va='bottom', fontsize=12, color='#000000')
            axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.06, f'({sample_counts[c]})', 
                        ha='center', va='bottom', fontsize=10, color='#000000')
        axes[0].axhline(y=avg_acc, color='#FDCA00', linestyle='--', linewidth=1.5, alpha=1, 
                    label=f'Macro average: {avg_acc:.3f}')
        axes[0].legend(loc='upper right', fontsize=10, framealpha=0.6)
        axes[0].set_ylabel('Per-class Accuracy', fontsize=12, labelpad=12, color='#000000', fontweight="normal")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(classes, fontsize=12, color="#000000")

        # Plot 2: Precision
        avg_prec = results["overall_metrics"]["precision_macro"]
        colors_prec = ['#4d4943' if val >= avg_prec else '#f5f5f5' for val in precisions]
        bars1 = axes[1].bar(x, precisions, color=colors_prec, edgecolor='black', linewidth=0.5, alpha=1)
        for i, (bar, val, c) in enumerate(zip(bars1, precisions, classes)):
            axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
                        ha='center', va='bottom', fontsize=12, color='#000000')
            axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.06, f'({sample_counts[c]})', 
                        ha='center', va='bottom', fontsize=10, color='#000000')
        axes[1].axhline(y=avg_prec, color='#FDCA00', linestyle='--', linewidth=1.5, alpha=1, 
                    label=f'Macro average: {avg_prec:.3f}')
        axes[1].legend(loc='upper right', fontsize=10, framealpha=0.6)
        axes[1].set_ylabel('Per-class Precision', fontsize=12, labelpad=12, color='#000000', fontweight='normal')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes, fontsize=12, color="#000000")
        
        # Plot 3: Recall
        avg_rec = results["overall_metrics"]["recall_macro"]
        colors_rec = ['#4d4943' if val >= avg_rec else '#f5f5f5' for val in recalls]
        bars2 = axes[2].bar(x, recalls, color=colors_rec, edgecolor='black', linewidth=0.5, alpha=1)
        for i, (bar, val, c) in enumerate(zip(bars2, recalls, classes)):
            axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
                        ha='center', va='bottom', fontsize=12, color='#000000')
            axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.06, f'({sample_counts[c]})', 
                        ha='center', va='bottom', fontsize=10, color='#000000')
        axes[2].axhline(y=avg_rec, color='#FDCA00', linestyle='--', linewidth=1.5, alpha=1, 
                    label=f'Macro Recall: {avg_rec:.3f}')
        axes[2].legend(loc='upper right', fontsize=10, framealpha=0.6)
        axes[2].set_ylabel('Per-class Recall', fontsize=12, labelpad=12, color='#000000', fontweight='normal')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(classes, fontsize=12, color="#000000")
        
        # Plot 4: F1 Score with average line
        avg_f1 = results["overall_metrics"]["f1_macro"]
        colors_f1 = ['#4d4943' if val >= avg_f1 else '#f5f5f5' for val in f1_scores]
        bars3 = axes[3].bar(x, f1_scores, color=colors_f1, edgecolor='black', linewidth=0.5, alpha=1)
        for i, (bar, val, c) in enumerate(zip(bars3, f1_scores, classes)):
            axes[3].text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
                        ha='center', va='bottom', fontsize=12, color='#000000')
            axes[3].text(bar.get_x() + bar.get_width()/2, val + 0.06, f'({sample_counts[c]})', 
                        ha='center', va='bottom', fontsize=10, color='#000000')
        
        # Add average line
        axes[3].axhline(y=avg_f1, color='#FDCA00', linestyle='--', linewidth=1.5, alpha=1, 
                    label=f'Macro average: {avg_f1:.3f}')
        axes[3].legend(loc='upper right', fontsize=10, framealpha=0.6)
        
        axes[3].set_ylabel('Per-class F1-score', fontsize=12, labelpad=12, color='#000000', weight='normal')
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(classes, fontsize=12, color="#000000")
        
        # White background
        fig.patch.set_facecolor('white')
        
        plt.tight_layout(pad=2.0)
        
        # Save both PDF and PNG
        png_path = self.save_results_dir_path / 'per_class_metrics.png'
        pdf_path = self.save_results_dir_path / 'per_class_metrics.pdf'
        
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.savefig(png_path, bbox_inches='tight', facecolor='white', dpi=300)

        logger.info("saved")
        plt.close(fig)

class TruckEvaluatorBenchmark(TruckEvaluator):
    def __init__(self, cfg, model, test_data, targets, save_dir):
        self.cfg = cfg
        self.model = model
        self.test_data = test_data
        self.targets = np.array(targets)
        self.save_results_dir_path = save_dir
        self.num_classes = self.cfg.training.num_classes
        self.label_col = self.cfg.training.label
        self.label_decoder = LABEL_DECODERS[self.label_col]
        self.class_names = self._get_class_names()

    def evaluate(self):
        
        preds = self.model.predict(self.test_data)
        preds = np.array(preds)

        results = self._compute_and_save_metrics(preds, self.targets)
        self._plot_confusion_matrix(results['confusion_matrix'])
        self._plot_per_class_metrics(results)
 
        return results

    