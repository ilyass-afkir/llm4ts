"""
Clean, Modern PyTorch Trainer with Beautiful Plots
Perfect for classification tasks
Uses OmegaConf/Hydra configs
"""

import logging
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math
import gc
from matplotlib import font_manager as fm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        save_dir: Path
    ):
        self.cfg = cfg
        
        self.device = self.cfg.training.device
        self.model = model.to(device=self.device, dtype=torch.bfloat16)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.save_results_dir_path = save_dir

        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.cfg.training.label_smoothing
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
            betas=self.cfg.training.betas
        )
        
        # Cosine Annealing with Warmup Scheduler
        #self.scheduler = self._setup_cosine_scheduler()
        #self.scheduler = self._setup_cosine_scheduler_restart()
        self.scheduler = self._setup_cosine_scheduler_multi_restart()
        
        # Tracking
        self.history = {
            "total_training_time": [],
            "train_loss": [],
            "val_loss": [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'lr': []
        }

        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.global_step = 0
        
    def _setup_cosine_scheduler(self, alpha_f: float = 0.1):
        """Setup cosine annealing scheduler with warmup and minimum LR fraction alpha_f"""
        total_steps = len(self.train_loader) * self.cfg.training.epochs
        warmup_steps = int(total_steps * self.cfg.training.warmup_ratio)
        alpha_f = self.cfg.training.alpha_f
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / max(1, warmup_steps)
            else:
                # Cosine decay to alpha_f fraction of initial LR
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return alpha_f + (1 - alpha_f) * 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _setup_cosine_scheduler_multi_restart(self):
        """Cosine annealing with warm restarts - define ONE cycle, repeat it"""
        total_steps = len(self.train_loader) * self.cfg.training.epochs
        restart_interval = self.cfg.training.restart_interval
        alpha_f = self.cfg.training.alpha_f
        
        # Steps per complete cycle (warmup + decay)
        steps_per_cycle = total_steps // restart_interval
        # Warmup steps within each cycle
        cycle_warmup_steps = int(steps_per_cycle * self.cfg.training.warmup_ratio)
        # Cosine decay steps within each cycle
        cycle_decay_steps = steps_per_cycle - cycle_warmup_steps
        
        def lr_lambda(step):
            # Which cycle are we in and where in that cycle?
            progress_in_cycle = (step % steps_per_cycle) / steps_per_cycle
            step_in_cycle = step % steps_per_cycle
            
            if step_in_cycle < cycle_warmup_steps:
                # WARMUP phase: go from alpha_f to 1.0
                warmup_progress = step_in_cycle / cycle_warmup_steps
                return alpha_f + (1 - alpha_f) * warmup_progress
            else:
                # DECAY phase: cosine from 1.0 to alpha_f
                decay_progress = (step_in_cycle - cycle_warmup_steps) / cycle_decay_steps
                return alpha_f + (1 - alpha_f) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    

    def _setup_cosine_scheduler_restart(self):
        steps_per_epoch = len(self.train_loader)
        alpha_f = self.cfg.training.alpha_f
        
        # Calculate minimum learning rate
        eta_min = alpha_f * self.cfg.training.learning_rate
        
        # T_0: Number of steps until first restart (e.g., 10 epochs)
        restart_interval_epochs = self.cfg.training.restart_interval
        T_0 = restart_interval_epochs * steps_per_epoch
        T_mult = self.cfg.training.T_mult
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=-1
        )
        
        return scheduler
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_x, batch_y in pbar:

            batch_x = batch_x.to(self.device, dtype=torch.bfloat16)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
     
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training.max_grad_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            
            total_loss += loss.item()
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })

        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """Validate the model and compute metrics"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc='Validating', leave=False)
        for batch_x, batch_y in pbar:

            batch_x = batch_x.to(self.device, dtype=torch.bfloat16)
            batch_y = batch_y.to(self.device)

            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
        
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint following best practices.
        
        Best model: lightweight, only what's needed for inference
        Full checkpoint: everything needed to resume training
        """
        
        # Always save full checkpoint (for resuming training)
        full_checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
        }
        
        #path = self.save_results_dir_path / 'checkpoint_epoch.pt'
        #torch.save(full_checkpoint, path)
        
        # Save best model (lightweight, for inference/evaluation)
        if is_best:
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'best_val_f1': self.best_val_f1,
                'best_val_loss': self.best_val_loss,
            }
            best_path = self.save_results_dir_path / 'best_model.pt'
            torch.save(best_checkpoint, best_path)
            logger.info(f'Saved best model (F1: {self.best_val_f1:.4f})')

    def plot_training_curves(self):
        
        # Load custom font
        font_path_normal = "llm-erange/src/utils/times.ttf"
        font_path_bold   = "llm-erange/src/utils/times_bold.ttf"
        fm.fontManager.addfont(font_path_normal)
        fm.fontManager.addfont(font_path_bold)
        prop_normal = fm.FontProperties(fname=font_path_normal)
        plt.rcParams["font.family"] = prop_normal.get_name()
        plt.rcParams["font.size"] = 12

        epochs = range(1, len(self.history['train_loss']) + 1)
        best_epoch = self.history['val_f1'].index(max(self.history['val_f1'])) + 1
        
        # ========== FIGURE 1: Loss and Learning Rate (2x1) ==========
        fig1, axes1 = plt.subplots(2, 1, figsize=(14, 10))
        
        for ax in axes1:
            ax.set_facecolor('#FFFFFF')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_color("#000000")
            ax.spines['bottom'].set_color("#000000")
            ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.6, color="#FFFFFF", axis='y')
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=12, colors="#000000")

        # Only bottom subplot gets Epoch label
        axes1[1].set_xlabel("Epoch", fontsize=12, labelpad=12, color='#000000', weight="normal")

        axes1[0].set_ylabel('Loss', fontsize=12, labelpad=12, color='#000000', weight="normal")
        axes1[1].set_ylabel('Learning rate', fontsize=12, labelpad=12, color='#000000', weight="normal")
        
        # ===== LOSS =====
        axes1[0].plot(epochs, self.history['train_loss'], linewidth=1.5, color="#000000")
        axes1[0].plot(epochs, self.history['val_loss'], linewidth=1.5, linestyle=':', color='#808080')
        axes1[0].axvline(x=best_epoch, color='#FDCA00', linestyle='--', linewidth=2)

        train_line = plt.Line2D([0], [0], color="#000000", linewidth=1.5)
        val_line = plt.Line2D([0], [0], color='#808080', linewidth=1.5, linestyle=':')
        best_line = plt.Line2D([0], [0], color='#FDCA00', linewidth=1.5, linestyle='--')
        time_line = plt.Line2D([0], [0], color='none')
        time_label = f"Training time: {self.history['total_training_time'][0]} min"
        leg = axes1[0].legend(
            [train_line, val_line, best_line, time_line], 
            ['Training loss', 'Validation loss', f'Best epoch (max F1-score): {best_epoch}', time_label], 
            fontsize=10,
            loc="upper right",
            frameon=True
        )
        
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_alpha(0.6)
        leg.get_frame().set_edgecolor('#000000')
        leg.get_frame().set_linewidth(0.5)
        
        # ===== LR =====
        axes1[1].plot(epochs, self.history['lr'], linewidth=1.5, color="#4d4943")
        axes1[1].set_yscale('log')
        axes1[1].axvline(x=best_epoch, color='#FDCA00', linestyle='--', linewidth=2)

        fig1.patch.set_facecolor('#FFFFFF')
        plt.tight_layout(pad=2.0)
        
        plot_path_png_1 = self.save_results_dir_path / 'training_curves_loss_lr.png'
        plot_path_pdf_1 = self.save_results_dir_path / 'training_curves_loss_lr.pdf'
        
        plt.savefig(plot_path_png_1, dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_pdf_1, bbox_inches='tight')
        plt.close(fig1)
        
        # ========== FIGURE 2: Metrics (2x2) ==========
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
        axes2 = axes2.flatten()
        
        for ax in axes2:
            ax.set_facecolor('#FFFFFF')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_color("#000000")
            ax.spines['bottom'].set_color("#000000")
            ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.6, color="#FFFFFF", axis='y')
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=10, colors='#666666')

        # Only bottom row gets Epoch label
        axes2[2].set_xlabel("Epoch", fontsize=12, labelpad=12, color='#000000', weight="normal")
        axes2[3].set_xlabel('Epoch', fontsize=12, labelpad=12, color='#000000', weight="normal")

        axes2[0].set_ylabel('Accuracy', fontsize=12, labelpad=12, color='#000000', weight="normal")
        axes2[1].set_ylabel('F1-score', fontsize=12, labelpad=12, color='#000000', weight="normal")
        axes2[2].set_ylabel('Precision', fontsize=12, labelpad=12, color='#000000', weight="normal")
        axes2[3].set_ylabel('Recall', fontsize=12, labelpad=12, color='#000000', weight="normal")
        
        # ===== Accuracy =====
        axes2[0].plot(epochs, self.history['val_acc'], linewidth=1.5, color="#000000")
        axes2[0].axvline(x=best_epoch, color='#FDCA00', linestyle='--', linewidth=2)
        axes2[0].set_ylim([0, 1.05])
        best_acc_epoch = self.history['val_acc'].index(max(self.history['val_acc'])) + 1
        best_acc = max(self.history['val_acc'])
        axes2[0].plot(best_acc_epoch, best_acc, marker='o', markersize=8, color="#000000", markerfacecolor="#000000", markeredgewidth=1, markeredgecolor="#000000")
        acc_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#000000", markersize=8, markeredgewidth=1, markeredgecolor="#000000")
        acc_label = f"Best Accuracy: {best_acc:.3f}"
        leg_acc = axes2[0].legend(
            [acc_marker], 
            [acc_label], 
            fontsize=10,
            loc="upper right",
            frameon=True
        )
        leg_acc.get_frame().set_facecolor('white')
        leg_acc.get_frame().set_alpha(0.6)
        leg_acc.get_frame().set_edgecolor('#000000')
        leg_acc.get_frame().set_linewidth(0.5)
        
        # ===== F1 =====
        axes2[1].plot(epochs, self.history['val_f1'], linewidth=1.5, color="#000000")
        axes2[1].axvline(x=best_epoch, color='#FDCA00', linestyle='--', linewidth=2)
        axes2[1].set_ylim([0, 1.05])
        best_f1 = max(self.history['val_f1'])
        axes2[1].plot(best_epoch, best_f1, marker='o', markersize=8, color="#000000", markerfacecolor="#000000", markeredgewidth=1, markeredgecolor="#000000")
        f1_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#000000", markersize=8, markeredgewidth=1, markeredgecolor="#000000")
        f1_label = f"Best F1-score: {best_f1:.3f}"
        leg_f1 = axes2[1].legend(
            [f1_marker], 
            [f1_label], 
            fontsize=10,
            loc="upper right",
            frameon=True
        )
        leg_f1.get_frame().set_facecolor('white')
        leg_f1.get_frame().set_alpha(0.6)
        leg_f1.get_frame().set_edgecolor('#000000')
        leg_f1.get_frame().set_linewidth(0.5)
        
        # ===== Precision =====
        axes2[2].plot(epochs, self.history['val_precision'], linewidth=1.5, color="#000000")
        axes2[2].axvline(x=best_epoch, color='#FDCA00', linestyle='--', linewidth=2)
        axes2[2].set_ylim([0, 1.05])
        best_precision_epoch = self.history['val_precision'].index(max(self.history['val_precision'])) + 1
        best_precision = max(self.history['val_precision'])
        axes2[2].plot(best_precision_epoch, best_precision, marker='o', markersize=8, color="#000000", markerfacecolor="#000000", markeredgewidth=1, markeredgecolor="#000000")
        prec_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#000000", markersize=8, markeredgewidth=1, markeredgecolor="#000000")
        prec_label = f"Best Precision: {best_precision:.3f}"
        leg_prec = axes2[2].legend(
            [prec_marker], 
            [prec_label], 
            fontsize=10,
            loc="upper right",
            frameon=True
        )
        leg_prec.get_frame().set_facecolor('white')
        leg_prec.get_frame().set_alpha(0.6)
        leg_prec.get_frame().set_edgecolor('#000000')
        leg_prec.get_frame().set_linewidth(0.5)
        
        # ===== Recall =====
        axes2[3].plot(epochs, self.history['val_recall'], linewidth=2, color="#000000")
        axes2[3].axvline(x=best_epoch, color='#FDCA00', linestyle='--', linewidth=2)
        axes2[3].set_ylim([0, 1.05])
        best_recall_epoch = self.history['val_recall'].index(max(self.history['val_recall'])) + 1
        best_recall = max(self.history['val_recall'])
        axes2[3].plot(best_recall_epoch, best_recall, marker='o', markersize=8, color="#000000", markerfacecolor="#000000", markeredgewidth=1, markeredgecolor="#4d4943")
        recall_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#000000", markersize=8, markeredgewidth=1, markeredgecolor="#000000")
        recall_label = f"Best Recall: {best_recall:.3f}"
        leg_recall = axes2[3].legend(
            [recall_marker], 
            [recall_label], 
            fontsize=10,
            loc="upper right",
            frameon=True
        )
        leg_recall.get_frame().set_facecolor('white')
        leg_recall.get_frame().set_alpha(0.6)
        leg_recall.get_frame().set_edgecolor('#000000')
        leg_recall.get_frame().set_linewidth(0.5)

        fig2.patch.set_facecolor('#FFFFFF')
        plt.tight_layout(pad=2.0)
        
        plot_path_png_2 = self.save_results_dir_path / 'training_curves_metrics.png'
        plot_path_pdf_2 = self.save_results_dir_path / 'training_curves_metrics.pdf'
        
        plt.savefig(plot_path_png_2, dpi=300, bbox_inches='tight')
        plt.savefig(plot_path_pdf_2, bbox_inches='tight')
        plt.close(fig2)

    def _log_training_config(self):
        logger.info("Starting Training")
        logger.info(f"Device: {self.device}")
        logger.info(f"LR: {self.cfg.training.learning_rate}")
        logger.info(f"Epochs: {self.cfg.training.epochs}")

    def fit(self):
        self._log_training_config()
        
        start_time = time.time()
        
        for epoch in range(1, self.cfg.training.epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch()
            
            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['lr'].append(current_lr)
            
            # Validation
            if self.val_loader is not None:
                # Validation
                val_metrics = self.validate()
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_precision'].append(val_metrics['precision'])
                self.history['val_recall'].append(val_metrics['recall'])
                self.history['val_f1'].append(val_metrics['f1'])
            
            # Print progress
            epoch_time = time.time() - epoch_start

            logger.info(f"Epoch [{epoch}/{self.cfg.training.epochs}] - Time: {epoch_time:.2f}s")
            logger.info(f"Train Loss: {train_loss:.6f}")
            logger.info(f"LR: {current_lr:.2e}")

            if self.val_loader is not None:
                logger.info(f"Val Loss: {val_metrics['loss']:.6f}")
                logger.info(f"Val Acc: {val_metrics['accuracy']:.4f}")
                logger.info(f"Val F1: {val_metrics['f1']:.4f}")
                
                # Check for improvement (based on F1 score)
                if val_metrics['f1'] > self.best_val_f1:
                    improvement = val_metrics['f1'] - self.best_val_f1
                    self.best_val_f1 = val_metrics['f1']
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                    logger.info(f"New best F1! ({improvement:.4f})")
                else:
                    self.patience_counter += 1
                    patience = self.cfg.training.patience
                    logger.info(f"Patience: {self.patience_counter}/{patience}")
                    
                    if self.patience_counter >= patience:
                        logger.info("Early stopping triggered!")
                        break
        
        # Training complete
        total_time = time.time() - start_time
        self.history["total_training_time"].append(round(total_time / 60, 3))

        if self.val_loader is not None:
            logger.info(f"Total Time: {total_time/60:.2f} minutes")
            logger.info(f"Best Val Loss: {self.best_val_loss:.6f}")
            logger.info(f"Best Val F1:   {self.best_val_f1:.4f}")
            
            self.plot_training_curves()
        
        # Save history
        history_path = self.save_results_dir_path/ 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
        
        # Load best model
        best_path = self.save_results_dir_path / 'best_model.pt'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_path}")

        return self.model, total_time/60
