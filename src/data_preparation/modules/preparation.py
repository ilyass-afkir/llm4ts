"""
Description here.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, List
import time
import json

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from einops import rearrange

from src.data_preparation.modules import constants as const

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class TruckDataPreparator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        self.verified_data_path = Path(self.cfg.data_preparation.verified_data_path)
        self.weather_labels_path = Path(self.cfg.data_preparation.weather_labels_path)
        self.highway_labels_path = Path(self.cfg.data_preparation.highway_labels_path)
        
        self.save_labeled_path = Path(self.cfg.data_preparation.save_labeled_path)
        self.save_windows_path = Path(self.cfg.data_preparation.save_windows_path)
        self.save_data_split_path = Path(self.cfg.data_preparation.save_data_split_path)
        self.save_reports_path = Path(self.cfg.data_preparation.save_reports_path)

        self.file_extension = self.cfg.data_preparation.file_extension
        self.window_extension = self.cfg.data_preparation.window_extension
        self.label_columns  = self.cfg.data_preparation.label_columns
        self.window_sizes_sec = self.cfg.data_preparation.window_sizes_sec
        self.train_pct = self.cfg.data_preparation.training_pct   
        self.val_pct = self.cfg.data_preparation.validation_pct
        self.test_pct = self.cfg.data_preparation.test_pct
        self.fewshot_pct = self.cfg.data_preparation.fewshot_pct
        self.signals_to_drop = self.cfg.data_preparation.signals_to_drop
        self.labels_to_drop = self.cfg.data_preparation.labels_to_drop
  
        self.label_encoders = const.LABEL_ENCODERS
        self.label_decoders = const.LABEL_DECODERS
        self.merge_signals = const.MERGE_SIGNALS
        self.train_signal_mapping = const.TRAIN_SIGNAL_MAPPING
        self.highway_columns = const.HIGHWAY_COLUMNS 
        self.weather_columns = const.WEATHER_COLUMNS
        self.signal_mapping = const.SIGNAL_MAPPING
   
        self._setup_dirs()

    def _setup_dirs(self):
        for path in [self.save_labeled_path, self.save_windows_path, self.save_reports_path, self.save_data_split_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _get_datasets(self, data_path: Path, file_extension: str) -> Tuple[List[Path], List[str]]:
        paths = sorted(data_path.glob(f"*{file_extension}"))
        filenames = [f.stem for f in paths]
        return paths, filenames
    
    def _verify_expected_features(self, df: pd.DataFrame, verification_mapping: dict) -> pd.DataFrame:
   
        expected_signals = set(verification_mapping.keys())
        actual_signals = set(df.columns)
        
        missing_signals = expected_signals - actual_signals
        if missing_signals:
            raise ValueError(f"Missing signals in dataframe: {missing_signals}")
        
        extra_signals = actual_signals - expected_signals
        if extra_signals:
            raise ValueError(f"Unexpected signals in dataframe: {extra_signals}")
        
        ordered_columns = sorted(verification_mapping, key=lambda k: verification_mapping[k])
        df = df[ordered_columns]

        assert len(df.columns) == len(verification_mapping)

        return df

    def _merge_and_verify_label_datasets(self):
        paths, filenames = self._get_datasets(self.verified_data_path, self.file_extension)
        highway_paths, highway_filenames = self._get_datasets(self.highway_labels_path, self.file_extension)
        weather_paths, weather_filenames = self._get_datasets(self.weather_labels_path, self.file_extension)

        path_map = dict(zip(filenames, paths))
        highway_map = dict(zip(highway_filenames, highway_paths))
        weather_map = dict(zip(weather_filenames, weather_paths))

        for filename in tqdm(filenames, total=len(paths), desc="Merging labels"):
            output_path = Path(self.save_labeled_path) / f"{filename}{self.file_extension}"
    
            df_main = pd.read_parquet(path_map[filename], engine="fastparquet")
            df_highway = pd.read_parquet(highway_map[filename], columns=self.merge_signals + self.highway_columns, engine="fastparquet")
            df_weather = pd.read_parquet(weather_map[filename], columns=self.merge_signals + self.weather_columns, engine="fastparquet")

            assert len(df_main) == len(df_highway) == len(df_weather), \
                f"Row mismatch: {len(df_main)}, {len(df_highway)}, {len(df_weather)}"

            df_main = df_main.merge(df_highway, on=self.merge_signals , how="left")
            df_main = df_main.merge(df_weather, on=self.merge_signals , how="left")
            df_main = self._verify_expected_features(df_main, self.signal_mapping)
            df_main.to_parquet(output_path)

    def _extract_and_save_windows(self):
        paths, filenames  = self._get_datasets(self.save_labeled_path, self.file_extension)

        for label_col in self.label_columns:
            
            for window_size in self.window_sizes_sec:
                print(f"Creating windows for {label_col} with size {window_size}")

                save_path = self.save_windows_path / f"{label_col}_{window_size}.npy"
     
                all_windows = []
                for path in tqdm(paths, total=len(paths), desc="Creating windows"):
                    windows = self._extract_windows_from_file(
                        path,
                        list(self.train_signal_mapping.keys()) + [label_col],
                        window_size,
                    )

                    if windows is not None:
                        all_windows.append(windows)

                if all_windows:
                    stacked = np.vstack(all_windows)
                    np.save(save_path, stacked)
                    logger.info(f"Saved {len(stacked)} windows")
                    self._compute_label_distribution(stacked, label_col, f"{label_col}_{window_size}", self.save_reports_path)
                    del all_windows, stacked
                    gc.collect()

    def _extract_windows_from_file(self, path, columns, window_size):
        # Load only the required columns
        df = pd.read_parquet(path, columns=columns)

        train_mapping = self.train_signal_mapping.copy()
        train_mapping[df.columns[-1]] = len(train_mapping)
        df = self._verify_expected_features(df, train_mapping)

        if len(df) < window_size:
            return None

        df[df.columns[-1]] = df[df.columns[-1]].map(self.label_encoders[df.columns[-1]])

        data = df.values

        # Find non-overlapping windows where all labels are identical
        label_windows = sliding_window_view(data[:, -1], window_size)
        valid_mask = (label_windows == label_windows[:, 0:1]).all(axis=1)
        valid_indices = np.where(valid_mask)[0][::window_size]

        if len(valid_indices) == 0:
            return None

        # Extract windows
        windows = np.array([data[i:i+window_size] for i in valid_indices])

        # Remove windows where the label column contains NaN
        windows = windows[~np.isnan(windows[:, :, -1]).any(axis=1)]

        if len(windows) == 0:
            return None

        return windows

    def _compute_label_distribution(self, data: np.ndarray, label_col: str, filename: str, save_path: Path):
        total_windows, window_length, _ = data.shape

        labels = np.array([self.label_decoders[label_col][i] for i in data[:, 0, -1]])
        unique_labels, counts = np.unique(labels, return_counts=True)
        proportions = {str(l): round(c / total_windows * 100, 2) for l, c in zip(unique_labels, counts)}

        # Sort labels by counts descending for better visualization
        sort_idx = np.argsort(counts)[::-1]
        sorted_labels = unique_labels[sort_idx]
        sorted_counts = counts[sort_idx]
        sorted_proportions = [proportions[str(l)] for l in sorted_labels]

        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_labels)))
        bars = ax.barh(sorted_labels, sorted_counts, color=colors, edgecolor="white", linewidth=0.8)

        # Annotate counts and percentages at end of bars
        for bar, count, perc in zip(bars, sorted_counts, sorted_proportions):
            ax.text(
                bar.get_width() + max(sorted_counts)*0.01,  # small offset to the right
                bar.get_y() + bar.get_height()/2,
                f"{count:,} ({perc:.3f}%)",
                ha="left", va="center", fontsize=10
            )

        ax.set_xlabel("Count", fontsize=12)
        ax.set_ylabel("Labels", fontsize=12)
        ax.set_title(f"{filename}\nTotal windows: {total_windows:,} | Window length: {window_length:,}", fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(save_path / f"{filename}.png", dpi=150)
        plt.close(fig)

        # Save JSON summary
        summary = {
            "file": filename,
            "window_length": int(window_length),
            "total_windows": int(total_windows),
            "counts": {str(k): int(v) for k, v in zip(unique_labels, counts)},
            "proportions_percent": proportions
        }

        with open(save_path / f"{filename}.json", "w") as f:
            json.dump(summary, f, indent=2)


    def _filter_split_and_normalize_windowed_data(self, random_seed: int = 42) -> Dict[str, np.ndarray]:

        paths, filenames = self._get_datasets(self.save_windows_path, self.window_extension)
        
        for path, filename in tqdm(zip(paths, filenames), total=len(paths), desc="Filter, split and normalize data"):
            label_col = "_".join(filename.split("_")[:-1])
            window_size = filename.split("_")[-1]
            
            data = np.load(path, mmap_mode='r')
            
            # Drop specified signals (features)
            signals_to_drop_list = self.signals_to_drop.get(label_col, [])
            signals_to_drop_idx = [self.train_signal_mapping[feat] for feat in signals_to_drop_list if feat in self.train_signal_mapping]
            data = np.delete(data, signals_to_drop_idx, axis=2)
            
            # Drop specified labels (filter out windows with certain labels)
            labels_to_drop_list = self.labels_to_drop.get(label_col, [])
            if labels_to_drop_list:  # Only run if list is not empty
                labels_to_drop_encoded = [self.label_encoders[label_col][i] for i in labels_to_drop_list]
                current_labels = data[:, 0, -1]
                mask_to_keep = ~np.isin(current_labels, labels_to_drop_encoded)
                data = data[mask_to_keep]

            assert abs(self.train_pct + self.val_pct + self.test_pct - 1.0) < 1e-6, \
                f"Percentages must sum to 1.0, got {self.train_pct + self.val_pct + self.test_pct}"
            assert 0 < self.fewshot_pct <= 1.0, "fewshot_pct must be between 0 and 1"
        
            np.random.seed(random_seed)
        
            labels = data[:, 0, -1].astype(int)  
            
            # Get unique classes and their counts
            unique_classes, class_counts = np.unique(labels, return_counts=True)
            n_classes = len(unique_classes)
        
            # Initialize splits
            train_indices = []
            val_indices = []
            test_indices = []
        
            # Split each class separately to maintain balance
            for cls in unique_classes:
                # Get indices for this class
                cls_indices = np.where(labels == cls)[0]
                n_samples = len(cls_indices)
                np.random.shuffle(cls_indices)
                
                # Calculate split sizes for THIS CLASS
                n_train = int(n_samples * self.train_pct)
                n_val = int(n_samples * self.val_pct)
                n_test = n_samples - n_train - n_val
                
                # Split indices
                train_indices.extend(cls_indices[:n_train])
                val_indices.extend(cls_indices[n_train:n_train + n_val])
                test_indices.extend(cls_indices[n_train + n_val:])
        
            # Convert to arrays and shuffle
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
            test_indices = np.array(test_indices)
        
            # Extract data for each split (keep labels as last column in each window)
            X_train = data[train_indices].copy()
            X_val = data[val_indices].copy()
            X_test = data[test_indices].copy()
        
            # Fit scaler on training data (exclude label column which is the last one)
            train_features = rearrange(X_train[:, :, :-1], 'n w f -> (n w) f')
            scaler = StandardScaler()
            scaler.fit(train_features)
            
            # Apply scaler to all splits (normalize features but keep labels unchanged)
            X_train[:, :, :-1] = rearrange(
                scaler.transform(rearrange(X_train[:, :, :-1], 'n w f -> (n w) f')),
                '(n w) f -> n w f', n=X_train.shape[0], w=X_train.shape[1]
            )
            
            X_val[:, :, :-1] = rearrange(
                scaler.transform(rearrange(X_val[:, :, :-1], 'n w f -> (n w) f')),
                '(n w) f -> n w f', n=X_val.shape[0], w=X_val.shape[1]
            )
            
            X_test[:, :, :-1] = rearrange(
                scaler.transform(rearrange(X_test[:, :, :-1], 'n w f -> (n w) f')),
                '(n w) f -> n w f', n=X_test.shape[0], w=X_test.shape[1]
            )
        
            X_train_fewshot = X_train
            if self.fewshot_pct < 1.0:
                print(f"\nCreating few-shot training set ({self.fewshot_pct*100}% of train data - CLASS BALANCED ✓)...")
                y_train = X_train[:, 0, -1].astype(int)
                fewshot_indices = []
                
                # Sample PROPORTIONALLY from each class in the training set
                for cls in unique_classes:
                    cls_train_mask = (y_train == cls)
                    cls_train_indices = np.where(cls_train_mask)[0]
                    n_cls_train = len(cls_train_indices)
                    
                    # Calculate how many samples to take for few-shot FROM THIS CLASS
                    n_fewshot = max(1, int(n_cls_train * self.fewshot_pct))
                    
                    print(f"  Class {cls}: Taking {n_fewshot}/{n_cls_train} samples ({n_fewshot/n_cls_train*100:.1f}%)")
                    
                    # Randomly sample
                    selected = np.random.choice(cls_train_indices, size=n_fewshot, replace=False)
                    fewshot_indices.extend(selected)
            
                fewshot_indices = np.array(fewshot_indices)
                np.random.shuffle(fewshot_indices)
            
                X_train_fewshot = X_train[fewshot_indices]

            save_path = self.save_data_split_path / window_size / label_col
            save_path.mkdir(parents=True, exist_ok=True)

            np.save(save_path / 'train.npy', X_train)
            np.save(save_path / 'fewshot_train.npy', X_train_fewshot)
            np.save(save_path / 'val.npy', X_val)
            np.save(save_path / 'test.npy', X_test)

            report_path = self.save_reports_path / window_size / label_col
            report_path.mkdir(parents=True, exist_ok=True)
            self._compute_label_distribution(X_train, label_col, f"{filename}_train", report_path)
            self._compute_label_distribution(X_train_fewshot, label_col, f"{filename}_fewshot_train", report_path)
            self._compute_label_distribution(X_val, label_col, f"{filename}_val", report_path)
            self._compute_label_distribution(X_test, label_col, f"{filename}_test", report_path)

    def _plot_label_distributions(self, data, label_col, save_path, filename):
        
        font_path_normal = "llm-erange/src/utils/times.ttf"
        font_path_bold   = "llm-erange/src/utils/times_bold.ttf"
        fm.fontManager.addfont(font_path_normal)
        fm.fontManager.addfont(font_path_bold)
        prop_normal = fm.FontProperties(fname=font_path_normal)
        plt.rcParams["font.family"] = prop_normal.get_name()
        plt.rcParams["font.size"] = 12

        labels_int = data[:, 0, -1].astype(int)
        labels = np.array([self.label_decoders[label_col][int(l)] for l in labels_int])

        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        proportions = {str(l): round(c / total_samples * 100, 3) for l, c in zip(unique_labels, counts)}
        
        # Sort by counts descending
        sort_idx = np.argsort(counts)[::-1]
        sorted_labels = unique_labels[sort_idx]
        sorted_counts = counts[sort_idx]
        sorted_proportions = [proportions[str(l)] for l in sorted_labels]
        
        # DYNAMIC figure width based on number of classes
        num_classes = len(sorted_labels)
        if num_classes > 5:
            width = num_classes * 1.2  # Much wider for >5 classes
        else:
            width = 7  # Standard width for ≤5 classes
        fig, ax = plt.subplots(1, 1, figsize=(width, 5))
        
        # Color by mean
        avg_count = np.mean(sorted_counts)
        colors = ['#4d4943' if val >= avg_count else '#f5f5f5' for val in sorted_counts]
        
        x = np.arange(len(sorted_labels))
        # WIDER bars with more spacing
        bar_width = 0.8 # Slightly narrower bars = more white space
        bars = ax.bar(x, sorted_counts, width=bar_width, color=colors, edgecolor='black', linewidth=0.5, alpha=1)
        
        # Styling - EXACT match to per_class_metrics
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_color('#000000')
        ax.spines['bottom'].set_color('#000000')
        ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.6, color="#FFFFFF", axis='y')
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=12, colors="#000000")
        ax.set_yscale('log')
        
        # Annotate bars - RIGHT above each bar with 3 decimal places
        for i, (bar, count, perc) in enumerate(zip(bars, sorted_counts, sorted_proportions)):
            ax.text(bar.get_x() + bar.get_width()/2, count * 1.15, 
                f"{count}\n({perc:.3f}%)",  
                ha='center', va='bottom', fontsize=12, color='#000000')
        
        # Set y-limits for more top space
        ax.set_ylim(bottom=min(sorted_counts)*0.5, top=max(sorted_counts)*2.5)
        
        # Add macro average line - matching style
        ax.axhline(y=avg_count, color='#FDCA00', linestyle='--', linewidth=2, alpha=1, 
                label=f'Average: {round(avg_count)}')
        
        # Labels - matching fontsize=12, labelpad=12, fontweight='bold'
        ax.set_ylabel("Number of Trips", fontsize=12, labelpad=12, color='#000000', fontweight='normal')
        ax.set_xlabel("Class Label", fontsize=12, labelpad=12, color='#000000', fontweight='normal')
        ax.set_xticks(x)
        # ROTATE labels 45 degrees, align right
        ax.set_xticklabels(sorted_labels, rotation=0, ha='center')
        
        # Legend - matching fontsize=10
        legend = ax.legend(loc='upper right', fontsize=10, framealpha=1)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(0.5)

        # White background
        fig.patch.set_facecolor('white')
        plt.tight_layout(pad=2.0)
        
        # Save outputs
        pdf_path = save_path / f'{filename}.pdf'
        png_path = save_path / f'{filename}.png'
        json_path = save_path / f'{filename}_summary.json'
        
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.savefig(png_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close(fig)
        
        # Store summary
        summary = {
            "total_samples": int(total_samples),
            "num_classes": int(len(unique_labels)),
            "counts": {str(k): int(v) for k, v in zip(unique_labels, counts)},
            "proportions_percent": proportions,
            "mean_count": float(avg_count)
        }
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def _plot_all(self):
        label_cols = ["air_temperature_label", "highway_label", "surface_condition_label", "weather_label"]
        sizes = ["100", "500"]
        paths, filenames = self._get_datasets(self.save_windows_path, self.window_extension)

        for path, filename  in zip(paths, filenames):
            array = np.load(path)
            label_col = filename.replace("_100", "").replace("_500", "")
            self._plot_label_distributions(array, label_col, self.save_reports_path, filename)
        
        for size in  sizes:
            for label_col in label_cols:
                paths, filenames = self._get_datasets(self.save_data_split_path / size / label_col, self.window_extension)
                for path, filename  in zip(paths, filenames):
                    array = np.load(path)
                    save_path = self.save_reports_path / size / label_col
                    save_path.mkdir(parents=True, exist_ok=True)
                    self._plot_label_distributions(array, label_col, save_path, filename)
                    
    def _preparation_pipeline(self):
        
        #self._merge_and_verify_label_datasets()
        #self._extract_and_save_windows()
        #self._filter_split_and_normalize_windowed_data()
        self._plot_all()
        
    def run(self):
        gc.collect()
        start_time = time.time()
        
        self._preparation_pipeline()
        
        elapsed_total = time.time() - start_time
        minutes, seconds = divmod(elapsed_total, 60)
        logger.info(f"Completed data preparation in {int(minutes)} min {int(seconds)} sec")

       

    