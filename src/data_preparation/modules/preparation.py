"""Data preparation pipeline for truck sensor time-series classification.

This module provides :class:`TruckDataPreparator`, which orchestrates the
full preprocessing pipeline from raw verified trip files to normalised,
split, and windowed NumPy arrays ready for model training.

The pipeline consists of four sequential stages:

1. **Label merging** – joins highway and weather label files onto the
   verified trip data and validates the resulting feature set.
2. **Window extraction** – applies a sliding window over each labeled trip
   and retains only windows where every time step shares the same label.
3. **Filtering, splitting, and normalisation** – drops configured signals
   and label classes, performs stratified train / val / test splitting, fits
   a :class:`~sklearn.preprocessing.StandardScaler` on training features, and
   creates a class-balanced few-shot training subset.
4. **Reporting** – saves per-split label distribution plots (PNG + PDF) and
   JSON summaries to the results directory.

Expected on-disk layout after stage 3::

    <data_split_dir>/
    └── <window_size>/
        └── <label_col>/
            ├── train.npy
            ├── fewshot_train.npy
            ├── val.npy
            └── test.npy

Each ``.npy`` file has shape ``(B, T, C+1)`` where ``B`` is the number of
windows, ``T`` is the window size, ``C`` is the number of feature channels,
and the last channel holds the integer-encoded class label.

Example:
    >>> preparator = TruckDataPreparator(cfg)
    >>> preparator.run()
"""

import logging
from pathlib import Path
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
    """Orchestrates the full truck sensor data preparation pipeline.

    Reads configuration from a Hydra/OmegaConf ``DictConfig``, resolves all
    input/output paths, and exposes the four pipeline stages as public
    methods. Call :meth:`run` to execute the complete pipeline.

    Attributes:
        cfg (DictConfig): Hydra configuration object. All 
            required keys are read from ``cfg.data_preparation``.

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.load("configs/data_preparation.yaml")
        >>> preparator = TruckDataPreparator(cfg)
        >>> preparator.run()
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.verified_data_dir = Path(self.cfg.data_preparation.verified_data_dir)
        self.weather_labels_dir = Path(self.cfg.data_preparation.weather_labels_dir)
        self.highway_labels_dir = Path(self.cfg.data_preparation.highway_labels_dir)

        self.labeled_data_dir = Path(self.cfg.data_preparation.labeled_data_dir)
        self.data_windows_dir = Path(self.cfg.data_preparation.data_windows_dir)
        self.data_split_dir = Path(self.cfg.data_preparation.data_split_dir)
        self.results_dir = Path(self.cfg.data_preparation.results_dir)

        self.file_extension = self.cfg.data_preparation.file_extension
        self.window_extension = self.cfg.data_preparation.window_extension
        self.label_columns = self.cfg.data_preparation.label_columns
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

        self.setup_dirs()

    def setup_dirs(self) -> None:
        """Creates all required output directories if they do not already exist.

        Applies ``mkdir(parents=True, exist_ok=True)`` to
        ``labeled_data_dir``, ``data_windows_dir``, ``results_dir``, and
        ``data_split_dir``.
        """
        for path in [self.labeled_data_dir, self.data_windows_dir, self.results_dir, self.data_split_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def get_datasets(self, data_dir: Path, ext: str) -> tuple[list[Path], list[str]]:
        """Returns all matching files in a directory, sorted alphabetically.

        Args:
            data_dir (Path): Directory to search for files.
            ext (str): File extension used to filter results (e.g.
                ``".parquet"``).

        Returns:
            tuple[list[Path], list[str]]: A 2-tuple ``(paths, filenames)``
            where

            * ``paths`` – sorted list of :class:`~pathlib.Path` objects
              whose names end with ``ext``.
            * ``filenames`` – corresponding list of file stems (filename
              without extension).
        """
        paths = sorted(data_dir.glob(f"*{ext}"))
        filenames = [f.stem for f in paths]
        return paths, filenames

    def verify_expected_features(
        self, df: pd.DataFrame, verification_mapping: dict
    ) -> pd.DataFrame:
        """Validates and reorders DataFrame columns against an expected mapping.

        Checks that the DataFrame contains exactly the signals in
        ``verification_mapping`` — no more and no fewer — then reorders
        the columns to match the index order defined by the mapping values.

        Args:
            df (pd.DataFrame): Trip DataFrame whose columns are validated.
            verification_mapping (dict[str, int]): Mapping of expected signal
                names to their target column indices.

        Returns:
            pd.DataFrame: The input DataFrame with columns reordered to match
            ``verification_mapping``.

        Raises:
            ValueError: If any expected signals are absent from ``df``.
            ValueError: If any unexpected signals are present in ``df``.
        """
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

    def merge_and_verify_label_datasets(self) -> None:
        """Merges highway and weather label files with verified trip data.

        For each trip file in ``verified_data_dir``, loads the corresponding
        highway and weather label files, asserts equal row counts, merges on
        ``merge_signals``, validates the resulting feature set via
        :meth:`verify_expected_features`, and writes the merged file to
        ``labeled_data_dir``.

        Raises:
            AssertionError: If row counts differ between the verified, highway,
                or weather DataFrames for any trip.
            KeyError: If a trip filename present in ``verified_data_dir`` is
                not found in ``highway_labels_dir`` or ``weather_labels_dir``.
        """
        paths, filenames = self.get_datasets(self.verified_data_dir, self.file_extension)
        highway_paths, highway_filenames = self.get_datasets(self.highway_labels_dir, self.file_extension)
        weather_paths, weather_filenames = self.get_datasets(self.weather_labels_dir, self.file_extension)

        path_map = dict(zip(filenames, paths))
        highway_map = dict(zip(highway_filenames, highway_paths))
        weather_map = dict(zip(weather_filenames, weather_paths))

        for filename in tqdm(filenames, total=len(paths), desc="Merging labels"):
            output_path = Path(self.labeled_data_dir) / f"{filename}{self.file_extension}"

            df_main = pd.read_parquet(path_map[filename], engine="fastparquet")
            df_highway = pd.read_parquet(highway_map[filename], columns=self.merge_signals + self.highway_columns, engine="fastparquet")
            df_weather = pd.read_parquet(weather_map[filename], columns=self.merge_signals + self.weather_columns, engine="fastparquet")

            assert len(df_main) == len(df_highway) == len(df_weather), \
                f"Row mismatch: {len(df_main)}, {len(df_highway)}, {len(df_weather)}"

            df_main = df_main.merge(df_highway, on=self.merge_signals, how="left")
            df_main = df_main.merge(df_weather, on=self.merge_signals, how="left")
            df_main = self.verify_expected_features(df_main, self.signal_mapping)
            df_main.to_parquet(output_path)

    def extract_and_save_windows(self) -> None:
        """Extracts fixed-length windows from all labeled datasets.

        Iterates over every combination of label column and window size defined
        in the configuration. For each combination, :meth:`extract_windows_from_file`
        is called on every labeled file; valid window arrays are vertically
        stacked and saved as a single ``.npy`` file. A label distribution
        plot and JSON summary are also written via
        :meth:`compute_label_distribution`.

        Files that produce no valid windows are silently skipped. Memory is
        explicitly freed between label/window-size iterations using
        :func:`gc.collect`.
        """
        paths, _ = self.get_datasets(self.labeled_data_dir, self.file_extension)

        for label_col in self.label_columns:
            for window_size in self.window_sizes_sec:
                print(f"Creating windows for {label_col} with size {window_size}")

                save_path = self.data_windows_dir / f"{label_col}_{window_size}.npy"

                all_windows = []
                for path in tqdm(paths, total=len(paths), desc="Creating windows"):
                    windows = self.extract_windows_from_file(
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
                    self.compute_label_distribution(stacked, label_col, f"{label_col}_{window_size}", self.results_dir)
                    del all_windows, stacked
                    gc.collect()

    def extract_windows_from_file(
        self, path: Path, columns: list[str], window_size: int
    ) -> np.ndarray | None:
        """Extracts valid non-overlapping windows from a single labeled file.

        Loads the requested columns from a parquet file, verifies the feature
        ordering, integer-encodes the label column, and uses a sliding window
        view to identify positions where all ``window_size`` consecutive rows
        share the same label. Only every ``window_size``-th valid start index
        is retained to avoid overlap. Windows containing ``NaN`` label values
        are subsequently removed.

        Args:
            path (Path): Path to the labeled parquet file.
            columns (list[str]): Columns to load; the last element is treated
                as the label column.
            window_size (int): Number of consecutive rows per window.

        Returns:
            np.ndarray | None: Float array of shape
            ``(num_valid_windows, window_size, num_features)`` containing
            valid windows, or ``None`` if the file has fewer rows than
            ``window_size`` or no valid windows remain after filtering.

        Raises:
            ValueError: If required signals are missing or unexpected columns
                are present during feature verification.
            KeyError: If no label encoder is registered for the label column.
        """
        df = pd.read_parquet(path, columns=columns)

        train_mapping = self.train_signal_mapping.copy()
        train_mapping[df.columns[-1]] = len(train_mapping)
        df = self.verify_expected_features(df, train_mapping)

        if len(df) < window_size:
            return None

        df[df.columns[-1]] = df[df.columns[-1]].map(self.label_encoders[df.columns[-1]])

        data = df.values

        label_windows = sliding_window_view(data[:, -1], window_size)
        valid_mask = (label_windows == label_windows[:, 0:1]).all(axis=1)
        valid_indices = np.where(valid_mask)[0][::window_size]

        if len(valid_indices) == 0:
            return None

        windows = np.array([data[i:i + window_size] for i in valid_indices])
        windows = windows[~np.isnan(windows[:, :, -1]).any(axis=1)]

        if len(windows) == 0:
            return None

        return windows

    def compute_label_distribution(
        self,
        data: np.ndarray,
        label_col: str,
        filename: str,
        save_path: Path,
    ) -> None:
        """Plots and saves the label distribution for a window array.

        Decodes integer labels using ``self.label_decoders``, computes per-class
        counts and percentage proportions, renders a horizontal bar chart, and
        writes both a PNG (150 dpi) and a JSON summary to ``save_path``.

        Args:
            data (np.ndarray): Window array of shape ``(B, T, C+1)``; the
                label is read from ``data[:, 0, -1]``.
            label_col (str): Label column name used to look up the decoder in
                ``self.label_decoders``.
            filename (str): Base name (without extension) for the output files.
            save_path (Path): Directory where the PNG and JSON are written.
                Must already exist.
        """
        total_windows, window_length, _ = data.shape

        labels = np.array([self.label_decoders[label_col][i] for i in data[:, 0, -1]])
        unique_labels, counts = np.unique(labels, return_counts=True)
        proportions = {str(l): round(c / total_windows * 100, 2) for l, c in zip(unique_labels, counts)}

        sort_idx = np.argsort(counts)[::-1]
        sorted_labels = unique_labels[sort_idx]
        sorted_counts = counts[sort_idx]
        sorted_proportions = [proportions[str(l)] for l in sorted_labels]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(sorted_labels)))
        bars = ax.barh(sorted_labels, sorted_counts, color=colors, edgecolor="white", linewidth=0.8)

        for bar, count, perc in zip(bars, sorted_counts, sorted_proportions):
            ax.text(
                bar.get_width() + max(sorted_counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({perc:.3f}%)",
                ha="left", va="center", fontsize=10
            )

        ax.set_xlabel("Count", fontsize=12)
        ax.set_ylabel("Labels", fontsize=12)
        ax.set_title(
            f"{filename}\nTotal windows: {total_windows:,} | Window length: {window_length:,}",
            fontsize=14
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(save_path / f"{filename}.png", dpi=150)
        plt.close(fig)

        summary = {
            "file": filename,
            "window_length": int(window_length),
            "total_windows": int(total_windows),
            "counts": {str(k): int(v) for k, v in zip(unique_labels, counts)},
            "proportions_percent": proportions,
        }
        with open(save_path / f"{filename}.json", "w") as f:
            json.dump(summary, f, indent=2)

    def filter_split_and_normalize_windowed_data(self, random_seed: int = 42) -> None:
        """Filters, stratifies, normalises, and persists windowed data splits.

        For every window file in ``data_windows_dir``:

        1. Drops configured feature signals (``signals_to_drop``).
        2. Filters out windows whose label belongs to ``labels_to_drop``.
        3. Performs stratified per-class splitting into train / val / test
           according to ``train_pct``, ``val_pct``, and ``test_pct``.
        4. Fits a :class:`~sklearn.preprocessing.StandardScaler` on training
           features and applies it to all three splits (label column excluded).
        5. Creates a class-balanced few-shot subset from the training split
           using ``fewshot_pct``.
        6. Saves all four arrays (``train``, ``fewshot_train``, ``val``,
           ``test``) as ``.npy`` files under ``data_split_dir``.
        7. Writes label distribution reports for every split via
           :meth:`compute_label_distribution`.

        Args:
            random_seed (int): Seed passed to :func:`numpy.random.seed` for
                reproducible shuffling and few-shot sampling. Defaults to
                ``42``.

        Raises:
            AssertionError: If ``train_pct + val_pct + test_pct`` does not
                equal ``1.0`` (within ``1e-6``) or if ``fewshot_pct`` is
                outside ``(0, 1]``.
        """
        paths, filenames = self.get_datasets(self.data_windows_dir, self.window_extension)

        for path, filename in tqdm(zip(paths, filenames), total=len(paths), desc="Filter, split and normalize data"):
            label_col = "_".join(filename.split("_")[:-1])
            window_size = filename.split("_")[-1]

            data = np.load(path, mmap_mode="r")

            signals_to_drop_list = self.signals_to_drop.get(label_col, [])
            signals_to_drop_idx = [self.train_signal_mapping[feat] for feat in signals_to_drop_list if feat in self.train_signal_mapping]
            data = np.delete(data, signals_to_drop_idx, axis=2)

            labels_to_drop_list = self.labels_to_drop.get(label_col, [])
            if labels_to_drop_list:
                labels_to_drop_encoded = [self.label_encoders[label_col][i] for i in labels_to_drop_list]
                current_labels = data[:, 0, -1]
                mask_to_keep = ~np.isin(current_labels, labels_to_drop_encoded)
                data = data[mask_to_keep]

            assert abs(self.train_pct + self.val_pct + self.test_pct - 1.0) < 1e-6, \
                f"Percentages must sum to 1.0, got {self.train_pct + self.val_pct + self.test_pct}"
            assert 0 < self.fewshot_pct <= 1.0, "fewshot_pct must be between 0 and 1"

            np.random.seed(random_seed)

            labels = data[:, 0, -1].astype(int)
            unique_classes, _ = np.unique(labels, return_counts=True)

            train_indices, val_indices, test_indices = [], [], []

            for cls in unique_classes:
                cls_indices = np.where(labels == cls)[0]
                n_samples = len(cls_indices)
                np.random.shuffle(cls_indices)

                n_train = int(n_samples * self.train_pct)
                n_val = int(n_samples * self.val_pct)

                train_indices.extend(cls_indices[:n_train])
                val_indices.extend(cls_indices[n_train:n_train + n_val])
                test_indices.extend(cls_indices[n_train + n_val:])

            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
            test_indices = np.array(test_indices)

            X_train = data[train_indices].copy()
            X_val = data[val_indices].copy()
            X_test = data[test_indices].copy()

            train_features = rearrange(X_train[:, :, :-1], "n w f -> (n w) f")
            scaler = StandardScaler()
            scaler.fit(train_features)

            X_train[:, :, :-1] = rearrange(
                scaler.transform(rearrange(X_train[:, :, :-1], "n w f -> (n w) f")),
                "(n w) f -> n w f", n=X_train.shape[0], w=X_train.shape[1]
            )
            X_val[:, :, :-1] = rearrange(
                scaler.transform(rearrange(X_val[:, :, :-1], "n w f -> (n w) f")),
                "(n w) f -> n w f", n=X_val.shape[0], w=X_val.shape[1]
            )
            X_test[:, :, :-1] = rearrange(
                scaler.transform(rearrange(X_test[:, :, :-1], "n w f -> (n w) f")),
                "(n w) f -> n w f", n=X_test.shape[0], w=X_test.shape[1]
            )

            X_train_fewshot = X_train
            if self.fewshot_pct < 1.0:
                print(f"\nCreating few-shot training set ({self.fewshot_pct * 100}% of train data - CLASS BALANCED ✓)...")
                y_train = X_train[:, 0, -1].astype(int)
                fewshot_indices = []

                for cls in unique_classes:
                    cls_train_indices = np.where(y_train == cls)[0]
                    n_cls_train = len(cls_train_indices)
                    n_fewshot = max(1, int(n_cls_train * self.fewshot_pct))
                    print(f"  Class {cls}: Taking {n_fewshot}/{n_cls_train} samples ({n_fewshot / n_cls_train * 100:.1f}%)")
                    selected = np.random.choice(cls_train_indices, size=n_fewshot, replace=False)
                    fewshot_indices.extend(selected)

                fewshot_indices = np.array(fewshot_indices)
                np.random.shuffle(fewshot_indices)
                X_train_fewshot = X_train[fewshot_indices]

            save_path = self.data_split_dir / window_size / label_col
            save_path.mkdir(parents=True, exist_ok=True)

            np.save(save_path / "train.npy", X_train)
            np.save(save_path / "fewshot_train.npy", X_train_fewshot)
            np.save(save_path / "val.npy", X_val)
            np.save(save_path / "test.npy", X_test)

            report_path = self.results_dir / window_size / label_col
            report_path.mkdir(parents=True, exist_ok=True)
            self.compute_label_distribution(X_train, label_col, f"{filename}_train", report_path)
            self.compute_label_distribution(X_train_fewshot, label_col, f"{filename}_fewshot_train", report_path)
            self.compute_label_distribution(X_val, label_col, f"{filename}_val", report_path)
            self.compute_label_distribution(X_test, label_col, f"{filename}_test", report_path)

    def plot_label_distributions(
        self,
        data: np.ndarray,
        label_col: str,
        save_path: Path,
        filename: str,
    ) -> None:
        """Renders a publication-quality label distribution bar chart.

        Decodes integer labels, sorts classes by descending count, renders a
        vertical bar chart with a log-scaled y-axis and an average-count
        reference line, and writes PNG (300 dpi), PDF, and a JSON summary to
        ``save_path``. Uses Times New Roman font if the font files are present
        at the configured paths.

        The figure width scales dynamically: ``num_classes * 1.2`` inches for
        more than five classes, ``7`` inches otherwise.

        Args:
            data (np.ndarray): Window array of shape ``(B, T, C+1)``; the
                label is read from ``data[:, 0, -1]``.
            label_col (str): Label column name used to look up the decoder in
                ``self.label_decoders``.
            save_path (Path): Directory where PNG, PDF, and JSON files are
                written. Must already exist.
            filename (str): Base name (without extension) for all output files.
        """
        font_path_normal = "llm-erange/src/utils/times.ttf"
        font_path_bold = "llm-erange/src/utils/times_bold.ttf"
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

        sort_idx = np.argsort(counts)[::-1]
        sorted_labels = unique_labels[sort_idx]
        sorted_counts = counts[sort_idx]
        sorted_proportions = [proportions[str(l)] for l in sorted_labels]

        num_classes = len(sorted_labels)
        width = num_classes * 1.2 if num_classes > 5 else 7
        fig, ax = plt.subplots(1, 1, figsize=(width, 5))

        avg_count = np.mean(sorted_counts)
        colors = ["#4d4943" if val >= avg_count else "#f5f5f5" for val in sorted_counts]

        x = np.arange(len(sorted_labels))
        bars = ax.bar(x, sorted_counts, width=0.8, color=colors, edgecolor="black", linewidth=0.5, alpha=1)

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
        ax.set_yscale("log")

        for bar, count, perc in zip(bars, sorted_counts, sorted_proportions):
            ax.text(
                bar.get_x() + bar.get_width() / 2, count * 1.15,
                f"{count}\n({perc:.3f}%)",
                ha="center", va="bottom", fontsize=12, color="#000000"
            )

        ax.set_ylim(bottom=min(sorted_counts) * 0.5, top=max(sorted_counts) * 2.5)
        ax.axhline(y=avg_count, color="#FDCA00", linestyle="--", linewidth=2, alpha=1,
                   label=f"Average: {round(avg_count)}")

        ax.set_ylabel("Number of Trips", fontsize=12, labelpad=12, color="#000000", fontweight="normal")
        ax.set_xlabel("Class Label", fontsize=12, labelpad=12, color="#000000", fontweight="normal")
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_labels, rotation=0, ha="center")

        legend = ax.legend(loc="upper right", fontsize=10, framealpha=1)
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(0.5)

        fig.patch.set_facecolor("white")
        plt.tight_layout(pad=2.0)

        plt.savefig(save_path / f"{filename}.pdf", bbox_inches="tight", facecolor="white", dpi=300)
        plt.savefig(save_path / f"{filename}.png", bbox_inches="tight", facecolor="white", dpi=300)
        plt.close(fig)

        summary = {
            "total_samples": int(total_samples),
            "num_classes": int(len(unique_labels)),
            "counts": {str(k): int(v) for k, v in zip(unique_labels, counts)},
            "proportions_percent": proportions,
            "mean_count": float(avg_count),
        }
        with open(save_path / f"{filename}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def _plot_all(self) -> None:
        """Regenerates label distribution plots for all window and split files.

        Iterates over every window file in ``data_windows_dir`` and every
        split file under ``data_split_dir`` for all configured label columns
        and window sizes, and calls :meth:`plot_label_distributions` for each.

        Note:
            This method references private helper names (``_get_datasets``,
            ``_plot_label_distributions``) which are aliases for the public
            :meth:`get_datasets` and :meth:`plot_label_distributions` methods.
            Ensure naming is consistent before calling this method.
        """
        label_cols = ["air_temperature_label", "highway_label", "surface_condition_label", "weather_label"]
        sizes = ["100", "500"]
        paths, filenames = self.get_datasets(self.data_windows_dir, self.window_extension)

        for path, filename in zip(paths, filenames):
            array = np.load(path)
            label_col = filename.replace("_100", "").replace("_500", "")
            self.plot_label_distributions(array, label_col, self.results_dir, filename)

        for size in sizes:
            for label_col in label_cols:
                paths, filenames = self.get_datasets(
                    self.data_split_dir / size / label_col, self.window_extension
                )
                for path, filename in zip(paths, filenames):
                    array = np.load(path)
                    save_path = self.results_dir / size / label_col
                    save_path.mkdir(parents=True, exist_ok=True)
                    self.plot_label_distributions(array, label_col, save_path, filename)

    def preparation_pipeline(self) -> None:
        """Defines and executes the ordered data preparation stages.

        Calls the four pipeline stages in sequence. Individual stages can be
        commented out to resume the pipeline from an intermediate checkpoint.

        Stages (in order):

        1. :meth:`merge_and_verify_label_datasets`
        2. :meth:`extract_and_save_windows`
        3. :meth:`_filter_split_and_normalize_windowed_data`
        4. :meth:`_plot_all`
        """
        self.merge_and_verify_label_datasets()
        self.extract_and_save_windows()
        self.filter_split_and_normalize_windowed_data()
        self._plot_all()

    def run(self) -> None:
        """Executes the full data preparation pipeline and logs elapsed time.

        Calls :meth:`_preparation_pipeline`, wraps it with
        :func:`gc.collect` before and after, and logs the total wall-clock
        duration at ``INFO`` level.
        """
        gc.collect()
        start_time = time.time()

        self.preparation_pipeline()

        elapsed_total = time.time() - start_time
        minutes, seconds = divmod(elapsed_total, 60)
        logger.info(f"Completed data preparation in {int(minutes)} min {int(seconds)} sec")

       

    