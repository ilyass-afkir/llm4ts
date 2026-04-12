"""PyTorch Dataset and DataLoader factories for truck sensor classification data.

This module provides three classes for loading windowed truck sensor data
stored as ``.npy`` files on disk:

* :class:`TruckDataset` – a map-style ``Dataset`` for deep-learning models.
* :class:`TruckDataFactory` – builds :class:`~torch.utils.data.DataLoader`
  instances from :class:`TruckDataset` for training, few-shot training,
  validation, and testing.
* :class:`TruckDataFactoryBenchmark` – loads raw NumPy arrays for
  scikit-learn-compatible classifiers (ROCKET, InceptionTime, etc.).

The expected on-disk layout is::

    <data_path>/
    └── <window_size>/
        └── <label_col>/
            ├── train.npy
            ├── fewshot_train.npy
            ├── val.npy
            └── test.npy

Each ``.npy`` file has shape ``(B, T, C+1)`` where ``B`` is the number of
windows, ``T`` is the sequence length (window size), ``C`` is the number of
feature channels, and the last channel index holds the integer class label
(constant across the time axis for each window).

Example:
    >>> from src.data_preparation.modules.truck_data_loaders import (
    ...     TruckDataset, TruckDataFactory, TruckDataFactoryBenchmark
    ... )
    >>>
    >>> # --- Deep-learning pipeline ---
    >>> factory = TruckDataFactory(
    ...     batch_size=64,
    ...     num_workers=4,
    ...     data_path="data/truck",
    ...     window_size=128,
    ...     label_col="highway_label",
    ... )
    >>> train_loader, fewshot_loader, val_loader, test_loader = (
    ...     factory.build_train_val_test_loaders()
    ... )
    >>> x_batch, y_batch = next(iter(train_loader))
    >>> x_batch.shape   # (64, 128, C)
    >>> y_batch.shape   # (64,)
    >>>
    >>> # --- Benchmark / scikit-learn pipeline ---
    >>> bench = TruckDataFactoryBenchmark(
    ...     data_path="data/truck",
    ...     window_size=128,
    ...     label_col="highway_label",
    ... )
    >>> X_train, y_train, X_fs, y_fs, X_test, y_test = bench.get_splits()
    >>> X_train.shape   # (B, C, T)  — channels-first for sktime/aeon
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from src.data_preparation.modules.constants import HIGHWAY_OLD_TO_NEW, WEATHER_OLD_TO_NEW


class TruckDataset(Dataset):
    """Map-style PyTorch Dataset for windowed truck sensor data.

    Loads a memory-mapped ``.npy`` file on construction and returns
    ``(features, label)`` pairs indexed by window position. Integer labels
    are optionally re-encoded to consecutive integers via lookup tables
    defined in :mod:`src.data_preparation.modules.constants`.

    Attributes:
        data_path (str | Path): Root directory that contains the
            ``<window_size>/<label_col>/`` sub-tree.
        window_size (int): Sequence length used to locate the correct
                sub-directory and stored as an instance attribute.
        label_col (str): Label column name; must be one of
            ``"highway_label"`` or ``"weather_label"`` for re-encoding to
            apply, otherwise raw integer labels are returned unchanged.
        mode (str): Split name corresponding to the ``.npy`` filename
            (e.g. ``"train"``, ``"fewshot_train"``, ``"val"``,
            ``"test"``).

    Example:
        >>> dataset = TruckDataset(
        ...     data_path="data/truck",
        ...     window_size=128,
        ...     label_col="highway_label",
        ...     mode="train",
        ... )
        >>> x, y = dataset[0]
        >>> x.shape   # torch.Size([128, C])
        >>> y.item()  # integer class index
    """

    def __init__(
        self,
        data_path: str | Path,
        window_size: int,
        label_col: str,
        mode: str,
    ) -> None:
        self.data_path = Path(data_path) / str(window_size) / label_col / f"{mode}.npy"
        self.data = np.load(self.data_path, mmap_mode="r")
        self.window_size = window_size
        self.label_col = label_col
        self.mode = mode

    def __len__(self) -> int:
        """Returns the total number of windows in the dataset.

        Returns:
            int: Number of windows (first dimension of the underlying array).
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a single ``(features, label)`` pair.

        Copies the memory-mapped window at ``idx`` into RAM, splits off the
        label column, applies label re-encoding if required, and converts both
        tensors to the appropriate dtypes.

        Args:
            idx (int): Window index in ``[0, len(self))``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A 2-tuple ``(x, y)`` where

            * ``x`` – float32 tensor of shape ``(T, C)`` containing the
              feature channels for every time step.
            * ``y`` – int64 scalar tensor holding the re-encoded class label.
        """
        window = self.data[idx].copy()

        x = torch.tensor(window[:, :-1], dtype=torch.float32)
        y = window[0, -1]

        if self.label_col == "highway_label":
            y = HIGHWAY_OLD_TO_NEW.get(int(y), int(y))
        elif self.label_col == "weather_label":
            y = np.vectorize(WEATHER_OLD_TO_NEW.get)(y.astype(int))

        y = torch.tensor(y, dtype=torch.long)

        return x, y


class TruckDataFactory:
    """Factory that creates :class:`~torch.utils.data.DataLoader` objects for truck data.

    Wraps :class:`TruckDataset` and exposes convenience methods for building
    individual loaders or the full set of train / few-shot-train / val / test
    loaders in a single call.

    Attributes:
        batch_size (int): Number of windows per mini-batch.
        num_workers (int): Subprocesses spawned by each
            :class:`~torch.utils.data.DataLoader` for prefetching.
        data_path (Path): Root directory that contains the
            ``<window_size>/<label_col>/`` sub-tree.
        window_size (int): Sequence length; selects the correct
            sub-directory under ``data_path``.
        label_col (str): Label column name forwarded to
            :class:`TruckDataset`.

    Example:
        >>> factory = TruckDataFactory(
        ...     batch_size=64,
        ...     num_workers=4,
        ...     data_path="data/truck",
        ...     window_size=128,
        ...     label_col="highway_label",
        ... )
        >>> train_loader, fewshot_loader, val_loader, test_loader = (
        ...     factory.build_train_val_test_loaders()
        ... )
        >>> x, y = next(iter(train_loader))
        >>> x.shape  # torch.Size([64, 128, C])
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_path: Path,
        window_size: int,
        label_col: str,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.window_size = window_size
        self.label_col = label_col

    def build_loader(self, mode: str) -> DataLoader:
        """Builds a single shuffled :class:`~torch.utils.data.DataLoader`.

        Creates a :class:`TruckDataset` for the requested split and wraps it
        in a :class:`~torch.utils.data.DataLoader` with pinned memory and
        persistent workers enabled for GPU training throughput.

        Args:
            mode (str): Split identifier matching a ``.npy`` filename under
                the configured data path (e.g. ``"train"``, ``"val"``,
                ``"test"``, ``"fewshot_train"``).

        Returns:
            DataLoader: Configured loader that yields ``(x, y)`` batches
            with ``x`` of shape ``(batch_size, T, C)`` and ``y`` of shape
            ``(batch_size,)``.
        """
        dataset = TruckDataset(
            data_path=self.data_path,
            window_size=self.window_size,
            label_col=self.label_col,
            mode=mode,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        return loader

    def build_train_val_test_loaders(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Builds all four standard data loaders in a single call.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader, DataLoader]: A 4-tuple
            ``(train_loader, fewshot_train_loader, val_loader, test_loader)``
            where each element is a shuffled :class:`~torch.utils.data.DataLoader`
            for the corresponding split.
        """
        train_loader = self.build_loader(mode="train")
        fewshot_train_loader = self.build_loader(mode="fewshot_train")
        val_loader = self.build_loader(mode="val")
        test_loader = self.build_loader(mode="test")
        return train_loader, fewshot_train_loader, val_loader, test_loader


class TruckDataFactoryBenchmark(Dataset):
    """Loads truck sensor data as raw NumPy arrays for scikit-learn classifiers.

    Unlike :class:`TruckDataFactory`, this class does **not** produce PyTorch
    :class:`~torch.utils.data.DataLoader` objects. Instead it returns
    channels-first NumPy arrays compatible with sktime / aeon estimators such
    as :class:`~sktime.classification.kernel_based.RocketClassifier` and
    :class:`~sktime.classification.deep_learning.InceptionTimeClassifier`.

    Attributes:
         data_path (str | Path): Root directory that contains the
            ``<window_size>/<label_col>/`` sub-tree.
        window_size (int): Sequence length; selects the correct
            sub-directory under ``data_path``.
        label_col (str): Label column name; must be one of
            ``"highway_label"`` or ``"weather_label"`` for re-encoding to
            apply, otherwise raw integer labels are returned unchanged.

    Example:
        >>> bench = TruckDataFactoryBenchmark(
        ...     data_path="data/truck",
        ...     window_size=128,
        ...     label_col="weather_label",
        ... )
        >>> X_train, y_train, X_fs, y_fs, X_test, y_test = bench.get_splits()
        >>> X_train.shape  # (B, C, T) — channels-first
        >>> y_train.dtype  # dtype('int64')
    """

    def __init__(
        self,
        data_path: str | Path,
        window_size: int,
        label_col: str,
    ) -> None:
        self.data_path = Path(data_path) / str(window_size) / label_col
        self.label_col = label_col

    def load_data(self, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Loads a single split from disk and returns channels-first arrays.

        Reads the ``.npy`` file for the requested split, transposes the
        feature array to channels-first order ``(B, C, T)`` using
        :func:`einops.rearrange`, extracts the scalar label for each window
        from the first time step, and applies label re-encoding if required.

        Args:
            mode (str): Split identifier matching a ``.npy`` filename
                (e.g. ``"train"``, ``"fewshot_train"``, ``"test"``).

        Returns:
            tuple[np.ndarray, np.ndarray]: A 2-tuple ``(X, y)`` where

            * ``X`` – float array of shape ``(B, C, T)`` containing the
              feature channels in channels-first order.
            * ``y`` – int array of shape ``(B,)`` holding the re-encoded
              class label for each window.
        """
        file_path = self.data_path / f"{mode}.npy"
        dataset = np.load(file_path, mmap_mode="r").copy()  # (B, T, C+1)

        X = dataset[:, :, :-1]           # (B, T, C)
        X = rearrange(X, "B T C -> B C T")
        y = dataset[:, 0, -1]            # (B,)

        if self.label_col == "highway_label":
            y = np.vectorize(HIGHWAY_OLD_TO_NEW.get)(y.astype(int))
        elif self.label_col == "weather_label":
            y = np.vectorize(WEATHER_OLD_TO_NEW.get)(y.astype(int))

        return X, y

    def get_splits(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads the train, few-shot-train, and test splits in one call.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A 6-tuple ``(X_train, y_train, X_fewshot_train, y_fewshot_train,
            X_test, y_test)`` where every ``X`` array has shape ``(B, C, T)``
            and every ``y`` array has shape ``(B,)``.
        """
        X_train, y_train = self.load_data("train")
        X_fewshot_train, y_fewshot_train = self.load_data("fewshot_train")
        X_test, y_test = self.load_data("test")

        return X_train, y_train, X_fewshot_train, y_fewshot_train, X_test, y_test
