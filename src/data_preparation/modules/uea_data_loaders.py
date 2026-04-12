"""PyTorch Dataset and DataLoader factory for UEA/UCR multivariate time-series.

This module provides two classes for loading and normalising datasets from
the UEA & UCR Time Series Classification repository via
:func:`sktime.datasets.load_UCR_UEA_dataset`:

* :class:`UEADataset` – a map-style :class:`~torch.utils.data.Dataset` that
  downloads, normalises, and label-encodes a single split (``"train"`` or
  ``"test"``).
* :class:`UEADataFactory` – coordinates scaler sharing between the train and
  test splits and exposes convenience methods for building individual or
  paired :class:`~torch.utils.data.DataLoader` objects.

The expected normalisation contract is:

1. A :class:`~sklearn.preprocessing.StandardScaler` is **fitted** on the
   flattened training windows ``(B*T, C)``.
2. The same fitted scaler is **passed in** when constructing the test
   :class:`UEADataset` to prevent data leakage.

Example:
    >>> from pathlib import Path
    >>> from src.data_preparation.modules.uea_data_loaders import UEADataFactory
    >>>
    >>> factory = UEADataFactory(
    ...     name="ArticularyWordRecognition",
    ...     data_path=Path("data/uea"),
    ...     batch_size=32,
    ...     num_workers=2,
    ... )
    >>> train_loader, test_loader = factory.build_train_test_loaders()
    >>> x_batch, y_batch = next(iter(train_loader))
    >>> x_batch.shape   # (32, T, C)
    >>> y_batch.shape   # (32,)
"""

from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sktime.datasets import load_UCR_UEA_dataset
from einops import rearrange
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

logger = logging.getLogger(__name__)


class UEADataset(Dataset):
    """Map-style PyTorch Dataset for a single UEA/UCR split.

    Downloads (if necessary) and caches the requested dataset via sktime,
    transposes the array to time-first order ``(B, T, C)``, applies
    :class:`~sklearn.preprocessing.StandardScaler` normalisation, and
    integer-encodes string class labels with
    :class:`~sklearn.preprocessing.LabelEncoder`.

    The scaler is **fitted** on the training split and must be **supplied**
    for the test split to prevent data leakage. Use
    :class:`UEADataFactory` to manage this automatically.

    Attributes:
        name (str): UEA/UCR dataset identifier passed directly to
            :func:`sktime.datasets.load_UCR_UEA_dataset`.
        data_path (Path): Directory where sktime caches downloaded
            datasets.
        split (str): One of ``"train"`` or ``"test"``. 
        scaler (StandardScaler | None): Pre-fitted scaler for the ``"test"`` split. 
            Must be ``None`` for ``"train"``. The scaler will be fitted and stored automatically. Defaults to ``None``.
    """

    def __init__(
        self,
        name: str,
        data_path: Path,
        split: str,
        scaler: StandardScaler | None = None,
    ) -> None:
        self.split = split
        self.name = name
        self.data_path = data_path
        self.scaler = scaler
        self.label_encoder = LabelEncoder()

        try:
            self.X, self.y = load_UCR_UEA_dataset(
                name=self.name,
                return_X_y=True,
                extract_path=self.data_path,
                split=self.split,
                return_type="numpy3D"
            )

            self.X = rearrange(self.X, "B C T -> B T C")

            if self.split == "train":
                X_flat = rearrange(self.X, "B T C -> (B T) C")
                self.scaler = StandardScaler()
                self.scaler.fit(X_flat)
                X_normalized = self.scaler.transform(X_flat)
                self.X = rearrange(X_normalized, "(B T) C -> B T C", B=self.X.shape[0])
                logger.info(f"Fitted scaler on {self.name} training data")

            elif self.split == "test":
                if self.scaler is None:
                    raise ValueError(
                        f"Scaler must be provided for test split! Got None for {self.name}"
                    )
                X_flat = rearrange(self.X, "B T C -> (B T) C")
                X_normalized = self.scaler.transform(X_flat)
                self.X = rearrange(X_normalized, "(B T) C -> B T C", B=self.X.shape[0])
                logger.info(f"Applied training scaler to {self.name} test data")

            self.X = torch.tensor(self.X, dtype=torch.float32)

            self.y = self.label_encoder.fit_transform(self.y)
            self.y = torch.tensor(self.y, dtype=torch.long)

            logger.info(
                f"Successfully loaded {self.name} ({self.split}) with shape {self.X.shape}"
            )

        except (ValueError, Exception) as e:
            logger.error(f"Skipping {name} due to error: {e}")

    def __len__(self) -> int:
        """Returns the total number of windows in the split.

        Returns:
            int: Number of windows ``B`` (first dimension of ``self.X``).
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a single normalised ``(features, label)`` pair.

        Args:
            idx (int): Window index in ``[0, len(self))``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A 2-tuple ``(X, y)`` where

            * ``X`` – float32 tensor of shape ``(T, C)`` containing
              normalised feature values for every time step.
            * ``y`` – int64 scalar tensor holding the encoded class label.
        """
        return self.X[idx], self.y[idx]

    def get_scaler(self) -> StandardScaler:
        """Returns the fitted scaler for use when constructing the test split.

        Returns:
            StandardScaler: The :class:`~sklearn.preprocessing.StandardScaler`
            fitted on the flattened training windows.

        Raises:
            ValueError: If called before the scaler has been fitted (i.e.
                on a ``"test"`` split instance or after a failed load).
        """
        if self.scaler is None:
            raise ValueError("Scaler is not fitted yet!")
        return self.scaler


class UEADataFactory:
    """Factory that builds normalised DataLoaders for a single UEA/UCR dataset.

    Manages scaler sharing between the training and test splits so that the
    caller never has to handle the fitted :class:`StandardScaler` manually.
    The train loader **must** be built before the test loader; this is
    enforced automatically by :meth:`build_train_test_loaders`.

    Attributes:
        name (str): UEA/UCR dataset identifier passed to
            :class:`UEADataset`.
        data_path (Path): Directory where sktime caches downloaded
            datasets.
        batch_size (int): Number of windows per mini-batch.
        num_workers (int): Subprocesses spawned by each
            :class:`~torch.utils.data.DataLoader` for prefetching.
    """

    def __init__(
        self,
        name: str,
        data_path: Path,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.name = name
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_scaler: StandardScaler | None = None

    def build_loader(self, split: str, shuffle: bool = True) -> DataLoader | None:
        """Builds a :class:`~torch.utils.data.DataLoader` for one split.

        Constructs a :class:`UEADataset`, passing the stored
        ``train_scaler`` automatically for the ``"test"`` split. After a
        successful ``"train"`` build the fitted scaler is cached in
        ``self.train_scaler`` for subsequent test builds.

        Any exception raised during dataset or loader construction is caught,
        logged at ``ERROR`` level, and ``None`` is returned so that the
        caller can skip the dataset gracefully.

        Args:
            split (str): One of ``"train"`` or ``"test"``.
            shuffle (bool): Whether to shuffle the dataset each epoch.
                Defaults to ``True``; set to ``False`` for evaluation splits.

        Returns:
            DataLoader | None: A configured
            :class:`~torch.utils.data.DataLoader` yielding ``(x, y)``
            batches with ``x`` of shape ``(batch_size, T, C)`` and ``y`` of
            shape ``(batch_size,)``, or ``None`` if construction failed.
        """
        try:
            scaler = self.train_scaler if split == "test" else None

            dataset = UEADataset(
                name=self.name,
                data_path=self.data_path,
                split=split,
                scaler=scaler,
            )

            if split == "train":
                self.train_scaler = dataset.get_scaler()

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=False,
            )

            return loader

        except Exception as e:
            logger.error(f"Failed to build loader for {self.name} ({split}): {e}")
            return None

    def build_train_test_loaders(self) -> tuple[DataLoader | None, DataLoader | None]:
        """Builds the train and test loaders in the correct order.

        The training loader is always built first so that the
        :class:`~sklearn.preprocessing.StandardScaler` is fitted before the
        test loader requests it. Shuffling is enabled for training and
        disabled for testing.

        Returns:
            tuple[DataLoader | None, DataLoader | None]: A 2-tuple
            ``(train_loader, test_loader)``. Either element may be ``None``
            if the corresponding split failed to load; callers should check
            for ``None`` before iterating.
        """
        train_loader = self.build_loader(split="train", shuffle=True)
        test_loader = self.build_loader(split="test", shuffle=False)
        return train_loader, test_loader
