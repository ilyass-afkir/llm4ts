"""Centralised seeding utilities for fully reproducible PyTorch experiments.

This module exposes :class:`SeedManager`, a stateful helper that applies a
single seed to every randomness source used in a typical training pipeline
(Python, NumPy, PyTorch CPU/GPU, cuDNN) and provides the generator and
worker-init function required by :class:`~torch.utils.data.DataLoader`.

Example:
    >>> from src.utils.seed import SeedManager
    >>> sm = SeedManager(seed=42)
    >>> loader = DataLoader(
    ...     dataset,
    ...     generator=sm.get_generator(),
    ...     worker_init_fn=sm.seed_worker,
    ... )
    >>> checkpoint = {"model": model.state_dict(), "rng": sm.save_rng_state()}
    >>> torch.save(checkpoint, "ckpt.pt")
    >>> # On resume:
    >>> sm.load_rng_state(torch.load("ckpt.pt")["rng"])
"""

import os
import random
import numpy as np
import torch


class SeedManager:
    """Stateful manager that centralises all seeding concerns for an experiment.

    On construction the global seed is applied immediately and a dedicated
    :class:`torch.Generator` is created. The same instance can then supply
    the generator and ``worker_init_fn`` directly to
    :class:`~torch.utils.data.DataLoader`, and can save / restore the full
    RNG state for mid-training checkpointing.

    Attributes:
        seed (int): Seed applied to all RNG sources. Defaults to ``2025``.
    """

    def __init__(self, seed: int = 2025) -> None:
        self.seed = seed
        self.generator = self.create_generator()
        self.set_global_seed()

    def set_global_seed(self) -> None:
        """Applies ``self.seed`` to every randomness source in the pipeline.

        Covers the Python hash seed (``PYTHONHASHSEED``), the ``random``
        module, NumPy, PyTorch CPU, all CUDA devices, and cuDNN
        (deterministic kernels, auto-tuning disabled). Called automatically
        during ``__init__`` and can be called again after loading a
        checkpoint to re-apply the seed.

        Note:
            ``torch.backends.cudnn.deterministic = True`` and
            ``torch.backends.cudnn.benchmark = False`` guarantee
            reproducibility at the cost of reduced GPU throughput. Disable
            these flags for inference-only workloads.
        """
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def seed_worker(self, worker_id: int) -> None:
        """Seeds a DataLoader worker process for reproducible data loading.

        Derives a unique, deterministic seed for each worker from PyTorch's
        ``initial_seed()`` (controlled by the main-process seed) offset by
        ``worker_id``. Pass this bound method as ``worker_init_fn`` to
        :class:`~torch.utils.data.DataLoader`.

        Args:
            worker_id (int): Zero-based index of the worker process, supplied
                automatically by :class:`~torch.utils.data.DataLoader`.

        Example:
            >>> sm = SeedManager(42)
            >>> loader = DataLoader(
            ...     dataset,
            ...     num_workers=4,
            ...     worker_init_fn=sm.seed_worker,
            ...     generator=sm.get_generator(),
            ... )
        """
        worker_seed = torch.initial_seed() % 2**32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def create_generator(self) -> torch.Generator:
        """Creates and seeds the internal :class:`torch.Generator`.

        Called once during ``__init__``; the result is stored as
        ``self.generator`` and exposed via :meth:`get_generator`.

        Returns:
            torch.Generator: A CPU generator seeded with ``self.seed``.
        """
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        return gen

    def get_generator(self) -> torch.Generator:
        """Returns the seeded :class:`torch.Generator` for DataLoader use.

        Using a dedicated generator isolates DataLoader shuffle / sampling
        randomness from the global PyTorch RNG, so that changing the number
        of data-loading steps does not affect model weight initialisation or
        dropout.

        Returns:
            torch.Generator: The CPU generator created during ``__init__``,
            seeded with ``self.seed``.

        Example:
            >>> sm = SeedManager(42)
            >>> loader = DataLoader(dataset, generator=sm.get_generator())
        """
        return self.generator

    def save_rng_state(self) -> dict:
        """Captures the current RNG state of every randomness source.

        Returns a snapshot that can be saved alongside a model checkpoint and
        later restored with :meth:`load_rng_state` to resume training with
        bit-exact reproducibility.

        Returns:
            dict: A dictionary with the following keys:

            * ``"python"`` – state object from :func:`random.getstate`.
            * ``"numpy"`` – state tuple from :func:`numpy.random.get_state`.
            * ``"torch"`` – CPU RNG state tensor from
              :func:`torch.get_rng_state`.
            * ``"cuda"`` – list of per-device GPU RNG state tensors from
              :func:`torch.cuda.get_rng_state_all` (empty list if no CUDA
              devices are available).

        Example:
            >>> sm = SeedManager(42)
            >>> torch.save({"rng": sm.save_rng_state()}, "ckpt.pt")
        """
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        }

    def load_rng_state(self, state: dict) -> None:
        """Restores RNG states from a checkpoint produced by :meth:`save_rng_state`.

        After this call, all subsequent random operations across Python,
        NumPy, PyTorch CPU, and all CUDA devices will proceed as if the
        training run had never been interrupted.

        Args:
            state (dict): Dictionary produced by :meth:`save_rng_state`,
                containing the keys ``"python"``, ``"numpy"``, ``"torch"``,
                and ``"cuda"``.

        Example:
            >>> ckpt = torch.load("ckpt.pt")
            >>> sm = SeedManager(42)
            >>> sm.load_rng_state(ckpt["rng"])
        """
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state_all(state["cuda"])