# seed.py

import os
import random
import numpy as np
import torch

def set_global_seed(seed: int):
    """
    Set all seeds for reproducibility.
    Includes Python, NumPy, PyTorch (CPU + GPU), cuDNN behavior.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """
    Ensures reproducibility in multi-process data loading.
    Used with worker_init_fn in DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def get_torch_generator(seed: int) -> torch.Generator:
    """
    Creates a seeded PyTorch Generator (used in DataLoader).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def save_rng_states() -> dict:
    """
    Save current RNG states (Python, NumPy, Torch CPU/GPU).
    Useful for exact checkpointing.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def load_rng_states(state: dict):
    """
    Restore RNG states.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])

class SeedManager:
    def __init__(self, seed: int = 2025):
        self.seed = seed
        self.generator = self._create_generator()
        self.set_global_seed()

    def set_global_seed(self):
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def seed_worker(self, worker_id: int):
        worker_seed = torch.initial_seed() % 2**32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _create_generator(self) -> torch.Generator:
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        return gen

    def get_generator(self) -> torch.Generator:
        return self.generator

    def save_rng_state(self) -> dict:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        }

    def load_rng_state(self, state: dict):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state_all(state["cuda"])