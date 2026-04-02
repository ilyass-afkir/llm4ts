from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data_preparation.modules.constants import HIGHWAY_OLD_TO_NEW,  WEATHER_OLD_TO_NEW
from einops import rearrange

class TruckDataset(Dataset):
    def __init__(
        self, 
        data_path: str | Path, 
        window_size: int, 
        label_col: str, 
        mode: str
    ):
      
        self.data_path = Path(data_path) / str(window_size) / label_col / f"{mode}.npy"
        self.data = np.load(self.data_path, mmap_mode="r")
        self.window_size = window_size
        self.label_col = label_col
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.data[idx].copy()

        # Separate features and label
        x = torch.tensor(window[:, :-1], dtype=torch.float32) 
        y = window[0, -1]
        
        # Re-encode highway labels to consecutive integers
        if self.label_col == "highway_label":
            y = HIGHWAY_OLD_TO_NEW.get(int(y), int(y))  # Map old → new, keep unchanged if not in mapping
        elif self.label_col == "weather_label":
            y = np.vectorize(WEATHER_OLD_TO_NEW.get)(y.astype(int))
        
        y = torch.tensor(y, dtype=torch.long)
        
        return x, y

class TruckDataFactory:
    def __init__(
        self, 
        batch_size: int, 
        num_workers: int,
        data_path: Path, 
        window_size: int, 
        label_col: str 
    ):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path 
        self.window_size= window_size 
        self.label_col = label_col 

    def _build_loader(self, mode: str) -> DataLoader:

        dataset = TruckDataset(
            data_path=self.data_path,
            window_size=self.window_size,
            label_col=self.label_col,
            mode=mode
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )

        return loader
    
    def build_train_val_test_loaders(self):
        train_loader = self._build_loader(mode="train")
        fewshot_train_loader = self._build_loader(mode="fewshot_train")
        val_loader = self._build_loader(mode="val")
        test_loader = self._build_loader(mode="test")

        return train_loader, fewshot_train_loader, val_loader, test_loader

    
class TruckDataFactoryBenchmark(Dataset):
    def __init__(
        self, 
        data_path: str | Path, 
        window_size: int, 
        label_col: str, 
    ):
        
        self.data_path = Path(data_path) / str(window_size) / label_col
        self.label_col = label_col

    def _load_data(self, mode: str) -> tuple:
        file_path = self.data_path / f"{mode}.npy"
        dataset = np.load(file_path, mmap_mode="r").copy()  # (B, T, channels)
        X = dataset[:, :, :-1]  # Remove label channel from features
        X = rearrange(X, "B T C -> B C T")
        y = dataset[:, 0, -1]   # Extract label from first timestep (same for all timesteps)
        
        # Re-encode highway labels to consecutive integers
        if self.label_col == "highway_label":
            y = np.vectorize(HIGHWAY_OLD_TO_NEW.get)(y.astype(int))
        elif self.label_col == "weather_label":
            y = np.vectorize(WEATHER_OLD_TO_NEW.get)(y.astype(int))
   
        return X, y
    
    def _get_splits(self):
        X_train, y_train = self._load_data("train")
        X_fewshot_train, y_fewshot_train = self._load_data("fewshot_train")
        X_test, y_test = self._load_data("test")
        
        return X_train, y_train, X_fewshot_train, y_fewshot_train, X_test, y_test
