"""
UEA Dataset with Normalization
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
    def __init__(self, name: str, data_path: Path, split: str, scaler=None):
        
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
                    raise ValueError(f"Scaler must be provided for test split! Got None for {self.name}")
                # Transform test data using training scaler
                X_flat = rearrange(self.X, "B T C -> (B T) C")
                X_normalized = self.scaler.transform(X_flat)
                self.X = rearrange(X_normalized, "(B T) C -> B T C", B=self.X.shape[0])
                logger.info(f"Applied training scaler to {self.name} test data")
            
            self.X = torch.tensor(self.X, dtype=torch.float32)

            # Encode labels to integers
            self.y = self.label_encoder.fit_transform(self.y)
            self.y = torch.tensor(self.y, dtype=torch.long)

            logger.info(f"Successfully loaded {self.name} ({self.split}) with shape {self.X.shape}")
            
        except (ValueError, Exception) as e:
            logger.error(f"Skipping {name} due to error: {e}")
            
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X = self.X[idx]
        y = self.y[idx]  
            
        return X, y
    
    def get_scaler(self):
        """Return the fitted scaler for use with test set"""
        if self.scaler is None:
            raise ValueError("Scaler is not fitted yet!")
        return self.scaler


class UEADataFactory:
    def __init__(self, name: str, data_path: Path, batch_size: int, num_workers: int):
        self.name = name
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_scaler = None  # Store scaler from training data
    
    def build_loader(self, split: str, shuffle: bool = True) -> DataLoader:
        
        try:
            # For test split, pass the training scaler
            scaler = self.train_scaler if split == "test" else None
            
            dataset = UEADataset(
                name=self.name, 
                data_path=self.data_path,
                split=split,
                scaler=scaler
            )
            
            # Store the scaler from training data
            if split == "train":
                self.train_scaler = dataset.get_scaler()
            
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=False
            )
            
            return loader
        
        except Exception as e:
            logger.error(f"Failed to build loader for {self.name} ({split}): {e}")
            return None
  
    def build_train_test_loaders(self) -> tuple[DataLoader, DataLoader]:
        # IMPORTANT: Build train loader FIRST to fit the scaler
        train_loader = self.build_loader(split="train", shuffle=True)
        # Then build test loader using the fitted scaler
        test_loader = self.build_loader(split="test", shuffle=False)
        return train_loader, test_loader
