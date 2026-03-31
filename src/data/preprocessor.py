"""
Data preprocessing: sliding windows, normalization, and train/test splitting
for time series anomaly detection.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesWindower:
    """Creates sliding windows from time series for model input."""

    def __init__(self, window_size: int = 50, stride: int = 1):
        self.window_size = window_size
        self.stride = stride
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        """Compute normalization statistics from training data."""
        self.mean = data.mean()
        self.std = data.std() + 1e-8
        return self

    def normalize(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("Call fit() first")
        return (data - self.mean) / self.std

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("Call fit() first")
        return data * self.std + self.mean

    def create_windows(self, data: np.ndarray) -> np.ndarray:
        """Create sliding windows from a 1D array."""
        n = len(data)
        windows = []
        for i in range(0, n - self.window_size + 1, self.stride):
            windows.append(data[i : i + self.window_size])
        return np.array(windows)

    def prepare(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize and window the data."""
        if fit:
            self.fit(data)
        normalized = self.normalize(data)
        return self.create_windows(normalized)


class WindowDataset(Dataset):
    """PyTorch Dataset for windowed time series."""

    def __init__(self, windows: np.ndarray):
        self.windows = torch.FloatTensor(windows).unsqueeze(-1)  # (N, W, 1)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.windows[idx]  # input = target (autoencoder)


def create_dataloader(windows: np.ndarray, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    dataset = WindowDataset(windows)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
