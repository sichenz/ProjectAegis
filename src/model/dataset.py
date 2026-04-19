import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class HAIDataset(Dataset):
    """
    PyTorch Dataset for HAI Security Dataset.
    Generates sliding windows of telemetry data for time-series anomaly detection.
    """
    def __init__(self, data_dir, window_size=60, is_train=True, scaler=None):
        self.window_size = window_size
        self.is_train = is_train
        
        # Find all CSV files for train or test
        pattern = "hai-train*.csv" if is_train else "hai-test*.csv"
        file_paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
        
        if not file_paths:
            raise FileNotFoundError(f"No files found matching {pattern} in {data_dir}")
        
        print(f"Loading {'training' if is_train else 'testing'} data from {len(file_paths)} files...")
        
        df_list = []
        for fp in file_paths:
            df = pd.read_csv(fp)
            # Drop timestamp
            if 'timestamp' in df.columns:
                df = df.drop(columns=['timestamp'])
            df_list.append(df)
            
        # Concatenate all files vertically
        self.data_df = pd.concat(df_list, axis=0, ignore_index=True)
        
        # Convert to numpy and scale
        raw_data = self.data_df.values
        
        if is_train:
            self.scaler = MinMaxScaler()
            self.scaled_data = self.scaler.fit_transform(raw_data)
        else:
            if scaler is None:
                raise ValueError("A scaler must be provided for test data.")
            self.scaler = scaler
            self.scaled_data = self.scaler.transform(raw_data)
            
        # Generate valid starting indices for windows
        self.num_samples = len(self.scaled_data) - self.window_size + 1
        
        # Convert to tensor
        self.tensor_data = torch.FloatTensor(self.scaled_data)
        
        self.num_features = self.tensor_data.shape[1]
        print(f"Loaded {self.num_samples} windows of shape ({self.window_size}, {self.num_features})")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Extract a sliding window of shape (window_size, num_features)
        window = self.tensor_data[idx : idx + self.window_size]
        return window

def get_dataloaders(data_dir, window_size=60, batch_size=256):
    """
    Returns train and test dataloaders, along with the scaler used for normalization.
    """
    train_dataset = HAIDataset(data_dir, window_size=window_size, is_train=True)
    test_dataset = HAIDataset(data_dir, window_size=window_size, is_train=False, scaler=train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader, train_dataset.scaler
