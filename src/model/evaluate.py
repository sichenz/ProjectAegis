import os
import torch
import torch.nn as nn
import pickle
import numpy as np
from dataset import get_dataloaders
from autoencoder import Conv1DAutoencoder

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data', 'hai-23.05')
    models_dir = os.path.join(base_dir, 'models')
    
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    model_path = os.path.join(models_dir, 'autoencoder.pth')
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print("Model or scaler not found. Please run train.py first.")
        return
        
    # Load Scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    window_size = 60
    batch_size = 256
    
    # We only need test_loader, but get_dataloaders loads train as well right now. 
    # For a robust script we'd separate them, but we'll use our existing func.
    print("Loading test datasets...")
    from dataset import HAIDataset
    from torch.utils.data import DataLoader
    test_dataset = HAIDataset(data_dir, window_size=window_size, is_train=False, scaler=scaler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    num_features = test_dataset.num_features
    model = Conv1DAutoencoder(num_features=num_features, seq_len=window_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    criterion = nn.MSELoss(reduction='none') # We want error per sample
    
    print("Evaluating test set...")
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            
            # Loss per window: mean over features and sequence length
            loss = criterion(outputs, batch)
            # mean over dimensions 1 and 2 (seq_len and features)
            loss_per_sample = loss.mean(dim=[1, 2]).cpu().numpy()
            reconstruction_errors.extend(loss_per_sample)
            
    reconstruction_errors = np.array(reconstruction_errors)
    
    # We can determine a threshold. Let's just say anything above the 95th percentile of test errors is an anomaly for demonstration
    # In a real scenario, this threshold is determined from the validation split of the normal data
    threshold = np.percentile(reconstruction_errors, 95)
    
    anomalies = reconstruction_errors > threshold
    num_anomalies = np.sum(anomalies)
    
    print(f"--- Evaluation Complete ---")
    print(f"Total Test Windows: {len(reconstruction_errors)}")
    print(f"Max Error: {np.max(reconstruction_errors):.6f}")
    print(f"Min Error: {np.min(reconstruction_errors):.6f}")
    print(f"Calculated Threshold (95th percentile): {threshold:.6f}")
    print(f"Number of Anomalies Detected: {num_anomalies} ({(num_anomalies/len(reconstruction_errors))*100:.2f}%)")

if __name__ == '__main__':
    evaluate()
