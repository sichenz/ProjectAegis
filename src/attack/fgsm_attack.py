import os
import sys
import torch
import torch.nn as nn
import pickle
import numpy as np

# Add src to path so we can import model
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_dir, 'src'))

from model.dataset import HAIDataset
from torch.utils.data import DataLoader
from model.autoencoder import Conv1DAutoencoder
from defense.ensemble import EnsembleAutoencoder

def fgsm_attack(model, criterion, batch, epsilon):
    """
    Fast Gradient Sign Method (FGSM) attack for Autoencoders.
    Goal: MINIMIZE reconstruction error to hide anomalies.
    """
    # Requires gradient for the input
    batch.requires_grad = True
    
    # Forward pass
    outputs = model(batch)
    
    # Calculate loss
    loss = criterion(outputs, batch).mean()
    
    # Zero gradients, backward pass
    model.zero_grad()
    loss.backward()
    
    # Get the sign of the gradient
    data_grad = batch.grad.data
    sign_data_grad = data_grad.sign()
    
    # Create the perturbed image by subtracting the gradient (minimizing loss)
    perturbed_batch = batch - epsilon * sign_data_grad
    
    # Clip the perturbed data to keep it in the valid [0, 1] range (MinMaxScaler range)
    perturbed_batch = torch.clamp(perturbed_batch, 0, 1)
    
    return perturbed_batch

def run_attack(model_name='autoencoder.pth', epsilon=0.05, threshold=0.016369):
    data_dir = os.path.join(base_dir, 'data', 'hai-23.05')
    models_dir = os.path.join(base_dir, 'models')
    
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    model_path = os.path.join(models_dir, model_name)
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print(f"Model or scaler not found. Please ensure {model_name} exists.")
        return
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    window_size = 60
    batch_size = 256
    
    print("Loading test datasets...")
    test_dataset = HAIDataset(data_dir, window_size=window_size, is_train=False, scaler=scaler)
    # We will use a smaller batch size and drop last to simplify
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    num_features = test_dataset.num_features
    if 'ensemble' in model_name:
        model = EnsembleAutoencoder(num_features=num_features, seq_len=window_size).to(device)
    else:
        model = Conv1DAutoencoder(num_features=num_features, seq_len=window_size).to(device)
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    criterion = nn.MSELoss(reduction='none') # Loss per sample
    
    print(f"Starting FGSM Attack with epsilon = {epsilon}...")
    
    original_anomalies = 0
    successful_hides = 0
    total_processed = 0
    
    # For speed in demonstration, we will only process a subset of the test dataset
    max_batches = 200 # Roughly 50,000 windows
    
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= max_batches:
            break
            
        batch = batch.to(device)
        
        # 1. Evaluate original data
        with torch.no_grad():
            outputs = model(batch)
            loss = criterion(outputs, batch)
            original_loss = loss.mean(dim=[1, 2]).cpu().numpy()
            
        # Find which indices in this batch are naturally considered anomalies (> threshold)
        anomalous_indices = np.where(original_loss > threshold)[0]
        
        if len(anomalous_indices) == 0:
            total_processed += len(batch)
            continue
            
        original_anomalies += len(anomalous_indices)
        
        # 2. Apply FGSM Attack
        # To attack, we only really need to perturb the anomalies, but we can do the whole batch for simplicity
        perturbed_batch = fgsm_attack(model, nn.MSELoss(reduction='none'), batch, epsilon)
        
        # 3. Evaluate perturbed data
        with torch.no_grad():
            outputs_adv = model(perturbed_batch)
            loss_adv = criterion(outputs_adv, perturbed_batch)
            adv_loss = loss_adv.mean(dim=[1, 2]).cpu().numpy()
            
        # Check how many of the originally anomalous samples are now below the threshold
        for idx in anomalous_indices:
            if adv_loss[idx] <= threshold:
                successful_hides += 1
                
        total_processed += len(batch)
        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}/{max_batches} - Hidden so far: {successful_hides}/{original_anomalies}")

    print(f"\n--- Attack Results (Epsilon: {epsilon}) ---")
    print(f"Windows Processed: {total_processed}")
    print(f"Original Anomalies Detected: {original_anomalies}")
    print(f"Anomalies Hidden (Success): {successful_hides} ({(successful_hides/original_anomalies)*100 if original_anomalies > 0 else 0:.2f}%)")
    print(f"Anomalies Still Detected: {original_anomalies - successful_hides}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='autoencoder.pth')
    parser.add_argument('--epsilon', type=float, default=0.05)
    args = parser.parse_args()
    
    run_attack(model_name=args.model, epsilon=args.epsilon)
