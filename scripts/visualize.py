import os
import sys
import torch
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))

from model.dataset import HAIDataset
from torch.utils.data import DataLoader
from model.autoencoder import Conv1DAutoencoder
from attack.fgsm_attack import fgsm_attack

def visualize_attack():
    data_dir = os.path.join(base_dir, 'data', 'hai-23.05')
    models_dir = os.path.join(base_dir, 'models')
    docs_dir = os.path.join(base_dir, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    model_path = os.path.join(models_dir, 'autoencoder.pth')
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    window_size = 60
    
    print("Loading test dataset for visualization...")
    test_dataset = HAIDataset(data_dir, window_size=window_size, is_train=False, scaler=scaler)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    model = Conv1DAutoencoder(num_features=test_dataset.num_features, seq_len=window_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    criterion = nn.MSELoss(reduction='none')
    threshold = 0.016369 # Derived from Phase 2
    epsilon = 0.05
    
    # Find the first anomaly
    anomalous_window = None
    original_loss = None
    
    for batch in test_loader:
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch)
            loss = criterion(outputs, batch).mean(dim=[1, 2])
            
        anomalies = torch.where(loss > threshold)[0]
        if len(anomalies) > 0:
            idx = anomalies[0]
            anomalous_window = batch[idx:idx+1]
            original_loss = loss[idx].item()
            break
            
    if anomalous_window is None:
        print("No anomalies found to visualize.")
        return
        
    print(f"Found anomaly with original MSE: {original_loss:.6f}")
    
    # Apply FGSM to hide it
    perturbed_window = fgsm_attack(model, nn.MSELoss(reduction='none'), anomalous_window, epsilon)
    
    with torch.no_grad():
        outputs_adv = model(perturbed_window)
        adv_loss = criterion(outputs_adv, perturbed_window).mean().item()
        
    print(f"Perturbed anomaly MSE: {adv_loss:.6f}")
    
    # Move to CPU for plotting
    orig_np = anomalous_window.detach().squeeze().cpu().numpy()
    pert_np = perturbed_window.detach().squeeze().cpu().numpy()
    noise_np = pert_np - orig_np
    
    # We have 86 features. Plotting all is too messy. Let's find the top 3 features with the highest variance.
    variances = np.var(orig_np, axis=0)
    top_features = np.argsort(variances)[-3:]
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Project Aegis: Adversarial Anomaly Hiding Attack (FGSM)', fontsize=16)
    
    colors = ['r', 'g', 'b']
    
    # Plot 1: Original
    for i, f_idx in enumerate(top_features):
        axs[0].plot(orig_np[:, f_idx], label=f'Sensor {f_idx}', color=colors[i])
    axs[0].set_title(f'Original Anomaly (Reconstruction MSE: {original_loss:.6f} > Threshold)')
    axs[0].set_ylabel('Normalized Value')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Noise
    for i, f_idx in enumerate(top_features):
        axs[1].plot(noise_np[:, f_idx], color=colors[i])
    axs[1].set_title(f'Adversarial Noise ($\epsilon$ = {epsilon})')
    axs[1].set_ylabel('Perturbation')
    axs[1].grid(True)
    
    # Plot 3: Perturbed
    for i, f_idx in enumerate(top_features):
        axs[2].plot(pert_np[:, f_idx], color=colors[i])
    axs[2].set_title(f'Poisoned Data (Reconstruction MSE: {adv_loss:.6f} < Threshold)')
    axs[2].set_xlabel('Time Step (Seconds)')
    axs[2].set_ylabel('Normalized Value')
    axs[2].grid(True)
    
    plt.tight_layout()
    out_path = os.path.join(docs_dir, 'anomaly_visualization.png')
    plt.savefig(out_path, dpi=300)
    print(f"Visualization saved to {out_path}")

if __name__ == '__main__':
    visualize_attack()
