import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_dir, 'src'))

from model.dataset import get_dataloaders
from model.autoencoder import Conv1DAutoencoder

def generate_adv_batch(model, criterion, batch, epsilon):
    batch.requires_grad = True
    outputs = model(batch)
    loss = criterion(outputs, batch).mean()
    
    model.zero_grad()
    loss.backward()
    
    data_grad = batch.grad.data
    # Maximize loss to create adversarial examples (False Alarms)
    perturbed_batch = batch + epsilon * data_grad.sign()
    perturbed_batch = torch.clamp(perturbed_batch, 0, 1)
    
    return perturbed_batch.detach()

def adv_train():
    data_dir = os.path.join(base_dir, 'data', 'hai-23.05')
    models_dir = os.path.join(base_dir, 'models')
    
    window_size = 60
    batch_size = 256
    num_epochs = 3 # Keep small for demonstration
    learning_rate = 1e-3
    epsilon = 0.05
    
    print("Loading datasets for Adversarial Training...")
    train_loader, _, _ = get_dataloaders(data_dir, window_size=window_size, batch_size=batch_size)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    num_features = train_loader.dataset.num_features
    # Start from scratch or fine-tune? Let's fine-tune the existing model to save time
    model = Conv1DAutoencoder(num_features=num_features, seq_len=window_size).to(device)
    model_path = os.path.join(models_dir, 'autoencoder.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Loaded baseline model for fine-tuning.")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting Adversarial Training with Epsilon {epsilon}...")
    
    # We will use a fraction of the dataset for demonstration speed
    max_batches = 500
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
                
            batch = batch.to(device)
            
            # Generate Adversarial Examples
            x_adv = generate_adv_batch(model, nn.MSELoss(reduction='none'), batch, epsilon)
            
            # Clear gradients from generation
            optimizer.zero_grad()
            
            # Forward pass on normal and adversarial data
            out_normal = model(batch)
            out_adv = model(x_adv)
            
            # Loss: We want the model to reconstruct the ORIGINAL clean batch in both cases
            loss_normal = criterion(out_normal, batch)
            loss_adv = criterion(out_adv, batch) 
            
            # Combined Loss
            total_loss = 0.5 * loss_normal + 0.5 * loss_adv
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{max_batches}], Total Loss: {total_loss.item():.6f}")
                
        avg_loss = train_loss / min(len(train_loader), max_batches)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.6f}")
        
    robust_model_path = os.path.join(models_dir, 'autoencoder_robust.pth')
    torch.save(model.state_dict(), robust_model_path)
    print(f"Saved robust model to {robust_model_path}")

if __name__ == '__main__':
    adv_train()
