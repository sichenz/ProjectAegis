import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from dataset import get_dataloaders
from autoencoder import Conv1DAutoencoder

def train():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data', 'hai-23.05')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Parameters
    window_size = 60
    batch_size = 256
    num_epochs = 5  # Keep small for demonstration
    learning_rate = 1e-3
    
    print("Loading datasets...")
    train_loader, _, scaler = get_dataloaders(data_dir, window_size=window_size, batch_size=batch_size)
    
    # Save scaler for evaluation
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")
    
    # Device setup (MPS for Mac, CUDA for Nvidia, otherwise CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for training.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")
        
    num_features = train_loader.dataset.num_features
    model = Conv1DAutoencoder(num_features=num_features, seq_len=window_size).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # Batch shape is (batch_size, window_size, num_features)
            batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            
            # Calculate loss (Reconstruction Error)
            loss = criterion(outputs, batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.6f}")
                
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.6f}")
        
    # Save model
    model_path = os.path.join(models_dir, 'autoencoder.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")

if __name__ == '__main__':
    train()
