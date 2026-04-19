import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_dir, 'src'))

from model.dataset import get_dataloaders
from model.autoencoder import Conv1DAutoencoder

class EnsembleAutoencoder(nn.Module):
    def __init__(self, num_features, seq_len, num_models=3):
        super(EnsembleAutoencoder, self).__init__()
        # Use ModuleList so PyTorch tracks the parameters
        self.models = nn.ModuleList([
            Conv1DAutoencoder(num_features, seq_len) for _ in range(num_models)
        ])
        
    def forward(self, x):
        # Average the outputs of all models in the ensemble
        outputs = [model(x) for model in self.models]
        # outputs is a list of tensors of shape (batch, seq_len, features)
        # Stack them along a new dimension and take the mean
        return torch.stack(outputs, dim=0).mean(dim=0)

def train_ensemble():
    data_dir = os.path.join(base_dir, 'data', 'hai-23.05')
    models_dir = os.path.join(base_dir, 'models')
    
    window_size = 60
    batch_size = 256
    # Train very quickly for demonstration (1 epoch)
    num_epochs = 1 
    learning_rate = 1e-3
    
    print("Loading datasets for Ensemble Training...")
    train_loader, _, _ = get_dataloaders(data_dir, window_size=window_size, batch_size=batch_size)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    num_features = train_loader.dataset.num_features
    ensemble = EnsembleAutoencoder(num_features=num_features, seq_len=window_size).to(device)
    
    criterion = nn.MSELoss()
    
    # We will train each model in the ensemble using slightly different subsets of the data
    # to encourage diversity (Bagging-like approach). For simplicity, we just use different seeds
    # or just train them sequentially on the same data since random initialization provides some diversity.
    
    # Max batches to speed up demo
    max_batches = 400
    
    for i, model in enumerate(ensemble.models):
        print(f"\n--- Training Ensemble Sub-Model {i+1}/3 ---")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= max_batches:
                    break
                    
                batch = batch.to(device)
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                if batch_idx % 100 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{max_batches}], Loss: {loss.item():.6f}")
                    
            avg_loss = train_loss / min(len(train_loader), max_batches)
            print(f"Sub-Model {i+1} Final Average Loss: {avg_loss:.6f}")
            
    ensemble_path = os.path.join(models_dir, 'ensemble.pth')
    torch.save(ensemble.state_dict(), ensemble_path)
    print(f"\nSaved trained Ensemble Autoencoder to {ensemble_path}")

if __name__ == '__main__':
    train_ensemble()
