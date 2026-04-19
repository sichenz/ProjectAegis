import torch
import torch.nn as nn

class Conv1DAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for multi-variate time-series anomaly detection.
    Expects input shape: (Batch, Features, SequenceLength)
    """
    def __init__(self, num_features, seq_len):
        super(Conv1DAutoencoder, self).__init__()
        
        self.num_features = num_features
        self.seq_len = seq_len
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Calculate latent sequence length after downsampling
        # seq_len=60 -> stride=2 -> 30 -> stride=2 -> 15
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=num_features, kernel_size=7, stride=2, padding=3, output_padding=1),
            # No final activation to allow reconstructing negative/scaled numbers, or Sigmoid if scaled [0,1]
            nn.Sigmoid() 
        )

    def forward(self, x):
        # Input shape expected: (Batch, Seq_Len, Features)
        # Conv1d expects (Batch, Channels, Seq_Len) -> We need to transpose
        x = x.transpose(1, 2)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Transpose back to (Batch, Seq_Len, Features)
        return decoded.transpose(1, 2)

if __name__ == '__main__':
    # Test model shape
    model = Conv1DAutoencoder(num_features=86, seq_len=60)
    test_input = torch.randn(16, 60, 86)  # Batch 16, Window 60, Features 86
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    assert test_input.shape == output.shape, "Input and output shapes must match!"
