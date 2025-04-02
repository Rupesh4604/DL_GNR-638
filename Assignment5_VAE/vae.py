import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "../Images"  # Path to your dataset
img_size = (200, 200)
batch_size = 32

# Data loading remains the same
def load_data(dataset_path, img_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

train_loader, val_loader = load_data(dataset_path, img_size, batch_size)

# VAE Model
class VAE(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 25 * 25, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space layers
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128 * 25 * 25),
            nn.ReLU(),
            nn.Unflatten(1, (128, 25, 25)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output range [0,1] for image pixels
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(self.decoder_input(z))
        return x_recon, mu, logvar

# Training function modified for VAE
def train_model(model, train_loader, val_loader, optimizer, epochs=10):
    model.to(device)
    loss_history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:  # Ignore labels since VAE is unsupervised
            images = images.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon_images, mu, logvar = model(images)
            
            # Loss: Reconstruction loss + KL divergence
            recon_loss = nn.functional.binary_cross_entropy(recon_images, images, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_div
            
            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Finished. Time taken: {training_time:.2f} seconds")
    return loss_history, training_time

# Initialize and train
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_history, training_time = train_model(model, train_loader, val_loader, optimizer, epochs=10)

# Plot loss
plt.plot(range(1, 11), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training Loss")
plt.grid(True)
plt.show()