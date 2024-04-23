# Import necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Corrected Generator class
class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.label_emb = nn.Embedding(num_classes, noise_dim)  # Updated embedding dimension to match noise
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim + noise_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # Output normalized between -1 and 1
        )

    def forward(self, noise, labels):
        batch_size = noise.shape[0]  # Determine batch size from input
        label_embedding = self.label_emb(labels).view(batch_size, self.noise_dim, 1, 1)  # Adjusted embedding
        noise_with_labels = torch.cat((noise.view(batch_size, self.noise_dim, 1, 1), label_embedding), 1)  # Concatenate noise and labels
        return self.main(noise_with_labels)


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 1)  # Updated embedding dimension to a scalar value
        self.main = nn.Sequential(
            nn.Conv2d(input_channels + 1, 128, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0, bias=True),  # Output a single scalar
        )

    def forward(self, image, labels):
        batch_size = image.shape[0]  # Determine batch size from input
        label_embedding = self.label_emb(labels).view(batch_size, 1, 1, 1).expand(-1, -1, 32, 32)  # Adjusted embedding shape
        image_with_labels = torch.cat((image, label_embedding), 1)  # Concatenate image and labels
        return self.main(image_with_labels).view(-1)
        
class WGAN_Discriminator(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(WGAN_Discriminator, self).__init__()
        self.num_classes = num_classes
        self.main = nn.Sequential(
            nn.Conv2d(input_channels + 1, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0)  # Output is a single scalar for WGAN
        )
    
    def forward(self, image, labels):
        batch_size = image.shape[0]
        label_embedding = nn.Embedding(self.num_classes, 1)(labels)  # Initial embedding
        label_embedding_expanded = label_embedding.unsqueeze(2).unsqueeze(3)  # Add two singleton dimensions
        label_embedding_expanded = label_embedding_expanded.expand(batch_size, 1, 32, 32)  # Expand to match image size
        
        image_with_labels = torch.cat((image, label_embedding_expanded), 1)  # Concatenate image and label embeddings
        return self.main(image_with_labels).view(-1) 

