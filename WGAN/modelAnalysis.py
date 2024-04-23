import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# Class to encapsulate WGAN sample generation and cosine similarity
class WGANAnalysis:
    def __init__(self, generator, noise_dim=100, num_classes=10):
        self.generator = generator
        self.noise_dim = noise_dim
        self.num_classes = num_classes

    # Method to generate samples from the generator
    def generate_samples(self, num_samples):
        noise = torch.randn(num_samples, self.noise_dim)  # Random noise
        labels = torch.randint(0, self.num_classes, (num_samples,))  # Random class labels
        generated_images = self.generator(noise, labels).detach()  # Detach for safety

        # Rescale and convert to [0, 1]
        generated_images = (generated_images * 0.5 + 0.5).permute(0, 2, 3, 1).numpy()

        # Display generated images
        fig, axes = plt.subplots(1, num_samples, figsize=(12, 2))
        for i in range(num_samples):
            axes[i].imshow(generated_images[i])
            axes[i].axis("off")

        plt.show()

        return generated_images  # Return generated images for further analysis

    def cosine_similarity(self, images1, images2):
        # Flatten images to compute cosine similarity in feature space
        flat_images1 = torch.tensor(images1).view(images1.shape[0], -1)  # Flatten first set
        flat_images2 = torch.tensor(images2).view(images2.shape[0], -1)  # Flatten second set

        # Compute cosine similarity between each pair of images
        similarities = []
        for img1 in flat_images1:
            cos_sims = [F.cosine_similarity(img1, img2, dim=0).item() for img2 in flat_images2]
            similarities.append(cos_sims)

        return similarities  # Return cosine similarity matrix
