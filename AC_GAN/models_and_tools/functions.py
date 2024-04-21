import torch
import torch.nn as nn
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.models import resnet50
from scipy.spatial.distance import cosine

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# All main hyperparameters
class Conf:
	def __init__(self):
		self.batch_size = 100
		self.num_classes = 10  # number of classes
		self.len_nz = 100  # number of values of noise for the generation
		self.epochs = 100  # number of training epochs as a basic value (100 to get 50k iterations like in the paper)
		self.img_size = 32  # resizing images if needed
		self.loss_adversarial = torch.nn.BCELoss()  # loss from the paper for the discriminator
		self.loss_classification = torch.nn.CrossEntropyLoss()  # loss for the classification
		self.optimizerG = 'Adam'  # optimized for the generator
		self.optimizerD = 'Adam'  # optimized for the discriminator
		self.learning_rate = 0.0001  # learning rate for both
		self.drop_out = 0.5
		self.beta1 = 0.5  # for Adam optimizers
		self.beta2 = 0.999  # for Adam optimizers
		self.relu_slope = 0.2  # sloe in RELU
		self.activation_noise_std = 0.1  # activation noise std
		self.mean = 0  # initialization mean
		self.std = 0.01  # initialization std


conf = Conf()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def plot_losses(lossesG, lossesD):
	"""
	Plot the training losses for the generator and discriminator.
	Parameters:
	    lossesG (list of floats): Losses from the Generator over all training epochs.
	    lossesD (list of floats): Losses from the Discriminator over all training epochs.
	"""
	epochs = range(1, len(lossesG) + 1)
	
	# Create the plot
	plt.figure(figsize=(10, 5))
	plt.plot(epochs, lossesG, label='Generator Loss')
	plt.plot(epochs, lossesD, label='Discriminator Loss')
	# Add title and labels
	plt.title('Loss per Epoch')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()


def show_images(images, num_images=None, nrow=10, names=None, name_scale=1):
	"""
	Show a grid of images optionally with names above each image.
	Parameters:
	    images (tensor): Tensor of images to display.
	    num_images (int, optional): Number of images to display (default is the minimum of 50 or total images).
	    nrow (int): Number of images per row in the grid.
	    names (list of str, optional): Names to display below each image.
	    name_scale (float): Scale factor for adjusting the name placement and size.
	"""
	if num_images is None:
		# all images of 50 if too many
		num_images = min(len(images), 50)
	# size is determined by the number of images
	plt.figure(figsize=(15 / 10 * nrow, 15))
	images = make_grid(images.cpu()[:num_images], nrow=nrow, padding=2, pad_value=1)
	images = np.clip(images, 0, 1)
	npimg = images.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.axis('off')
	# the size of the image to use it for the grid and names later
	single_img_height = npimg.shape[1] // (num_images // nrow) * name_scale
	if names is not None:
		for i, name in enumerate(names):
			if i >= num_images:
				break  # Break if there are more names than images
			row = i // nrow - 1
			col = i % nrow
			plt.text(col * single_img_height + single_img_height / 2,
			         (row + 1) * single_img_height - single_img_height * 0.14,
			         name, ha='center', va='top', fontsize=12, color='black',
			         bbox=dict(facecolor='white', alpha=0.5, edgecolor='white', boxstyle='round,pad=0.1'))
	plt.show()


def generate_images(net_gen, num_images, image_classes, noise_dim=conf.len_nz):
	"""
	Generate images from a generator network using random noise and specified class labels.
	Parameters:
	    net_gen (torch.nn.Module): The generator network.
	    num_images (int): Number of images to generate.
	    image_classes (list of int): Class labels for each image to generate.
	    noise_dim (int): Dimension of the noise vector.
	    device (str): Device to perform computations on.
	Returns:
	    torch.Tensor: Tensor of generated images.
	"""
	# Generate random noise
	noise = torch.randn(num_images, noise_dim, device=device)
	classes_one_hot = np.zeros((num_images, conf.num_classes))
	for i in range(num_images):
		classes_one_hot[i, image_classes[i]] = 1
	classes_one_hot = torch.tensor(classes_one_hot).to(device).float()
	# Generate images from noise
	with torch.no_grad():
		net_gen.eval()
		images = net_gen(noise, classes_one_hot)
		images = (images + 1) / 2  # Rescale images from [-1,1] to [0,1] if needed
	return images


def show_images_quick(net_gen, num_rows=1):
	"""
	Quickly generate and display a grid of images from a generator network for quick visualization.
	Assumes there are 10 classes (0-9).
	Parameters:
	    net_gen (torch.nn.Module): The generator network.
	    num_rows (int): Number of rows of images to generate, each containing one image per class.
	"""
	# Generate class indices for 10 classes, repeated for the number of rows needed
	images_classes = [it for it in range(10)] * num_rows
	# Assuming 'classes' is a list of class names corresponding to indices
	names_classes = [classes[it] for it in images_classes[:10]]
	# Generate images using the specified network
	generated_images = generate_images(net_gen, 10 * num_rows, images_classes)
	# Display these images using the `show_images` function
	show_images(generated_images, num_images=10 * num_rows, nrow=10, names=names_classes, name_scale=0.975)


def get_images_labels(num_images, dataloader):
	"""
	Fetches a specific number of images and labels from a dataloader.
	Parameters:
	    num_images (int): The number of images and labels to retrieve.
	    dataloader (DataLoader): The DataLoader to fetch from.
	Returns:
	    tuple: Two tensors, the first with images and the second with corresponding labels.
	"""
	images, labels = [], []
	count = 0
	
	for batch_images, batch_labels in dataloader:
		# Calculate how many samples are needed to reach n
		if count + batch_images.size(0) > num_images:
			excess = num_images - count
			images.append(batch_images[:excess])
			labels.append(batch_labels[:excess])
			break
		else:
			images.append(batch_images)
			labels.append(batch_labels)
			count += batch_images.size(0)
	
	# Concatenate all collected batches into single tensors
	images = torch.cat(images, 0)
	labels = torch.cat(labels, 0)
	
	return images, labels
