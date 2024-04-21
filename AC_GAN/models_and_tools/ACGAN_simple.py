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


class Generator_simple(nn.Module):
    def __init__(self):
        super(Generator_simple, self).__init__()
        self.label_emb = nn.Embedding(conf.num_classes, conf.num_classes)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(110, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
            # nn.Sigmoid()
        )

    def forward(self, noise, labels):
        # input = torch.cat((self.label_emb(labels), noise), dim=1)
        # labels_one_hot = self.nn.functional.one_hot(labels, num_classes=10)
        # input = torch.cat((labels_one_hot, noise), dim=1)
        input = torch.cat((noise, labels), dim=1)
        input_reshaped = input.view(input.shape[0], -1, 1, 1)
        output = self.main(input_reshaped)
        return output


class Discriminator_simple(nn.Module):
    def __init__(self):
        super(Discriminator_simple, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.img_real = nn.Sequential(
            nn.Conv2d(256, 1, 4, 1, 0),
            nn.Sigmoid()
        )
        self.classification = nn.Sequential(
            nn.Linear(256*4*4, 256),
            nn.Linear(256, conf.num_classes)
        )

    def forward(self, input):
        output = self.main(input)
        # print(output.shape)
        img_real = self.img_real(output).view(-1)
        classification = self.classification(output.view(input.shape[0], -1)).squeeze()
        # print(classification.shape)
        # print(input.shape)
        return img_real, classification


def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0.0)