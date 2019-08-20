import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

#https://www.youtube.com/watch?v=u0hC8gmpUDw
kwarqs = {}

#Trainingdaten
train = louder = torch.utils.data.DataLoader(datasets.MNIST("data", train=True, download=True,
                                                    transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,),(0.3081,))])),
                                     batch_size=64, shuffle=True, **kwarqs)

#Testdaten
test = louder = torch.utils.data.DataLoader(datasets.MNIST("data", train=False,
                                                    transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,),(0.3081,))])),
                                     batch_size=64, shuffle=True, **kwarqs)
