import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50 as res50,  resnet101 as res101, resnet152 as res152
import sys
    
class resize(nn.Module):
	def __init__(self):
		super(resize, self).__init__()

	def forward(self, x):
		x = nn.Upsample((256,256))(x)
		return x  

def resnet50(hyperparams, channels):
	model = nn.Sequential(resize(), res50(num_classes = channels))

	model = nn.Sequential(model, nn.Sigmoid())
	return model