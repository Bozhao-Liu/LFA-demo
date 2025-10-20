import os
from typing import Union
import torch
import torch.nn as nn
import logging
from sklearn import metrics
import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

epslon = 1e-8

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __call__(self):
        return self.avg


class ECE_loss(nn.Module):
	def __init__(self, weights: Union[tuple, int] = (1, 1)):
		super(ECE_loss, self).__init__()
		if type(weights) is int:
			self.weight = [weights, weights]
		else:
			self.weight = [weights[0], weights[1]]

	def forward(self, pred, label):
		pred = torch.clamp(pred, min=epslon, max=1-epslon)
		pos_loss = torch.mul(label, 1.0/(pred + epslon) - 1)
		neg_loss = torch.mul(1 - label, 1.0/(1 - pred + epslon) - 1)
		loss =  self.weight[0]*pos_loss + self.weight[1]*neg_loss
		return torch.mean(loss)
		
		
class Focal_loss(nn.Module):
	def __init__(self, gamma: Union[tuple, int] =  (2, 2)):
		super(Focal_loss, self).__init__()
		if type(gamma) is int:
			self.gamma = [gamma, gamma]
		else:
			self.gamma = [gamma[0], gamma[1]]
			
	def forward(self, pred, label):
		pred = torch.clamp(pred, min=epslon, max=1-epslon)
		exp_pos = torch.mul(label, -torch.pow(1-pred, self.gamma[0]))
		pos_loss = torch.mul(exp_pos, torch.log(pred))
		exp_neg = torch.mul(1 - label, -torch.pow(pred, self.gamma[0]))
		neg_loss = torch.mul(exp_neg, torch.log(1-pred))
		loss =  pos_loss + neg_loss
		loss = torch.mean(loss) 
		if not np.isnan(loss.cpu().data.numpy().any()):	
			return loss
		else:
			print('#####NAN#####')
			return 1e5
		
class F_ECE_loss(nn.Module):
	def __init__(self, gamma: int = 2):
		super(F_ECE_loss, self).__init__()
		self.gamma = gamma

	def forward(self, pred, label):
		pred = torch.clamp(pred, min=epslon, max=1-epslon)
		pos_loss = torch.mul(label, 1.0/(pred + epslon) - 1)
		
		exp_neg = torch.mul(1 - label, -torch.pow(pred, self.gamma))
		neg_loss = torch.mul(exp_neg, torch.log(1-pred))
		loss =  pos_loss + neg_loss
		return torch.mean(loss)

class ASL(nn.Module):
	''' Notice - optimized version, minimizes memory allocation and gpu uploading,
	favors inplace operations'''

	def __init__(self, gamma_neg=4, gamma_pos=1):
		super(ASL, self).__init__()

		self.gamma_neg = gamma_neg
		self.gamma_pos = gamma_pos

	def forward(self, pred, label):
		""""
		Parameters
		----------
		x: input logits
		y: targets (multi-label binarized vector)
		"""
		pred = torch.clamp(pred, min=epslon, max=1-epslon)
		neg_label = 1 - label #1-y

		# Calculating Probabilities
		P_pos = pred
		P_neg = 1.0 - pred

		P_pos.clamp_(min=epslon)
		P_neg.add_(0.05).clamp_(max=1, min=epslon)

		# Basic CE calculation
		loss_pos = -torch.mul(label, torch.log(P_pos))
		loss_neg = -torch.mul(neg_label, torch.log(P_neg))

		# Asymmetric Focusing
		if self.gamma_neg > 0 or self.gamma_pos > 0:
			loss_pos = torch.mul(torch.pow(P_neg, self.gamma_pos), loss_pos)
			loss_neg = torch.mul(torch.pow(P_pos, self.gamma_neg), loss_neg)
						  
		loss = torch.add(loss_pos, loss_neg)
                
		return loss.mean()


def get_loss(loss_name, Hyperparam):
	loss_name = loss_name.lower()
	loss_name = loss_name.replace('\r', '')
	loss_name = loss_name.replace(' ', '')
	if loss_name == 'bce':
		return nn.BCELoss()
	elif loss_name == 'ece': 
		return ECE_loss()
	elif loss_name == 'focal': 
		Hyperparam.gamma = 2
		return Focal_loss(Hyperparam.gamma)
	elif loss_name == 'f-ece': 
		Hyperparam.gamma = 2
		return F_ECE_loss(Hyperparam.gamma)
	elif loss_name == 'asl':
		return ASL()
	else:
		logging.error("No loss function with the name {} found, please check your spelling.".format(loss_name))
		logging.error("loss function List:")
		logging.error("    BCE")
		logging.error("    ECE")
		logging.error("    focal")
		logging.error("    ASL")
		logging.error("    F-ECE")
		import sys
		sys.exit()
		
		
	
def get_threshold(outputs):
	thresholds = []
	for i in range(outputs[0].shape[1]):
		precision, recall, threshold = metrics.precision_recall_curve(outputs[1][:, i], outputs[0][:, i])
		f1_scores = 2*recall*precision/(recall+precision + 1e-17)
		ind = np.argmax(f1_scores)
		thresholds.append(threshold[ind])
		
	return thresholds
	
def get_eval_multi(outputs, thresholds: list):
	result = defaultdict(list)

	for i in range(outputs[0].shape[1]):
		fpr, tpr, _ = metrics.roc_curve(outputs[1][:, i], outputs[0][:, i], pos_label=1)
		result['AUC'].append(metrics.auc(fpr, tpr))		
		outputs[0][:, i] = outputs[0][:, i] > thresholds[i]
		
		result['threshold'].append(thresholds[i])
		result['acc'].append(metrics.accuracy_score(outputs[1][:, i], outputs[0][:, i]))
		result['Precision'].append(metrics.precision_score(outputs[1][:, i], outputs[0][:, i], average='weighted', zero_division = 0))
		result['Recall'].append(metrics.recall_score(outputs[1][:, i], outputs[0][:, i], average='weighted'))
		result['F0.5'].append(metrics.fbeta_score(outputs[1][:, i], outputs[0][:, i], average='weighted', beta=0.5))
		result['F0'].append(metrics.fbeta_score(outputs[1][:, i], outputs[0][:, i], average='weighted', beta=0))
		result['F1'].append(metrics.f1_score(outputs[1][:, i], outputs[0][:, i], average='weighted'))
	
	return result
	