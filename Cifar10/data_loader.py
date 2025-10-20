import os

import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import Tensor
from itertools import chain
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

np.random.seed(230)

def Encode_label(label, label_list):
	encode = np.zeros(len(label_list))
	ind = label_list.index(label)
	encode[ind] = 1
	return encode

class DatasetWrapper(object):	
	class __DatasetWrapper:
		def __init__(self, params, CViters):

			assert params.CV_iters > 2, 'Cross validation folds must be more than 2 folds'
			self.cv_iters = params.CV_iters
			
			datapath = './data'
			label_list = os.listdir(datapath)
			label_list.sort()
			self.dataset = [[os.path.join(os.path.join(datapath, label), image), Encode_label(label, label_list)] 
							for label in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, label))
								for image in os.listdir(os.path.join(datapath, label)) if '.png' in image]
			self.shuffle()
				

		def shuffle(self):
			"""
			categorize sample ID by label
			"""
			self.ind = np.arange(len(self.dataset))
			np.random.shuffle(self.ind)
			self.ind = self.ind[:int(len(self.ind)/self.cv_iters) * self.cv_iters].reshape((self.cv_iters, -1))

			
	instance = None
	def __init__(self, params, CViters):
		super(DatasetWrapper, self).__init__()
			
		if not DatasetWrapper.instance:
			DatasetWrapper.instance =  DatasetWrapper.__DatasetWrapper(params, CViters)

		self.cv_iters = params.CV_iters
		self.dataset = DatasetWrapper.instance.dataset
		self.Testindex = CViters[0]
		self.CVindex = CViters[1]
	
	def features(self, key):
		"""
		Args: 
			key:(string) value from dataset	
		Returns:
			features in list	
		"""
		return DatasetWrapper.instance.dataset[key][0]

	def label(self, key):
		"""
		Args: 
			key:(string) the sample key/id	
		Returns:
			arrayed label
		"""
		return np.array(DatasetWrapper.instance.dataset[key][1])


	def __trainSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of trainning set
		"""

		ind = list(range(self.cv_iters))
		ind = np.delete(ind, [self.CVindex, self.Testindex])

		trainSet = DatasetWrapper.instance.ind[ind].flatten()
		np.random.shuffle(trainSet)
		return trainSet
	
	def __valSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of validation set
		"""

		valSet = DatasetWrapper.instance.ind[self.CVindex].flatten()
		np.random.shuffle(valSet)
		return valSet

	def __testSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of full dataset
		"""

		testSet = DatasetWrapper.instance.ind[self.Testindex].flatten()
		np.random.shuffle(testSet)
		return testSet

	def getDataSet(self, dataSetType = 'train'):
		"""
		Args: 
			dataSetType: (string) 'train' or 'val'	
		Returns:
			dataset: (np.ndarray) array of key/id of data set
		"""

		if dataSetType == 'train':
			return self.__trainSet()

		if dataSetType == 'val':
			return self.__valSet()

		if dataSetType == 'test':
			return self.__testSet()

		return self.__testSet()
		


class imageDataset(Dataset):
	"""
	A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
	"""
	def __init__(self, dataSetType, params, CViters):
		"""
		initialize DatasetWrapper
		"""
		super(imageDataset, self).__init__()
		self.DatasetWrapper = DatasetWrapper(params, CViters)

		self.samples = self.DatasetWrapper.getDataSet(dataSetType)

		self.transformer = transforms.Compose([ transforms.ToTensor(), transforms.Resize((512, 512)), transforms.RandomRotation(degrees = 180)])


	def __len__(self):
		# return size of dataset
		return len(self.samples)



	def __getitem__(self, idx):
		"""
		Fetch feature and labels from dataset using index of the sample.

		Args:
		    idx: (int) index of the sample

		Returns:
		    feature: (Tensor) feature array
		    label: (int) corresponding label of sample
		"""
		sample = self.samples[idx]
		try:
			image = Image.open(self.DatasetWrapper.features(sample))
		except Exception as error:
			print("An exception occurred:", error)
			print('Cannot load image: ', self.DatasetWrapper.features(sample))
		data = self.transformer(image)

		label = Tensor(self.DatasetWrapper.label(sample).astype(np.uint8))

		return data, label


def fetch_dataloader(types, params, CViters):
	"""
	Fetches the DataLoader object for each type in types.

	Args:
	types: (list) has one or more of 'train', 'val'depending on which data is required '' to get the full dataSet
	params: (Params) hyperparameters

	Returns:
	data: (dict) contains the DataLoader object for each type in types
	"""
	dataloaders = {}
	assert CViters[0] != CViters[1], 'ERROR! Test set and validation set cannot be the same!'
	
	if len(types)>0:
		for split in types:
			if split in ['train', 'val', 'test']:
				dl = DataLoader(imageDataset(split, params, CViters), 
						batch_size=params.batch_size, 
						shuffle=True,
						num_workers=params.num_workers,
						pin_memory=params.cuda)

				dataloaders[split] = dl
	else:
		dl = DataLoader(imageDataset('', params, CViters), 
				batch_size=params.batch_size, 
				shuffle=True,
				num_workers=params.num_workers,
				pin_memory=params.cuda)

		return dl

	return dataloaders

