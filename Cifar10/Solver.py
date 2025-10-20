from numpy import isnan, inf, savetxt
import torch
import logging
import gc

import torch.nn as nn
import torch.backends.cudnn as cudnn
from Evaluation_Matix import *
from utils import *
import model_loader
from data_loader import fetch_dataloader
from tqdm import tqdm
from datetime import datetime

class Solver:
	def __init__(self, args, params, CViter):
		def init_weights(m):
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				m.bias.data.fill_(0.01)
		torch.cuda.empty_cache() 
		self.args = args
		self.params = params
		self.CViter = CViter
		self.dataloaders = fetch_dataloader(['train', 'val', 'test'], params, CViter) 
		self.model = model_loader.loadModel(params.hyperparam, args.network, params.channels).cuda()
		self.model.apply(init_weights)
		self.optimizer = torch.optim.Adam(	self.model.parameters(), 
							params.learning_rate, 
							betas=(0.9, 0.999), 
							eps=1e-08, 
							weight_decay = params.weight_decay, 
							amsgrad=False)
		self.loss_fn = get_loss(args.loss, params.hyperparam)
		
		
	def __step__(self):
		torch.cuda.empty_cache() 
		logging.info("Training")
		losses = AverageMeter()
		# switch to train mode
		self.model.train()
		with tqdm(total=len(self.dataloaders['train']), unit = 'epoch', leave = 0) as t:
			for i, (datas, label) in enumerate(self.dataloaders['train']):
				logging.info("        Loading Varable")
				# compute output
				logging.info("        Compute output")
				output = self.model(torch.autograd.Variable(datas.cuda())).double()
				# measure record cost
				cost = self.loss_fn(output, torch.autograd.Variable(label.cuda()).double())
				
				del output

				# compute gradient and do SGD step
				logging.info("        Compute gradient and do SGD step")
				self.optimizer.zero_grad()
				cost.backward()
				self.optimizer.step()
				
				losses.update(cost.cpu().data.numpy(), len(datas))
				gc.collect()
				#self.validate()
				
				t.set_postfix(loss = '{:05.3f}'.format(cost.cpu().data.numpy()), refresh=True)
				t.update()

		return losses()
	
	
	def validate(self, dataset_type = 'val'):
		torch.cuda.empty_cache() 
		logging.info("Validating")
		losses = AverageMeter()
		if dataset_type =='test':
			self.__resume_checkpoint__('best')
			
		# switch to evaluate mode
		self.model.eval()
		with tqdm(total=len(self.dataloaders[dataset_type]), leave = 0) as t:
			for i, (datas, label) in (enumerate(self.dataloaders[dataset_type])):
				logging.info("        Compute output")
				output = self.model(torch.autograd.Variable(datas.cuda())).double()
				label_var = torch.autograd.Variable(label.cuda()).double()
				logging.info("        Computing loss")
				loss = self.loss_fn(output, torch.autograd.Variable(label.cuda()).double())
				
				# measure record cost
				losses.update(loss.cpu().data.numpy(), len(datas))
				
				del output
				del label_var
				
				gc.collect()
				t.update()
		

		return losses()
		
	def test(self, dataset_type = 'test'):
		torch.cuda.empty_cache() 
		logging.info("testing")
		if dataset_type =='test':
			self.__resume_checkpoint__('best')
			
		outputs = [np.empty((0, self.params.channels), float), np.empty((0, self.params.channels), float)]
		# switch to evaluate mode
		self.model.eval()
		with tqdm(total=len(self.dataloaders[dataset_type]), leave = 0) as t:
			for i, (datas, label) in (enumerate(self.dataloaders[dataset_type])):
				logging.info("        Compute output")
				output = self.model(torch.autograd.Variable(datas.cuda())).double()
				label_var = torch.autograd.Variable(label.cuda()).double()
				outputs[0] = np.concatenate((outputs[0], output.cpu().data.numpy()), axis=0)
				outputs[1] = np.concatenate((outputs[1], label_var.cpu().data.numpy()), axis=0)
				
				del output
				del label_var
				
				gc.collect()
				t.update()
		
		return outputs


	def train(self):
		start_epoch = 0
		best_loss = inf	
		if self.args.resume:
			logging.info('Resuming Checkpoint')
			start_epoch, best_loss = self.__resume_checkpoint__('')
			if not start_epoch < self.params.epochs:
				logging.info('Skipping training for finished model\n')
				return 0			
		
		assert best_loss > 0, 'ERROR! Best loss is 0'
		logging.info('    Starting With Best loss = {loss:.4f}'.format(loss = best_loss))
		logging.info('Initialize training from {} to {} epochs'.format(start_epoch, self.params.epochs))

		with tqdm(total=self.params.epochs - start_epoch, leave = 0) as t:
			for epoch in range(start_epoch, self.params.epochs):
				logging.info('CV [{}], Training Epoch: [{}/{}]'.format('_'.join(tuple(map(str, self.CViter))), epoch+1, self.params.epochs))
				
				
				self.__step__()
				gc.collect()
				# evaluate on validation set
				loss = self.validate()
				
				
				gc.collect()

				# remember best model and save checkpoint
				logging.info('    loss {loss:.4f};\n'.format(loss = loss))		
				if loss < best_loss:
					self.__save_checkpoint__({
						'epoch': epoch + 1,
						'state_dict': self.model.state_dict(),
						'loss': loss,
						'optimizer' : self.optimizer.state_dict(),
						}, 'best')
					best_loss = loss
					logging.info('    Saved Best model with  \n{} \n'.format(loss))

				self.__save_checkpoint__({
						'epoch': epoch + 1,
						'state_dict': self.model.state_dict(),
						'loss': best_loss,
						'optimizer' : self.optimizer.state_dict(),
						}, '')
						
				if epoch % 5 == 4 and epoch > 0:
					self.__learning_rate_decay__(self.optimizer, self.params.hyperparam.lrDecay)
				
				t.set_postfix(best_loss = best_loss)
				t.update()

		gc.collect()
		logging.info('Training finalized with best average loss {}\n'.format(best_loss))
		return best_loss
		
	def __save_checkpoint__(self, state, checkpoint_type):
		checkpointpath, checkpointfile = get_checkpointname(	self.args, 
									checkpoint_type, 
									self.CViter)
		if not os.path.isdir(checkpointpath):
			os.mkdir(checkpointpath)
			
		torch.save(state, checkpointfile)


	def __resume_checkpoint__(self, checkpoint_type):
		_, checkpointfile = get_checkpointname(self.args, checkpoint_type, self.CViter)
		
		if not os.path.isfile(checkpointfile):
			return 0, inf
		else:
			logging.info("Loading checkpoint {}".format(checkpointfile))
			checkpoint = torch.load(checkpointfile)
			start_epoch = checkpoint['epoch']
			loss = checkpoint['loss']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
				
			return start_epoch, loss
			
	def __learning_rate_decay__(self, optimizer, decay_rate):
		if decay_rate < 1:
			for param_group in optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * decay_rate
