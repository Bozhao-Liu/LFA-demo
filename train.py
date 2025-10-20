import argparse

import torch
import logging
import os

import numpy as np
from itertools import permutations, product
from tqdm import tqdm
from Cifar10.Solver import Solver
import torch.backends.cudnn as cudnn
from Cifar10.model_loader import get_model_list
from Cifar10.utils import set_logger, set_params, Params
from Cifar10.Evaluation_Matix import get_eval_multi, get_threshold
from collections import defaultdict
import json
from result.get_overleaf_table import create_table

CUDA_VISIBLE_DEVICES = 0

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
parser = argparse.ArgumentParser(description='PyTorch Deep Neural Net Training')
parser.add_argument('--train', default = False, type=str2bool, 
			help="specify whether train the model or not (default: False)")
parser.add_argument('--model_dir', default='./', 
			help="Directory containing params.json")
parser.add_argument('--resume', default = True, type=str2bool, 
			help='resume from latest checkpoint (default: True)')
parser.add_argument('--network', type=str, default = '',
			help='select network to train on. leave it blank means train on all model')
parser.add_argument('--log', default='warning', type=str,
			help='set logging level')
parser.add_argument('--gamma', default=2, type=float,
			help='gamma value for focal')	
parser.add_argument('--loss', type=str, default = 'F_ECE',
			help='select loss function to train with. ')



def main():
	assert torch.cuda.is_available(), "ERROR! GPU is not available."
	cudnn.benchmark = True
	args,unknown = parser.parse_known_args()
	print(args)
	print(unknown)
	args.loss = args.loss.replace('\r', '')
	args.loss = args.loss.replace(' ', '')
	netlist = get_model_list(args.network)
	eval_matrix = {}
	for network in netlist:
		args.network = network
		set_logger(os.path.join(args.model_dir,'Model'), args.network, args.log)
		params = set_params(os.path.join(args.model_dir,'./Model'), network, paramtype = 'params')
		params.hyperparam = Params(model_dir = os.path.join(args.model_dir,'./Model'), network = network, paramtype = 'Hyperparams', loss_fn = args.loss)
		#CV_iters = list(permutations(list(range(params.CV_iters)), 2))
		CV_iters = [(0, 1),(1, 2),(2, 3),(3, 4),(4, 0)]
		eval_matrix = defaultdict(list)
		with tqdm(total = len(CV_iters)) as t:
			for i, CViter in enumerate(CV_iters):
				logging.info('Cross Validation on iteration {}/{CV_iters}, {network}, {loss}'.format(i+1, CV_iters = len(CV_iters), network = args.network, loss = args.loss))
				
				solver = Solver(args, params, CViter)
				best_loss = 0
				if args.train:
					best_loss = solver.train()
					
				matrix = get_eval_multi(solver.test('test'), get_threshold(solver.test('val')))
				for key, value in matrix.items():
					eval_matrix[key].append(value)
					logging.info('{}: {}\n'.format(key, value))
					
				del matrix
				t.set_postfix(loss = args.loss, network = args.network, AUC = np.mean(eval_matrix['AUC']))
				t.update()
				
		jsn = json.dumps(eval_matrix)
		with open("./Result/eval_matrix_{network}_{loss}.json".format(network = args.network, loss = args.loss), "w") as f:
			f.write(jsn)

		stats = defaultdict(dict)
		for key, value in eval_matrix.items():
			eval_matrix[key] = np.array(eval_matrix[key]).reshape((1,-1))
			eval_matrix[key] = eval_matrix[key][~np.isnan(eval_matrix[key])]
			stats[key]['mean'] = np.mean(eval_matrix[key])
			stats[key]['SD'] = np.std(eval_matrix[key])
			
		jsn = json.dumps(stats)
		with open("./Result/stats_{network}_{loss}.json".format(network = args.network, loss = args.loss), "w") as f:
			f.write(jsn)
		
		del eval_matrix
	
	#create_table('./Result')


		
if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"]="0"
	main()
