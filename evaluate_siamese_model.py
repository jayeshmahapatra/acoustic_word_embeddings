#Core Python, Pandas, and kaldi_io
import numpy as np
import pandas as pd
import string
from collections import Counter
import kaldi_io
import argparse
import sys

#Scikit
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances,average_precision_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.dummy import DummyClassifier
from scipy import stats
from scipy.spatial.distance import pdist

#Plotting
from matplotlib import pyplot as plt
import seaborn as sns



#Torch and utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader,random_split,ConcatDataset

#Import User defined classes
from data_helpers import DataHelper
from models import SimpleNet, SimpleNet_with_dropout, SiameseNet
from train_test_helpers import evaluate_siamese_model, test_siamese_model
from datasets import AMI_dataset, SiameseTriplets

if __name__ == '__main__':

	

	parser = argparse.ArgumentParser()
	parser.add_argument('-n','--noisy',help = "Noisy dataset", action = "store_true")
	parser.add_argument('-ne','--num_examples', type = int, default = 11000,  help = "Intger : Number of test examples to evaluate on")
	parser.add_argument('-snr', '--snr', type = int, default = 0, help = "SNR of the AMI Noisy data (required if noisy)")
	args = parser.parse_args()

	####Check Parser Arguments ############
	parser_invalid = False #Flag to exit if parser arguments invalid
	allowed_snr_values = [-5, 0, 5, 20] #Allowed values for snr
	if args.noisy:
		if args.snr not in allowed_snr_values:
			print("Only snr values allowed are -5, 0, 5 and 20")
			parser_invalid = True

	if parser_invalid:
		sys.exit()
	else:
		print("Parser Arguments Valid")

	#######End of Checking Parser Arguments################


	dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	bs = 64
	num_examples = np.Inf
	print('Loading the Data')

	#Set snr to infinity if clean
	if not args.noisy:
		snr = np.Inf
	else:
		snr = args.snr


	
	#test_ds = SiameseTriplets(split_set = "test", frequency_bounds = (0,np.Inf), snr = snr, cluster = True)
	test_ds = AMI_dataset(num_examples = num_examples, split_set = "test", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = snr, cluster = True)
	
	

	#Dataloaders
	test_dl = DataLoader(test_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)
		
	
	print('Creating the Neural Net')

	net = SiameseNet()
	net = net.float()
	net.to(dev)

	print('Loading best model')
	#Load the best model

	save_path = "/data/users/jmahapatra/models/"

	model_name = "siamese"

	if args.noisy:
		model_name += "_noisy_snr%d"%(args.snr)
	else:
		model_name += "_clean"
	

	model_name += ".pth"

	model_save_path = save_path + model_name

	print("Evaluating ", model_name)

	net.load_state_dict(torch.load(model_save_path))

	print("Test")
	test_loss = test_siamese_model(net, test_dl, dev)
	print("Test Loss", test_loss)

	average_precision = evaluate_model(net,test_dl,dev, num_examples = args.num_examples)
	print("average precision", average_precision)
	#avg_p_paper = evaluate_model_paper(net,evaluate_dl,dev, False)





