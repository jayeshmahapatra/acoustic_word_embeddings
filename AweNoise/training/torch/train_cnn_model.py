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
from models import SimpleNet, SimpleNet_with_dropout
from train_test_helpers import plot_learning_curves,train_loop
from datasets import CNN_dataset, SiameseTriplets

if __name__ == '__main__':


	parser = argparse.ArgumentParser()
	parser.add_argument('-n','--noisy',help = "Noisy dataset", action = "store_true")
	parser.add_argument('-d','--dropout', help = "Dropout", action = "store_true")
	parser.add_argument('-p','--probability', type = float, help = "Float : Dropout probability")
	parser.add_argument('-ne','--num_examples', type = int, default = 11000,  help = "Intger : Number of test examples to evaluate on")
	parser.add_argument('-snr', '--snr', type = int, default = 0, help = "SNR of the AMI Noisy data (required if noisy)")
	args = parser.parse_args()


	####Check Parser Arguments ############
	parser_invalid = False #Flag to exit if parser arguments invalid
	allowed_snr_values = [-5, 0, 5, 20] #Allowed values for snr
	if args.dropout:
		if not args.probability:
			print("Specify probability of dropout using -p in command line")
			parser_invalid = True
		else:
			dropout_probability = args.probability
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
	
	
	train_ds = CNN_dataset(split_set = "train", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = snr, k = np.Inf, cluster = True)
	val_ds = CNN_dataset(split_set = "val", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = snr, k = np.Inf, cluster = True)
	

	#DataLoaders
	train_dl = DataLoader(train_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)
	val_dl = DataLoader(val_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)

	

	dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	print('Using Device ',dev.type)

	print('Creating the Neural Net')

	num_output = len(train_ds.c.keys())


	if args.dropout:
		net = SimpleNet_with_dropout(num_output, p = dropout_probability)
	else:
		net = SimpleNet(num_output)

	net = net.float()
	net.to(dev)

	#Defining training criterion
	criterion = nn.NLLLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	#optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	num_epochs = 150
	#Training the model
	
	save_path = "/data/users/jmahapatra/models/"

	model_name = "cnn"

	if args.noisy:
		model_name += "_noisy_snr%d"%(args.snr)
	else:
		model_name += "_clean"
	if args.dropout:
		model_name += "_dropout_%d"%(int(args.probability*100))

	model_name += ".pth"

	model_save_path = save_path + model_name

	print("Training ",model_name)


	hist = train_loop(net,num_epochs,train_dl,val_dl,optimizer,criterion,dev,save_path=model_save_path,verbose = True)
	#hist = train_model(net,num_epochs,train_dl,val_dl,optimizer,criterion,dev,save_path="./Models/test/",verbose = True)
	
	lc_save_path = "/data/users/jmahapatra/data/learning_curves/"

	lc_name = "learning_curves"
	if args.noisy:
		lc_name += "_noisy_snr%d"%(args.snr)
	else:
		lc_name += "_clean"
	if args.dropout:
		lc_name += "_dropout_%d"%(int(args.probability*100))
	lc_name += ".png"

	lc_save_path += lc_name


	plot_learning_curves(hist,lc_save_path, show = False)
	#plot_learning_curves(hist,'./Data/learning_curves.png', show = False)















