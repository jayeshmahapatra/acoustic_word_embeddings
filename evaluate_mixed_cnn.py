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
from train_test_helpers import evaluate_model,test_model, baseline, clean_mapping
from datasets import CNN_dataset, SiameseTriplets

if __name__ == '__main__':

	

	parser = argparse.ArgumentParser()
	parser.add_argument('-n','--noisy',help = "Noisy dataset", action = "store_true")
	parser.add_argument('-d','--dropout', help = "Dropout", action = "store_true")
	parser.add_argument('-bl', '--baseline', help = "Baseline Test Acc", action = "store_true")
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


	#Load Clean and Noisy
	clean_test_ds = CNN_dataset(split_set = "test", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = np.Inf, k = np.Inf, cluster = True)
	noisy_test_ds = CNN_dataset(split_set = "test", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = snr, k = np.Inf, cluster = True)
	clean_num_to_word, clean_word_to_num = clean_test_ds.num_to_word.copy(),clean_test_ds.word_to_num.copy()
	del clean_test_ds
	noisy_num_to_word, noisy_word_to_num = noisy_test_ds.num_to_word.copy(),noisy_test_ds.word_to_num.copy()



	#Transform Labels of Noisy
	noisy_test_ds.labels = clean_mapping(noisy_test_ds.labels, clean_word_to_num, noisy_num_to_word)

	
	test_ds = noisy_test_ds
		
	
	if args.baseline:
			train_ds = CNN_dataset(split_set = "train", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = snr, k= np.Inf, cluster = True)
	

	#Dataloaders
	test_dl = DataLoader(test_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)
		
	
	print('Creating the Neural Net')

	#num_output = 9974
	num_output = len(clean_word_to_num.keys())

	if args.dropout:
		net = SimpleNet_with_dropout(num_output, p = dropout_probability)
	else:
		net = SimpleNet(num_output)


	net = net.float()
	net.to(dev)
	net.eval()

	print('Loading best model')
	#Load the best model

	save_path = "/data/users/jmahapatra/models/"

	model_name = "cnn_mixed"

	if args.noisy:
		model_name += "_noisy_snr%d"%(args.snr)
	else:
		model_name += "_clean"
	if args.dropout:
		model_name += "_dropout_%d"%(int(args.probability*100))

	model_name += ".pth"

	model_save_path = save_path + model_name

	print("Evaluating ", model_name)

	net.load_state_dict(torch.load(model_save_path))
	test_acc  = test_model(net,test_dl,dev)
	print("test acc %f "%(test_acc))

	#Baseline
	if args.baseline:
		baseline_acc = baseline(train_ds,test_ds)
		print("Baseline acc %f "%(baseline_acc))


	average_precision = evaluate_model(net,test_dl,dev, num_examples = args.num_examples)
	print("average precision", average_precision)

	#k values
	k_values = [10, 100, 500, 1000, 2000,3000, 4000, 5000]

	evaluate_on_k_values = False

	if evaluate_on_k_values:
		for k in k_values:

			#Get the top k words
			inputs,labels = test_ds.give_top_k_words(k)

			#Create a dataset and dataloader with instances belonging to the top k words
			k_ds = TensorDataset(inputs, labels)
			k_dl = DataLoader(k_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)


			print("Evaluating at top %d words"%(k))

			k_avg_precision = evaluate_model(net,k_dl,dev, num_examples = args.num_examples)

			print("average precision \n", k_avg_precision)


	#avg_p_paper = evaluate_model_paper(net,evaluate_dl,dev, False)





