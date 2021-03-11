#Train and Evaluate AWE models at all noise levels
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
from train_test_helpers import accuracy,train_loop,evaluate_model,evaluate_model_paper,test_model,plot_learning_curves
from datasets import CNN_dataset, SiameseTriplets

def train_model(run, k, train_dl, val_dl, snr, dropout_probability):

	dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	print('Creating the Neural Net')

	num_output = len(train_ds.c.keys())


	if dropout_probability > 0:
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

	noisy = True if snr < np.Inf else False
	dropout = True if dropout_probability > 0 else False

	if noisy:
		model_name += "_noisy_snr%d"%(snr)
	else:
		model_name += "_clean"
	
	if dropout:
		model_name += "_dropout_%d"%(int(dropout_probability*100))

	#Add run number
	model_name += "_top%d_words_upper_bound"%(k)

	model_name += "_run_%d"%(run)

	model_name += ".pth"

	model_save_path = save_path + model_name

	print("Training ",model_name)


	hist = train_loop(net,num_epochs,train_dl,val_dl,optimizer,criterion,dev,save_path=model_save_path,verbose = True)
	#hist = train_model(net,num_epochs,train_dl,val_dl,optimizer,criterion,dev,save_path="./Models/test/",verbose = True)
	
	lc_save_path = "/data/users/jmahapatra/data/learning_curves/"

	lc_name = "learning_curves"

	if noisy:
		lc_name += "_noisy_snr%d"%(snr)
	else:
		lc_name += "_clean"
	if dropout:
		lc_name += "_dropout_%d"%(int(dropout_probability*100))

	#Add run number
	lc_name += "_top%d_words_upper_bound"%(k)

	lc_name += "_run_%d"%(run)

	lc_name += ".png"

	lc_save_path += lc_name


	plot_learning_curves(hist,lc_save_path, show = False)

def test_and_evaluate_model(run, k, test_dl, snr, dropout_probability):

	noisy = True if snr < np.Inf else False
	dropout = True if dropout_probability > 0 else False

	dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	print('Creating the Neural Net')


	num_output = len(test_ds.c.keys())


	if dropout:
		net = SimpleNet_with_dropout(num_output, p = dropout_probability)
	else:
		net = SimpleNet(num_output)

	net = net.float()
	net.to(dev)
	net.eval()

	print('Loading best model')
	#Load the best model

	save_path = "/data/users/jmahapatra/models/"


	model_name = "cnn"

	if noisy:
		model_name += "_noisy_snr%d"%(snr)
	else:
		model_name += "_clean"
	if dropout:
		model_name += "_dropout_%d"%(int(dropout_probability*100))

	#Add run number
	model_name += "_top%d_words_upper_bound"%(k)

	model_name += "_run_%d"%(run)	

	precision_recall_curve_path = save_path + model_name+".png"

	model_name += ".pth"

	model_save_path = save_path + model_name

	print("Evaluating ", model_name)

	net.load_state_dict(torch.load(model_save_path))
	test_acc  = test_model(net,test_dl,dev)
	print("test acc %f "%(test_acc))


	average_precision = evaluate_model(net,test_dl,dev, np.Inf)
	print("average precision", average_precision)

	return test_acc, average_precision

if __name__ == '__main__':

	#Snr values of different datasets
	snr_values = [np.Inf, 20, 5, 0, -5]


	#Dropout values
	dropout_values = [0, 0.2, 0.5]

	#k values
	k_values = [10, 100, 250, 500, 1000, 2500, 5000]

	num_runs = 2

	evaluation_dict = {}
	evaluation_dict["Dataset"] = []
	evaluation_dict["Dropout"] = []
	evaluation_dict["K"] = []

	for run in range(num_runs):
		evaluation_dict["Test Accuracy Run %d"%(run)] = []
		evaluation_dict["Same-Different Task Run %d"%(run)] = []

	bs = 64 #Batch Size

	cluster = True


	for snr in snr_values:

		
		for k in k_values:
		

			#Load the Data
			train_ds = CNN_dataset(split_set = "train", char_threshold = 5, frequency_bounds = (0,30), snr = snr, k = k, cluster = cluster)
			val_ds = CNN_dataset(split_set = "val", char_threshold = 5, frequency_bounds = (0,30), snr = snr, k = k, cluster = cluster)
			test_ds = CNN_dataset(split_set = "test", char_threshold = 5, frequency_bounds = (0,30), snr = snr, k = k, cluster = cluster)

			#DataLoaders
			train_dl = DataLoader(train_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)
			val_dl = DataLoader(val_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)
			test_dl = DataLoader(test_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)


			for dropout_probability in dropout_values:
				
				run_test_accuracies = []
				run_avg_precisions = []

				for run in range(num_runs):

					#Train the model
					train_model(run, k, train_dl, val_dl, snr, dropout_probability)

					#Evaluate the model
					test_acc,avg_p = test_and_evaluate_model(run, k, test_dl, snr, dropout_probability)

					run_test_accuracies.append(test_acc)
					run_avg_precisions.append(avg_p)


				dataset = "Clean" if snr == np.Inf else "Noisy SNR %d"%(snr)
				evaluation_dict["Dataset"].append(dataset)
				evaluation_dict["Dropout"].append(dropout_probability)
				evaluation_dict["K"].append(k)

				for run in range(num_runs):
					evaluation_dict["Test Accuracy Run %d"%(run)].append(run_test_accuracies[run])
					evaluation_dict["Same-Different Task Run %d"%(run)].append(run_avg_precisions[run])

	#Save the evaluation Dict as a csv
	evaluation_df = pd.DataFrame(evaluation_dict)
	evaluation_df_filepath = "/data/users/jmahapatra/data/cnn_top_k_evaluation.csv"
	evaluation_df.to_csv(evaluation_df_filepath, index = False)








