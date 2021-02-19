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
from models import SimpleNet, SimpleNet_with_dropout, SiameseNet
from train_test_helpers import evaluate_model, test_siamese_model, siamese_train_loop, plot_learning_curves

#datsets
from datasets import CNN_dataset, SiameseTriplets, Siamese_top_k, CNN_top_k
from datasets import BalancedBatchSampler,TopK_WordsSampler



#Train and Evaluate Siamese models for top k words

def train_siamese_model(k, train_dl, val_dl, snr):

	dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	print('Creating the Neural Net')

	print('Creating the Neural Net')

	net = SiameseNet()
	net = net.float()
	net.to(dev)




	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	#optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	num_epochs = 150
	#Training the model
	
	save_path = "/data/users/jmahapatra/models/"

	model_name = "siamese"

	noisy = True if snr < np.Inf else False
	

	if noisy:
		model_name += "_noisy_snr%d"%(snr)
	else:
		model_name += "_clean"
	

	#Add run number
	model_name += "_top%d_words"%(k)

	model_name += ".pth"

	model_save_path = save_path + model_name

	print("Training ",model_name)


	hist = siamese_train_loop(net,num_epochs,train_dl,val_dl,optimizer,dev,save_path=model_save_path,verbose = True)
	#hist = train_model(net,num_epochs,train_dl,val_dl,optimizer,criterion,dev,save_path="./Models/test/",verbose = True)
	
	lc_save_path = "/data/users/jmahapatra/data/learning_curves/"

	lc_name = "learning_curves_siamese"

	if noisy:
		lc_name += "_noisy_snr%d"%(snr)
	else:
		lc_name += "_clean"
	

	#Add run number
	model_name += "_top%d_words"%(k)

	lc_name += ".png"

	lc_save_path += lc_name


	plot_learning_curves(hist,lc_save_path, show = False)

def test_and_evaluate_siamese_model(k, test_dl, evaluate_dl, snr):

	noisy = True if snr < np.Inf else False
	

	dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	print('Creating the Neural Net')

	net = SiameseNet()
	net = net.float()
	net.to(dev)


	print('Loading best model')
	#Load the best model

	save_path = "/data/users/jmahapatra/models/"

	model_name = "siamese"

	noisy = True if snr < np.Inf else False
	

	if noisy:
		model_name += "_noisy_snr%d"%(snr)
	else:
		model_name += "_clean"
	

	#Add run number
	model_name += "_top%d_words"%(k)

	precision_recall_curve_path = save_path + model_name+".png"

	model_name += ".pth"

	model_save_path = save_path + model_name

	print("Evaluating ", model_name)

	net.load_state_dict(torch.load(model_save_path))
	print("Test")
	test_loss = test_siamese_model(net, test_dl, dev)
	print("Test Loss", test_loss)


	average_precision = evaluate_model(net,evaluate_dl,dev, num_examples = 11000)
	print("average precision", average_precision)

	return test_loss, average_precision

if __name__ == '__main__':

	#Snr values of different datasets
	snr_values = [np.Inf, 20, 5, 0, -5]
	#snr_values = [5]

	k_values = [100,500,1000,5000]





	cluster = True


	evaluation_dict = {}
	evaluation_dict["Dataset"] = []
	evaluation_dict["Test Loss"] = []
	evaluation_dict["Same-Different Task"] = []
	evaluation_dict["K"] = []

	bs = 64 #Batch Size


	for snr in snr_values:

	
		model_loss = []
		model_avg_p = []
		for k in k_values:

			print("Top %d words"%(k))

			#Load the Data
			train_ds = Siamese_top_k(k = k, split_set = "train", frequency_bounds = (0,np.Inf), snr = snr, cluster = cluster)
			val_ds = Siamese_top_k(k = k, split_set = "val", frequency_bounds = (0,np.Inf), snr = snr, cluster = cluster)
			test_ds = Siamese_top_k(k = k, split_set = "test", frequency_bounds = (0,np.Inf), snr = snr, cluster = cluster)
			evaluate_ds = CNN_top_k(k = k, split_set = "test", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = snr, cluster = cluster)

			#DataLoaders
			train_dl = DataLoader(train_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)
			val_dl = DataLoader(val_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)
			test_dl = DataLoader(test_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)
			evaluate_dl = DataLoader(evaluate_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)
			


			#Train the model
			train_siamese_model(k, train_dl, val_dl, snr)

			#Evaluate the model
			test_loss,avg_p = test_and_evaluate_siamese_model(k, test_dl, evaluate_dl, snr)

			model_loss.append(test_loss)
			model_avg_p.append(avg_p)

		#Update the evaluation dict with average values
		for loss,avg_p in zip(model_loss,model_avg_p):
			dataset = "Clean" if snr == np.Inf else "Noisy SNR %d"%(snr)
			evaluation_dict["Dataset"].append(dataset)
			evaluation_dict["K"].append(k)
			evaluation_dict["Test Loss"].append(loss)
			evaluation_dict["Same-Different Task"].append(avg_p)

	#Save the evaluation Dict as a csv
	evaluation_df = pd.DataFrame(evaluation_dict)
	evaluation_df_filepath = "/data/users/jmahapatra/data/siamese_top_k_evaluation.csv"
	evaluation_df.to_csv(evaluation_df_filepath, index = False)








