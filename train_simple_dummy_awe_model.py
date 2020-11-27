#Core Python, Pandas, and kaldi_io
import numpy as np
import pandas as pd
import string
from collections import Counter
import kaldi_io

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
from models import SimpleNet
from train_test_helpers import accuracy,train_model,evaluate_model,evaluate_model_paper,test_model,plot_learning_curves
from ami_dummy_dataset import AMI_dataset

if __name__ == '__main__':

	print('Loading the Data')
	'''
	#load_list = ['/data/users/jmahapatra/data/feats_cmvn.ark']
	load_list = ['Data/feats_cmvn.ark']
	num_examples = np.Inf

	dh = DataHelper(load_list,num_examples)
	dh.load_data(char_threshold = 5, frequency_bounds = (0,np.Inf))
	dh.process_data()
	c,word_to_num,num_to_word = dh.generate_key_dicts()

	inputs,labels = dh.give_inputs_and_labels()
	del dh
	'''

	dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	print('Using Device ',dev.type)

	'''

	print('Splitting the data into train/val/test and creating dataloaders')
	x_trainval,x_test,y_trainval,y_test = train_test_split(inputs, labels, test_size=0.2, random_state=32)
	x_train,x_val,y_train,y_val = train_test_split(x_trainval,y_trainval,test_size =0.25, random_state = 32)

	x_train,y_train = torch.tensor(x_train,dtype= torch.float),torch.tensor(y_train, dtype= torch.float)
	x_val,y_val = torch.tensor(x_val, dtype= torch.float),torch.tensor(y_val, dtype= torch.float)
	x_test,y_test = torch.tensor(x_test, dtype= torch.float),torch.tensor(y_test, dtype= torch.float)
	print(x_train.shape,y_train.shape)
	print(x_val.shape,y_val.shape)
	print(x_test.shape,y_test.shape)

	'''

	bs = 64
	num_examples = np.Inf
	data_filepath = "/nethome/achingacham/apiai/data/AMI_Noisy/feats.scp"
	train_ds = AMI_dataset(num_examples = num_examples, split_set = "train", data_filepath = data_filepath, char_threshold = 5, frequency_bounds = (0,np.Inf))
	train_dl = DataLoader(train_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)

	val_ds = AMI_dataset(num_examples = num_examples, split_set = "val", data_filepath = data_filepath, char_threshold = 5, frequency_bounds = (0,np.Inf))
	val_dl = DataLoader(val_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)

	#test_ds = TensorDataset(x_test, y_test)
	#test_dl = DataLoader(test_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)

	print('Creating the Neural Net')

	num_output = len(train_ds.c.keys())
	net = SimpleNet(num_output)
	net = net.float()
	net.to(dev)

	#Load the model weights
	#clean_speech_model_weights = "/data/users/jmahapatra/models/awe_best_model.pth"
	#net.load_state_dict(torch.load(clean_speech_model_weights, map_location = torch.device('cpu') ))

	#Defining training criterion
	criterion = nn.NLLLoss()
	#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	num_epochs = 150
	#Training the model
	
	hist = train_model(net,num_epochs,train_dl,val_dl,optimizer,criterion,dev,save_path="/data/users/jmahapatra/models/dummy_noisy/",verbose = True)
	#hist = train_model(net,num_epochs,train_dl,val_dl,optimizer,criterion,dev,save_path="./Models/test/",verbose = True)
	
	plot_learning_curves(hist,'/data/users/jmahapatra/data/learning_curves.png', show = False)
	#plot_learning_curves(hist,'./Data/learning_curves.png', show = False)















