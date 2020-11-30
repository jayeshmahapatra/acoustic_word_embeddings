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
from ami_clean_dataset import AMI_clean_dataset

if __name__ == '__main__':

	print('Loading the Data')
	#load_list = ['/data/users/jmahapatra/data/feats_cmvn.ark']
	'''
	load_list = ['Data/feats_cmvn.ark']
	num_examples = np.Inf

	dh = DataHelper(load_list,num_examples)
	dh.load_data(char_threshold = 5, frequency_bounds = (0,np.Inf))
	dh.process_data()
	c,word_to_num,num_to_word = dh.generate_key_dicts()

	inputs,labels = dh.give_inputs_and_labels()
	del dh

	print('Splitting the data into train/val/test and creating dataloaders')
	x_trainval,x_test,y_trainval,y_test = train_test_split(inputs, labels, test_size=0.2, random_state=32)
	x_train,x_val,y_train,y_val = train_test_split(x_trainval,y_trainval,test_size =0.25, random_state = 32)

	x_train,y_train = torch.tensor(x_train,dtype= torch.float),torch.tensor(y_train, dtype= torch.float)
	x_val,y_val = torch.tensor(x_val, dtype= torch.float),torch.tensor(y_val, dtype= torch.float)
	x_test,y_test = torch.tensor(x_test, dtype= torch.float),torch.tensor(y_test, dtype= torch.float)
	print(x_train.shape,y_train.shape)
	print(x_val.shape,y_val.shape)
	print(x_test.shape,y_test.shape)
	bs = 64
	num_examples = np.Inf
	test_ds = TensorDataset(x_test, y_test)
	test_dl = DataLoader(test_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)
	'''

	dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	bs = 64
	num_examples = np.Inf
	test_ds = AMI_clean_dataset(num_examples = num_examples, split_set = "test", data_filepath = "Data/feats_cmvn.ark", char_threshold = 5, frequency_bounds = (0,np.Inf))
	test_dl = DataLoader(test_ds, batch_size=bs, pin_memory = True, shuffle = True, drop_last = True)



	print('Creating the Neural Net')

	#num_output = 9974
	num_output = len(test_ds.c.keys())
	#num_output = len(c.keys())
	net = SimpleNet(num_output)
	net = net.float()
	net.to(dev)

	print('Loading best model')
	#Load the best model
	best_model_path = "./Models/awe_best_model.pth"

	net.load_state_dict(torch.load(best_model_path))
	evaluate_dl = DataLoader(test_ds, batch_size=1024, pin_memory = True, drop_last = False)
	test_acc = test_model(net,test_dl,dev)
	print(test_acc)
	#average_precision = evaluate_model(net,evaluate_dl,dev, num_examples = 11000)
	#avg_p_paper = evaluate_model_paper(net,evaluate_dl,dev, False)



