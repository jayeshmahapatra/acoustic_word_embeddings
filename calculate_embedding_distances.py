#Core Python, Pandas, and kaldi_io
import numpy as np
import pandas as pd

#Torch and utilities
import torch
import torch.nn as nn
import torch.nn.functional as F

#scikit
from sklearn.metrics import pairwise_distances,average_precision_score
from sklearn.metrics.pairwise import pairwise_kernels,paired_distances

#Import User defined classes
from data_helpers import DataHelper
from models import SimpleNet

def generate_word_embedding_dict(words,inputs,labels,word_to_num):
	word_embedding_dict = {}
	#Calculate embeddings
	for i,word in enumerate(words):

		#Find the mfcc features of the acoustic representation of the word in the data
		word_features = inputs[np.where(np.isin(labels,word_to_num[word]))]
		
		#Calculate embeddings for the feature
		word_embedding = net.give_embeddings(torch.tensor(word_features, device = dev, dtype=torch.float),dev)
		
		#If the number of representation is more than one, take the average embedding
		word_embedding_dict[word] = np.mean(word_embedding, axis = 0).reshape(1,-1)
	
	return word_embedding_dict

def calculate_embedding_distance(homophone_df,word_embedding_dict,metrics = ['cosine']):

	word1_embeddings = None
	word2_embeddings = None
	
	metric_distance_dict = {}
	for metric in metrics:
		metric_distance_dict[metric] = []
		
	for row in homophone_df.itertuples():
		word1, word2 = map(lambda x: x.strip(' \''),row.word_pairs.strip('()').split(','))
		
		for metric in metrics:
			metric_distance_dict[metric].append(paired_distances(word_embedding_dict[word1],word_embedding_dict[word2], metric = metric)[0])
		
		
		#if word1_embeddings is None and word2_embeddings is None:
		#    word1_embeddings = word_embedding_dict[word1]
		#    word2_embeddings = word_embedding_dict[word2]
		#else:
		#    word1_embeddings = np.vstack((word1_embeddings, word_embedding_dict[word1]))
		#    word2_embeddings = np.vstack((word2_embeddings, word_embedding_dict[word2]))
			
		

	#Calculate the distance
	#print(word1_embeddings.shape)
	for metric in metrics:
		#metric_distance = paired_distances(word1_embeddings,word2_embeddings, metric = metric)
		homophone_df.insert(len(homophone_df.columns),"%s_distance"%(metric), metric_distance_dict[metric], True)
	
	return homophone_df
	
	

if __name__ == '__main__':

	print('Loading the Data')

	load_list = ['/data/users/jmahapatra/data/feats_cmvn.ark']
	#load_list = ['./Data/feats_cmvn.ark']


	num_examples = np.Inf

	dh = DataHelper(load_list,num_examples)
	dh.load_data()
	dh.process_data()
	c,word_to_num,num_to_word = dh.generate_key_dicts()
	inputs,labels = dh.give_inputs_and_labels()
	del dh



	print('Loading the Model')
	#Load the Model
	#num_output = len(c.keys())
	dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	num_output = len(c.keys())
	net = SimpleNet(num_output)
	net = net.float()
	if dev.type == 'cuda':
		net.to(dev)

	#Load the best model
	#best_model_path = "./Models/awe_best_model.pth"
	best_model_path = "/data/users/jmahapatra/models/awe_best_model.pth"
	net.load_state_dict(torch.load(best_model_path))


	print('Loading the DataFrame')
	#Load the word_pairs DataFrame
	#wordpairs_df = pd.read_csv('./Data/wordpairs.txt')
	wordpairs_df = pd.read_csv('/data/users/jmahapatra/data/wordpairs.txt')

	
	#Calculate all the unique words
	words = set(wordpairs_df["word_1"].to_list()).union(set(wordpairs_df["word_2"].to_list()))

	word_embedding_dict = generate_word_embedding_dict(words,inputs,labels,word_to_num)

	np.save('/data/users/jmahapatra/data/word_embedding_dict.npy',word_embedding_dict)
	#np.save('./Data/test_embedding_dict.npy',word_embedding_dict)

	print('Calculated and saved the embeddings')
	
	#df = calculate_embedding_distance(wordpairs_df,word_embedding_dict,metrics = ['cosine', 'euclidean'])

	#print('Calculated Distances, saving the dataframe with distances')
	#df.to_csv('Data/wordpairs_with_distances.txt',sep = ',', index = False)


