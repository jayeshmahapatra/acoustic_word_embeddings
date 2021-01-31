#Core Python, Pandas, and kaldi_io
import numpy as np
import pandas as pd
import random
import string
from collections import Counter
import random
from random import choice
import kaldi_io


#Torch and utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader,random_split,ConcatDataset
#scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class AMI_dataset(torch.utils.data.Dataset):
	'''Dataset that on each iteration provides a triplet pair of acosutic segments, out of which
	two belong to the same word, and one belongs to a different word.'''
	
	def __init__(self, split_set = "train", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = np.Inf, cluster = True):
	
		self.char_threshold = char_threshold
		self.frequency_bounds = frequency_bounds
		self.split_set = split_set
		self.snr = snr
		self.cluster = cluster
		

		#Data Structures to store data
		self.keys = []
		self.matrices = []
		self.mat_lengths = []
		self.c = None
		self.word_to_num = None
		self.num_to_word = None
		self.inputs = None
		self.labels = None
		
		

		#Load and Process the data
		self._load_data()
		self._pad_and_truncate_data()
		self._generate_key_dicts()
		self._generate_inputs_and_labels()


		#Shuffle the array
		self.inputs,self.labels = shuffle(self.inputs,self.labels, random_state = 3)

		x_trainval,x_test,y_trainval,y_test = train_test_split(self.inputs, self.labels, test_size=0.2, random_state=32)
		x_train,x_val,y_train,y_val = train_test_split(x_trainval,y_trainval,test_size =0.25, random_state = 32)

		x_train,y_train = torch.tensor(x_train,dtype= torch.float),torch.tensor(y_train, dtype= torch.float)
		x_val,y_val = torch.tensor(x_val, dtype= torch.float),torch.tensor(y_val, dtype= torch.float)
		x_test,y_test = torch.tensor(x_test, dtype= torch.float),torch.tensor(y_test, dtype= torch.float)


		print(x_train.shape,x_val.shape,x_test.shape)
		
		#Split the dataset
		if self.split_set == "train":
			self.inputs,self.labels = x_train,y_train
		elif self.split_set == "val":
			self.inputs,self.labels = x_val,y_val
		else:
			self.inputs,self.labels = x_test,y_test


		
	def __getitem__(self, index):
		
		return self.inputs[index],self.labels[index]
		
	def __len__(self):
		
		return self.inputs.shape[0]
		
	
	
	
	
	################################## Helper Functions #####################################################################


	def _load_data(self):
		'''Loads the data from the file into the data object'''
		
		#filetype = self.load_list[0].split(".")[-1] 
		#read_function = kaldi_io.read_mat_ark if filetype == "ark" else kaldi_io.read_mat_scp


		#Create the keyword to word dict
		if self.cluster:

			if self.snr == np.Inf:
				#Clean
				data_directory = "/nethome/achingacham/apiai/data/AMI_Clean/"
			else:
				data_directory = "/nethome/achingacham/apiai/data/AMI_White_SNR%d_v2/"%(self.snr)
			
			keyword_path = data_directory + "text"
			data_load_list = [data_directory+"feats.scp"]
			#data_load_list = [data_directory+"feats_cmvn.ark"]

		else:
			if self.snr == 0:
				keyword_path = "./Data/Noisy/text"
				data_load_list = ["./Data/Noisy/feats.scp"]


		keywords_df = pd.read_csv(keyword_path, sep = " ", header = None)
		keywords_df.columns = ["keyword", "key"]
		keyword_to_key = {}

		#clean_speech_keys_list = set(list(self.word_to_num.keys()))

		for row in keywords_df.itertuples():
			keyword_to_key[row.keyword] = row.key
		

		for load_file in data_load_list:
			file_keys,file_matrices,file_mat_lengths = [],[],[]
			for i,(keyword,matrix) in enumerate(kaldi_io.read_mat_scp(load_file)):
				file_keys.append(keyword_to_key[keyword])
				file_matrices.append(matrix)
				file_mat_lengths.append(matrix.shape[0])
	
			#Filter the data
			file_keys,file_matrices = self._filter_on_character_length(file_keys,file_matrices ,char_threshold = self.char_threshold)


			#Add to the main list
			self.keys.extend(file_keys)
			self.matrices.extend(file_matrices)
			self.mat_lengths.extend(file_mat_lengths)

		self.keys,self.matrices = self._filter_on_frequency_bounds(self.keys,self.matrices,frequency_bounds = self.frequency_bounds)



		print('Finished Loading the Data, %d examples'%(len(self.keys)))

	def _pad_and_truncate_data(self):
		'''Processes the loaded data'''

		#Truncate the dimensions of the data
		self.matrices = self._truncate_shapes(self.matrices,max_length=200,num_mfcc_features=40)
		#Pad the matrices
		self.matrices,self.mat_lengths = self._pad_sequences(self.matrices,n_padded = 100,center_padded = True)
		self.matrices = np.transpose(self.matrices,(0,2,1))

		#delete mat_lengths
		del self.mat_lengths
		
	def _generate_inputs_and_labels(self):
		'''Uses the stored matrices and keys to generate numpy array of inputs and labels'''

		label_list = []
		for key in self.keys:
			label_list.append(self.word_to_num[key])

		inputs = np.stack(self.matrices)
		labels = np.array(label_list)
		
		self.inputs = inputs
		self.labels = labels
		
		#Delete other datastructures to free memory
		#del self.keys,self.matrices
		
		return None

			
	
	# Function to truncate and limit dimensionality
	def _truncate_shapes(self,matrices,max_length = 100,num_mfcc_features = 40):

		mat_lengths = []
		for i, seq in enumerate(matrices):
			matrices[i] = matrices[i][:max_length, :num_mfcc_features]

		return matrices

	#Function for padding
	def _pad_sequences(self,x, n_padded, center_padded=True):
		"""Return the padded sequences and their original lengths."""
		padded_x = np.zeros((len(x), n_padded, x[0].shape[1]))
		lengths = []
		for i_data, cur_x in enumerate(x):
			length = cur_x.shape[0]
			if center_padded:
				padding = int(np.round((n_padded - length) / 2.))
				if length <= n_padded:
					padded_x[i_data, padding:padding + length, :] = cur_x
				else:
					# Cut out snippet from sequence exceeding n_padded
					padded_x[i_data, :, :] = cur_x[-padding:-padding + n_padded]
				lengths.append(min(length, n_padded))
			else:
				length = min(length, n_padded)
				padded_x[i_data, :length, :] = cur_x[:length, :]
				lengths.append(length)
		return padded_x, lengths
	
	def _filter_on_character_length(self,keys,matrices, char_threshold = 5):
		'''Takes in matrices and keys. Filters the data by making all keys lowercase, removing words
		with number of letters less than a threshold.'''

		print('Length before filtering on char length %d'%(len(keys)))
		#Lowercase all keys
		keys = list(map(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower(),keys))

		#Filter if the characters are smaller than the character threshold
		keys,matrices = zip(*filter(lambda x: len(x[0])>=char_threshold, zip(keys,matrices)))

		keys,matrices = list(keys),list(matrices)

		print('Length after filtering on char length %d'%(len(keys)))


		return keys,matrices

	def _filter_on_frequency_bounds(self,keys,matrices,frequency_bounds = (0,np.Inf)):
		'''Filter words that have frequnecy less than a lower bound threshold or more than an upper bound threshold'''

		print('Length before filtering on frequency_bounds %d'%(len(keys)))

		
		
		lower_bound,upper_bound = frequency_bounds[0],frequency_bounds[1]

		#If bounds are at thresholds, return as it is
		if lower_bound == 0 and upper_bound == np.Inf:
			return keys,matrices
		
		#Create a Counter
		c = Counter(keys)
		
		data_class = {}

		for key in c.keys():
			ids = [index for index, element in enumerate(keys) if element == key]
			data_class[key] = [matrices[index] for index in ids]
		

		#Get the words whose frequency is below a lower bound threshold
		remove_list = []

		for key,value in c.items():
			if value < lower_bound:
				remove_list.append(key)

		#Remove the words from the Counter
		for word in remove_list:
			del data_class[word]
			
		keys,matrices = [],[]
			
		#Limit the frequency of words according to an upper limit
		for word,word_matrices in data_class.items():
			num_examples = min(len(word_matrices),upper_bound)
			keys.extend([word] * num_examples)
			matrices.extend(word_matrices[:num_examples])



		print('Length after filtering on frequency_bounds %d'%(len(keys)))

		return keys,matrices
	
	def _generate_key_dicts(self):
		'''Arguments:
		keys : A list of words corresponding to the mfcc feature matrices
		-------------
		Returns:
		labels : A list of numbers correspoding to the words in the list keys'''
		self.c = Counter(self.keys)
		#print(c)
		num_words = len(self.c.keys())
		self.word_to_num = {}
		self.num_to_word = {}

		index = 0
		for key in self.c.keys():
			self.word_to_num[key] = index
			self.num_to_word[index] = key
			index+=1




		print('Number of Unique words ',len(self.c.keys()))
		return None
	################################## End of Helper Functions #####################################################################



class SiameseTriplets(torch.utils.data.Dataset):
	'''Dataset that on each iteration provides a triplet pair of acosutic segments, out of which
	two belong to the same word, and one belongs to a different word.'''
	
	def __init__(self, split_set = "train", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = np.Inf, cluster = True):
	
		self.char_threshold = char_threshold
		self.frequency_bounds = frequency_bounds
		self.split_set = split_set
		self.snr = snr
		self.cluster = cluster
		

		#Data Structures to store data
		self.keys = []
		self.matrices = []
		self.mat_lengths = []
		self.c = None
		self.word_to_num = None
		self.num_to_word = None
		self.inputs = None
		self.labels = None
		
		self.examples_per_class = 20

		self.triplets = []
		self.triplets_labels = []


		#Load and Process the data
		self._load_data()
		self._pad_and_truncate_data()
		self._generate_key_dicts()
		self._generate_inputs_and_labels()

		#Siamese specific processing
		self._split_dataset()
		self._generate_data_class()
		self._create_triplets()

		
		
		
		print("Triplet Shape")
		print(self.triplets.shape, self.triplets_labels.shape)
		
	def __getitem__(self, index):
		
		return self.triplets[index],self.triplets_labels[index]
		
	def __len__(self):
		
		return self.triplets.shape[0]
		
	
	
	
	
	################################## Helper Functions #####################################################################
	def _load_data(self):
		'''Loads the data from the file into the data object'''
		
		#filetype = self.load_list[0].split(".")[-1] 
		#read_function = kaldi_io.read_mat_ark if filetype == "ark" else kaldi_io.read_mat_scp


		#Create the keyword to word dict
		if self.cluster:

			if self.snr == np.Inf:
				#Clean
				data_directory = "/nethome/achingacham/apiai/data/AMI_Clean/"
			else:
				data_directory = "/nethome/achingacham/apiai/data/AMI_White_SNR%d_v2/"%(self.snr)
			
			keyword_path = data_directory + "text"
			data_load_list = [data_directory+"feats.scp"]

		else:
			if self.snr == 0:
				keyword_path = "./Data/Noisy/text"
				data_load_list = ["./Data/Noisy/feats.scp"]


		keywords_df = pd.read_csv(keyword_path, sep = " ", header = None)
		keywords_df.columns = ["keyword", "key"]
		keyword_to_key = {}

		#clean_speech_keys_list = set(list(self.word_to_num.keys()))

		for row in keywords_df.itertuples():
			keyword_to_key[row.keyword] = row.key
		

		for load_file in data_load_list:
			file_keys,file_matrices,file_mat_lengths = [],[],[]
			for i,(keyword,matrix) in enumerate(kaldi_io.read_mat_scp(load_file)):
				file_keys.append(keyword_to_key[keyword])
				file_matrices.append(matrix)
				file_mat_lengths.append(matrix.shape[0])
	
			#Filter the data
			file_keys,file_matrices = self._filter_on_character_length(file_keys,file_matrices ,char_threshold = self.char_threshold)


			#Add to the main list
			self.keys.extend(file_keys)
			self.matrices.extend(file_matrices)
			self.mat_lengths.extend(file_mat_lengths)

		self.keys,self.matrices = self._filter_on_frequency_bounds(self.keys,self.matrices,frequency_bounds = self.frequency_bounds)



		print('Finished Loading the Data, %d examples'%(len(self.keys)))

	def _pad_and_truncate_data(self):
		'''Processes the loaded data'''

		#Truncate the dimensions of the data
		self.matrices = self._truncate_shapes(self.matrices,max_length=200,num_mfcc_features=40)
		#Pad the matrices
		self.matrices,self.mat_lengths = self._pad_sequences(self.matrices,n_padded = 100,center_padded = True)
		self.matrices = np.transpose(self.matrices,(0,2,1))

		#delete mat_lengths
		del self.mat_lengths
		
	def _generate_inputs_and_labels(self):
		'''Uses the stored matrices and keys to generate numpy array of inputs and labels'''

		label_list = []
		for key in self.keys:
			label_list.append(self.word_to_num[key])

		inputs = np.stack(self.matrices)
		labels = np.array(label_list)
		
		self.inputs = inputs
		self.labels = labels
		
		#Delete other datastructures to free memory
		del self.keys,self.matrices
		
		return None
	
	def _generate_data_class(self):
		
		self.data_class = {}
		
		for key in self.c.keys():
			ids = np.where(np.isin(self.labels,self.word_to_num[key]))
			if ids[0].shape[0] > 0:
				self.data_class[self.word_to_num[key]] = torch.tensor(self.inputs[ids], dtype = torch.float).cpu()

		del self.inputs,self.labels


			
	
	# Function to truncate and limit dimensionality
	def _truncate_shapes(self,matrices,max_length = 100,num_mfcc_features = 40):

		
		for i, seq in enumerate(matrices):
			matrices[i] = matrices[i][:max_length, :num_mfcc_features]
			

		return matrices

	#Function for padding
	def _pad_sequences(self,x, n_padded, center_padded=True):
		"""Return the padded sequences and their original lengths."""
		padded_x = np.zeros((len(x), n_padded, x[0].shape[1]))
		lengths = []
		for i_data, cur_x in enumerate(x):
			length = cur_x.shape[0]
			if center_padded:
				padding = int(np.round((n_padded - length) / 2.))
				if length <= n_padded:
					padded_x[i_data, padding:padding + length, :] = cur_x
				else:
					# Cut out snippet from sequence exceeding n_padded
					padded_x[i_data, :, :] = cur_x[-padding:-padding + n_padded]
				lengths.append(min(length, n_padded))
			else:
				length = min(length, n_padded)
				padded_x[i_data, :length, :] = cur_x[:length, :]
				lengths.append(length)
		return padded_x, lengths
	
	def _filter_on_character_length(self,keys,matrices, char_threshold = 5):
		'''Takes in matrices and keys. Filters the data by making all keys lowercase, removing words
		with number of letters less than a threshold.'''

		print('Length before filtering on char length %d'%(len(keys)))
		#Lowercase all keys
		keys = list(map(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower(),keys))

		#Filter if the characters are smaller than the character threshold
		keys,matrices = zip(*filter(lambda x: len(x[0])>=char_threshold, zip(keys,matrices)))

		keys,matrices = list(keys),list(matrices)

		print('Length after filtering on char length %d'%(len(keys)))


		return keys,matrices

	def _filter_on_frequency_bounds(self,keys,matrices,frequency_bounds = (0,np.Inf)):
		'''Filter words that have frequnecy less than a lower bound threshold or more than an upper bound threshold'''

		print('Length before filtering on frequency_bounds %d'%(len(keys)))

		
		
		lower_bound,upper_bound = frequency_bounds[0],frequency_bounds[1]

		#If bounds are at thresholds, return as it is
		if lower_bound == 0 and upper_bound == np.Inf:
			return keys,matrices
		
		#Create a Counter
		c = Counter(keys)
		
		data_class = {}

		for key in c.keys():
			ids = [index for index, element in enumerate(keys) if element == key]
			data_class[key] = [matrices[index] for index in ids]
		

		#Get the words whose frequency is below a lower bound threshold
		remove_list = []

		for key,value in c.items():
			if value < lower_bound:
				remove_list.append(key)

		#Remove the words from the Counter
		for word in remove_list:
			del data_class[word]
			
		keys,matrices = [],[]
			
		#Limit the frequency of words according to an upper limit
		for word,word_matrices in data_class.items():
			num_examples = min(len(word_matrices),upper_bound)
			keys.extend([word] * num_examples)
			matrices.extend(word_matrices[:num_examples])



		print('Length after filtering on frequency_bounds %d'%(len(keys)))

		return keys,matrices
	
	def _generate_key_dicts(self):
		'''Arguments:
		keys : A list of words corresponding to the mfcc feature matrices
		-------------
		Returns:
		labels : A list of numbers correspoding to the words in the list keys'''
		self.c = Counter(self.keys)
		#print(c)
		num_words = len(self.c.keys())
		self.word_to_num = {}
		self.num_to_word = {}

		index = 0
		for key in self.c.keys():
			self.word_to_num[key] = index
			self.num_to_word[index] = key
			index+=1




		print('Number of Unique words ',len(self.c.keys()))
		return None

	def _split_dataset(self):

		#Shuffle the array
		self.inputs,self.labels = shuffle(self.inputs,self.labels, random_state = 3)

		x_trainval,x_test,y_trainval,y_test = train_test_split(self.inputs, self.labels, test_size=0.2, random_state=32)
		x_train,x_val,y_train,y_val = train_test_split(x_trainval,y_trainval,test_size =0.25, random_state = 32)

		x_train,y_train = torch.tensor(x_train,dtype= torch.float),torch.tensor(y_train, dtype= torch.float)
		x_val,y_val = torch.tensor(x_val, dtype= torch.float),torch.tensor(y_val, dtype= torch.float)
		x_test,y_test = torch.tensor(x_test, dtype= torch.float),torch.tensor(y_test, dtype= torch.float)

		print(x_train.shape,x_val.shape,x_test.shape)
		
		#Split the dataset
		if self.split_set == "train":
			self.inputs,self.labels = x_train,y_train
		elif self.split_set == "val":
			self.inputs,self.labels = x_val,y_val
		else:
			self.inputs,self.labels = x_test,y_test

	def _create_triplets(self):

		#Number of Classes (unique labels)
		num_classes = len(self.data_class.keys())

		
		#Create Triplets of the form (key[j], key[k], random_key[l]), where j,k,l are indices to choose examples belonging to a key





		for i,key in enumerate(self.data_class.keys()):
			#Generate 5 examples per training class
			for index in range(self.examples_per_class):

				#Pick j 
				j = choice(range(self.data_class[key].shape[0]))

				#Pick index k as something other than j (ONLY if there are options)
				allowed_indices = list(range(self.data_class[key].shape[0]))
				if len(allowed_indices) > 1:
					allowed_indices.remove(j) #Any other index than j is allowed
				#Randomly pick one of the allowed index
				k = choice(allowed_indices)
				

				#Pick a random key for the triplet
				#random key
				allowed_keys = list(self.data_class.keys()).copy()
				del allowed_keys[i]
				random_key = choice(allowed_keys)

				#random example for the random key
				l = choice(range(self.data_class[random_key].shape[0]))



				#Append the triplet
				self.triplets.append(torch.stack([self.data_class[key][j],self.data_class[key][k],self.data_class[random_key][l]]))
				self.triplets_labels.append([key,random_key])

		self.triplets = torch.stack(self.triplets).cpu()
		self.triplets_labels = torch.tensor(self.triplets_labels).cpu()
		del self.data_class

	################################## End of Helper Functions #####################################################################
		
		