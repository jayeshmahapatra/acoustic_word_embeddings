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
from torch.utils.data import Dataset, TensorDataset,DataLoader,random_split,ConcatDataset
from torch.utils.data.sampler import BatchSampler
#scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class Base_AMI(Dataset):
	'''Base Dataset Class containing data loading, and filtering '''

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
			if self.snr == np.Inf:

				data_directory = "./Data/AMI_Clean/"
			else:
				data_directory = "./Data/AMI_White_SNR%d_v2/"%(self.snr)


			keyword_path = data_directory + "text"
			data_load_list = [data_directory+"feats.scp"]


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

		#self.keys,self.matrices = self._filter_on_frequency_bounds(self.keys,self.matrices,frequency_bounds = self.frequency_bounds)



		print('Finished Loading the Data, %d examples'%(len(self.keys)))

	def _pad_and_truncate_data(self):
		'''Processes the loaded data'''

		#Truncate the dimensions of the data
		self.matrices = self._truncate_shapes(self.matrices,max_length=200,num_mfcc_features=40)
		#Pad the matrices
		self.matrices,self.mat_lengths = self._pad_sequences(self.matrices,n_padded = 128,center_padded = True)
		self.matrices = np.transpose(self.matrices,(0,2,1))

		#delete mat_lengths
		del self.mat_lengths
		
	def _generate_inputs_and_labels(self):
		'''Uses the stored matrices and keys to generate numpy array of inputs and labels'''

		label_list = [self.word_to_num[key] for key in self.keys]

		inputs = np.stack(self.matrices)
		labels = np.array(label_list)
		
		self.inputs = inputs
		self.labels = labels
		
		#Delete other datastructures to free memory
		del self.keys,self.matrices
		
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

	def _filter_on_frequency_bounds(self, frequency_bounds = (0,np.Inf)):
		'''Filter words that have frequnecy less than a lower bound threshold or more than an upper bound threshold'''


		print('Length before filtering on frequency_bounds ', (self.labels.shape))

		lower_bound,upper_bound = frequency_bounds[0],frequency_bounds[1]

		#If bounds are at thresholds, return as it is
		if lower_bound == 0 and upper_bound == np.Inf:
			print("Not filtering")
			return None
		
		#Create a Counter
		c = Counter(self.labels.tolist())

		labels_set = set(list(c.keys()))
		label_to_indices = {label: np.where(self.labels == label)[0]
								 for label in labels_set}

		#Lower bound filtering (remove words that occur lower than a set threshold)
		#and upper bound filtering (keep only n instances of a label)
		allowed_indices = [label_to_indices[label][:min(upper_bound,int(label_to_indices[label].shape[0]))] for label in labels_set if label_to_indices[label].shape[0] >= lower_bound]
		allowed_indices = np.concatenate(allowed_indices)

		self.inputs, self.labels = self.inputs[allowed_indices],self.labels[allowed_indices]

		print('Length after filtering on frequency_bounds ' , self.labels.shape)

		return None

	def give_top_k_words(self, k):

		'''Returns instances belonging to the top k most common words'''

		#Create a Counter
		c = Counter(self.labels.tolist())

		labels_set = set(list(c.keys()))
		top_k_labels,_ = zip(*c.most_common(k))
		top_k_labels_set = set(top_k_labels)

		label_to_indices = {label: np.where(self.labels == label)[0]
								 for label in labels_set}

		present_top_k_labels_set = labels_set.intersection(top_k_labels_set)

		print("Giving top %d words"%(len(present_top_k_labels_set)))


		#Allowed indices are indices belonging to the allowed classes
		allowed_indices = [label_to_indices[word] for word in present_top_k_labels_set]
		allowed_indices = np.concatenate(allowed_indices)
		


		#only keep data for top k classes
		return self.inputs[allowed_indices],self.labels[allowed_indices]



	def _filter_top_k_words(self, k):
		'''Only keep the top k most common words'''

		#Filter only if K is less than np.Inf
		if self.k < np.Inf:

			print('Length before filtering for top %d words %d'%(k, self.labels.shape[0]))
			#only keep data for top k classes
			self.inputs, self.labels = self.give_top_k_words(k)
			print('Length after filtering for top %d words %d'%(k, self.labels.shape[0]))

		return None

	

	
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

		
		print(x_train.shape,x_val.shape,x_test.shape)
		
		if self.split_set == "train":
			#Create tensors
			x_train,y_train = torch.tensor(x_train,dtype= torch.float),torch.tensor(y_train, dtype= torch.float)
			#assing to the base class
			self.inputs,self.labels = x_train,y_train
		#Val
		elif self.split_set == "val":
			#create tensors
			x_val,y_val = torch.tensor(x_val, dtype= torch.float),torch.tensor(y_val, dtype= torch.float)
			#assign to the base class
			self.inputs,self.labels = x_val,y_val
		#Test
		else:
			#create tensors
			x_test,y_test = torch.tensor(x_test, dtype= torch.float),torch.tensor(y_test, dtype= torch.float)
			#assign to the base class
			self.inputs,self.labels = x_test,y_test

	################################## End of Helper Functions #####################################################################

class CNN_dataset(Base_AMI):
	''' Base CNN dataset class'''

	def __init__(self, split_set = "train", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = np.Inf, k = np.Inf, cluster = True):
		

		super().__init__(split_set, char_threshold, frequency_bounds, snr, cluster)

		self.k = k # Top k most common words will be kept (if k is np.Inf all words will be kept)

		#Load and Process the data
		self._load_data()
		self._pad_and_truncate_data()
		self._generate_key_dicts()
		self._generate_inputs_and_labels()

		#Filter based on frequency bounds
		self._filter_on_frequency_bounds(self.frequency_bounds)

		#Keep only top k most common words (if k is np.Inf, no filtering is done, and all words are kept)
		self._filter_top_k_words(self.k)

		#Shuffle the array
		self._split_dataset()



class SiameseTriplets(Base_AMI):
	""" Base Siamese Class"""
	def __init__(self, split_set = "train", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = np.Inf, k = np.Inf, cluster = True):
	
		super().__init__(split_set, char_threshold, frequency_bounds, snr, cluster)

		#Siamese triplet arguments
		
		self.examples_per_class = 15 #Default

		self.k = k # Top k most common words will be kept (if k is np.Inf all words will be kept)

		self.triplets = []
		self.triplets_labels = []

		#Load and Process the data
		self._load_data()
		self._pad_and_truncate_data()
		self._generate_key_dicts()
		self._generate_inputs_and_labels()

		#Keep only top k most common words (if k is np.Inf, no filtering is done, and all words are kept)
		self._filter_top_k_words(self.k)

		#Siamese specific processing
		self._split_dataset()
		self._generate_data_class()
		self._create_triplets()

		
		print("Triplet Shape")
		print(self.inputs.shape, self.labels.shape)



	###### Siamese Triplet specific helpers ########################################

	



	def _generate_data_class(self):
		
		self.data_class = {}
		
		for key in self.c.keys():
			ids = np.where(np.isin(self.labels,self.word_to_num[key]))
			if ids[0].shape[0] > 0:
				self.data_class[self.word_to_num[key]] = torch.tensor(self.inputs[ids], dtype = torch.float).cpu()

		del self.inputs,self.labels





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

		self.inputs = torch.stack(self.triplets).cpu()
		self.labels = torch.tensor(self.triplets_labels).cpu()
		del self.data_class


################################Samplers##############################################
	
	
   

class BalancedBatchSampler(BatchSampler):
	"""
	BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
	Returns batches of size n_classes * n_samples
	"""

	def __init__(self, labels, n_classes, n_samples):
		self.labels = labels
		self.labels_set = list(set(self.labels.numpy()))
		self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
								 for label in self.labels_set}
		for l in self.labels_set:
			np.random.shuffle(self.label_to_indices[l])
		self.used_label_indices_count = {label: 0 for label in self.labels_set}
		self.count = 0
		self.n_classes = n_classes
		self.n_samples = n_samples
		self.n_dataset = len(self.labels)
		self.batch_size = self.n_samples * self.n_classes

	def __iter__(self):
		self.count = 0
		while self.count + self.batch_size < self.n_dataset:
			classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
			indices = []
			for class_ in classes:
				indices.extend(self.label_to_indices[class_][
							   self.used_label_indices_count[class_]:self.used_label_indices_count[
																		 class_] + self.n_samples])
				self.used_label_indices_count[class_] += self.n_samples
				if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
					np.random.shuffle(self.label_to_indices[class_])
					self.used_label_indices_count[class_] = 0
			yield indices
			self.count += self.n_classes * self.n_samples

	def __len__(self):
		return self.n_dataset // self.batch_size



class TopK_WordsSampler(BatchSampler):
	"""
	BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
	Returns batches of size n_classes * n_samples
	"""

	def __init__(self, labels, k, batch_size, drop_last = True):
		self.labels = labels
		self.k = k
		self.labels_set = list(set(self.labels.numpy()))
		self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
								 for label in self.labels_set}
	  
		self.batch_size = batch_size
		self.drop_last = drop_last

		self.allowed_classes,_ = zip(*Counter(self.labels.numpy()).most_common(self.k))
		

		#Allowed indices are indices belonging to the allowed classes
		self.allowed_indices = [self.label_to_indices[class_] for class_ in self.allowed_classes]
		self.allowed_indices = np.concatenate(self.allowed_indices)
		np.random.shuffle(self.allowed_indices)

		#Create a sampler
		self.sampler = torch.utils.data.sampler.RandomSampler(self.allowed_indices)





	def __iter__(self):
		batch = []
		for idx in self.sampler:
			batch.append(self.allowed_indices[idx])
			if len(batch) == self.batch_size:
				yield batch
				batch = []
		if len(batch) > 0 and not self.drop_last:
			yield batch

	def __len__(self):
		if self.drop_last:
			return len(self.sampler) // self.batch_size  # type: ignore
		else:
			return (len(self.sampler) + self.batch_size - 1) // self.batch_size 





		
		

