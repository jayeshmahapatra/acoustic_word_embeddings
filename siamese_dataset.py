#Core Python, Pandas, and kaldi_io
import numpy as np
import pandas as pd
import string
from collections import Counter
import random
import kaldi_io


#Torch and utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader,random_split,ConcatDataset
#scikit-learn
from sklearn.model_selection import train_test_split

class SiameseTriplets(torch.utils.data.Dataset):
	'''Dataset that on each iteration provides a triplet pair of acosutic segments, out of which
	two belong to the same word, and one belongs to a different word.'''
	
	def __init__(self, num_examples = np.Inf, split_set = "train", data_filepath = "Data/feats_cmvn.ark", char_threshold = 5, frequency_bounds = (0,np.Inf)):
	
		self.load_list = [data_filepath]
		self.char_threshold = char_threshold
		self.frequency_bounds = frequency_bounds
		self.num_examples = num_examples
		self.split_set = split_set
		

		#Data Structures to store data
		self.keys = []
		self.matrices = []
		self.mat_lengths = []
		self.c = None
		self.word_to_num = None
		self.num_to_word = None
		self.inputs = None
		self.labels = None
		
		self.examples_per_class = 10

		#Load and Process the data
		self._load_data()
		self._process_data()
		self._generate_key_dicts()
		self._generate_inputs_and_labels()
		self._generate_data_class()
		self._pick_data_class()

		#Number of Classes (unique labels)
		num_classes = len(self.data_class.keys())

		self.triplets = []
		self.triplets_labels = []

		#Create Triplets
		for i,word_num in enumerate(self.data_class.keys()):
			#Generate 5 examples per training class
			for j in range(self.examples_per_class):

				#Pick a random word other than the current word
				rnd_cls = random.randint(0,num_classes-2)

				if rnd_cls >=i:
					rnd_cls += 1

				#The random word_num
				rnd_word_num = list(self.data_class.keys())[rnd_cls]

				#Pick a random sample other than j for the same class
				if self.data_class[word_num].shape[0]-2 >0 :
					sample_same_cls = random.randint(0,self.data_class[word_num].shape[0]-2)
					if sample_same_cls >= j:
						sample_same_cls += 1
				else:
					sample_same_cls = (j%self.data_class[word_num].shape[0] + 1)%self.data_class[word_num].shape[0]

				#Pick a random sample for the different class
				sample_diff_cls = random.randint(0,self.data_class[rnd_word_num].shape[0]-1)


				#Append the triplet
				self.triplets.append(torch.stack([self.data_class[word_num][j%self.data_class[word_num].shape[0]],self.data_class[word_num][sample_same_cls],self.data_class[rnd_word_num][sample_diff_cls]]))
				self.triplets_labels.append([word_num,rnd_word_num])

		self.triplets = torch.stack(self.triplets).cpu()
		self.triplets_labels = torch.tensor(self.triplets_labels).cpu()
		del self.data_class
		print(self.triplets.shape)
		
	def __getitem__(self, index):
		
		return self.triplets[index],self.triplets_labels[index]
		
	def __len__(self):
		
		return self.triplets.shape[0]
		
	
	
	
	
	################################## Helper Functions #####################################################################
	def _load_data(self):
		'''Loads the data from the file into the data object'''
		
		filetype = self.load_list[0].split(".")[-1] 
		read_function = kaldi_io.read_mat_ark if filetype == "ark" else kaldi_io.read_mat_scp

		for load_file in self.load_list:
			file_keys,file_matrices,file_mat_lengths = [],[],[]
			for i,(key,matrix) in enumerate(kaldi_io.read_mat_ark(load_file)):
				file_keys.append(key.split('_')[1])
				file_matrices.append(matrix)
				file_mat_lengths.append(matrix.shape[0])
				if i+1 == self.num_examples:
					break
			#Filter the data
			file_matrices,file_mat_lengths,file_keys = self._filter_on_character_length(file_matrices,file_mat_lengths,file_keys,char_threshold = self.char_threshold)


			#Add to the main list
			self.keys.extend(file_keys)
			self.matrices.extend(file_matrices)
			self.mat_lengths.extend(file_mat_lengths)

		file_matrices,file_mat_lengths,file_keys = self.filter_on_character_length(file_matrices,file_mat_lengths,file_keys,char_threshold = char_threshold)

		print('Finished Loading the Data, %d examples'%(len(self.keys)))

	def _process_data(self):
		'''Processes the loaded data'''

		#Truncate the dimensions of the data
		self.matrices,self.mat_lengths = self._truncate_shapes(self.matrices,self.mat_lengths,max_length=200,num_mfcc_features=40)
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
			self.data_class[self.word_to_num[key]] = torch.tensor(self.inputs[ids], dtype = torch.float).cpu()

		del self.inputs,self.labels

	def _pick_data_class(self):


		#Split the numbers into train, val and test
		words = list(self.num_to_word.keys())
		trainval,test = train_test_split(words, test_size=0.2, random_state=32)
		train,val = train_test_split(trainval, test_size=0.25, random_state=32) 

		print(list(map(len,[train,val,test])))


		if self.split_set == "train":
			del_list = val + test
		elif self.split_set == "val":
			del_list = train + test
		else:
			del_list = train + val

		for key in (del_list):
				del self.data_class[key]

		print("For %sset number of unique words are %d"%(self.split_set,len(self.data_class.keys())))

			
	
	# Function to truncate and limit dimensionality
	def _truncate_shapes(self,matrices,mat_lengths,max_length = 100,num_mfcc_features = 40):

		for i, seq in enumerate(matrices):
			matrices[i] = matrices[i][:max_length, :num_mfcc_features]
			mat_lengths[i] = min(mat_lengths[i], max_length)

		return matrices,mat_lengths

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
	
	def _filter_on_character_length(self,matrices,mat_lengths,keys, char_threshold = 5):
		'''Takes in matrices and keys. Filters the data by making all keys lowercase, removing words
		with number of letters less than a threshold.'''

		print('Length before filtering on char length %d'%(len(keys)))
		#Lowercase all keys
		keys = list(map(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower(),keys))

		#Filter if the characters are smaller than the character threshold
		matrices,mat_lengths,keys = zip(*filter(lambda x: len(x[2])>=char_threshold, zip(matrices,mat_lengths,keys)))

		matrices,mat_lengths,keys = list(matrices),list(mat_lengths),list(keys)

		print('Length after filtering on char length %d'%(len(keys)))


		return matrices,mat_lengths,keys

	def _filter_on_character_frequency(self, matrices,mat_lengths,keys,frequency_bounds = (0,np.Inf)):
		'''Filter words that have frequnecy less than a lower bound threshold or more than an upper bound threshold'''

		print('Length before filtering on frequency_bounds %d'%(len(keys)))

		#Create a Counter
		c = Counter(keys)

		#Get the words whose frequency is below a lower bound threshold or above an upper bound threshold
		remove_list = []

		for key,value in c.items():
			if value < frequency_bounds[0] or value > frequency_bounds[1]:
				remove_list.append(key)

		#Remove the words from the Counter
		for word in remove_list:
			del c[word]

		#Remove the words from data
		matrices,mat_lengths,keys = zip(*filter(lambda x: x[2] not in remove_list, zip(matrices,mat_lengths,keys)))


		print('Length after filtering on frequency_bounds %d'%(len(keys)))

		return map(list,(matrices,mat_lengths,keys))
	
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
		