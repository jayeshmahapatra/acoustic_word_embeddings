#Core Python, Pandas, and kaldi_io
import numpy as np
import pandas as pd
import string
from collections import Counter
import kaldi_io

class DataHelper():


	def __init__(self,load_list,num_examples = np.Inf):
		'''Initializes the DataLoader Function'''

		self.keys = []
		self.matrices = []
		self.mat_lengths = []
		self.word_to_num = {}
		self.num_to_word = {}


		self.load_list = load_list
		self.num_examples = num_examples

		self.filetype = load_list[0].split(".")[-1] 



	def filter_on_character_length(self,matrices,mat_lengths,keys, char_threshold = 5):
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

	def filter_on_frequency_bounds(self, matrices,mat_lengths,keys,frequency_bounds = (0,np.Inf)):
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

	# Function to truncate and limit dimensionality
	def truncate_shapes(self,matrices,mat_lengths,max_length = 100,num_mfcc_features = 40):

		for i, seq in enumerate(matrices):
			matrices[i] = matrices[i][:max_length, :num_mfcc_features]
			mat_lengths[i] = min(mat_lengths[i], max_length)

		return matrices,mat_lengths

	#Function for padding
	def pad_sequences(self,x, n_padded, center_padded=True):
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

	def generate_key_dicts(self):
		'''Arguments:
		keys : A list of words corresponding to the mfcc feature matrices
		-------------
		Returns:
		labels : A list of numbers correspoding to the words in the list keys'''
		c = Counter(self.keys)
		#print(c)
		num_words = len(c.keys())
		self.word_to_num = {}
		self.num_to_word = {}

		index = 0
		for key in c.keys():
			self.word_to_num[key] = index
			self.num_to_word[index] = key
			index+=1




		print('Number of Unique words ',len(c.keys()))
		return c,self.word_to_num,self.num_to_word

	def load_data(self, char_threshold = 5,frequency_bounds= (0,np.Inf)):
		'''Loads the data from the file into the data object'''

		read_function = kaldi_io.read_mat_ark if self.filetype == "ark" else kaldi_io.read_mat_scp

		for load_file in self.load_list:
			file_keys,file_matrices,file_mat_lengths = [],[],[]
			for i,(key,matrix) in enumerate(read_function(load_file)):
				file_keys.append(key.split('_')[1])
				file_matrices.append(matrix)
				file_mat_lengths.append(matrix.shape[0])
				if i+1 == self.num_examples:
					break
			#Filter the data
			file_matrices,file_mat_lengths,file_keys = self.filter_on_character_length(file_matrices,file_mat_lengths,file_keys,char_threshold = char_threshold)


			#Add to the main list
			self.keys.extend(file_keys)
			self.matrices.extend(file_matrices)
			self.mat_lengths.extend(file_mat_lengths)

		self.matrices,self.mat_lengths,self.keys = self.filter_on_frequency_bounds(self.matrices,self.mat_lengths,self.keys,frequency_bounds = frequency_bounds)

		print('Finished Loading the Data, %d examples'%(len(self.keys)))

	def process_data(self):
		'''Processes the loaded data'''

		#Truncate the dimensions of the data
		self.matrices,self.mat_lengths = self.truncate_shapes(self.matrices,self.mat_lengths,max_length=200,num_mfcc_features=40)
		#Pad the matrices
		self.matrices,self.mat_lengths = self.pad_sequences(self.matrices,n_padded = 100,center_padded = True)
		self.matrices = np.transpose(self.matrices,(0,2,1))

		#delete mat_lengths
		del self.mat_lengths

	def give_inputs_and_labels(self):
		'''Uses the stored matrices and keys to generate numpy array of inputs and labels'''

		label_list = []
		for key in self.keys:
			label_list.append(self.word_to_num[key])

		inputs = np.stack(self.matrices)
		labels = np.array(label_list)

		return inputs,labels	










