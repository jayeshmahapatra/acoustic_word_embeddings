#Core Python, Pandas, and kaldi_io
import numpy as np
import pandas as pd
import string
from collections import Counter
import kaldi_io

from models import SimpleNet, SimpleNet_with_dropout


def generate_word_embedding_dict(words):
    word_embedding_dict = OrderedDict()
    #Calculate embeddings
    for word in words:
        #Find the mfcc features of the acoustic representation of the word in the data
        word_features = inputs[np.where(np.isin(labels,word_to_num[word]))]
        
        #Calculate embeddings for the feature
        word_embedding = net(torch.tensor(word_features),dev)
        
        #If the number of representation is more than one, take the average embedding
        word_embedding_dict[word] = np.mean(word_embedding, axis = 0).reshape(1,-1)
    
    return word_embedding_dict

if __name__ == '__main__':

	
	#Load all the data
	train_ds = AMI_noisy_dataset(num_examples = num_examples, split_set = "train", data_filepath = data_filepath, char_threshold = 5, frequency_bounds = (0,np.Inf))
	val_ds = AMI_noisy_dataset(num_examples = num_examples, split_set = "val", data_filepath = data_filepath, char_threshold = 5, frequency_bounds = (0,np.Inf))
	test_ds = AMI_noisy_dataset(num_examples = num_examples, split_set = "test", data_filepath = data_filepath, char_threshold = 5, frequency_bounds = (0,np.Inf))


	#Merge all data into inputs and labels
	inputs,labels = torch.vstack(train_ds.inputs,val_ds.inputs), torch.vstack(train_ds.labels,val_ds.labels)
	inputs,labels = torch.vstack(inputs,test_ds.inputs), torch.vstack(labels,test_ds.labels)

	
	#Get the dictionaries
	word_to_num,num_to_word = train_ds.word_to_num, train_ds.num_to_word


	#Delete the datasets
	del train_ds,val_ds,test_ds

	dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	
	#Number of unique words
	num_output = len(word_to_num.keys())

	#Load the Neural Network
	model_path = ""
	net = SimpleNet(num_output)
	net.load_state_dict(torch.load(model_path))
	net.eval()

	#Unique words
	words = list(word_to_num.keys())

	#Generate the word embedding dict
	word_embedding_dict = generate_word_embedding_dict(words)

	word_embedding_dict_save_path = ""

	#Save the word embedding dict
	np.save(word_embedding_dict_save_path, word_embedding_dict)




