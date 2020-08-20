#Core Python, Pandas
import numpy as np
import pandas as pd
import string
from collections import Counter,OrderedDict

#Scikit learn
from sklearn.metrics.pairwise import pairwise_kernels,paired_distances
from scipy import stats

def alphabet_commas(string):
	'''Takes a string and returns a filtered string with only alphabet and commas'''
	
	return ''.join(e for e in string if (e.isalpha() or e == ","))

def sim_score(ser):
	return 10 - min(10,ser/10)

def similarity_task(wordpairs,word_embedding_dict,column_name = "phonetic", use_sim_score = True):
	'''Function to return spearmans rank coefficient between similarity score/symbol error rate and cosine similarity/distance of word pairs'''


	if use_sim_score == True:
		#Create the ranked sim_score
		rho_1,p_value_1 = stats.spearmanr(wordpairs["word_1_%s_ser"%(column_name)].apply(lambda x: sim_score(100*x)).to_list(),wordpairs["cosine_similarity"].to_list())
		rho_2,p_value_2 = stats.spearmanr(wordpairs["word_2_%s_ser"%(column_name)].apply(lambda x: sim_score(100*x)).to_list(),wordpairs["cosine_similarity"].to_list())

	else:
		rho_1,p_value_1 = stats.spearmanr(wordpairs["word_1_%s_ser"%(column_name)].to_list(),wordpairs["cosine_similarity"].apply(lambda x: 1-x).to_list())
		rho_2,p_value_2 = stats.spearmanr(wordpairs["word_2_%s_ser"%(column_name)].to_list(),wordpairs["cosine_similarity"].apply(lambda x: 1-x).to_list())



	
	return (rho_1+rho_2)/2

def homophone_task(homophones):
	words = set(homophones["word"].to_list())
	homophones["homophone_words"] = homophones["homophone_words"].apply(alphabet_commas) 

	avg_precision = 0
	for word in words:
		homophone_query = homophones.query("word == '%s'"%(word))
		awe_nn_words_query = em_cosine_nn.query("word == '%s'"%(word))

		homophone_words = list(filter(lambda x: x.isalpha(), homophone_query["homophone_words"].item().split(",")))
		awe_nn_words = list(awe_nn_words_query["neighbours"].item().split(","))[:len(homophone_words)]

		#print(awe_nn_words)

		#Set of homophone words
		set_homophone_words = set(homophone_words)

		#Set of Nearest neighbours based on cosine_similarity of embeddings
		set_awe_nn_words = set(awe_nn_words)


		#print(word,set_homophone_words,set_awe_nn_words)

		#Calculate precision score
		word_precision = len(set_homophone_words.intersection(set_awe_nn_words))/len(set_homophone_words)

		avg_precision += word_precision

	avg_precision = avg_precision/len(words)
	return avg_precision

if __name__ == '__main__':

	#Load the wordpairs dataframe
	wordpairs_with_ser = pd.read_csv('/data/users/jmahapatra/data/wordpairs_with_ser.txt')
	#wordpairs_with_ser = pd.read_csv('Data/wordpairs_test_with_ser.txt')


	



	#Load the embedding nearest neighbour dataframe
	em_cosine_nn = pd.read_csv('/data/users/jmahapatra/data/em_nearest_neighbours_cosine.txt')
	#em_cosine_nn = pd.read_csv('Data/em_nearest_neighbours_cosine.txt')



	#Load the homophones dataframe
	homophones = pd.read_csv('/data/users/jmahapatra/data/homophones.txt')
	#homophones = pd.read_csv('Data/homophones.txt')

	#Load the word embedding dict
	word_embedding_dict = np.load('/data/users/jmahapatra/data/word_embedding_dict.npy', allow_pickle = True)
	#word_embedding_dict = np.load('Data/word_embedding_dict.npy', allow_pickle = True)

	#Perform Similarity Tasks
	sim_ph_spearman = similarity_task(wordpairs_with_ser, word_embedding_dict, "phonetic" ,use_sim_score = True)
	sim_orthographic_spearman = similarity_task(wordpairs_with_ser, word_embedding_dict, "orthographic", use_sim_score = True)

	print('Similarity Tasks for NN words based on Similarity')
	print('Phonetic Similarity %f'%(sim_ph_spearman))
	print('Orthographic Similarity %f'%(sim_orthographic_spearman))


	#Perform Similarity Tasks
	edit_ph_spearman = similarity_task(wordpairs_with_ser, word_embedding_dict, "phonetic", use_sim_score = False)
	edit_orthographic_spearman = similarity_task(wordpairs_with_ser, word_embedding_dict, "orthographic", use_sim_score = False)

	print('Similarity Tasks for NN words based on Edit Distance')
	print('Phonetic Similarity %f'%(edit_ph_spearman))
	print('Orthographic Similarity %f'%(edit_orthographic_spearman))


	avg_precision = homophone_task(homophones)

	print('Homophone Task')
	print('Average Precision of %f on homophones'%(avg_precision))



