#Core Python, Pandas
import numpy as np
import pandas as pd
import string
from collections import Counter,OrderedDict

#Scikit learn
from sklearn.metrics.pairwise import pairwise_kernels,paired_distances
from scipy import stats

import matplotlib.pyplot as plt

from sfba4.utils import alignSequences

def alphabet_commas(string):
	'''Takes a string and returns a filtered string with only alphabet and commas'''
	
	return ''.join(e for e in string if (e.isalpha() or e == ","))


def similarity_task(wordpairs):
	'''Function to return spearmans rank coefficient between similarity score/symbol error rate and cosine similarity/distance of word pairs'''

	rho_1,p_value_1 = stats.spearmanr(wordpairs["word_1_%s_ser"%(column_name)].to_list(),wordpairs["cosine_similarity"].apply(lambda x: 1-x).to_list())
	rho_2,p_value_2 = stats.spearmanr(wordpairs["word_2_%s_ser"%(column_name)].to_list(),wordpairs["cosine_similarity"].apply(lambda x: 1-x).to_list())

	return (rho_1+rho_2)/2
	

def homophone_task(homophones, word_phoneme_dict):
	words = set(homophones["word"].to_list())
	homophones["homophone_words"] = homophones["homophone_words"].apply(alphabet_commas) 


	
	phoneme_eDistance_list = []
	result_strings = []

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

		for awe_nn_word in set_awe_nn_words:
			
			aligned_seq1, aligned_seq2, eDistance = alignSequences.align(word_phoneme_dict.item().get(word),word_phoneme_dict.item().get(awe_nn_word))
			#print('{%s , %s } Phoneme eDistance %d'%(word,awe_nn_word,eDistance))
			phoneme_eDistance_list.append(eDistance/len(word_phoneme_dict.item().get(word).split()))

		result_string = str(word) + " " + str(set_homophone_words) + " " + str(set_awe_nn_words)
		print(result_string)
		result_strings.append(result_string)
		#Calculate precision score
		word_precision = len(set_homophone_words.intersection(set_awe_nn_words))/len(set_homophone_words)

		avg_precision += word_precision

	avg_precision = avg_precision/len(words)

	c = Counter(phoneme_eDistance_list)

	print('Phonemic Distance of NN words in embedding space')
	print(sorted(c.items()))

	plt.hist(phoneme_eDistance_list)

	plt.title('Frequency of Relative Phonetic Distance of Nearest Neighbours')
	plt.ylabel('Frequency')
	plt.xlabel('Relative Phonetic Edit Distance')

	plt.savefig('Data/nn_phonetic_distance_histogram.png')


	print(len(result_string))

	return avg_precision

if __name__ == '__main__':

	#Load the wordpairs dataframe
	#wordpairs_with_ser = pd.read_csv('/data/users/jmahapatra/data/wordpairs_with_ser.txt')
	wordpairs_with_ser = pd.read_csv('Data/bla_wordpairs_test_with_ser.txt')


	



	#Load the embedding nearest neighbour dataframe
	#em_cosine_nn = pd.read_csv('/data/users/jmahapatra/data/em_nearest_neighbours_cosine.txt')
	em_cosine_nn = pd.read_csv('Data/em_nearest_neighbours_cosine_freq_5.txt')



	#Load the homophones dataframe
	#homophones = pd.read_csv('/data/users/jmahapatra/data/homophones.txt')
	homophones = pd.read_csv('Data/homophones.txt')

	#Load the word embedding dict
	#word_embedding_dict = np.load('/data/users/jmahapatra/data/word_embedding_dict.npy', allow_pickle = True)
	word_embedding_dict = np.load('Data/word_embedding_dict_freq_5.npy', allow_pickle = True)


	word_phoneme_dict = np.load('Data/word_phoneme_dict.npy', allow_pickle = True)
	'''


	#Perform Similarity Tasks
	edit_ph_spearman = similarity_task(wordpairs_with_ser, word_embedding_dict, "phonetic")

	print('Similarity Tasks for NN words based on Edit Distance')
	print('Phonetic Similarity %f'%(edit_ph_spearman))
	'''

	avg_precision = homophone_task(homophones, word_phoneme_dict)

	print('Homophone Task')
	print('Average Precision of %f on homophones'%(avg_precision))



