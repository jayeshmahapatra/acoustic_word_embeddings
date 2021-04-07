from sfba4.utils import alignSequences
from itertools import product
from big_phoney import BigPhoney
from data_helpers import DataHelper
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import pairwise_kernels,paired_distances

def filter_alphabets(string):
	return ''.join(e for e in string if (e.isalpha() or e.isspace()))

def swap_columns(query):
	#Swap columns
	initial_cols = list(query)
	cols = list(query)
	cols.insert(0, cols.pop(cols.index('word_2')))
	query = query.loc[:, cols]
	#Change column names
	query.columns = initial_cols
	return query

def generate_pairs(phoney,data_filepath = '/data/users/jmahapatra/data/feats_cmvn.ark', wordpairs_filepath = '/data/users/jmahapatra/data/wordpairs.txt'):

	
	load_list = [data_filepath]
	
	num_examples = np.Inf

	dh = DataHelper(load_list,num_examples)
	dh.load_data()
	dh.process_data()
	c,word_to_num,num_to_word = dh.generate_key_dicts()

	inputs,labels = dh.give_inputs_and_labels()
	del dh

		

	words = set([num_to_word[labels[i].item()] for i in range(labels.shape[0])])

	print('Number of unique words',len(words))


	word_phoneme_dict = {}
	word_spaced_dict = {}

	#Calculate the word phonemes
	for word in words:
		word_spaced_dict[word] = ' '.join(word)
		phonemes = phoney.phonize(word)
		word_phoneme_dict[word] = filter_alphabets(phonemes)

	print('Finished Calculating embeddings')

	

	word_pairs = list(product(words,words))

	print('Number of word pairs before filtering',len(word_pairs))

	#Unique word pairs
	word_pairs = [tuple(sorted(word_pair)) for word_pair in word_pairs if word_pair[0]!=word_pair[1]]
	word_pairs = set(word_pairs)

	wordpairs_dict = {}
	wordpairs_dict["word_1"] = []
	wordpairs_dict["word_2"] = []
	wordpairs_dict["orthographic_edit_distance"] = []
	wordpairs_dict["phonetic_edit_distance"] = []

	print('Number of word pairs after filtering',len(word_pairs))

	for word_pair in word_pairs:
		aligned_seq1, aligned_seq2, or_eDistance = alignSequences.align(word_spaced_dict[word_pair[0]], word_spaced_dict[word_pair[1]])	
		aligned_seq1, aligned_seq2, ph2_eDistance = alignSequences.align(word_phoneme_dict[word_pair[0]], word_phoneme_dict[word_pair[1]])		

		
		wordpairs_dict["word_1"].append(word_pair[0])
		wordpairs_dict["word_2"].append(word_pair[1])
		wordpairs_dict["orthographic_edit_distance"].append(or_eDistance)
		wordpairs_dict["phonetic_edit_distance"].append(ph2_eDistance)


	#Calculate the orthographic and phonetic ser
	word_columns = ["word_1","word_2"]

	for word_column in word_columns:
		wordpairs["%s_orthographic_ser"%(word_column)] = wordpairs.apply(lambda row: row["orthographic_edit_distance"]/len(row[word_column]), axis = 1)
		wordpairs["%s_phonetic_ser"%(word_column)] = wordpairs.apply(lambda row: row["phonetic_edit_distance"]/len(row[word_column]), axis = 1)

	wordpairs_df = pd.DataFrame(wordpairs_dict)
	
	wordpairs_df.to_csv(wordpairs_filepath, index= False)

	print('Finished Generating Word Pairs')
	print('--------------------------------------------------------------------------------------')
	
def generate_embedding_similarity(wordpairs_filepath = '/data/users/jmahapatra/data/wordpairs.txt', word_embedding_dict_filepath = '/data/users/jmahapatra/data/word_embedding_dict.npy', embedding_similarity_filepath = '/data/users/jmahapatra/data/wordpairs_embedding_similarity.txt',metric = 'cosine'):

	#Load the wordpairs
	wordpairs_df = pd.read_csv(wordpairs_filepath)

	wordpairs = list(zip(wordpairs_df["word_1"].to_list(),wordpairs_df["word_2"].to_list()))

	del wordpairs_df

	#Load the embedding dict
	word_embedding_dict = np.load(word_embedding_dict_filepath, allow_pickle = True)

	embedding_similarity_dict = {}
	embedding_similarity_dict["word_1"] = []
	embedding_similarity_dict["word_2"] = []
	embedding_similarity_dict["%s_similarity"%(metric)] = []


	for i,wordpair in enumerate(wordpairs):
	    
	    word_1_embedding = word_embedding_dict.item().get(wordpair[0]).squeeze().reshape(1,-1)
	    word_2_embedding = word_embedding_dict.item().get(wordpair[1]).squeeze().reshape(1,-1)
	    similarity = pairwise_kernels(word_1_embedding,word_2_embedding, metric = metric )
	    
	    embedding_similarity_dict["word_1"].append(wordpair[0])
	    embedding_similarity_dict["word_2"].append(wordpair[1])
	    embedding_similarity_dict["%s_similarity"%(metric)].append(similarity.item())

	    #if(i==1000):
	    #	break
	    

	embedding_similarity_df = pd.DataFrame(embedding_similarity_dict)
	embedding_similarity_df.to_csv(embedding_similarity_filepath, index = False)

	print("Finished Generating Embedding Similarity")
	print('--------------------------------------------------------------------------------------')


def generate_homophones(wordpairs_filepath = '/data/users/jmahapatra/data/wordpairs.txt',homophones_filepath = '/data/users/jmahapatra/data/homophones.txt'):

	#Load the word pairs dataframe
	wordpairs_df = pd.read_csv(wordpairs_filepath, sep = ',')

	#Get the homophone pairs, discard the rest
	homophone_df = wordpairs_df.query("phonetic_edit_distance == 0")
	del wordpairs_df

	#Get the set of words for whom homophones exist
	words = set(homophone_df["word_1"].to_list()).union(set(homophone_df["word_2"].to_list()))

	print('Number of filtered phoneme distance homophones', len(words))

	homophones_dict = {}
	homophones_dict["word"] = []
	homophones_dict["homophone_words"] = []


	print('Calculating Homophones')


	#Create a set to keep track of already added words to the homophone_dict to avoid repetitions
	added_words = set()

	for word in words:

		if word in added_words:
			continue
		query = pd.concat([homophone_df.query("word_1 == '%s'"%(word)),swap_columns(homophone_df.query("word_2 == '%s'"%(word)))])
		
		homophones_dict["word"].append(word)
		homophones_dict["homophone_words"].append(tuple(query["word_2"].to_list()))

		added_words.add(word)
		for homophone_word in query["word_2"].to_list():
			added_words.add(homophone_word)


	
	
	#Create the DataFrames
	homophone_df = pd.DataFrame(homophones_dict)


	#Save the homophones
	homophone_df.to_csv(homophones_filepath, index = False)

	


	print('Finished Generating homophones')
	print('--------------------------------------------------------------------------------------')
