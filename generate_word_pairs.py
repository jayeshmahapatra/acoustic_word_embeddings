from sfba4.utils import alignSequences
from itertools import product
from big_phoney import BigPhoney
from data_helpers import DataHelper
import numpy as np
import pandas as pd

def filter_alphabets(string):
	return ''.join(e for e in string if (e.isalpha() or e.isspace()))

def generate_pairs(phoney,data_filepath = '/data/users/jmahapatra/data/feats_cmvn.ark', wordpairs_filepath = '/data/users/jmahapatra/data/wordpairs.txt'):

	
	load_list = [data_filepath]
	
	num_examples = 10000

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
	


if __name__ == '__main__':


	phoney = BigPhoney()

	data_filepath = 'Data/feats_cmvn.ark'
	wordpairs_filepath = 'Data/wordpairs_test.txt'
	generate_pairs(phoney,data_filepath,wordpairs_filepath)

	
	








