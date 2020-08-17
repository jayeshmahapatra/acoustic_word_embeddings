from sfba4.utils import alignSequences
from itertools import product
from big_phoney import BigPhoney
from data_helpers import DataHelper
import numpy as np
import pandas as pd

def filter_alphabets(string):
	return ''.join(e for e in string if (e.isalpha() or e.isspace()))

def generate_pairs(phoney):

	
	load_list = ['/data/users/jmahapatra/data/feats_cmvn.ark']
	#load_list = ['Data/feats_cmvn.ark']
	
	num_examples = np.Inf

	dh = DataHelper(load_list,num_examples)
	dh.load_data()
	dh.process_data()
	c,word_to_num,num_to_word = dh.generate_key_dicts()

	inputs,labels = dh.give_inputs_and_labels()
	del dh

		

	words = set([num_to_word[labels[i].item()] for i in range(labels.shape[0])])

	print('Number of unique words',len(words))


	word_filtered_phoneme_dict = {}
	word_raw_phoneme_dict = {}
	word_spaced_dict = {}

	#Calculate the word phonemes
	for word in words:
		word_spaced_dict[word] = ' '.join(word)
		phonemes = phoney.phonize(word)
		word_raw_phoneme_dict[word] = phonemes
		word_filtered_phoneme_dict[word] = filter_alphabets(phonemes)

	print('Finished Calculating embeddings')

	

	word_pairs = list(product(words,words))

	print('Number of word pairs before filtering',len(word_pairs))

	word_pairs = [tuple(sorted(word_pair)) for word_pair in word_pairs if word_pair[0]!=word_pair[1]]
	word_pairs = set(word_pairs)

	homophone_dict = {}
	homophone_dict["word_1"] = []
	homophone_dict["word_2"] = []
	homophone_dict["orthographic_edit_distance"] = []
	homophone_dict["raw_phonetic_edit_distance"] = []
	homophone_dict["filtered_phonetic_edit_distance"] = []

	print('Number of word pairs after filtering',len(word_pairs))

	for word_pair in word_pairs:
		aligned_seq1, aligned_seq2, or_eDistance = alignSequences.align(word_spaced_dict[word_pair[0]], word_spaced_dict[word_pair[1]])	
		aligned_seq1, aligned_seq2, ph1_eDistance = alignSequences.align(word_raw_phoneme_dict[word_pair[0]], word_raw_phoneme_dict[word_pair[1]])
		aligned_seq1, aligned_seq2, ph2_eDistance = alignSequences.align(word_filtered_phoneme_dict[word_pair[0]], word_filtered_phoneme_dict[word_pair[1]])		

		
		homophone_dict["word_1"].append(word_pair[0])
		homophone_dict["word_2"].append(word_pair[1])
		homophone_dict["orthographic_edit_distance"].append(or_eDistance)
		homophone_dict["raw_phonetic_edit_distance"].append(ph1_eDistance)
		homophone_dict["filtered_phonetic_edit_distance"].append(ph2_eDistance)

	homophone_df = pd.DataFrame(homophone_dict)
	
	homophone_df.to_csv('/data/users/jmahapatra/data/wordpairs.txt', index= False)
	#homophone_df.to_csv('Data/wordpairs_test.txt', index= False)
	


if __name__ == '__main__':


	phoney = BigPhoney()
	generate_pairs(phoney)

	
	








