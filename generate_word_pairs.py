from sfba4.utils import alignSequences
from itertools import product
from big_phoney import BigPhoney
from data_helpers import DataHelper
import pandas as pd



if __name__ == '__main__':

	#load_list = ['Data/feats_cmvn.ark']
	load_list = ['/data/users/jmahapatra/data/feats_cmvn.ark']
	
	num_examples = 200

	dh = DataHelper(load_list,num_examples)
	dh.load_data()
	dh.process_data()
	c,word_to_num,num_to_word = dh.generate_key_dicts()

	inputs,labels = dh.give_inputs_and_labels()
	del dh

	phoney = BigPhoney()	

	words = set([num_to_word[labels[i].item()] for i in range(labels.shape[0])])

	print('Number of unique words',len(words))


	word_phoneme_dict = {}
	word_spaced_dict = {}

	#Calculate the word phonemes
	for word in words:
		word_spaced_dict[word] = ' '.join(word)
		word_phoneme_dict[word] = phoney.phonize(word)

	print('Finished Calculating embeddings')

	

	word_pairs = list(product(words,words))

	print('Number of word pairs before filtering',len(word_pairs))

	word_pairs = [tuple(sorted(word_pair)) for word_pair in word_pairs if word_pair[0]!=word_pair[1]]
	word_pairs = set(word_pairs)

	homophone_dict = {}
	homophone_dict["word_pairs"] = []
	homophone_dict["orthographic_edit_distance"] = []
	homophone_dict["phonetic_edit_distance"] = []

	print('Number of word pairs after filtering',len(word_pairs))

	for word_pair in word_pairs:
		aligned_seq1, aligned_seq2, or_eDistance = alignSequences.align(word_spaced_dict[word_pair[0]], word_spaced_dict[word_pair[1]])	
		aligned_seq1, aligned_seq2, ph_eDistance = alignSequences.align(word_phoneme_dict[word_pair[0]], word_phoneme_dict[word_pair[1]])		

		
		homophone_dict["word_pairs"].append(word_pair)
		homophone_dict["orthographic_edit_distance"].append(or_eDistance)
		homophone_dict["phonetic_edit_distance"].append(ph_eDistance)

	homophone_df = pd.DataFrame(homophone_dict)
	#homophone_df.to_csv('Data/wordpairs_test.txt', index= False)
	homophone_df.to_csv('/data/users/jmahapatra/data/homophones.txt', index= False)
	








