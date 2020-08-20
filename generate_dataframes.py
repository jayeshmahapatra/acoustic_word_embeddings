#Core python libraries
from datetime import datetime


#BigPhoney
#from big_phoney import BigPhoney

#User defined libraries
#from generate_word_pairs import filter_alphabets, generate_pairs
#from generate_nearest_neighbours import *
#from generate_homophones import generate_homophones
from add_ser import add_ser_and_cosine_similarity


def show_time():
	now = datetime.now()
	current_time = now.strftime("%d/%m/%Y %H:%M:%S")
	print("Current Date and Time =", current_time)


if __name__ == '__main__':

	#phoney = BigPhoney()

	#data_filepath = 'Data/feats_cmvn.ark'
	wordpairs_filepath = 'Data/wordpairs_test.txt'
	wordpairs_ser_filepath = 'Data/wordpairs_test_with_ser.txt'
	word_embedding_dict_filepath = 'Data/word_embedding_dict.npy'
	word_phoneme_dict_filepath = 'Data/word_phoneme_dict.npy'
	#homophones_filepath = 'Data/homophones.txt'
	#sim_nearest_neighbours_filepath = 'Data/sim_nearest_neighbours.txt'
	#edit_nearest_neighbours_filepath = 'Data/edit_nearest_neighbours.txt'

	#generate_pairs(phoney, data_filepath,wordpairs_filepath)
	add_ser(wordpairs_filepath, word_phoneme_dict_filepath, wordpairs_ser_filepath)
	#generate_homophones(wordpairs_filepath,homophones_filepath)
	#generate_nn(phoney, wordpairs_filepath ,sim_nearest_neighbours_filepath, edit_nearest_neighbours_filepath)

	
	#add_ser()
	#generate_homophones()

