#Core python libraries
from datetime import datetime


#BigPhoney
from big_phoney import BigPhoney

#User defined libraries

from dataframe_methods import filter_alphabets, generate_pairs, generate_embedding_similarity, generate_homophones


def show_time():
	now = datetime.now()
	current_time = now.strftime("%d/%m/%Y %H:%M:%S")
	print("Current Date and Time =", current_time)


if __name__ == '__main__':

	phoney = BigPhoney()

	data_filepath = 'Data/feats_cmvn.ark'
	wordpairs_filepath = 'Data/wordpairs_test.txt'
	word_embedding_dict_filepath = 'Data/word_embedding_dict.npy'
	embedding_similarity_filepath = '/data/users/jmahapatra/data/wordpairs_embedding_similarity.txt'
	word_phoneme_dict_filepath = 'Data/word_phoneme_dict.npy'
	homophones_filepath = 'Data/homophones.txt'
	
	#Generate a dataframe containing all possible wordpairs and saving it in "wordpairs_filepath" (RUN ONLY ONCE)
	#generate_pairs(phoney, data_filepath,wordpairs_filepath)

	#Generating a dataframe containing embedding distance between all wordpairs saved in "wordpairs_filepath" and saving it in "embedding_similarity filepath"
	generate_embedding_similarity(wordpairs_filepath, word_embedding_dict_filepath, embedding_similarity_filepath,metric = 'cosine'):
	

	#Generating a dataframe containing words and lists of their homophones and saving it in "homophones_filepath"
	generate_homophones(wordpairs_filepath,homophones_filepath)

