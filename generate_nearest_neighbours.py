#Core Python, Pandas, and kaldi_io
import numpy as np
import pandas as pd
import string
from collections import Counter,OrderedDict 

#BigPhoney
from big_phoney import BigPhoney




def sim_score(ser):
	return 10 - min(10,ser/10)

def swap_columns(query):
	#Swap columns
	initial_cols = list(query)
	cols = list(query)
	cols.insert(0, cols.pop(cols.index('word_2')))
	query = query.loc[:, cols]
	#Change column names
	query.columns = initial_cols
	return query

def generate_word_phoneme_dict(words, phoney):
	'''Given a list of words generate a dictionary with their phonetic expansions'''
	word_phoneme_dict = {}
	for word in words:
		word_phoneme_dict[word] = phoney.phonize(word)

	return word_phoneme_dict

def generate_nn(phoney, wordpairs_filepath = '/data/users/jmahapatra/data/wordpairs.txt', sim_nearest_neighbours_filepath = '/data/users/jmahapatra/data/sim_nearest_neighbours.txt' , edit_nearest_neighbours_filepath = '/data/users/jmahapatra/data/edit_nearest_neighbours.txt'):

	wordpairs = pd.read_csv(wordpairs_filepath)

	#Get list of words
	words = set(wordpairs["word_1"].to_list()).union(set(wordpairs["word_2"].to_list()))
	print('Number of Unique words ',len(words))


	word_phoneme_dict = generate_word_phoneme_dict(words,phoney)

	#Similarity based NNs
	sim_nn_dict = {}
	sim_nn_dict["word"] = []
	sim_nn_dict["orthographic"] = []
	sim_nn_dict["phonetic"] = []


	#Edit Distance based NNs
	edit_nn_dict = {}
	edit_nn_dict["word"] = []
	edit_nn_dict["orthographic"] = []
	edit_nn_dict["phonetic"] = []


	for word in words:

		query = pd.concat([wordpairs.query("word_1 == '%s'"%(word)),swap_columns(wordpairs.query("word_2 == '%s'"%(word)))])

		query["orthographic_sim"] = query.apply(lambda row: sim_score(100*row["orthographic_edit_distance"]/len(row["word_1"])), axis = 1)
		query["phonetic_sim"] = query.apply(lambda row: sim_score(100*row["phonetic_edit_distance"]/len(word_phoneme_dict[row["word_1"]])), axis = 1)
		

		

		sim_orthographic_nn = tuple(query.sort_values( "orthographic_sim", ascending =False).iloc[:10]["word_2"].to_list())
		sim_phonetic_nn = tuple(query.sort_values( "phonetic_sim", ascending =False).iloc[:10]["word_2"].to_list())


		edit_orthographic_nn = tuple(query.sort_values( "orthographic_edit_distance", ascending =True).iloc[:10]["word_2"].to_list())
		edit_phonetic_nn = tuple(query.sort_values( "phonetic_edit_distance", ascending =True).iloc[:10]["word_2"].to_list())

		sim_nn_dict["word"].append(word)
		sim_nn_dict["orthographic"].append(sim_orthographic_nn)
		sim_nn_dict["phonetic"].append(sim_phonetic_nn)
		


		edit_nn_dict["word"].append(word)
		edit_nn_dict["orthographic"].append(edit_orthographic_nn)
		edit_nn_dict["phonetic"].append(edit_phonetic_nn)

		del query

	sim_nn_df = pd.DataFrame(sim_nn_dict)
	edit_nn_df = pd.DataFrame(edit_nn_dict)

	

	sim_nn_df.to_csv(sim_nearest_neighbours_filepath, index = False)
	edit_nn_df.to_csv(edit_nearest_neighbours_filepath, index = False)

	print('Finished Creating Nearest Neighbours')
	print('--------------------------------------------------------------------------------------')


if __name__ == '__main__':

	wordpairs_filepath = 'Data/wordpairs_test.txt'
	sim_nearest_neighbours_filepath = 'Data/sim_nearest_neighbours.txt'
	edit_nearest_neighbours_filepath = 'Data/edit_nearest_neighbours.txt'

	generate_nn(wordpairs_filepath,sim_nearest_neighbours_filepath,edit_nearest_neighbours_filepath)

	





