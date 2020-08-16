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
	cols = list(query)
	cols.insert(0, cols.pop(cols.index('word_2')))
	query = query.loc[:, cols]
	#Change column names
	query.columns = ["word_1","word_2","orthographic_edit_distance","raw_phonetic_edit_distance","filtered_phonetic_edit_distance"]
	return query

def generate_word_phoneme_dict(words, phoney):
	'''Given a list of words generate a dictionary with their phonetic expansions'''
	word_phoneme_dict = {}
	for word in words:
		word_phoneme_dict[word] = phoney.phonize(word)

	return word_phoneme_dict

def generate_nn(phoney):

	homophones = pd.read_csv('/data/users/jmahapatra/data/homophones.txt')
	#homophones = pd.read_csv('Data/wordpairs_test.txt')

	#Get list of words
	words = set(homophones["word_1"].to_list()).union(set(homophones["word_2"].to_list()))
	print('Number of Unique words ',len(words))


	word_phoneme_dict = generate_word_phoneme_dict(words,phoney)

	#Similarity based NNs
	sim_nn_dict = {}
	sim_nn_dict["word"] = []
	sim_nn_dict["orthographic"] = []
	sim_nn_dict["raw_phonetic"] = []
	sim_nn_dict["filtered_phonetic"] = []


	#Edit Distance based NNs
	edit_nn_dict = {}
	edit_nn_dict["word"] = []
	edit_nn_dict["orthographic"] = []
	edit_nn_dict["raw_phonetic"] = []
	edit_nn_dict["filtered_phonetic"] = []


	for word in words:

		query = pd.concat([homophones.query("word_1 == '%s'"%(word)),swap_columns(homophones.query("word_2 == '%s'"%(word)))])

		query["orthographic_sim"] = query.apply(lambda row: sim_score(100*row["orthographic_edit_distance"]/len(row["word_1"])), axis = 1)
		query["raw_phonetic_sim"] = query.apply(lambda row: sim_score(100*row["raw_phonetic_edit_distance"]/len(word_phoneme_dict[row["word_1"]])), axis = 1)
		query["filtered_phonetic_sim"] = query.apply(lambda row: sim_score(100*row["filtered_phonetic_edit_distance"]/len(word_phoneme_dict[row["word_1"]])), axis = 1)

		sim_orthographic_nn = tuple(query.sort_values( "orthographic_sim", ascending =False).iloc[:10]["word_2"].to_list())
		sim_raw_phonetic_nn = tuple(query.sort_values( "raw_phonetic_sim", ascending =False).iloc[:10]["word_2"].to_list())
		sim_filtered_phonetic_nn = tuple(query.sort_values( "filtered_phonetic_sim", ascending =False).iloc[:10]["word_2"].to_list())


		edit_orthographic_nn = tuple(query.sort_values( "orthographic_edit_distance", ascending =True).iloc[:10]["word_2"].to_list())
		edit_raw_phonetic_nn = tuple(query.sort_values( "raw_phonetic_edit_distance", ascending =True).iloc[:10]["word_2"].to_list())
		edit_filtered_phonetic_nn = tuple(query.sort_values( "filtered_phonetic_edit_distance", ascending =True).iloc[:10]["word_2"].to_list())

		sim_nn_dict["word"].append(word)
		sim_nn_dict["orthographic"].append(sim_orthographic_nn)
		sim_nn_dict["raw_phonetic"].append(sim_raw_phonetic_nn)
		sim_nn_dict["filtered_phonetic"].append(sim_filtered_phonetic_nn)


		edit_nn_dict["word"].append(word)
		edit_nn_dict["orthographic"].append(edit_orthographic_nn)
		edit_nn_dict["raw_phonetic"].append(edit_raw_phonetic_nn)
		edit_nn_dict["filtered_phonetic"].append(edit_filtered_phonetic_nn)

		del query

	sim_nn_df = pd.DataFrame(sim_nn_dict)
	edit_nn_df = pd.DataFrame(edit_nn_dict)

	print('Finished Creating Nearest Neighbours')

	sim_nn_df.to_csv('/data/users/jmahapatra/data/sim_nearest_neighbours.txt', index = False)
	edit_nn_df.to_csv('/data/users/jmahapatra/data/edit_nearest_neighbours.txt', index = False)
	#sim_nn_df.to_csv('Data/sim_nearest_neighbours.txt', index = False)
	#edit_nn_df.to_csv('Data/edit_nearest_neighbours.txt', index = False)



if __name__ == '__main__':

	generate_nn()

	





