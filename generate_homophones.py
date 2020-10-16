from data_helpers import DataHelper
import numpy as np
import pandas as pd


def swap_columns(query):
	#Swap columns
	initial_cols = list(query)
	cols = list(query)
	cols.insert(0, cols.pop(cols.index('word_2')))
	query = query.loc[:, cols]
	#Change column names
	query.columns = initial_cols
	return query


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

if __name__ == '__main__':


	#wordpairs_filepath = 'Data/wordpairs_test.txt'
	#homophones_filepath = 'Data/homophones.txt'

	wordpairs_filepath = '/data/users/jmahapatra/data/wordpairs.txt'
	homophones_filepath = '/data/users/jmahapatra/data/homophones.txt'

	generate_homophones(wordpairs_filepath,homophones_filepath)
