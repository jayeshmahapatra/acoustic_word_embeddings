from data_helpers import DataHelper
import numpy as np
import pandas as pd

#User defined libraries
from generate_nearest_neighbours import swap_columns



def generate_homophones():

	#Load the word pairs dataframe
	wordpairs_df = pd.read_csv('/data/users/jmahapatra/data/wordpairs.txt')
	#wordpairs_df = pd.read_csv('Data/wordpairs_test.txt', sep = ',')

	#Get the homophone pairs, discard the rest
	raw_ph_homophone_df = wordpairs_df.query("raw_phonetic_edit_distance == 0")
	filtered_ph_homophone_df = wordpairs_df.query("filtered_phonetic_edit_distance == 0")
	del wordpairs_df

	#Get the set of words for whom homophones exist
	raw_ph_words = set(raw_ph_homophone_df["word_1"].to_list()).union(set(raw_ph_homophone_df["word_2"].to_list()))
	filtered_ph_words = set(filtered_ph_homophone_df["word_1"].to_list()).union(set(filtered_ph_homophone_df["word_2"].to_list()))

	raw_ph_homophone_dict = {}
	raw_ph_homophone_dict["word"] = []
	raw_ph_homophone_dict["homophone_words"] = []


	filtered_ph_homophone_dict = {}
	filtered_ph_homophone_dict["word"] = []
	filtered_ph_homophone_dict["homophone_words"] = []

	for word in raw_ph_words:
		query = pd.concat([raw_ph_homophone_df.query("word_1 == '%s'"%(word)),swap_columns(raw_ph_homophone_df.query("word_2 == '%s'"%(word)))])
		
		raw_ph_homophone_dict["word"].append(word)
		raw_ph_homophone_dict["homophone_words"].append(tuple(query["word_2"].to_list()))



	for word in filtered_ph_words:
		query = pd.concat([filtered_ph_homophone_df.query("word_1 == '%s'"%(word)),swap_columns(filtered_ph_homophone_df.query("word_2 == '%s'"%(word)))])
		
		filtered_ph_homophone_dict["word"].append(word)
		filtered_ph_homophone_dict["homophone_words"].append(tuple(query["word_2"].to_list()))

	
	#Create the DataFrames
	raw_ph_homophone_df = pd.DataFrame(raw_ph_homophone_dict)
	filtered_ph_homophone_df = pd.DataFrame(filtered_ph_homophone_dict)


	#Save the homophones
	raw_ph_homophone_df.to_csv('/data/users/jmahapatra/data/raw_ph_homophones.txt', index = False)
	#raw_ph_homophone_df.to_csv('Data/raw_ph_homophones.txt', index = False)
	filtered_ph_homophone_df.to_csv('/data/users/jmahapatra/data/filtered_ph_homophones.txt', index = False)
	#filtered_ph_homophone_df.to_csv('Data/filtered_ph_homophones.txt', index = False)



	print('Finished Saving homophones')

if __name__ == '__main__':

	generate_homophones()
