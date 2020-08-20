#Script to load the wordpairs dataframes and add the phonetic and orthographic symbol error rate w.r.t word_1 and word_2
import pandas as pd
import numpy as np



def add_ser(wordpairs_filepath = '/data/users/jmahapatra/data/wordpairs.txt', wordpairs_ser_filepath = '/data/users/jmahapatra/data/wordpairs_with_ser.txt'):

	#Read the wordpairs file
	wordpairs_df = pd.read_csv(wordpairs_filepath)

	print('Calculating phonetic ser')

	#Add the phonetic ser w.r.t word_1 and word_2 respectively
	wordpairs_df["word_1_phonetic_ser"] = wordpairs_df.apply(lambda row: row["phonetic_edit_distance"]/len(row["word_1"]), axis = 1)
	wordpairs_df["word_2_phonetic_ser"] = wordpairs_df.apply(lambda row: row["phonetic_edit_distance"]/len(row["word_2"]), axis = 1)

	print('Calculating orthographic ser')

	#Add the orthographic ser w.rt word_1 and word_2
	wordpairs_df["word_1_orthographic_ser"] = wordpairs_df.apply(lambda row: row["orthographic_edit_distance"]/len(row["word_1"]), axis = 1)
	wordpairs_df["word_2_orthographic_ser"] = wordpairs_df.apply(lambda row: row["orthographic_edit_distance"]/len(row["word_2"]), axis = 1)



	#save the wordpairs dataframe
	wordpairs_df.to_csv(wordpairs_ser_filepath, index = False)

	print('Finished saving wordpairs with ser')
	print('--------------------------------------------------------------------------------------')


if __name__ == '__main__':

	#wordpairs_filepath = 'Data/wordpairs_test.txt'
	#wordpairs_ser_filepath = 'Data/wordpairs_test_with_ser.txt'
	#add_ser(wordpairs_filepath, wordpairs_ser_filepath)

	wordpairs_filepath = '/data/users/jmahapatra/data/wordpairs.txt'
	wordpairs_ser_filepath = '/data/users/jmahapatra/data/wordpairs_with_ser.txt'
	
	add_ser(wordpairs_filepath, wordpairs_ser_filepath)


	
	




