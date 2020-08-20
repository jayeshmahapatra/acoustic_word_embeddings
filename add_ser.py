#Script to load the wordpairs dataframes and add the phonetic and orthographic symbol error rate w.r.t word_1 and word_2
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import pairwise_kernels,paired_distances


def add_ser(wordpairs_filepath = '/data/users/jmahapatra/data/wordpairs.txt', word_phoneme_dict_filepath = '/data/users/jmahapatra/data/word_phoneme_dict.npy',wordpairs_ser_filepath = '/data/users/jmahapatra/data/wordpairs_with_ser.txt'):

	#Read the wordpairs file
	wordpairs_df = pd.read_csv(wordpairs_filepath)



	#Read the Word phoneme dict
	word_phoneme_dict = np.load(word_phoneme_dict_filepath, allow_pickle = True)


	print('Calculating phonetic ser')

	#Add the phonetic ser w.r.t word_1 and word_2 respectively
	wordpairs_df["word_1_phonetic_ser"] = wordpairs_df.apply(lambda row: row["phonetic_edit_distance"]/len(word_phoneme_dict.item().get(row["word_1"]).split()), axis = 1)
	wordpairs_df["word_2_phonetic_ser"] = wordpairs_df.apply(lambda row: row["phonetic_edit_distance"]/len(word_phoneme_dict.item().get(row["word_2"]).split()), axis = 1)

	print('Calculating orthographic ser')

	#Add the orthographic ser w.rt word_1 and word_2
	wordpairs_df["word_1_orthographic_ser"] = wordpairs_df.apply(lambda row: row["orthographic_edit_distance"]/len(row["word_1"]), axis = 1)
	wordpairs_df["word_2_orthographic_ser"] = wordpairs_df.apply(lambda row: row["orthographic_edit_distance"]/len(row["word_2"]), axis = 1)

	wordpairs = list(zip(wordpairs_df["word_1"].to_list(),wordpairs_df["word_2"].to_list()))

	print('Finished Calculating SER, Calculating Cosine similarity')

	metric = 'cosine'

	#Load the word embedding dict
	word_embedding_dict_filepath = '/data/users/jmahapatra/data/word_embedding_dict.npy'
	word_embedding_dict = np.load(word_embedding_dict_filepath, allow_pickle = True)

	embedding_similarity = []

	for wordpair in wordpairs:
		word_1_embedding = word_embedding_dict.item().get(wordpair[0]).squeeze().reshape(1,-1)
		word_2_embedding = word_embedding_dict.item().get(wordpair[1]).squeeze().reshape(1,-1)
		similarity = pairwise_kernels(word_1_embedding,word_2_embedding, metric = metric )
		
		embedding_similarity.append(similarity.item())

	wordpairs_df["%s_similarity"%(metric)] = embedding_similarity

	#save the wordpairs dataframe
	wordpairs_df.to_csv(wordpairs_ser_filepath, index = False)

	print('Finished saving wordpairs with ser')
	print('--------------------------------------------------------------------------------------')


if __name__ == '__main__':

	#wordpairs_filepath = 'Data/wordpairs_test.txt'
	#word_phoneme_dict_filepath = 'Data/word_phoneme_dict.npy'
	#wordpairs_ser_filepath = 'Data/bla_wordpairs_test_with_ser.txt'

	wordpairs_filepath = '/data/users/jmahapatra/data/wordpairs.txt'
	word_phoneme_dict_filepath = '/data/users/jmahapatra/data/word_phoneme_dict.npy'
	wordpairs_ser_filepath = '/data/users/jmahapatra/data/wordpairs_with_ser.txt'
	
	add_ser(wordpairs_filepath, word_phoneme_dict_filepath, wordpairs_ser_filepath)


	
	




