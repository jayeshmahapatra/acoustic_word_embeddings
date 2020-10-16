import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import pairwise_kernels,paired_distances


def generate_embedding_similarity(wordpairs_filepath = '/data/users/jmahapatra/data/wordpairs.txt', word_embedding_dict_filepath = '/data/users/jmahapatra/data/word_embedding_dict.npy', embedding_similarity_filepath = '/data/users/jmahapatra/data/wordpairs_embedding_similarity.txt',metric = 'cosine'):

	#Load the wordpairs
	wordpairs_df = pd.read_csv(wordpairs_filepath)

	wordpairs = list(zip(wordpairs_df["word_1"].to_list(),wordpairs_df["word_2"].to_list()))

	del wordpairs_df

	#Load the embedding dict
	word_embedding_dict = np.load(word_embedding_dict_filepath, allow_pickle = True)

	embedding_similarity_dict = {}
	embedding_similarity_dict["word_1"] = []
	embedding_similarity_dict["word_2"] = []
	embedding_similarity_dict["%s_similarity"%(metric)] = []


	for i,wordpair in enumerate(wordpairs):
	    
	    word_1_embedding = word_embedding_dict.item().get(wordpair[0]).squeeze().reshape(1,-1)
	    word_2_embedding = word_embedding_dict.item().get(wordpair[1]).squeeze().reshape(1,-1)
	    similarity = pairwise_kernels(word_1_embedding,word_2_embedding, metric = metric )
	    
	    embedding_similarity_dict["word_1"].append(wordpair[0])
	    embedding_similarity_dict["word_2"].append(wordpair[1])
	    embedding_similarity_dict["%s_similarity"%(metric)].append(similarity.item())

	    #if(i==1000):
	    #	break
	    

	embedding_similarity_df = pd.DataFrame(embedding_similarity_dict)
	embedding_similarity_df.to_csv(embedding_similarity_filepath, index = False)

	print("Finished Generating Embedding Similarity")
	print('--------------------------------------------------------------------------------------')


if __name__ == '__main__':


	#wordpairs_filepath = 'Data/wordpairs_test.txt'
	#word_embedding_dict_filepath = 'Data/word_embedding_dict.npy'
	#embedding_similarity_filepath = 'Data/wordpairs_test_embedding_similarity.txt'

	wordpairs_filepath = '/data/users/jmahapatra/data/wordpairs.txt'
	word_embedding_dict_filepath = '/data/users/jmahapatra/data/word_embedding_dict.npy'
	embedding_similarity_filepath = '/data/users/jmahapatra/data/wordpairs_embedding_similarity.txt'


	generate_embedding_similarity(wordpairs_filepath,word_embedding_dict_filepath,embedding_similarity_filepath)

