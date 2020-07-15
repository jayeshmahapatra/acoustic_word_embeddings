Store misc functions

import pandas
import kaldi_io
import sys,os

def scp_to_csv(source_path,destination_path):

	source_path = 'Data/'
	destination_path = 'Data/'

	source_files = list(map(lambda x: source_path+x, filter(lambda x:x.endswith("scp"),os.listdir(source_path))))
	print(source_files)

	destination_files = list(map(lambda x: destination_path+x[len(source_path):].rstrip('scp')+'csv', source_files))
	print(destination_files)

	for scp_file,destination_file in zip(source_files,destination_files):
	    #Load the data
	    keys = []
	    matrices = []
	    mat_lengths = []

	    for key,matrix in kaldi_io.read_mat_scp('Data/raw_mfcc_AMI_Segments.9.scp'):
	        keys.append(key.split('_')[1])
	        matrices.append(matrix)
	        mat_lengths.append(matrix.shape[0])
	    
	    #Truncate the dimensions of the data
	    matrices,mat_lengths = truncate_shapes(matrices,mat_lengths,max_length=100,num_mfcc_features=40)
	    
	    #Pad the matrices
	    matrices,mat_lengths = pad_sequences(matrices,n_padded = 100,center_padded = True)
	    matrices = np.transpose(matrices,(0,2,1))
	    
	    #Create a dataframe with the keys and matrices
	    df = pd.DataFrame(columns = ["key","matrix"])
	    df["keys"] = keys
	    df["matrices"] = matrices
	    
	    #Save the dataframe as csv
	    df.to_csv(destination_file, index = False)



