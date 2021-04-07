#Core Python, Pandas, and kaldi_io
import numpy as np
import pandas as pd
import string
from collections import Counter
import kaldi_io
import argparse
import sys
from numpy.random import default_rng
from matplotlib import pyplot as plt

#scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn.metrics
from sklearn.metrics import pairwise_distances,average_precision_score
from sklearn.metrics import PrecisionRecallDisplay,precision_recall_curve
from sklearn.metrics.pairwise import pairwise_kernels

#Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


#Callbacks
def create_callbacks(checkpoint_path, early_stopping_patience = 10):
	#Early Stopping
	es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10)

	#Model Checkpoint
	modelckpt_callback = keras.callbacks.ModelCheckpoint(
		monitor="val_loss",
		filepath=checkpoint_path,
		verbose=1,
		save_weights_only=False,
		save_best_only=True,
	)
	
	return es_callback,modelckpt_callback

# Testing Models
def test_ae(model,x_test, plot = False):

	
	#Predict based on Test Data
	y_pred = model.predict(tf.convert_to_tensor(x_test), batch_size = 128)
	
	
	#Calculate the metrics
	mse_func = tf.keras.losses.MeanSquaredError()
	mse = mse_func(x_test, y_pred).numpy()
	print("Test MSE ",mse)
	
	if plot:
		print("Plotting Random Reconstructed Traces")

		num_plots = 3
		fig, axs = plt.subplots(num_plots,2, figsize = (10,10))
		
		print("test")
		selected_trace_ids = np.random.randint(y_pred.shape[0], size = num_plots)
		
		for i in range(num_plots):
			
			trace_id = selected_trace_ids[i]
			
			#Plot Trace
			axs[i,0].imshow(x_test[trace_id].T)
			#axs[i,0].set(xlim = (0,100), ylim = (0,100))
			
			#Plot Reconstruction
			axs[i,1].imshow(y_pred[trace_id].T)
			#axs[i,1].set(xlim = (0,100), ylim = (0,100))
		print("test2")
		axs[0][0].set_title("Actual MFCC")
		axs[0][1].set_title("Reconstructed MFCC")
		fig.suptitle('MFCC Reconstructions')
		plt.show()

#Evaluate Embeddings
def evaluate_embeddings(embeddings, labels, curve = False):
	
	'''Evaluate AWEs based on Same-Different Task
	Same-Different Task is to classify if a pair of embeddings belong to the same label,
	given their distance. The metric reported is average precision.

	Arguments:
		embeddings (Numpy Array): An Array of N x embedding_dimension for embeddings of N word instances
		labels (Numpy Array): Labels of the N word instances
		curve (Boolean, default -> False) : Whether the whole precision-recall curve is to be returned

	Returns:
		average_precision (Float) : Average Precision calculated on the Same-Different Task
		curve_points (Tuple of List) : A tuple containing precision and recall points for the Precision-Recall Curve
	
	'''

	#Calculate pairwise cosine distance
	distances = pairwise_distances(embeddings, metric='cosine')
	#Calculate pairwise cosine similarity
	similarity = pairwise_kernels(embeddings, metric = 'cosine')
	
	
	
	#Create labels of whether the words are same or not
	eval_labels = (labels[:,None]==labels).astype(float)
	
	
	
	#Remove the diagonal elements (word pairs with themselves)
	mask = np.array(np.tril(np.ones((similarity.shape[0],similarity.shape[0]), dtype= int),-1),dtype = bool)
	similarity = similarity[mask]
	distances = distances[mask]
	eval_labels = eval_labels[mask]
		
	#flatten the pairwise arrays
	distances = np.ravel(distances)
	similarity = np.ravel(similarity)
	#Flatten the labels
	eval_labels = np.ravel(eval_labels)
	
	num_positive = sum(eval_labels==1)
	num_negative = eval_labels.shape[0]-num_positive
	print('The number of positive examples %d and negative examples %d'%(num_positive,num_negative))
	#Calculate the Average Precision
	average_precision = average_precision_score(eval_labels,similarity)
	if curve:
		precision, recall, _ = precision_recall_curve(eval_labels,similarity)
		curve_points = (precision, recall)
	else:
		curve_points = (None, None)

	return average_precision, curve_points



####################### Misc #######################
def limit_gpu_memory_growth(gpus):
	'''Function to limit GPU Memory Growth (Helpful in Windows to avoid runtime errors)'''
	if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


