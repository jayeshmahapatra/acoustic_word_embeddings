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

#Import User defined classes
from AweNoise.datasets import CNN_dataset
from AweNoise.keras_utils import save_keras_learning_curves, create_callbacks, test_ae_model, evaluate_embeddings
from AweNoise.keras_utils import limit_gpu_memory_growth
from AweNoise.models.keras_models import create_lstm_autoencoder




if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-n','--noisy',help = "Noisy dataset", action = "store_true")
	parser.add_argument('-ne','--num_examples', type = int, default = 11000,  help = "Intger : Number of test examples to evaluate on")
	parser.add_argument('-snr', '--snr', type = int, default = -10, help = "SNR of the AMI Noisy data (required if noisy)")
	parser.add_argument('-lg', '--limit_gpu_memory_growth', action = "store_true", help = "Limit GPU Memory Growth (helpful for windows)")
	args = parser.parse_args()

	####Check Parser Arguments ############
	parser_invalid = False #Flag to exit if parser arguments invalid
	allowed_snr_values = [-5, 0, 5, 20] #Allowed values for snr
	if args.noisy:
		if args.snr not in allowed_snr_values:
			print("Only snr values allowed are -5, 0, 5 and 20, please specify a valid snr value using the -snr command line argument")
			parser_invalid = True

	if parser_invalid:
		sys.exit()
	else:
		print("Parser Arguments Valid")

	############# GPU ###################
	print("Using Tensorflow GPU ", tf.test.is_built_with_cuda()) #If TF is using GPU
	gpus = tf.config.list_physical_devices('GPU')
	print("Using GPUs ", gpus)

	if args.limit_gpu_memory_growth:
		limit_gpu_memory_growth(gpus)


	snr, num_examples = args.snr, args.num_examples
	#Load Data using torch dataset
	train_ds = CNN_dataset(split_set = "train", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = snr, k = np.Inf, cluster = False)
	val_ds = CNN_dataset(split_set = "val", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = snr, k = np.Inf, cluster = False)
	test_ds = CNN_dataset(split_set = "test", char_threshold = 5, frequency_bounds = (0,np.Inf), snr = snr, k = np.Inf, cluster = False)


	#Get the numpy arrays and delete the dataset object (Keras only needs numpy arrays)
	x_train,train_labels = train_ds.inputs.numpy(), train_ds.labels.numpy()
	x_val,val_labels = val_ds.inputs.numpy(), val_ds.labels.numpy()
	x_test,val_labels = val_ds.inputs.numpy(), val_ds.labels.numpy()
	del train_ds,val_ds,test_ds


	#Create an AE Model
	input_shape = x_train.shape[1:]
	lstm_autoencoder, lstm_encoder = create_lstm_autoencoder(input_shape)

	#Compile the model and define the training parameters

	#Optimizer
	learning_rate = 0.001
	optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

	#Compile Model
	lstm_autoencoder.compile(optimizer = optimizer, loss = "mse")

	#Number of Epochs and Batch Size
	num_epochs = 30
	batch_size = 64

	#Model Callbacks for Checkpointing and Early Stopping

	#Model Checkpoint Path
	path_checkpoint = "lstm_autoencoder_checkpoint.h5"

	#Get the callbacks
	es_callback,modelckpt_callback = create_callbacks(path_checkpoint, early_stopping_patience = 10)

	#Model Summary
	lstm_autoencoder.summary()

	#Train the model
	history = lstm_autoencoder.fit(
	    x = x_train, y = x_train,
	    verbose = 1,
	    epochs=num_epochs,
	    batch_size = batch_size,
	    validation_data=(x_val,x_val),
	    callbacks=[es_callback, modelckpt_callback]
	)

	#Load the best model
	lstm_autoencoder = keras.models.load_model(path_checkpoint)

	#Save the learning Curves and Save
	save_keras_learning_curves(history, lc_save_path)

	#Test Model
	test_ae_model(lstm_autoencoder, x_test)

	#Evaluate the Quality of Embeddings
	test_embeddings = lstm_autoencoder.predict(x_test, batch_size = 128)
	evaluate_embeddings(test_embeddings, test_labels)

