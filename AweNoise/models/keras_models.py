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



def create_lstm_autoencoder(input_shape):
	#Define Autoencoder
	
    # Simple LSTM Autoencoder
    timesteps = input_shape[0]
    latent_dim = 256

    #Model Definition
    inputs = keras.layers.Input(shape=input_shape)
    mask = keras.layers.Masking(mask_value=0) (inputs)

    encoded = keras.layers.LSTM(latent_dim)(mask)

    x = keras.layers.RepeatVector(timesteps)(encoded)
    x = keras.layers.LSTM(latent_dim, return_sequences=True)(x)
    decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_shape[-1]))(x)

    lstm_autoencoder = keras.Model(inputs, decoded)
    lstm_encoder = keras.Model(inputs, encoded)
    return lstm_autoencoder,lstm_encoder