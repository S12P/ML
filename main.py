import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras.layers.recurrent import SimpleRNN
from keras.models import Model

inputs = Input(shape=(784,5,))

h1 = Dense(64, activation='relu')(inputs)
h2 = Dense(64, activation='relu')(h1)
h3 = Dense(64, activation='relu')(h2)

lb = SimpleRNN(64, go_backwards = True)(h3)
lf = SimpleRNN(64)(h3)

h4 = keras.layers.Add()([lb,lf]) #merge

h5 = Dense(64, activation='relu')(h4)

#output = ?


def h_6(k, l, t, W, b, input):
    """k est la colonne"""
    den = 0
    for k in range(len(W[6])):
        den += np.exp(np.dot(W[6].transpose()[k]*h(5, t, W, b, input)) + b[6][k])
    return( (np.exp(np.dot(W[6].transpose()[k]*h(5, t, W, b, input)) + b[6][k])) / den)
