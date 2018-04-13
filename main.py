import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Add
from keras.layers.recurrent import SimpleRNN
from keras.models import Model

inputs = Input(shape=(784,5,))

h1 = Dense(64, activation='relu')(inputs)
h2 = Dense(64, activation='relu')(h1)
h3 = Dense(64, activation='relu')(h2)

lb = SimpleRNN(64, go_backwards = True)(h3)
lf = SimpleRNN(64)(h3)

h4 = Add()([lb,lf]) #merge

h5 = Dense(64, activation='relu')(h4)

outputs =Dense(10, activation = 'softmax')(h5)

model = keras.models.Model(inputs = inputs, outputs = outputs)
