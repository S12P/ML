import numpy as np
import tensorflow as tf
import keras
from sound
from keras.layers import Input, Dense, Add
from keras.layers.recurrent import GRU
from keras.models import Model

# model.load_weights('models/')

NB_FREQUENCIES = 9000
MAX_TIME_FRAMES = 500

inputs = Input(shape=(NB_FREQUENCIES, MAX_TIMES_FRAMES,))

h1 = Dense(64, activation='relu')(inputs)
h2 = Dense(64, activation='relu')(h1)
h3 = Dense(64, activation='relu')(h2)

lb = GRU(64, go_backwards = True)(h3)
lf = GRU(64)(h3)

h4 = Add()([lb,lf]) #merge

h5 = Dense(64, activation='relu')(h4)

outputs = Dense(10, activation = 'softmax')(h5)

model = keras.models.Model(inputs=inputs, outputs=outputs)


# model.save_weights('models/')
