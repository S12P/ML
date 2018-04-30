import numpy as np
import tensorflow as tf
import keras
import sound
import outils_entrainement as tt # stands for train tools
from keras.layers import Input, Dense, Add, Lambda, TimeDistributed
from keras.layers.recurrent import GRU
from keras.models import Model
import keras.backend as K
from keras.optimizers import SGD
from keras.models import load_model
import sys

def ctc_loss_lambda(args):
    y_pred, y_true, input_length, label_length = args

    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def ctc(y_true, y_pred):
    return K.mean(y_pred)

def clipped_relu(x):
    return keras.activations.relu(x, max_value=20)

arg = sys.argv
fichier = str(arg[1])
freq = 161

NB_FREQUENCIES = 161

inputs = Input(shape=(None, NB_FREQUENCIES), name='main_input')
labels = Input(shape=(None,), name='labels')
input_length = Input(shape=(1,), name='input_length')
label_length = Input(shape=(1,), name='label_length')


h1 = TimeDistributed(Dense(128, activation=clipped_relu))(inputs)
h2 = TimeDistributed(Dense(128, activation=clipped_relu))(h1)
h3 = TimeDistributed(Dense(128, activation=clipped_relu))(h2)


lb = GRU(128, go_backwards = True, return_sequences = True)(h3)
lf = GRU(128, return_sequences = True)(h3)


h4 = Add()([lb,lf]) # add the two layers

h5 = TimeDistributed(Dense(128, activation=clipped_relu))(h4)
h6 = TimeDistributed(Dense(29, activation='softmax'), name='aux_output')(h5)

loss_out = Lambda(ctc_loss_lambda, output_shape=(1, ), name='main_output')([h6, labels, input_length, label_length])

model = keras.models.Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out, h6])
model.summary()

sgd = SGD(nesterov=True)

model.compile(loss={'main_output': ctc, 'aux_output': lambda x, y: K.constant([0])}, metrics=['accuracy'], optimizer=sgd)

batch, lab, input_len, lab_len = tt.get_sound_examples('t.wav')
#out = K.ctc_decode(
model.predict([batch, lab, input_len, lab_len])
#[1], input_len)

fichier = sound.spectogram(fichier, freq)

output = model.predict(fichier)


