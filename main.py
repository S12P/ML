import numpy as np
import tensorflow as tf
import keras
import sound
from keras.layers import Input, Dense, Add, Lambda
from keras.layers.recurrent import GRU
from keras.models import Model
import keras.backend as K
from keras.optimizers import SGD


def ctc_loss_lambda(args):
    y_pred, y_true, input_length, label_length = args

    return K.mean(K.ctc_batch_cost(y_true, y_pred, input_length, label_length), axis=-1)

def ctc(y_true, y_pred):
    return y_pred


# model.load_weights('models/')

NB_FREQUENCIES = 2
MAX_TIME_FRAMES = 2
MAX_LABEL_SIZE = 100

inputs = Input(shape=(MAX_TIME_FRAMES, NB_FREQUENCIES))
labels = Input(shape=[None])
input_length = Input(shape=[1,])
label_length = Input(shape=[1,])

h1 = Dense(64, activation='relu')(inputs)
h2 = Dense(64, activation='relu')(h1)
h3 = Dense(64, activation='relu')(h2)

lb = GRU(64, go_backwards=True, return_sequences=True)(h3)
lf = GRU(64, return_sequences=True)(h3)

h4 = Add()([lb,lf]) #merge

h5 = Dense(64, activation='relu')(h4)
h6 = Dense(29, activation='softmax')(h5)

loss_out = Lambda(ctc_loss_lambda, output_shape=(1, ))([h6, labels, input_length, label_length])

model = keras.models.Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
model.summary()

sgd = SGD(nesterov=True)
model.compile(loss=ctc, metrics=['accuracy'], optimizer=sgd)

ex1 = np.random.random_sample((2,2))
ex2 = np.random.random_sample((2,2))
batch = np.array([ex1, ex2])

lab = np.array([[1,2],[1,2]])
input_len = np.array([2, 2])
lab_len = np.array([2, 2])

model.fit([batch, lab, input_len, lab_len], lab)


# model.save_weights('models/')
