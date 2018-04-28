import numpy as np
import tensorflow as tf
import keras
import sound
from keras.layers import Input, Dense, Add
from keras.layers.recurrent import GRU
from keras.models import Model
import keras.backend as K
from keras.optimizers import SGD


def _ctc_lambda(prediction_batch, label_batch, prediction_lengths, label_lengths):
        #prediction_batch, label_batch, prediction_lengths, label_lengths = args
        return K.ctc_batch_cost(y_true=label_batch, y_pred=prediction_batch,input_length=prediction_lengths, label_length=label_lengths)

def ctc_loss(y_true, y_pred):
    input_length = np.zeros((5,2)) # a changer
    label_length = np.zeros((5,2)) # a changer

    input_length = tf.convert_to_tensor(input_length, dtype=tf.float32)
    label_length = tf.convert_to_tensor(label_length, dtype=tf.float32)

    x = _ctc_lambda(y_true, y_pred, input_length, label_length)

    return K.mean(x, axis=-1)


# model.load_weights('models/')

NB_FREQUENCIES = 9000
MAX_TIME_FRAMES = 500
MAX_LABEL_SIZE = 100

inputs = Input(shape=(MAX_TIME_FRAMES, NB_FREQUENCIES,))

h1 = Dense(64, activation='relu')(inputs)
h2 = Dense(64, activation='relu')(h1)
h3 = Dense(64, activation='relu')(h2)

lb = GRU(64, go_backwards = True, return_sequences = True)(h3)
lf = GRU(64, return_sequences = True)(h3)

h4 = Add()([lb,lf]) #merge

h5 = Dense(64, activation='relu')(h4)

outputs = Dense(10, activation = 'softmax')(h5)

model = keras.models.Model(inputs=inputs, outputs=outputs)

sgd = SGD(nesterov=True)
model.summary()
model.compile(loss=ctc_loss, metrics=['accuracy'], optimizer=sgd)
#model.summary()


# model.save_weights('models/')
