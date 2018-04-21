import numpy as np
import tensorflow as tf
import keras
import sound
from keras.layers import Input, Dense, Add
from keras.layers.recurrent import GRU
from keras.models import Model
import keras.backend as Kfrom keras.optimizers import SGD


def ctc_loss(y_true, y_pred):
    input_length = np.zeros((5, 1)) # a changer
    label_length = np.zeros((5, 1)) # a changer

    input_length = tf.convert_to_tensor(input_length, dtype=tf.float32)
    label_length = tf.convert_to_tensor(label_length, dtype=tf.float32)
    return K.mean(K.ctc_batch_cost(y_true, y_pred, input_length, label_length), axis=-1)


# model.load_weights('models/')

NB_FREQUENCIES = 9000
MAX_TIME_FRAMES = 500
MAX_LABEL_SIZE = 100

inputs = Input(shape=(NB_FREQUENCIES, MAX_TIME_FRAMES,))

h1 = Dense(64, activation='relu')(inputs)
h2 = Dense(64, activation='relu')(h1)
h3 = Dense(64, activation='relu')(h2)

lb = GRU(64, go_backwards = True)(h3)
lf = GRU(64)(h3)

h4 = Add()([lb,lf]) #merge

h5 = Dense(64, activation='relu')(h4)

outputs = Dense(10, activation = 'softmax')(h5)

model = keras.models.Model(inputs=inputs, outputs=outputs)

sgd = SGD(nesterov=True)
model.compile(loss=ctc_loss, metrics=['accuracy'], optimizer=sgd)



# model.save_weights('models/')
