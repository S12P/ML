import numpy as np
import tensorflow as tf
import keras
import sound
from keras.layers import Input, Dense, Add
from keras.layers.recurrent import GRU
from keras.models import Model
import keras.backend as K
from keras.optimizers import SGD


def ctc_loss(y_true, y_pred):
    input_length = np.array([[2], [2]]) # a changer
    label_length = np.array([[2], [2]]) # a changer

    input_length = tf.convert_to_tensor(input_length, dtype=tf.float32)
    label_length = tf.convert_to_tensor(label_length, dtype=tf.float32)
    return K.mean(K.ctc_batch_cost(tf.squeeze(y_true), y_pred, input_length, label_length), axis=-1)


# model.load_weights('models/')

NB_FREQUENCIES = 2
MAX_TIME_FRAMES = 2
MAX_LABEL_SIZE = 100

inputs = Input(shape=(MAX_TIME_FRAMES, NB_FREQUENCIES))

h1 = Dense(64, activation='relu')(inputs)
h2 = Dense(64, activation='relu')(h1)
h3 = Dense(64, activation='relu')(h2)

lb = GRU(64, go_backwards=True, return_sequences=True)(h3)
lf = GRU(64, return_sequences=True)(h3)

h4 = Add()([lb,lf]) #merge

h5 = Dense(64, activation='relu')(h4)

outputs = Dense(29, activation = 'softmax')(h5)

model = keras.models.Model(inputs=inputs, outputs=outputs)
model.summary()

sgd = SGD(nesterov=True)
model.compile(loss=ctc_loss, metrics=['accuracy'], optimizer=sgd)

ex1 = np.random.random_sample((2,2))
ex2 = np.random.random_sample((2,2))
batch = np.array([ex1, ex2])

truth = np.array([[[0, 1]], [[0,1]]])

model.fit(batch, truth, epochs=1)


# model.save_weights('models/')
