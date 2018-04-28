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

def ctc_loss_lambda(args):
    y_pred, y_true, input_length, label_length = args

    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def ctc(y_true, y_pred):
    return y_pred



# model.load_weights('models/')

NB_FREQUENCIES = 161

inputs = Input(shape=(None, NB_FREQUENCIES), name='main_input')
labels = Input(shape=(None,), name='labels')
input_length = Input(shape=(1,), name='input_length')
label_length = Input(shape=(1,), name='label_length')


h1 = TimeDistributed(Dense(64, activation='relu'))(inputs)
h2 = TimeDistributed(Dense(64, activation='relu'))(h1)
h3 = TimeDistributed(Dense(64, activation='relu'))(h2)


lb = GRU(64, go_backwards = True, return_sequences = True)(h3)
lf = GRU(64, return_sequences = True)(h3)


h4 = Add()([lb,lf]) #merge

h5 = TimeDistributed(Dense(64, activation='relu'))(h4)
h6 = TimeDistributed(Dense(29, activation='softmax'))(h5)

loss_out = Lambda(ctc_loss_lambda, output_shape=(1, ))([h6, labels, input_length, label_length])

model = keras.models.Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
model.summary()

sgd = SGD(nesterov=True)

model.compile(loss=ctc, metrics=['accuracy'], optimizer=sgd)

batch, lab, input_len, lab_len = tt.get_batch()

[x_train, x_test] = np.split(batch, [int(.8 * len(batch))])
[y_train, y_test] = np.split(lab, [int(.8 * len(batch))])
[input_len_train, input_len_test] = np.split(input_len, [int(.8 * len(batch))])
[lab_len_train, lab_len_test] = np.split(lab_len, [int(.8 * len(batch))])


model.fit([x_train, y_train, input_len_train, lab_len_train], y_train, batch_size=100, epochs=1)

score = model.evaluate([x_test, y_test, input_len_test, lab_len_test], y_test)

print('The final score is {}'.format(score))



# model.save_weights('models/')
