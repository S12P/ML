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

arg = sys.argv
fichier = str(arg[1])
freq = 161

model = load_model('my_model.h5')

fichier = sound.spectogram(fichier, freq)
