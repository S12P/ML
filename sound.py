from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

import numpy as np
import wave


def spectogram(file, freq):
    window_size=20
    step_size=10
    sample_rate, audio = wavfile.read(file)
    nperseg = window_size * sample_rate / 1e3
    noverlap = freq
    f, t, spectogram = signal.spectrogram(audio, fs=sample_rate, window='hann', nperseg=int(nperseg), noverlap=int(noverlap), detrend=False)
    #print(f.shape, t.shape, spectogram.shape)
    return(spectogram.T)
