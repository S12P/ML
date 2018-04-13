from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import wave

def sound(file_name):
    s = wave.open(file_name, 'r')

    nframes = s.getnframes()
    samp_rate = s.getframerate()
    duration = nframes / samp_rate

    audio = np.frombuffer(s.readframes(-1), dtype=np.int16)
    samp_rate = s.getframerate()
    s.close()
    return(audio, samp_rate, duration) # duration en seconde
