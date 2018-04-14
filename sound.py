from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import wave

def sound(file_name):
    """
    input : wav file

    output : (audio : n frames of audio ) (framerate : sampling frequency) (duration : duration in seconde)

    """
    file = wave.open(file_name, 'r')

    nframes = file.getnframes()
    framerate = file.getframerate()
    duration = nframes / framerate

    audio = np.frombuffer(file.readframes(-1), dtype=np.int16)

    file.close()

    return(audio, framerate, duration)

print(sound("yes.wav"))
