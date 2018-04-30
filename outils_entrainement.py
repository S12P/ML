import os
import sound
import numpy as np
from map_char import *

#https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz


rm = [".", "_", "LICENSE"] # elements pour supprimer dossier inutile
nom_du_dossier = "data-speech_commands_v0.01"
dic = dict() #dictionnaire avec cle = resultat et valeurs = liste des spectogram de tous les fichiers audio


def test(elmt, rm):
    for k in range(len(rm)):
        if rm[k] in elmt:
            return(False)
    return(True)

def audio_train(nom_du_dossier, freq=200):
    files = os.listdir(nom_du_dossier)
    for k in range(len(files)):
        if test(files[k], rm):
            dic[files[k]] = []
    n = len(dic)

    for (keys, i) in zip(dic, range(n)):
        print('Loading examples for {}... [{}/{}]'.format(keys, i+1, n))
        nom_chemin = nom_du_dossier + "/" + str(keys)
        audio = os.listdir(nom_chemin)
        for k in range(len(audio)):
            nom_fichier = nom_chemin + "/" + str(audio[k])
            audio[k] = sound.spectogram(nom_fichier, freq)
        dic[keys] = audio
        print('Finished loading examples for {}!'.format(keys))

    print('\nAll the contents of {} has been loaded!\n'.format(nom_du_dossier))

    return(dic)

def text_to_number(text):
    """
    This function takes a text and encodes it as a list of number according to the the dictionary map_char
    """
    return [map_char[c] for c in text]

def int_list_to_text(int_list):
    """
    This function takes an it list which encodes a string and returns the associated string
    """
    str = ""
    for c in int_list:
        str = str + int_to_char[c]

    return stc


def dict_to_examples(dataset_dict):
    """
    This functions takes an array mapping a word to all its spectrograms and outputs a tuple containing and array with all the spectrograms and an array with the labels.
    It also regularises the data so all the labels have the same length, all the spectrograms have the same numbers of time slices.

    """

    keys = [key for key in dataset_dict]

    # Get the max number of time slices
    max_time_slices = max([len(spectro) for key in keys for spectro in dataset_dict[key]])
    print('The greatest number of time slices is {}'.format(max_time_slices))

    # Get the max length of the labels
    max_len_labels = max([len(key) for key in keys])

    # Get the number of frequencies
    nb_freqs = len(dataset_dict[keys[0]][0][0])
    print('The number of frequencies is {}'.format(nb_freqs))

    # Now put all the examples in a numpy array for the features, an other array for the labels
    batch = []
    labels = []

    print('Creating the batch...')

    for key in keys:
        for spectro in dataset_dict[key]:
            batch.append(np.concatenate([spectro, np.zeros((max_time_slices - len(spectro), nb_freqs))]))
            labels.append(text_to_number(key) + [0 for k in range(max_len_labels - len(key))])

    print('Done!\n')

    # Create input_length array and labels_length
    batch_size = len(batch)
    input_length = batch_size * [max_time_slices]
    labels_length = batch_size * [max_len_labels]

    return (np.array(batch), np.array(labels), np.array(input_length), np.array(labels_length))



def shuffle(batch, labels):
    """
    This function shuffles the data
    """

    assert(len(batch) == len(labels))
    p = np.random.permutation(len(labels))

    return (batch[p], labels[p])


def get_batch():
    """
    This function returns a batch
    """
    dictionary = audio_train(nom_du_dossier)

    batch, labels, input_length, labels_length = dict_to_examples(dictionary)
    batch, labels = shuffle(batch, labels)


    return (batch, labels, input_length, labels_length)


def get_sound_examples(dirname, freq=200):
    spectrograms = []
    audio = os.listdir(dirname)
    for k in range(len(audio)):
        filename = dirname + "/" + str(audio[k])
        audio[k] = sound.spectogram(filename, freq)

    examples = dict()
    examples['tests'] = audio

    return dict_to_examples(examples)
