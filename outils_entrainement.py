import os
import sound
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

def audio_train(nom_du_dossier, freq):
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

    print('All the contents of {} has been loaded!'.format(nom_du_dossier))

    return(dic)
print(audio_train(nom_du_dossier, 200))

def text_to_number(text):
    """
    This function takes a text and encodes it as a list of number according to the the dictionary map_char
    """
    return [map_char[c] for c in text]
