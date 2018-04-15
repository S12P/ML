import os
import sound

#https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz


rm = [".", "_", "LICENSE"] # elements pour supprimer dossier inutile
nom_du_dossier = "data-speech_commands_v0.01"
dic = dict() #dictionnaire avec cle = resultat et valeurs = liste des spectogram de tous les fichiers audio


def test(elmt, rm):
    for k in range(len(rm)):
        if rm[k] in elmt:
            return(False)
    return(True)

def audio_train(nom_du_dossier):
    files = os.listdir(nom_du_dossier)
    for k in range(len(files)):
        if test(files[k], rm):
            dic[files[k]] = []
    for keys in dic:
        nom_chemin = nom_du_dossier + "/" + str(keys)
        audio = os.listdir(nom_chemin)
        for k in range(len(audio)):
            nom_fichier = nom_chemin + "/" + str(audio[k])
            audio[k] = sound.spectogram(nom_fichier)
        dic[keys] = audio
        print(dic)
    return(dic)

print(audio_train(nom_du_dossier))
