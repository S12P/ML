# Machine Learning Project

We can find the article [here](https://arxiv.org/pdf/1412.5567.pdf)

We can find resources [here](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz)


To train the model it is necessary to put the folder of the audio files in the same folder that train.py and to change the name of nom_du_dossier in the file outils_entrainement.py

And then we can train the model with :

python3 train.py

To test it is necessary to put the audio files in the folder examples and execute :

python3 test.py

If you want to calculate again the weigth change b to False in test.py