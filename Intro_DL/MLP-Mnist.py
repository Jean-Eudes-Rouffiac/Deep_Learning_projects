

from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, Activation
from keras.optimizers import SGD,  Adam

plt.close('all')



#%% Fonctions utiles
# Visualisation chiffre
def plot_digit(img, label=" "):
    img = img.reshape(28,28)
    plt.imshow(img, cmap=plt.get_cmap("gray"),  interpolation="nearest")
    plt.title("Chiffre : {}".format(label))
    plt.show()
    
#%% Chargement des donnees (MNIST est disponible dans la libraire Keras)
(Xa, Ya), (Xt, Yt) = mnist.load_data()
print("\nDonnees apprentissage")
print("Nb de points : %d" % Ya.shape[0])
print("Lignes : %d, colonnes : %d" % (Xa.shape[1], Xa.shape[2]))

print("\nDonnees test")
print("Nb de points : %d" % Yt.shape[0])
print("Lignes : %d, colonnes : %d" % (Xt.shape[1], Xt.shape[2]))
    
#%% Visualisation de quelques chiffres choisis aleatoirement
idx = np.random.permutation(Ya.shape[0])[0:12]
plt.figure(1)
for i in range(len(idx)):
    plt.subplot(4,3,i+1)
    plot_digit(Xa[idx[i]], Ya[idx[i]])
    
    
#%% Preparation des donnees : reshape en vecteur de taille 784
# flatten 28*28 images to a 784 vector for each image and conversion to float32 for normalization sake
nb_pixels = Xa.shape[1] * Xa.shape[2]
Xa = Xa.reshape(Xa.shape[0], nb_pixels).astype('float32')
Xt = Xt.reshape(Xt.shape[0], nb_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
Xa = Xa / 255
Xt = Xt / 255

# one-hot class encoding
nbclasses = len(np.unique(Ya))
Ya = to_categorical(Ya, nbclasses)
Yt = to_categorical(Yt, nbclasses)