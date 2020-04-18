
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, Activation
from keras.optimizers import SGD


plt.close("all")
#%%
def plot_decision_2d(X, y, classifier, resolution=0.02, titre=' '):
    # setup marker generator and color map
    markers = ('s', 'v', 'o', '^', 'x')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 0, X[:, 0].max() + 0
    x2_min, x2_max = X[:, 1].min() - 0, X[:, 1].max() + 0
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict_classes(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.figure()
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.45, c=cmap(idx),
                    marker=markers[idx], label= 'classe {}'.format(cl))
    plt.legend(loc='best')
    plt.title(titre, fontsize=12)
    
# %% extraction des donnees
# apprentissage
datatrain = pd.read_csv("mixtureexampleTrain.csv", delimiter = "\t", header=None)
Ya = datatrain[2].values
Ya[Ya < 0] = 0
Xa = datatrain.drop(2, axis=1).values
print("\nDonnees apprentissage")
print("Labels: %d" % Ya.shape[0])
print("Lignes : %d, colonnes : %d" % (Xa.shape[0], Xa.shape[1]))

# test
datatest = pd.read_csv("mixtureexampleTest.csv", delimiter = "\t", header=None)
Yt = datatest[2].values
Yt[Yt < 0] = 0
Xt = datatest.drop(2, axis=1).values
print("\nDonnees apprentissage")
print("Labels: %d" % Yt.shape[0])
print("Lignes : %d, colonnes : %d" (Xt.shape[0], Xt.shape[1]))