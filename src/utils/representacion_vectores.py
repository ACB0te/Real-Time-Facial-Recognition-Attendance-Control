import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def dibujar_vectores(vectores, etiquetas=None, titulo="Vectores"):
    """
    Dibuja una lista de vectores en 2D.

    vectores:  lista o array de N vectores (del tamaño que sean)
    etiquetas: lista de N etiquetas para colorear (opcional)
    metodo:    "pca" o "tsne"
    """
    X = np.asarray(vectores)

    P = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(7, 5))
    if etiquetas is None:
        plt.scatter(P[:, 0], P[:, 1], s=25)
    else:
        etiquetas = np.asarray(etiquetas)
        for e in np.unique(etiquetas):
            m = etiquetas == e
            plt.scatter(P[m, 0], P[m, 1], s=25, label=str(e))
        plt.legend()
    plt.title(f"{titulo} ({metodo.upper()})")
    plt.show()