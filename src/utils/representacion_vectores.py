import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def dibujar_vectores(caras, titulo="Vectores"):
    """
    Dibuja los vectores en 2D, con un color por persona.

    caras:  diccionario {nombre: [vector1, vector2, ...]}
    metodo: "pca" o "tsne"
    """
    nombres = list(caras)
    X = np.vstack([np.asarray(caras[n]) for n in nombres])

    P = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(7, 5))
    inicio = 0
    for n in nombres:
        fin = inicio + len(caras[n])
        plt.scatter(P[inicio:fin, 0], P[inicio:fin, 1], s=25, label=n)
        inicio = fin
    plt.legend()
    plt.title(f"{titulo}")
    plt.show()