import cv2
import numpy as np
from pathlib import Path
import joblib
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Ajusta la ruta a donde tengáis el modelo
RUTA_SFACE = Path(__file__).resolve().parents[2] / "modelos" / "SFace" / "face_recognition_sface_2021dec.onnx"
red = cv2.FaceRecognizerSF.create(str(RUTA_SFACE), "")
# Carpeta donde se guarda el modelo de k-means
CARPETA_KNN = Path(__file__).resolve().parents[2] / "modelos" / "knn"
# Carpeta donde se guarda el svm
CARPETA_SVM = Path(__file__).resolve().parents[2] / "modelos" / "svm"

def sacar_vector(frame, caja, margen=0.1):
    """
    Saca el vector (embedding) de una cara.

    frame: imagen BGR completa
    caja:  (x, y, w, h) que devuelve viola_jones
    Devuelve: vector de 128 números con norma 1
    """
    x, y, w, h = caja
    m = int(margen * w)
    x0, y0 = max(0, x - m), max(0, y - m) #LLega muy ampliada de viola-jons
    recorte = frame[y0:y + h + m, x0:x + w + m]

    cara = cv2.resize(recorte, (112, 112))     # tamaño de entrada de SFace
    vector = red.feature(cara).flatten()       # (1, 128) → (128,)
    return vector / np.linalg.norm(vector)


#Modelos:



def entrenar_svm(caras, nombre_modelo="svm.joblib"):
    """
    Entrena una SVM con los vectores del registro y la guarda.

    caras: diccionario {nombre: [vector1, vector2, ...]}
    Devuelve: la SVM entrenada
    """
    # 1. Pasar el diccionario a X (vectores) e y (nombres)
    X, y = [], []
    for nombre, vectores in caras.items():
        X.extend(vectores)
        y.extend([nombre] * len(vectores))
    X = np.asarray(X)
    y = np.asarray(y)

    # 2. Entrenar
    # La calibración de probabilidades usa validación cruzada: no puede haber
    # más particiones que vectores tenga la persona con menos fotos.
    _, cuentas = np.unique(y, return_counts=True)
    minimo = int(cuentas.min())
    if minimo < 2:
        raise ValueError("Cada persona necesita al menos 2 vectores para entrenar la SVM")
    if minimo < 5:
        print(f"Aviso: la persona con menos fotos tiene {minimo} vectores (se recomiendan 5 o más)")

    svm = CalibratedClassifierCV(SVC(kernel="linear", C=10), ensemble=False, cv=min(5, minimo))
    svm.fit(X, y)

    # 3. Guardar
    CARPETA_SVM.mkdir(parents=True, exist_ok=True)
    joblib.dump(svm, CARPETA_SVM / nombre_modelo)
    print(f"SVM entrenada con {len(X)} vectores de {len(caras)} personas")

    return svm

def cargar_svm(nombre_modelo="svm.joblib"):
    """Carga la SVM entrenada desde modelos/svm/."""
    return joblib.load(CARPETA_SVM / nombre_modelo)


def clasificar_svm(svm, vector):
    """
    Clasifica un vector con la SVM.

    svm:    la SVM entrenada (de entrenar_svm o cargar_svm)
    vector: vector de 128 de una cara (de sacar_vector)
    Devuelve: (nombre, probabilidad)
    """
    # 1. La SVM espera una matriz (filas = vectores), no un vector suelto
    X = vector.reshape(1, -1)

    # 2. Probabilidad de cada persona
    probs = svm.predict_proba(X)[0]

    # 3. La persona con más probabilidad
    i = np.argmax(probs)
    return str(svm.classes_[i]), float(probs[i])

#knn con similitud coseno

def guardar_knn(caras, nombre_archivo="knn.npz"):
    """
    Guarda todos los vectores del registro con sus nombres.

    caras: diccionario {nombre: [vector1, vector2, ...]}
    """
    X, y = [], []
    for nombre, lista in caras.items():
        X.extend(lista)
        y.extend([nombre] * len(lista))

    CARPETA_KNN.mkdir(parents=True, exist_ok=True)
    np.savez(CARPETA_KNN / nombre_archivo, X=np.asarray(X), y=np.asarray(y))
    print(f"Guardados {len(X)} vectores de {len(caras)} personas")


def cargar_knn(nombre_archivo="knn.npz"):
    """Carga los vectores y etiquetas. Devuelve (X, y)."""
    datos = np.load(CARPETA_KNN / nombre_archivo)
    return datos["X"], datos["y"]


def clasificar_knn(knn, vector, k=5):
    """
    k-NN con similitud del coseno.

    knn:    (X, y) que devuelve cargar_knn
    vector: vector de 128 de una cara (de sacar_vector)
    Devuelve: (nombre, fracción de votos)
    """
    X, y = knn

    # 1. Similitud con todos y los k más parecidos
    sims = X @ vector
    vecinos = np.argsort(sims)[::-1][:k]

    # 2. Contar votos
    votos = {}
    for i in vecinos:
        nombre = str(y[i])
        votos[nombre] = votos.get(nombre, 0) + 1

    # 3. El que tenga más votos
    ganador = max(votos, key=votos.get)
    return ganador, votos[ganador] / k