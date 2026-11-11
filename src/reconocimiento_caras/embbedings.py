import cv2
import numpy as np
from pathlib import Path
import joblib
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# Ajusta la ruta a donde tengáis el modelo
RUTA_SFACE = Path("modelos") / "face_recognition_sface_2021dec.onnx"
red = cv2.FaceRecognizerSF.create(str(RUTA_SFACE), "")
# Carpeta donde se guarda el modelo de k-means
CARPETA_KNN = Path(__file__).resolve().parents[1] / "modelos" / "knn"
# Carpeta donde se guarda el svm
CARPETA_SVM = Path(__file__).resolve().parents[1] / "modelos" / "svm"

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
    svm = SVC(kernel="linear", C=10, probability=True)
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


