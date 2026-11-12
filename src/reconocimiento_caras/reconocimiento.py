import cv2

from src.reconocimiento_caras.deteccion import viola_jones
from src.reconocimiento_caras.embbedings import sacar_vector, clasificar_svm, clasificar_knn


def procesar_frame(frame, modelo, metodo="svm"):
    """
    Detecta y reconoce todas las caras de un fotograma.

    frame:  imagen BGR
    modelo: la SVM (de cargar_svm) o el k-NN (de cargar_knn)
    metodo: "svm" o "knn"
    Devuelve: lista de (caja, nombre, puntuación), una por cara
    """
    resultados = []
    for caja in viola_jones(frame):
        v = sacar_vector(frame, caja)
        if metodo == "svm":
            nombre, punt = clasificar_svm(modelo, v)
        else:
            nombre, punt = clasificar_knn(modelo, v)
        resultados.append((caja, nombre, punt))
    return resultados


def dibujar_resultados(frame, resultados):
    """Dibuja un recuadro y el nombre de cada cara sobre el fotograma."""
    for (x, y, w, h), nombre, punt in resultados:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{nombre} {punt:.2f}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame