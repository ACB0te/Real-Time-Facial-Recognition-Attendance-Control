import cv2
from pathlib import Path


# En OpenCV 5 las cascadas ya no vienen en cv2.data: se carga la copia de modelos/
RUTA_CASCADA = Path(__file__).resolve().parents[2] / "modelos" / "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(str(RUTA_CASCADA))

def viola_jones(frame, scale_factor=1.1, min_neighbors=5, tam_min=80):
    """
    Detecta caras en un fotograma con Viola-Jones.

    frame: imagen BGR (la que da cv2.VideoCapture o cv2.imread)
    Devuelve: lista de recuadros (x, y, ancho, alto), uno por cara.
    """
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris) #Aumentar el contraste de la imagen.
    caras = detector.detectMultiScale(
        gris,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(tam_min, tam_min),
    )
    return [tuple(int(v) for v in c) for c in caras]

