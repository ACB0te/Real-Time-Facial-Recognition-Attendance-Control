import cv2

from src.reconocimiento_caras.embbedings import cargar_svm
from src.reconocimiento_caras.reconocimiento import procesar_frame, dibujar_resultados

svm = cargar_svm()
CAMARA = 1   # en este Mac la 0 da imagen en negro (cámara virtual); la 1 es la webcam
cap = cv2.VideoCapture(CAMARA, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 1920x1080 es más lento de procesar
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    resultados = procesar_frame(frame, svm, metodo="svm")
    dibujar_resultados(frame, resultados)

    cv2.imshow("FaceCheck", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()