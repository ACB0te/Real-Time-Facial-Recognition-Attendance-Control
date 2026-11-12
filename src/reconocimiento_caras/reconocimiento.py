import cv2

from src.reconocimiento_caras.deteccion import viola_jones
from src.reconocimiento_caras.embbedings import sacar_vector, probabilidades_svm, probabilidades_knn


def procesar_frame(frame, modelo, metodo="svm"):
    """Devuelve [(caja, {nombre: probabilidad}), ...], una por cara."""
    detecciones = []
    for caja in viola_jones(frame):
        v = sacar_vector(frame, caja)
        if metodo == "svm":
            probs = probabilidades_svm(modelo, v)
        else:
            probs = probabilidades_knn(modelo, v)
        detecciones.append((caja, probs))
    return detecciones


def dibujar_pistas(frame, pistas):
    """Verde: confirmada. Naranja: todavía decidiendo."""
    for p in pistas:
        if p.sin_ver > 0:          # no se ve en este fotograma
            continue
        x, y, w, h = p.caja
        if p.confirmada:
            color, texto = (0, 200, 0), f"#{p.id} {p.nombre}"
        else:
            nombre, suma = p.mejor()
            color, texto = (0, 165, 255), f"#{p.id} {nombre}? {suma:.1f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, texto, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame