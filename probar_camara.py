"""
Prueba qué cámaras hay y si dan imagen.

Uso:  python probar_camara.py
"""

import cv2

for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Cámara {i}: no existe")
        continue
    ok, frame = False, None
    for _ in range(30):              # deja que arranque (~1 s)
        ok, frame = cap.read()
    if ok:
        print(f"Cámara {i}: {frame.shape[1]}x{frame.shape[0]}, brillo medio {frame.mean():.1f}")
    else:
        print(f"Cámara {i}: abierta pero no da imágenes")
    cap.release()
