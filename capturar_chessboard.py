"""Captura hasta 30 fotos del tablero pulsando Espacio.

Uso: python capturar_chessboard.py
     python capturar_chessboard.py --camara 0 --cantidad 30

Con la ventana de cámara seleccionada, pulsa Espacio para guardar una foto.
Pulsa Q o Escape para terminar antes. Mueve e inclina el tablero entre
capturas, manteniéndolo completo y quieto cuando se tome cada foto.
La resolución por defecto coincide con main.py (1280 x 720).
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2


CARPETA = Path(__file__).resolve().parent / "calibracion" / "fotos_chessboard"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camara", type=int, default=1 if sys.platform == "darwin" else 0)
    parser.add_argument("--cantidad", type=int, default=30)
    parser.add_argument("--ancho", type=int, default=1280)
    parser.add_argument("--alto", type=int, default=720)
    args = parser.parse_args()
    if args.cantidad < 1 or args.ancho < 1 or args.alto < 1:
        parser.error("La cantidad y la resolución deben ser positivas.")

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camara, backend)
    ventana = "Fotos chessboard - Espacio: foto | Q: salir"
    guardadas = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {args.camara}. Prueba --camara 1.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.alto)
        CARPETA.mkdir(parents=True, exist_ok=True)
        resolucion = None
        print(f"Destino: {CARPETA}\nPulsa Espacio para guardar una foto; Q o Escape para salir.")

        while guardadas < args.cantidad:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("La cámara dejó de proporcionar imágenes.")
            tamano = (frame.shape[1], frame.shape[0])
            if resolucion is None:
                resolucion = tamano
                print(f"Resolución real de captura: {tamano[0]} x {tamano[1]}")
            elif tamano != resolucion:
                raise RuntimeError("La cámara cambió de resolución durante la captura.")

            preview = frame.copy()
            texto = f"Fotos: {guardadas}/{args.cantidad} | Espacio: foto | Q: salir"
            cv2.putText(preview, texto, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(preview, texto, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(ventana, preview)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), ord("Q"), 27):
                break
            if tecla == ord(" "):
                marca = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                destino = CARPETA / f"chessboard_{marca}.png"
                # Guardamos el frame original, sin el texto de la vista previa.
                if not cv2.imwrite(str(destino), frame):
                    raise OSError(f"No se pudo guardar {destino}")
                guardadas += 1
                print(f"Foto {guardadas}/{args.cantidad}: {destino.name}")
            if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                break
    except (RuntimeError, OSError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Fotos guardadas en esta sesión: {guardadas}")


if __name__ == "__main__":
    main()
