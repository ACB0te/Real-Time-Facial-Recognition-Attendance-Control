"""Calibración de una cámara basada en el apartado A del lab 2.

Añade fotos del tablero a calibracion/fotos_chessboard y ejecuta:
    python calibrar_camara.py --columnas 8 --filas 6 --lado-mm 30

Columnas y filas son ESQUINAS INTERIORES, no casillas. Cambia estos
valores y el lado real de las casillas para ajustarlos a tu tablero.
Toma al menos 10-15 fotos nítidas con distintas posiciones e inclinaciones,
cubriendo el centro y los bordes, con la misma resolución y enfoque que
usarás en el proyecto. No mezcles cámaras ni resoluciones.

Los resultados se guardan en calibracion/resultado/parametros.npz:
    datos = np.load("calibracion/resultado/parametros.npz")
    corregida = cv2.undistort(frame, datos["intrinsics"], datos["dist_coeffs"])
El frame debe tener la resolución guardada en image_size (ancho, alto).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent / "calibracion"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_chessboard_points(chessboard_shape, dx, dy):
    """Coordenadas 3D del patrón plano, en las unidades de dx y dy."""
    columnas, filas = chessboard_shape
    puntos = np.zeros((columnas * filas, 3), dtype=np.float32)
    puntos[:, :2] = np.mgrid[:columnas, :filas].T.reshape(-1, 2)
    puntos[:, 0] *= dx
    puntos[:, 1] *= dy
    return puntos


def calibrar(carpeta, salida, patron, lado_mm):
    rutas = sorted(p for p in carpeta.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSIONES)
    if not rutas:
        raise ValueError(f"Añade fotos del chessboard a: {carpeta}")

    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    puntos_patron = get_chessboard_points(patron, lado_mm, lado_mm)
    puntos_3d, puntos_2d, nombres, detecciones = [], [], [], []
    image_size = None

    for ruta in rutas:
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            print(f"Omitida (no se puede leer): {ruta.name}")
            continue
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        tamano = gris.shape[::-1]  # OpenCV espera (ancho, alto).
        if image_size is not None and tamano != image_size:
            raise ValueError(f"Resolución distinta en {ruta.name}: {tamano}; "
                             f"se esperaba {image_size}.")
        image_size = tamano
        encontrado, esquinas = cv2.findChessboardCorners(gris, patron)
        if not encontrado:
            print(f"Tablero no detectado: {ruta.name}")
            continue
        esquinas = cv2.cornerSubPix(gris, esquinas, (5, 5), (-1, -1), criterios)
        puntos_3d.append(puntos_patron.copy())
        puntos_2d.append(esquinas)
        nombres.append(ruta.name)
        cv2.drawChessboardCorners(imagen, patron, esquinas, True)
        detecciones.append((ruta.name + "_esquinas.png", imagen))
        print(f"Tablero detectado: {ruta.name}")

    if len(puntos_2d) < 3:
        raise ValueError("Se necesitan al menos 3 detecciones válidas. "
                         "Recomendación: 10-15 o más vistas variadas.")

    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        puntos_3d, puntos_2d, image_size, None, None
    )
    extrinsics = np.array([
        np.hstack((cv2.Rodrigues(rvec)[0], tvec))
        for rvec, tvec in zip(rvecs, tvecs)
    ])
    errores = []
    for objeto, observados, rvec, tvec in zip(puntos_3d, puntos_2d, rvecs, tvecs):
        proyectados, _ = cv2.projectPoints(objeto, rvec, tvec, intrinsics, dist_coeffs)
        errores.append(np.sqrt(np.mean(np.sum((observados - proyectados) ** 2, axis=2))))

    salida.mkdir(parents=True, exist_ok=True)
    carpeta_esquinas = salida / "esquinas"
    carpeta_esquinas.mkdir(exist_ok=True)
    for nombre, imagen in detecciones:
        destino = carpeta_esquinas / nombre
        if not cv2.imwrite(str(destino), imagen):
            raise OSError(f"No se pudo guardar {destino}")
    destino = salida / "parametros.npz"
    np.savez(destino, intrinsics=intrinsics, dist_coeffs=dist_coeffs,
             rvecs=np.asarray(rvecs), tvecs=np.asarray(tvecs), extrinsics=extrinsics,
             image_size=np.array(image_size), chessboard_shape=np.array(patron),
             square_size_mm=lado_mm, rms=rms, image_names=np.array(nombres),
             reprojection_errors_px=np.array(errores))
    print(f"\nImágenes válidas: {len(nombres)}/{len(rutas)}")
    print("Matriz intrínseca:\n", intrinsics)
    print("Coeficientes de distorsión:\n", dist_coeffs)
    print(f"Error RMS de reproyección: {rms:.4f} píxeles")
    print(f"Parámetros guardados en: {destino}")
    print(f"Comprueba visualmente las esquinas en: {carpeta_esquinas}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fotos", type=Path, default=BASE_DIR / "fotos_chessboard")
    parser.add_argument("--salida", type=Path, default=BASE_DIR / "resultado")
    parser.add_argument("--columnas", type=int, default=6)
    parser.add_argument("--filas", type=int, default=4)
    parser.add_argument("--lado-mm", type=float, default=30.0)
    args = parser.parse_args()
    if args.columnas < 2 or args.filas < 2 or not np.isfinite(args.lado_mm) or args.lado_mm <= 0:
        parser.error("El patrón debe tener al menos 2 × 2 esquinas y un lado positivo.")
    if not args.fotos.is_dir():
        parser.error(f"No existe la carpeta de fotos: {args.fotos}")
    try:
        calibrar(args.fotos, args.salida, (args.columnas, args.filas), args.lado_mm)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"Error de calibración: {error}\n")


if __name__ == "__main__":
    main()
