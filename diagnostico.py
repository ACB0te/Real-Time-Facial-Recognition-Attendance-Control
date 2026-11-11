"""
Diagnóstico del registro: ¿los vectores separan bien a las personas?

1. Guarda en diagnostico/ el recorte de cara que ha usado cada foto (para revisarlos a ojo).
2. Similitud media dentro de cada persona y con las demás.
3. Fotos "raras": las que se parecen poco a su propia persona.
4. Acierto dejando una foto fuera (cada foto se clasifica con las demás).

Uso:  python diagnostico.py
"""

from pathlib import Path

import cv2
import numpy as np

from src.reconocimiento_caras.deteccion import viola_jones
from src.reconocimiento_caras.embbedings import sacar_vector

CARPETA_DATOS = Path(__file__).resolve().parent / "datos"
CARPETA_SALIDA = Path(__file__).resolve().parent / "diagnostico"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    CARPETA_SALIDA.mkdir(exist_ok=True)
    vectores, nombres, fotos = [], [], []

    # 1. Vectores y recortes
    for carpeta in sorted(CARPETA_DATOS.iterdir()):
        if not carpeta.is_dir():
            continue
        for foto in sorted(carpeta.iterdir()):
            if foto.suffix.lower() not in EXTENSIONES:
                continue
            img = cv2.imread(str(foto))
            if img is None:
                continue
            caras = viola_jones(img)
            if len(caras) != 1:
                continue
            x, y, w, h = caras[0]
            cv2.imwrite(str(CARPETA_SALIDA / f"{carpeta.name}_{foto.stem}.jpg"), img[y:y + h, x:x + w])
            vectores.append(sacar_vector(img, caras[0]))
            nombres.append(carpeta.name)
            fotos.append(foto.name)

    X, y = np.asarray(vectores), np.asarray(nombres)
    S = X @ X.T                      # similitud del coseno entre todas las fotos
    personas = list(dict.fromkeys(nombres))
    print(f"Recortes guardados en {CARPETA_SALIDA}/  ({len(X)} fotos)\n")

    # 2. Similitud dentro de cada persona y con las demás
    print(f"{'persona':20s} {'fotos':>5s} {'consigo':>8s} {'con otros':>10s} {'más parecido':>14s}")
    for p in personas:
        m = y == p
        dentro = S[np.ix_(m, m)][~np.eye(m.sum(), dtype=bool)].mean() if m.sum() > 1 else float("nan")
        fuera = {q: S[np.ix_(m, y == q)].mean() for q in personas if q != p}
        q = max(fuera, key=fuera.get)
        print(f"{p:20s} {m.sum():5d} {dentro:8.2f} {np.mean(list(fuera.values())):10.2f}   {q} ({fuera[q]:.2f})")

    # 3. Fotos raras
    print("\nFotos que se parecen poco a su persona (revisa su recorte):")
    for i in range(len(X)):
        otras = (y == y[i]) & (np.arange(len(X)) != i)
        if otras.any():
            s = S[i, otras].mean()
            if s < 0.3:
                print(f"  {y[i]}/{fotos[i]}: similitud media {s:.2f}")

    # 4. Acierto dejando una fuera (coseno con la media del resto)
    aciertos = 0
    for i in range(len(X)):
        resto = np.arange(len(X)) != i
        mejor, mejor_sim = None, -2
        for p in personas:
            m = resto & (y == p)
            if not m.any():
                continue
            media = X[m].mean(axis=0)
            sim = media @ X[i] / np.linalg.norm(media)
            if sim > mejor_sim:
                mejor, mejor_sim = p, sim
        aciertos += mejor == y[i]
    print(f"\nAcierto dejando una foto fuera: {aciertos}/{len(X)} = {aciertos / len(X):.0%}")


if __name__ == "__main__":
    main()
