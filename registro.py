from pathlib import Path

import cv2

from src.reconocimiento_caras.deteccion import viola_jones
from src.reconocimiento_caras.embbedings import sacar_vector, entrenar_svm, guardar_knn
from src.utils.representacion_vectores import dibujar_vectores

CARPETA_DATOS = Path(__file__).resolve().parent / "datos"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}

def registrar_desde_carpeta(carpeta=CARPETA_DATOS):
    """
    Recorre datos/<nombre>/ y saca un vector por foto con exactamente una cara.
    Devuelve: diccionario {nombre: [vectores]}
    """
    caras = {}

    for carpeta_persona in sorted(carpeta.iterdir()):
        if not carpeta_persona.is_dir():
            continue
        nombre = carpeta_persona.name
        vectores = []
        descartadas = 0

        for foto in sorted(carpeta_persona.iterdir()):
            if foto.suffix.lower() not in EXTENSIONES:
                continue
            img = cv2.imread(str(foto))
            if img is None:
                continue

            detectadas = viola_jones(img)
            if len(detectadas) != 1:
                print(f"  descartada {foto.name}: {len(detectadas)} caras")
                descartadas += 1
                continue

            vectores.append(sacar_vector(img, detectadas[0]))

        print(f"{nombre}: {len(vectores)} vectores, {descartadas} fotos descartadas")
        if vectores:
            caras[nombre] = vectores

    return caras


def main():
    caras = registrar_desde_carpeta()
    if not caras:
        print("No se ha registrado a nadie")
        return

    guardar_knn(caras)
    if len(caras) >= 2:
        entrenar_svm(caras)
    else:
        print("La SVM necesita al menos 2 personas")

    dibujar_vectores(caras, metodo="pca", titulo="Registro")


if __name__ == "__main__":
    main()