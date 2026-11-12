import csv
from datetime import datetime
from pathlib import Path

from src.decision.pista import Pista
from src.decision.seguimiento import emparejar


class Seguidor:
    """Lleva las pistas de todas las caras y decide quién ha entrado."""

    def __init__(self, max_sin_ver=10, min_vista=8, umbral=4.0, margen=2.0, umbral_iou=0.3):
        self.pistas = []          # pistas activas
        self.asistencia = {}      # {nombre: hora de entrada}

        self.max_sin_ver = max_sin_ver   # fotogramas sin ver antes de borrar una pista
        self.min_vista = min_vista
        self.umbral = umbral
        self.margen = margen
        self.umbral_iou = umbral_iou

    def actualizar(self, detecciones):
        """
        Se llama una vez por fotograma.

        detecciones: [(caja, probs), ...]  lo que devuelve procesar_frame
        Devuelve: lista de nombres confirmados en este fotograma
        """
        # 1. Emparejar los recuadros de este fotograma con las pistas
        cajas = [caja for caja, _ in detecciones]
        parejas, cajas_nuevas, pistas_no_vistas = emparejar(cajas, self.pistas, self.umbral_iou)

        # 2. Pistas vistas: sumar sus probabilidades
        for i, pista in parejas:
            caja, probs = detecciones[i]
            pista.actualizar(caja, probs)

        # 3. Caras nuevas: abrir una pista para cada una
        for i in cajas_nuevas:
            caja, probs = detecciones[i]
            pista = Pista(caja)
            pista.actualizar(caja, probs)
            self.pistas.append(pista)

        # 4. Pistas no vistas: contar y borrar las que llevan mucho sin aparecer
        for pista in pistas_no_vistas:
            pista.no_vista()
        self.pistas = [p for p in self.pistas if p.sin_ver <= self.max_sin_ver]

        # 5. Decidir
        nuevos = []
        for pista in self.pistas:
            if pista.confirmada:
                continue
            if pista.cumple_criterios(self.min_vista, self.umbral, self.margen):
                pista.confirmar()
                if pista.nombre not in self.asistencia:          # apuntar una sola vez
                    self.asistencia[pista.nombre] = datetime.now()
                    nuevos.append(pista.nombre)

        return nuevos

    def guardar_csv(self, carpeta="asistencia"):
        """Guarda la lista de asistencia en asistencia/AAAA-MM-DD_HH-MM.csv"""
        carpeta = Path(carpeta)
        carpeta.mkdir(exist_ok=True)
        ruta = carpeta / f"{datetime.now():%Y-%m-%d_%H-%M}.csv"
        with open(ruta, "w", newline="") as f:
            escritor = csv.writer(f)
            escritor.writerow(["nombre", "hora_entrada"])
            for nombre, hora in sorted(self.asistencia.items(), key=lambda x: x[1]):
                escritor.writerow([nombre, f"{hora:%H:%M:%S}"])
        return ruta