class Pista:
    """Una cara que se sigue de un fotograma a otro, con sus votos acumulados."""

    _siguiente_id = 1   # contador compartido por todas las pistas

    def __init__(self, caja):
        self.id = Pista._siguiente_id
        Pista._siguiente_id += 1

        self.caja = caja          # último recuadro (x, y, w, h)
        self.sumas = {}           # {nombre: suma de probabilidades}
        self.vista = 0            # fotogramas en los que se ha visto
        self.sin_ver = 0          # fotogramas seguidos sin verla
        self.confirmada = False
        self.nombre = None        # nombre confirmado

    def actualizar(self, caja, probs):
        """La pista se ha visto en este fotograma: nuevo recuadro y suma de probabilidades."""
        self.caja = caja
        self.vista += 1
        self.sin_ver = 0
        for nombre, p in probs.items():
            self.sumas[nombre] = self.sumas.get(nombre, 0.0) + p

    def no_vista(self):
        """La pista no ha aparecido en este fotograma."""
        self.sin_ver += 1

    def mejor(self):
        """Persona con la suma más alta: (nombre, suma)."""
        if not self.sumas:
            return None, 0.0
        nombre = max(self.sumas, key=self.sumas.get)
        return nombre, self.sumas[nombre] / self.vista

    def cumple_criterios(self, min_vista=8, umbral=0.4, margen=0.15):
        """
        ¿Se puede confirmar ya a la persona?
        Se usa la MEDIA por fotograma: no crece con el tiempo.
        """
        if self.confirmada or self.vista < min_vista:
            return False
        medias = sorted((s / self.vista for s in self.sumas.values()), reverse=True)
        primera = medias[0]
        segunda = medias[1] if len(medias) > 1 else 0.0
        return primera >= umbral and primera - segunda >= margen

    def confirmar(self):
        self.nombre, _ = self.mejor()
        self.confirmada = True
