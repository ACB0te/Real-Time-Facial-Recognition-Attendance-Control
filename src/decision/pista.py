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
        return nombre, self.sumas[nombre]

    def cumple_criterios(self, min_vista=8, umbral=4.0, margen=2.0):
        """¿Se puede confirmar ya a la persona?"""
        if self.confirmada or self.vista < min_vista:
            return False
        sumas = sorted(self.sumas.values(), reverse=True)
        primera = sumas[0]
        segunda = sumas[1] if len(sumas) > 1 else 0.0
        return primera >= umbral and primera - segunda >= margen

    def confirmar(self):
        self.nombre, _ = self.mejor()
        self.confirmada = True

    def reiniciar(self):
        """Empieza a contar de cero (p. ej. si pasa la ventana sin confirmarse)."""
        self.sumas = {}
        self.vista = 0