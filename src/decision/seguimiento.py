def iou(a, b):
    """Solapamiento entre dos recuadros (x, y, w, h): 0 = no se tocan, 1 = idénticos."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    # Rectángulo de la intersección
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def emparejar(cajas, pistas, umbral=0.3):
    """
    Empareja los recuadros de este fotograma con las pistas que ya había.
    Cada recuadro va como mucho a una pista, y cada pista recibe como mucho un recuadro.

    cajas:  lista de recuadros (x, y, w, h) del fotograma
    pistas: lista de objetos Pista
    Devuelve:
        parejas:          [(i, pista), ...]  el recuadro cajas[i] es esa pista
        cajas_nuevas:     [i, ...]           recuadros que no encajan con ninguna pista
        pistas_no_vistas: [pista, ...]       pistas que no han aparecido en este fotograma
    """
    # 1. Todas las combinaciones recuadro-pista que se solapan lo suficiente
    candidatos = []
    for i, caja in enumerate(cajas):
        for pista in pistas:
            s = iou(caja, pista.caja)
            if s >= umbral:
                candidatos.append((s, i, pista))

    # 2. De mayor a menor solapamiento, emparejar si ninguno de los dos está ya usado
    candidatos.sort(key=lambda c: c[0], reverse=True)
    parejas = []
    cajas_usadas, pistas_usadas = set(), set()
    for s, i, pista in candidatos:
        if i in cajas_usadas or pista.id in pistas_usadas:
            continue
        parejas.append((i, pista))
        cajas_usadas.add(i)
        pistas_usadas.add(pista.id)

    # 3. Lo que ha quedado sin pareja
    cajas_nuevas = [i for i in range(len(cajas)) if i not in cajas_usadas]
    pistas_no_vistas = [p for p in pistas if p.id not in pistas_usadas]

    return parejas, cajas_nuevas, pistas_no_vistas