"""Kombinacijski prediktori (PLAN_PROGNOZA_KOMBINACIJE §2–§3).

Svaki metod predlaže JEDNU kompletnu kombinaciju od 7 brojeva za sledeće kolo.
Interfejs analogno jednobrojnim (prediktori.py), ali vraća sortiranu 7-torku:

    fn(istorija, period, ciljno_kolo=None) -> tuple[int, ...]   # 7 jedinstvenih, rastuće

Garancije (identične jednobrojnim, testira se u tests/test_prognoza.py):
  - bez curenja budućnosti: funkcija vidi isključivo kola strogo pre ciljnog;
  - determinizam: sva tie-break pravila deterministička (osim k_random, koji je
    determinističan kroz seed = ciljno_kolo * 2 — različit od jednobrojnog random-a);
  - kanonski oblik: uvek sorted(...) radi poređenja i UNIQUE ograničenja.
"""

import random as _random

from . import konfig
from .prediktori import (_prozor, _frekvencija_i_poslednji, _bajes_skorovi, _povezanost)

MAX_BROJ = konfig.MAX_BROJ
K = konfig.BROJEVA_U_KOMBINACIJI


def _sedam(brojevi):
    return tuple(sorted(brojevi[:K]))


def k_hot7(istorija, period, ciljno_kolo=None):
    """Top-7 vrućih. Tie-break: skoriji poslednji nastup, pa manji broj."""
    w = _prozor(istorija, period)
    if not w:
        return None
    count, poslednji = _frekvencija_i_poslednji(w)
    rang = sorted(range(1, MAX_BROJ + 1),
                  key=lambda b: (-count[b], -poslednji.get(b, -1), b))
    return _sedam(rang)


def _globalni_poslednji(istorija):
    gp = {}
    for i, (_kolo, brojevi) in enumerate(istorija):
        for b in brojevi:
            gp[b] = i
    return gp


def k_cold7(istorija, period, ciljno_kolo=None):
    """Top-7 hladnih (prvo neizvučeni u periodu). Tie-break: najduže neizvučen, pa manji."""
    if not istorija:
        return None
    w = _prozor(istorija, period)
    count, _ = _frekvencija_i_poslednji(w)
    gp = _globalni_poslednji(istorija)
    rang = sorted(range(1, MAX_BROJ + 1),
                  key=lambda b: (count[b], gp.get(b, -1), b))
    return _sedam(rang)


def k_bayes7(istorija, period, ciljno_kolo=None):
    """Vrh Bajesove rang-liste (reuse _bajes_skorovi)."""
    w = _prozor(istorija, period)
    if not w:
        return None
    v = _bajes_skorovi(w)
    rang = sorted(range(1, MAX_BROJ + 1), key=lambda b: (-v[b], b))
    return _sedam(rang)


def k_hybrid7(istorija, period, ciljno_kolo=None):
    """Vrh hibridne rang-liste (80% Bajes + 20% povezanost sa top-20)."""
    w = _prozor(istorija, period)
    if not w:
        return None
    v = _bajes_skorovi(w)
    parovi, suma = _povezanost(w)
    top20 = [b for b, _ in sorted(v.items(), key=lambda x: (-x[1], x[0]))[:20]]
    maks_bonus = max(suma.values()) if suma else 0

    def finalni(b):
        bonus = sum(parovi.get((min(b, t), max(b, t)), 0) for t in top20 if t != b)
        norm = (bonus / maks_bonus) if maks_bonus > 0 else 0
        return v[b] * 0.8 + norm * 0.2

    rang = sorted(range(1, MAX_BROJ + 1), key=lambda b: (-finalni(b), b))
    return _sedam(rang)


def k_rhythm7(istorija, period, ciljno_kolo=None):
    """Top-7 po ritmu (najveći D/R). Ako <7 brojeva ima definisan ritam,
    dopuna po dužini kašnjenja D (najduže neizvučen), pa manji broj."""
    w = _prozor(istorija, period)
    if not w:
        return None
    n = len(w)
    pojave = {b: [] for b in range(1, MAX_BROJ + 1)}
    for i, (_kolo, brojevi) in enumerate(w):
        for b in brojevi:
            pojave[b].append(i)

    sa_ritmom = []       # (odnos, broj)
    kasnjenje = {}       # b -> D (kola od poslednjeg pojavljivanja; nikad = n+1)
    for b in range(1, MAX_BROJ + 1):
        p = pojave[b]
        kasnjenje[b] = (n - p[-1]) if p else (n + 1)
        if len(p) >= 2:
            razmaci = [p[i + 1] - p[i] for i in range(len(p) - 1)]
            r = sum(razmaci) / len(razmaci)
            odnos = (n - p[-1]) / r if r > 0 else 0
            sa_ritmom.append((odnos, b))

    sa_ritmom.sort(key=lambda t: (-t[0], t[1]))
    izabrani = [b for _o, b in sa_ritmom[:K]]
    if len(izabrani) < K:
        preostali = [b for b in range(1, MAX_BROJ + 1) if b not in izabrani]
        preostali.sort(key=lambda b: (-kasnjenje[b], b))
        izabrani += preostali[:K - len(izabrani)]
    return _sedam(izabrani)


def matrica_cooc(w):
    """Ko-okurencijska matrica (dict parova) + frekvencije, nad prozorom w.
    Reuse _povezanost (isti brojači kao strana Različitost, ali samo nad istorijom < N)."""
    parovi, _suma = _povezanost(w)
    count, _ = _frekvencija_i_poslednji(w)
    return parovi, count


def _cooc_iz_matrice(parovi, count):
    """Pohlepni izbor 7 brojeva iz gotove matrice (§3). Izdvojeno radi reuse u retro-petlji."""
    def C(a, b):
        return parovi.get((min(a, b), max(a, b)), 0)

    # 1) startni par sa najvećim C; tie: manji a, pa manji b
    najbolji_par, najbolji_c = None, -1
    for a in range(1, MAX_BROJ + 1):
        for b in range(a + 1, MAX_BROJ + 1):
            c = C(a, b)
            if c > najbolji_c or (c == najbolji_c and (najbolji_par is None or (a, b) < najbolji_par)):
                najbolji_c, najbolji_par = c, (a, b)
    if najbolji_par is None:
        return _sedam(list(range(1, K + 1)))
    S = set(najbolji_par)

    # 2) pohlepno dodavanje: max Σ C[x][s]; tie: veća frekvencija x, pa manji x
    while len(S) < K:
        najbolji_x, najbolji_skor = None, None
        for x in range(1, MAX_BROJ + 1):
            if x in S:
                continue
            skor = sum(C(x, s) for s in S)
            kljuc = (skor, count.get(x, 0), -x)
            if najbolji_skor is None or kljuc > najbolji_skor:
                najbolji_skor, najbolji_x = kljuc, x
        S.add(najbolji_x)
    return _sedam(sorted(S))


def k_cooc(istorija, period, ciljno_kolo=None):
    """Ko-okurencijski pohlepni algoritam (§3) — jedini algoritamski nov prediktor."""
    w = _prozor(istorija, period)
    if not w:
        return None
    parovi, count = matrica_cooc(w)
    return _cooc_iz_matrice(parovi, count)


def k_random(istorija, period, ciljno_kolo=None):
    """Kontrolna grupa: 7 nasumičnih jedinstvenih brojeva, seed = kolo*2
    (različit od jednobrojnog random prediktora da ne dele slučajnost)."""
    seme = (ciljno_kolo * 2) if ciljno_kolo is not None else 0
    return _sedam(sorted(_random.Random(seme).sample(range(1, MAX_BROJ + 1), K)))


# Registar: id -> (naziv, funkcija, kratak opis) — odvojen od jednobrojnog PREDIKTORI
PREDIKTORI_KOMB = {
    "k_hot7":    ("Top-7 vrućih",   k_hot7,    "7 najfrekventnijih brojeva u periodu."),
    "k_cold7":   ("Top-7 hladnih",  k_cold7,   "7 najređih (prvo neizvučeni u periodu; due hipoteza)."),
    "k_bayes7":  ("Top-7 Bajes",    k_bayes7,  "Vrh Bajesove rang-liste."),
    "k_hybrid7": ("Top-7 hibrid",   k_hybrid7, "Vrh hibridne rang-liste (80% Bajes + 20% povezanost)."),
    "k_rhythm7": ("Top-7 po ritmu", k_rhythm7, "7 brojeva sa najvećim odnosom kašnjenja i ritma (D/R)."),
    "k_cooc":    ("Ko-okurencijski", k_cooc,   "Pohlepno bira brojeve koji najčešće izlaze zajedno."),
    "k_random":  ("Nasumični (kontrola)", k_random, "Kontrolna grupa: 7 nasumičnih brojeva (seed = kolo·2)."),
}
