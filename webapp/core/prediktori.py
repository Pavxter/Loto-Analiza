"""Prediktori za stranu „Prognoza" — svaki predlaže JEDAN broj za sledeće kolo.

Interfejs (PLAN_PROGNOZA.md §2): svaka funkcija prima istoriju STROGO PRE ciljnog
kola i vraća jedan broj 1..MAX_BROJ (ili None ako nema dovoljno podataka).

    istorija:    lista (kolo, brojevi) hronološki; brojevi = tuple u redosledu izvlačenja
    period:      broj poslednjih kola koja se gledaju (None ili 0 = sva)
    ciljno_kolo: broj ciljnog kola — koristi ga samo 'random' (seed) radi
                 reproducibilnosti; ostali ga ignorišu

Sva tie-break pravila su deterministička (osim logike 'random', koja je
deterministička kroz seed = ciljno_kolo): pri jednakom skoru bira se pravilo iz
plana, a kao poslednji kriterijum uvek najmanji broj.

VAŽNO (anti-curenje): funkcije ne smeju da vide ciljno kolo niti bilo šta posle
njega. Ne drže nikakvo stanje između poziva. Testira se u tests/test_prognoza.py.
"""

import random as _random

from . import konfig

MAX_BROJ = konfig.MAX_BROJ


def _prozor(istorija, period):
    """Poslednjih `period` kola (ili sva ako je period None/0)."""
    if period and period > 0:
        return istorija[-period:]
    return istorija


def _frekvencija_i_poslednji(prozor):
    """Vraća (count, poslednji_indeks) po broju, indeksi lokalni u prozoru."""
    count = {b: 0 for b in range(1, MAX_BROJ + 1)}
    poslednji = {}
    for i, (_kolo, brojevi) in enumerate(prozor):
        for b in brojevi:
            count[b] += 1
            poslednji[b] = i
    return count, poslednji


def hot(istorija, period, ciljno_kolo=None):
    """Najvrući: najviše pojavljivanja u periodu. Tie-break: skoriji nastup, pa manji broj."""
    w = _prozor(istorija, period)
    if not w:
        return None
    count, poslednji = _frekvencija_i_poslednji(w)
    return max(range(1, MAX_BROJ + 1),
               key=lambda b: (count[b], poslednji.get(b, -1), -b))


def cold(istorija, period, ciljno_kolo=None):
    """Najhladniji („due"): najmanje pojavljivanja u periodu; prioritet neizvučenim.

    Tie-break: najduže neizvučen u CELOJ istoriji (nikad izvučen = najstariji), pa manji broj.
    """
    if not istorija:
        return None
    w = _prozor(istorija, period)
    count, _ = _frekvencija_i_poslednji(w)
    # poslednje pojavljivanje u celoj istoriji (globalni indeks; -1 = nikad)
    globalni_poslednji = {}
    for i, (_kolo, brojevi) in enumerate(istorija):
        for b in brojevi:
            globalni_poslednji[b] = i
    return min(range(1, MAX_BROJ + 1),
               key=lambda b: (count[b], globalni_poslednji.get(b, -1), b))


def _bajes_skorovi(prozor):
    """Bajesovska „verovanja" nad datim prozorom — ista formula kao rangiranje.bajes_verovanja
    (learning rate 0.005, normalizacija po kolu), ali nad eksplicitnom listom kola
    (bez pandas-a, radi brzine i bez ikakvog spoljnog stanja)."""
    lr = 0.005
    v = {b: 1.0 / MAX_BROJ for b in range(1, MAX_BROJ + 1)}
    for _kolo, brojevi in prozor:
        s = set(brojevi)
        for b in range(1, MAX_BROJ + 1):
            v[b] *= (1 + lr) if b in s else (1 - lr)
        suma = sum(v.values())
        if suma > 0:
            v = {b: x / suma for b, x in v.items()}
    return v


def bayes(istorija, period, ciljno_kolo=None):
    """Najviši skor Bajesovog modela (formula sa strane Rangiranje)."""
    w = _prozor(istorija, period)
    if not w:
        return None
    v = _bajes_skorovi(w)
    return max(range(1, MAX_BROJ + 1), key=lambda b: (v[b], -b))


def _povezanost(prozor):
    """Par-brojači: koliko puta su dva broja izvučena zajedno u prozoru.
    Vraća (parovi, suma_po_broju) — suma_po_broju odgovara zbiru kolone matrice
    povezanosti iz rangiranje.matrica_povezanosti."""
    parovi = {}
    suma = {b: 0 for b in range(1, MAX_BROJ + 1)}
    for _kolo, brojevi in prozor:
        bs = sorted(set(brojevi))
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                k = (bs[i], bs[j])
                parovi[k] = parovi.get(k, 0) + 1
                suma[bs[i]] += 1
                suma[bs[j]] += 1
    return parovi, suma


def hybrid(istorija, period, ciljno_kolo=None):
    """Najviši hibridni skor: 80% Bajes + 20% normalizovana povezanost sa top-20 Bajes brojeva
    (ista fuzija kao rangiranje.hibrid_rang)."""
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

    return max(range(1, MAX_BROJ + 1), key=lambda b: (finalni(b), -b))


def rhythm(istorija, period, ciljno_kolo=None):
    """Ritam koji kasni: najveći odnos D/R, gde je R prosečan razmak ponavljanja
    u periodu, a D broj kola od poslednjeg pojavljivanja. Preskaču se brojevi
    sa manje od 2 pojavljivanja u periodu (R nije definisan)."""
    w = _prozor(istorija, period)
    if not w:
        return None
    pojave = {}
    for i, (_kolo, brojevi) in enumerate(w):
        for b in brojevi:
            pojave.setdefault(b, []).append(i)
    najbolji, najbolji_odnos = None, -1.0
    for b in range(1, MAX_BROJ + 1):
        p = pojave.get(b, [])
        if len(p) < 2:
            continue
        razmaci = [p[i + 1] - p[i] for i in range(len(p) - 1)]
        r = sum(razmaci) / len(razmaci)
        d = len(w) - p[-1]  # kola od poslednjeg pojavljivanja (poslednje kolo prozora -> 1)
        odnos = d / r if r > 0 else 0
        if odnos > najbolji_odnos or (odnos == najbolji_odnos and (najbolji is None or b < najbolji)):
            najbolji, najbolji_odnos = b, odnos
    return najbolji


def fresh(istorija, period, ciljno_kolo=None):
    """Najsvežiji: broj izvučen najskorije — poslednje kolo, prva pozicija izvlačenja."""
    if not istorija:
        return None
    return int(istorija[-1][1][0])


def random(istorija, period, ciljno_kolo=None):
    """Kontrolna grupa: nasumičan broj, seedovan ciljnim kolom (reproducibilno)."""
    seme = ciljno_kolo if ciljno_kolo is not None else 0
    return _random.Random(seme).randint(1, MAX_BROJ)


# Registar: id -> (naziv, funkcija, kratak opis za tooltip u UI)
PREDIKTORI = {
    "hot":    ("Najvrući",         hot,    "Broj sa najviše pojavljivanja u periodu."),
    "cold":   ("Najhladniji",      cold,   "Broj sa najmanje pojavljivanja; prioritet neizvučenim (tzv. due hipoteza)."),
    "bayes":  ("Bajesovski",       bayes,  "Najviši skor iterativnog Bajesovog modela (isti kao na strani Rangiranje)."),
    "hybrid": ("Hibridni",         hybrid, "Najviši hibridni skor: 80% Bajes + 20% povezanost sa top-20."),
    "rhythm": ("Ritam koji kasni", rhythm, "Broj koji najviše kasni u odnosu na svoj prosečan ritam ponavljanja (D/R)."),
    "fresh":  ("Najsvežiji",       fresh,  "Prvi izvučen broj iz poslednjeg kola — testira hipotezu da vrući ostaju vrući."),
    "random": ("Nasumični (kontrola)", random, "Kontrolna grupa: nasumičan broj (seed = kolo). Referenca za poređenje."),
}
