"""Mapa kombinacija: indeksiranje prostora 7/39 i raspored na 2D
(plan_mapa_kombinacija.md, Faza 1).

Pojmovi (fiksni za ceo modul i sve generisane pločice):
  rang    = leksikografski redni broj kombinacije, 0 .. 15.380.936. Jednak je
            indeksu u itertools.combinations(1..39, 7), pa se ceo prostor može
            nabrojati u rastućem rangu bez ijednog poziva formule.
  (x, y)  = ćelija kvadrata 4096x4096 dobijena Hilbertovom krivom reda 12 nad
            rangom; x je kolona, y je red (raste nadole, kao kod pločica).

Kvadrat ima 16.777.216 ćelija, a kombinacija je 15.380.937, pa je poslednjih
oko 8% krive prazno — na mapi je to providna oblast, ne podatak.

Raspored je deterministički i trajan: promena RED_KRIVE ili DIMENZIJE znači
regeneraciju svih pločica.

Modul ne čita bazu i ne računa statistiku. Radi samo nad samom kombinacijom:
indeks, koordinate i osobine (zbir, raspon, parni, dekade).
"""

from math import comb

import numpy as np

from . import konfig

MAX_BROJ = konfig.MAX_BROJ
BROJEVA = konfig.BROJEVA_U_KOMBINACIJI
UKUPNO_KOMBINACIJA = comb(MAX_BROJ, BROJEVA)      # 15.380.937 za 7/39

RED_KRIVE = 12                                     # Hilbertova kriva reda 12
DIMENZIJA = 1 << RED_KRIVE                         # 4096 x 4096 ćelija
VELICINA_PLOCICE = 256
MAX_ZOOM = RED_KRIVE - 8                           # 4: na njemu je 1 piksel = 1 kombinacija

# Osobine kojima se boji pozadina mape. `ocena` (skor Generatora) dolazi u
# kasnijoj fazi jer zavisi od stanja baze, ne samo od kombinacije.
OSOBINE = {
    "zbir":   {"opis": "Zbir sedam brojeva", "tip": "sekvencijalna",
               "opseg": (sum(range(1, BROJEVA + 1)),
                         sum(range(MAX_BROJ - BROJEVA + 1, MAX_BROJ + 1)))},
    "raspon": {"opis": "Razlika najvećeg i najmanjeg broja", "tip": "sekvencijalna",
               "opseg": (BROJEVA - 1, MAX_BROJ - 1)},
    "parni":  {"opis": "Koliko je parnih brojeva", "tip": "diskretna",
               "opseg": (0, BROJEVA)},
    "dekade": {"opis": "Koliko dekada kombinacija dodiruje", "tip": "diskretna",
               "opseg": (1, (MAX_BROJ - 1) // 10 + 1)},
}

# Seed kontrolnog (slučajnog) sloja. Fiksan je da bi se ista „lažna istorija"
# uvek iscrtala isto i da bi poređenje sa stvarnim tačkama bilo ponovljivo.
SEED_KONTROLE = 20260905

# Kontrolne tačke skale (viridis). Ovde su, a ne u generatoru pločica, da bi
# legenda u browseru i ispečene boje došle iz istog izvora.
SKALA_BOJA = [
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
    (180, 222, 44), (253, 231, 37),
]


def proveri_kombinaciju(komb):
    """Vraća sortiranu n-torku ili diže ValueError ako kombinacija nije ispravna."""
    b = sorted(int(x) for x in komb)
    if len(b) != BROJEVA:
        raise ValueError(f"Kombinacija mora imati {BROJEVA} brojeva, dobijeno {len(b)}.")
    if len(set(b)) != BROJEVA:
        raise ValueError("Brojevi u kombinaciji moraju biti različiti.")
    if b[0] < 1 or b[-1] > MAX_BROJ:
        raise ValueError(f"Brojevi moraju biti u opsegu 1..{MAX_BROJ}.")
    return tuple(b)


# ----------------------------------------------------------------------------
# Rang: kombinacija <-> redni broj u leksikografskom poretku
# ----------------------------------------------------------------------------

def rang(komb):
    """Leksikografski rang kombinacije, 0 .. UKUPNO_KOMBINACIJA-1.

    Za svaku poziciju i sabira koliko kombinacija ima isti prefiks a manji broj
    na toj poziciji; zbir tih blokova (hokej-štap identitet) daje zatvoren oblik
    C(n-p, k-i) - C(n-c+1, k-i), gde je p prethodni broj u kombinaciji.
    """
    b = proveri_kombinaciju(komb)
    r, prethodni = 0, 0
    for i, c in enumerate(b):
        preostalo = BROJEVA - i
        r += comb(MAX_BROJ - prethodni, preostalo) - comb(MAX_BROJ - c + 1, preostalo)
        prethodni = c
    return r


def unrang(r):
    """Kombinacija sa datim leksikografskim rangom (inverz funkcije `rang`)."""
    r = int(r)
    if not 0 <= r < UKUPNO_KOMBINACIJA:
        raise ValueError(f"Rang mora biti u opsegu 0..{UKUPNO_KOMBINACIJA - 1}.")
    b, prethodni = [], 0
    for i in range(BROJEVA):
        preostalo = BROJEVA - i - 1
        c = prethodni + 1
        while True:
            blok = comb(MAX_BROJ - c, preostalo)   # koliko kombinacija ima taj broj na toj poziciji
            if r < blok:
                break
            r -= blok
            c += 1
        b.append(c)
        prethodni = c
    return tuple(b)


def sve_kombinacije():
    """Sve kombinacije u rastućem rangu kao niz (UKUPNO_KOMBINACIJA, 7) uint8.

    Redni broj vrste JESTE rang, pa generisanje pločica ne poziva `rang` ni
    jednom. Zauzima oko 107 MB i pravi se za nekoliko sekundi.
    """
    from itertools import chain, combinations
    ravno = chain.from_iterable(combinations(range(1, MAX_BROJ + 1), BROJEVA))
    return np.fromiter(ravno, dtype=np.uint8,
                       count=UKUPNO_KOMBINACIJA * BROJEVA).reshape(-1, BROJEVA)


# ----------------------------------------------------------------------------
# Hilbertova kriva (radi i nad skalarom i nad numpy nizom)
# ----------------------------------------------------------------------------

def _kao_niz(v):
    return np.atleast_1d(np.asarray(v, dtype=np.int64)).copy()


def hilbert_xy(d, red=RED_KRIVE):
    """Rang -> (x, y) na kvadratu 2^red. Skalar daje int, niz daje dva niza."""
    skalar = np.ndim(d) == 0
    t = _kao_niz(d)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    n = 1 << red
    s = 1
    while s < n:
        rx = (t >> 1) & 1
        ry = (t ^ rx) & 1
        okreni = ry == 0                       # rotacija kvadranta
        obrni = okreni & (rx == 1)
        x[obrni] = s - 1 - x[obrni]
        y[obrni] = s - 1 - y[obrni]
        x[okreni], y[okreni] = y[okreni], x[okreni]
        x += s * rx
        y += s * ry
        t >>= 2
        s <<= 1
    if skalar:
        return int(x[0]), int(y[0])
    return x, y


def hilbert_d(x, y, red=RED_KRIVE):
    """(x, y) -> rang na krivi (inverz funkcije `hilbert_xy`)."""
    skalar = np.ndim(x) == 0 and np.ndim(y) == 0
    x = _kao_niz(x)
    y = _kao_niz(y)
    n = 1 << red
    d = np.zeros(np.broadcast(x, y).shape, dtype=np.int64)
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        okreni = ry == 0
        obrni = okreni & (rx == 1)
        x[obrni] = n - 1 - x[obrni]
        y[obrni] = n - 1 - y[obrni]
        x[okreni], y[okreni] = y[okreni], x[okreni]
        s >>= 1
    if skalar:
        return int(d[0])
    return d


def koordinate(r):
    """Rang kombinacije -> (x, y) ćelija na mapi."""
    return hilbert_xy(r)


def rang_iz_koordinata(x, y):
    """(x, y) -> rang kombinacije, ili None ako ćelija pada u prazan deo krive."""
    r = hilbert_d(x, y)
    return None if r >= UKUPNO_KOMBINACIJA else r


def kombinacija_na_koordinati(x, y):
    """(x, y) -> kombinacija na toj ćeliji, ili None za praznu ćeliju."""
    r = rang_iz_koordinata(x, y)
    return None if r is None else unrang(r)


# ----------------------------------------------------------------------------
# Osobine kombinacije (boja pozadine)
# ----------------------------------------------------------------------------

def _dekada(b):
    """Indeks dekade broja: 1-10 -> 0, 11-20 -> 1, 21-30 -> 2, 31-39 -> 3."""
    return min((b - 1) // 10, (MAX_BROJ - 1) // 10)


def osobine(komb):
    """Sve osobine jedne kombinacije kao dict (isti nazivi kao u OSOBINE)."""
    b = proveri_kombinaciju(komb)
    return {
        "zbir": sum(b),
        "raspon": b[-1] - b[0],
        "parni": sum(1 for x in b if x % 2 == 0),
        "dekade": len({_dekada(x) for x in b}),
    }


def slucajni_rangovi(n, seed=SEED_KONTROLE):
    """n rangova izvučenih ravnomerno iz celog prostora (kontrolni sloj mape).

    Ravnomeran izbor po rangu je isto što i ravnomeran izbor kombinacije, jer je
    rang bijekcija. Isti seed uvek daje isti set, pa je slika ponovljiva.
    """
    if n < 0:
        raise ValueError("Broj tačaka ne može biti negativan.")
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, UKUPNO_KOMBINACIJA, size=int(n), dtype=np.int64)


def detalj_kombinacije(brojevi):
    """Sve što se o kombinaciji zna bez baze: rang, ćelija na mapi i osobine."""
    b = proveri_kombinaciju(brojevi)
    r = rang(b)
    x, y = koordinate(r)
    return {"brojevi": list(b), "rang": r, "x": x, "y": y, "osobine": osobine(b)}


def detalj_celije(x, y):
    """Isto, ali polazi od ćelije; za praznu ćeliju vraća samo koordinate."""
    x, y = int(x), int(y)
    if not (0 <= x < DIMENZIJA and 0 <= y < DIMENZIJA):
        raise ValueError(f"Koordinate moraju biti u opsegu 0..{DIMENZIJA - 1}.")
    r = rang_iz_koordinata(x, y)
    if r is None:
        return {"brojevi": None, "rang": None, "x": x, "y": y, "osobine": None}
    d = detalj_kombinacije(unrang(r))
    d["x"], d["y"] = x, y
    return d


def osobina_niz(naziv, kombinacije):
    """Vektorska verzija `osobine` za niz (N, 7): vraća N vrednosti kao int16.

    Koristi je generator pločica; rezultat je za svaki red isti kao
    `osobine(komb)[naziv]` (pokriveno testom).
    """
    k = np.asarray(kombinacije)
    if naziv == "zbir":
        return k.sum(axis=1, dtype=np.int16)
    if naziv == "raspon":
        return k.max(axis=1).astype(np.int16) - k.min(axis=1).astype(np.int16)
    if naziv == "parni":
        return (k % 2 == 0).sum(axis=1).astype(np.int16)
    if naziv == "dekade":
        dek = np.minimum((k.astype(np.int16) - 1) // 10, (MAX_BROJ - 1) // 10)
        broj = np.zeros(len(k), dtype=np.int16)
        for i in range((MAX_BROJ - 1) // 10 + 1):
            broj += (dek == i).any(axis=1)
        return broj
    raise ValueError(f"Nepoznata osobina: {naziv}")
