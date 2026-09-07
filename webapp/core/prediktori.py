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


# ----------------------------------------------------------------------------
# Ansambl (PLAN_SINTEZA §2.2): linearna kombinacija ocena postojecih metoda
# ----------------------------------------------------------------------------
# Sest komponenti, sest tezina, bez ijedne ML biblioteke. Komponente su iste
# velicine koje vec koriste postojeci metodi (frekvencija, Bajes, hibrid, ritam,
# svezina, parovi) — ansambl ne uvodi nov nacin predvidjanja, samo ih sabira.
#
# Odstupanje od plana: plan nabraja „poziciju" kao sestu komponentu, ali pozicija
# izvlacenja nije deo interfejsa prediktora (funkcije vide kolo kao skup). Umesto
# nje je „svezina" — osa koju testiraju postojeci metodi fresh i cold.
#
# Ucenje tezina ne sme da vidi kolo koje ocenjuje. Plan (§ 2.2) daje dve opcije;
# ovde je uzeta druga, jer prva cini retro-bektest kvadratnim: tezine se uce
# JEDNOM, na prvom delu istorije (kola [UCENJE_OD, UCENJE_DO), prorijedjena svako
# UCENJE_KORAK-to), i vaze za sva kola posle toga. Dok istorija ne dosegne
# UCENJE_DO, sve komponente imaju istu tezinu — nema jos nicega da se nauci.
# Tako nijedno ocenjeno kolo nije uslo u ucenje sopstvenih tezina.

KOMPONENTE = ("frekvencija", "bajes", "hibrid", "ritam", "svezina", "parovi")

UCENJE_OD = 100      # pre ovog kola nema dovoljno istorije za pun prozor
UCENJE_DO = 400      # posle ovog kola su tezine zamrznute (~30% tipicne istorije)
UCENJE_KORAK = 5     # proredjivanje skupa za ucenje (60 tacaka je dovoljno za 6 tezina)

# Kesevi: ucenje i bodovanje su ciste funkcije podataka, pa se smeju kesirati.
# Kljuc je jeftin otisak isecka (duzina + prvo i poslednje kolo) umesto celog
# sadrzaja — kola su jedinstvena i rastuca, pa taj trojac odredjuje isecak. Bez
# ovoga bi retro-bektest hesirao 400 kola na svakom od 1.400 koraka.
_KES_TEZINA = {}     # otisak isecka za ucenje -> tezine
_KES_MAX = 8
_KES_SKOR = {}       # otisak poslednjeg poziva -> (skor, tezine); deli ga ensemble i k_ensemble


def _primitivi(prozor):
    """Sve sto komponentama treba, izracunato jednom nad datim prozorom kola."""
    n = len(prozor)
    count, _poslednji = _frekvencija_i_poslednji(prozor)
    v = _bajes_skorovi(prozor) if n else {b: 0.0 for b in range(1, MAX_BROJ + 1)}
    parovi, suma = _povezanost(prozor)
    pojave = {b: [] for b in range(1, MAX_BROJ + 1)}
    for i, (_kolo, brojevi) in enumerate(prozor):
        for b in brojevi:
            pojave[b].append(i)
    kasnjenje, ritam = {}, {}
    for b in range(1, MAX_BROJ + 1):
        p = pojave[b]
        kasnjenje[b] = (n - p[-1]) if p else (n + 1)
        if len(p) >= 2:
            razmaci = [p[i + 1] - p[i] for i in range(len(p) - 1)]
            ritam[b] = sum(razmaci) / len(razmaci)
        else:
            ritam[b] = None
    return {"n": n, "count": count, "bajes": v, "parovi": parovi, "suma_par": suma,
            "kasnjenje": kasnjenje, "ritam": ritam}


def _normalizuj(vrednosti):
    """Min-max na [0, 1] preko svih 39 brojeva; ravan vektor daje same nule.

    Zaokruzivanje na 9 decimala drzi skor stabilnim bez obzira na to da li je
    Bajes racunat sveze ili klizno — inace bi razlika reda 1e-15 mogla da okrene
    argmax kod prakticno izjednacenih brojeva.
    """
    najmanje = min(vrednosti.values())
    najvece = max(vrednosti.values())
    raspon = najvece - najmanje
    if raspon <= 0:
        return {b: 0.0 for b in vrednosti}
    return {b: round((x - najmanje) / raspon, 9) for b, x in vrednosti.items()}


def skorovi_komponenti(prim):
    """Sest normalizovanih skor-vektora (svaki broj dobija ocenu 0..1)."""
    v = prim["bajes"]
    parovi, suma = prim["parovi"], prim["suma_par"]
    top20 = [b for b, _ in sorted(v.items(), key=lambda x: (-x[1], x[0]))[:20]]
    maks_bonus = max(suma.values()) if suma else 0

    def hibrid_skor(b):
        bonus = sum(parovi.get((min(b, t), max(b, t)), 0) for t in top20 if t != b)
        norm = (bonus / maks_bonus) if maks_bonus > 0 else 0
        return v[b] * 0.8 + norm * 0.2

    brojevi = range(1, MAX_BROJ + 1)
    sirovo = {
        "frekvencija": {b: float(prim["count"][b]) for b in brojevi},
        "bajes": {b: v[b] for b in brojevi},
        "hibrid": {b: hibrid_skor(b) for b in brojevi},
        # ritam: koliko broj kasni u odnosu na sopstveni prosecan razmak
        "ritam": {b: (prim["kasnjenje"][b] / prim["ritam"][b] if prim["ritam"][b] else 0.0)
                  for b in brojevi},
        # svezina: skorije izvucen = veci skor (suprotan smer od kasnjenja)
        "svezina": {b: float(-prim["kasnjenje"][b]) for b in brojevi},
        "parovi": {b: float(suma.get(b, 0)) for b in brojevi},
    }
    return {k: _normalizuj(sirovo[k]) for k in KOMPONENTE}


def _lift(skorovi, dobitni):
    """Koliko je komponenta podigla izvucene brojeve iznad proseka svih brojeva.

    Pozitivan lift znaci da je komponenta u tom kolu davala vise ocene brojevima
    koji su zaista izvuceni. Pod slucajnoscu lift osciluje oko nule.
    """
    izlaz = {}
    for k, s in skorovi.items():
        svi = sum(s.values()) / len(s)
        pogodjeni = sum(s[b] for b in dobitni) / len(dobitni)
        izlaz[k] = pogodjeni - svi
    return izlaz


def _tezine_iz_lifta(zbir, n):
    """Tezina komponente = njen prosecan lift, odsecen na nulu i normalizovan.

    Ako nijedna komponenta nema pozitivan lift (ocekivan ishod na slucajnim
    podacima), sve tezine su jednake — ansambl tada nije nista drugo do prosek.
    """
    if n <= 0:
        return {k: round(1.0 / len(KOMPONENTE), 9) for k in KOMPONENTE}
    pozitivni = {k: max(0.0, zbir[k] / n) for k in KOMPONENTE}
    ukupno = sum(pozitivni.values())
    if ukupno <= 0:
        return {k: round(1.0 / len(KOMPONENTE), 9) for k in KOMPONENTE}
    return {k: round(x / ukupno, 9) for k, x in pozitivni.items()}


def _otisak(istorija, granica, period):
    """Jeftin identitet isečka istorija[:granica] za keš (vidi komentar uz keševe)."""
    if granica <= 0:
        return (0, None, None, period)
    return (granica, istorija[0][0], istorija[granica - 1][0], period)


def nauci_tezine(istorija, period):
    """Tezine iz prvih UCENJE_DO kola date istorije; nikad iz kola koje se ocenjuje.

    Za istoriju kracu od UCENJE_DO nema skupa za ucenje pa su sve tezine jednake.
    Rezultat zavisi samo od tog prefiksa, pa se kesira: u retro-bektestu se uci
    tacno jednom umesto na svakom koraku.
    """
    if len(istorija) < UCENJE_DO:
        return _tezine_iz_lifta({k: 0.0 for k in KOMPONENTE}, 0)

    granica = UCENJE_DO
    kljuc = _otisak(istorija, granica, period)
    if kljuc in _KES_TEZINA:
        return _KES_TEZINA[kljuc]

    zbir = {k: 0.0 for k in KOMPONENTE}
    n = 0
    for i in range(UCENJE_OD, granica, UCENJE_KORAK):
        prozor = _prozor(istorija[:i], period)
        if not prozor:
            continue
        skorovi = skorovi_komponenti(_primitivi(prozor))
        for k, v in _lift(skorovi, set(istorija[i][1])).items():
            zbir[k] += v
        n += 1
    tezine = _tezine_iz_lifta(zbir, n)

    if len(_KES_TEZINA) >= _KES_MAX:
        _KES_TEZINA.clear()
    _KES_TEZINA[kljuc] = tezine
    return tezine


def skor_ansambla(istorija, period):
    """(skor po broju, tezine) za ciljno kolo — ulaz i za jednobrojni i za komb.

    Rezultat se pamti za poslednji isečak da ga `ensemble` i `k_ensemble`, koji se
    u retro-bektestu pozivaju jedan za drugim nad istim podacima, ne računaju dvaput.
    """
    prozor = _prozor(istorija, period)
    if not prozor:
        return None, None
    kljuc = _otisak(istorija, len(istorija), period)
    if _KES_SKOR.get("kljuc") == kljuc:
        return _KES_SKOR["skor"], _KES_SKOR["tezine"]

    tezine = nauci_tezine(istorija, period)
    skorovi = skorovi_komponenti(_primitivi(prozor))
    skor = {b: round(sum(tezine[k] * skorovi[k][b] for k in KOMPONENTE), 9)
            for b in range(1, MAX_BROJ + 1)}
    _KES_SKOR.update(kljuc=kljuc, skor=skor, tezine=tezine)
    return skor, tezine


def ensemble(istorija, period, ciljno_kolo=None):
    """Ansambl: broj sa najvisim tezinskim zbirom ocena sest komponenti."""
    skor, _tezine = skor_ansambla(istorija, period)
    if skor is None:
        return None
    return max(range(1, MAX_BROJ + 1), key=lambda b: (skor[b], -b))


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
    "ensemble": ("Ansambl", ensemble, "Težinski zbir ocena šest komponenti; težine naučene isključivo na ranijim kolima."),
    "random": ("Nasumični (kontrola)", random, "Kontrolna grupa: nasumičan broj (seed = kolo). Referenca za poređenje."),
}
