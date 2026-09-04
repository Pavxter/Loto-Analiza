"""Logika strane „Prognoza": uživo predlozi, evaluacija, retro-bektest, statistika.

Implementira PLAN_PROGNOZA.md. Ključne garancije:
  - bez curenja budućnosti: prediktor za kolo N vidi isključivo kola strogo pre N
    (retro petlja prvo predviđa, pa tek onda dodaje kolo u stanje);
  - determinizam: retro-bektest daje identičan rezultat pri svakom pokretanju
    (uključujući 'random' kroz seed = kolo);
  - performanse: inkrementalno stanje (klizni prozor + Bajes množenje/deljenje)
    umesto O(N²) preračunavanja — ceo bektest za ~1.400 kola u sekundi.
"""

import time
from collections import deque
from datetime import datetime

from scipy.stats import binomtest

from . import baza, konfig, razlicitost_teorija as T
from .prediktori import PREDIKTORI
from .prediktori_komb import PREDIKTORI_KOMB

MAX_BROJ = konfig.MAX_BROJ
BROJEVA_U_KOMBINACIJI = konfig.BROJEVA_U_KOMBINACIJI

BASELINE = BROJEVA_U_KOMBINACIJI / MAX_BROJ      # 7/39 ≈ 0.1795
RETRO_PERIOD = 100    # fiksiran period retro-bektesta (PLAN §5)
MIN_START = 50        # preskoči prvih N kola (premalo istorije)
PRAG = 0.05 / len(PREDIKTORI)   # Bonferroni korekcija za paralelne metode (PLAN §7.3)
# Bonferroni preko SVIH metoda strane (jednobrojni + kombinacijski su isti eksperiment,
# PLAN_PROGNOZA_KOMBINACIJE §1)
PRAG_KOMB = 0.05 / (len(PREDIKTORI) + len(PREDIKTORI_KOMB))
MU_PREKL = T.ocekivano_preklapanje()   # očekivano preklapanje pod slučajnošću (1,256)


def istorija_iz_conn(conn):
    """Istorija kao lista (kolo, brojevi_tuple u redosledu izvlačenja), hronološki."""
    redovi = conn.execute(
        "SELECT kolo, b1, b2, b3, b4, b5, b6, b7 FROM istorijski_rezultati ORDER BY id ASC"
    ).fetchall()
    return [(int(r[0]), tuple(int(x) for x in r[1:8])) for r in redovi]


def ciljno_kolo(istorija):
    """Sledeće očekivano kolo = poslednje uneto + 1 (numeracija godina*1000+br).

    Napomena: prelazak u novu godinu (npr. 2026104 -> 2027001) ne može da se
    predvidi unapred jer broj kola po godini varira; prognoze za kolo koje se
    nikad ne unese prosto ostaju neocenjene (PLAN §4, edge case).
    """
    if not istorija:
        return None
    return istorija[-1][0] + 1


# ----------------------------------------------------------------------------
# Uživo tok
# ----------------------------------------------------------------------------

def _predlozi_za(istorija, cilj, period):
    per = period if period and period > 0 else None
    rezultat = {}
    for metod, (_naziv, fn, _opis) in PREDIKTORI.items():
        rezultat[metod] = fn(istorija, per, ciljno_kolo=cilj)
    return rezultat


def generisi_uzivo(conn, period=0):
    """Vraća uživo predloge za sledeće kolo; računa i upisuje ih ako ne postoje (PLAN §4.1)."""
    istorija = istorija_iz_conn(conn)
    cilj = ciljno_kolo(istorija)
    if cilj is None:
        return {"ciljno_kolo": None, "predlozi": []}

    postojece = {p["metod"]: p for p in baza.prognoze_za_kolo(conn, cilj, "uzivo")}
    if not postojece:
        for metod, broj in _predlozi_za(istorija, cilj, period).items():
            if broj is not None:
                baza.sacuvaj_prognozu(conn, cilj, metod, broj, period or None, "uzivo")
        postojece = {p["metod"]: p for p in baza.prognoze_za_kolo(conn, cilj, "uzivo")}

    predlozi = []
    for metod, (naziv, _fn, opis) in PREDIKTORI.items():
        p = postojece.get(metod)
        if p:
            predlozi.append({"metod": metod, "naziv": naziv, "opis": opis,
                             "broj": p["broj"], "period": p["period"],
                             "kreirano": p["kreirano"]})
    return {"ciljno_kolo": cilj, "predlozi": predlozi}


def preracunaj_uzivo(conn, period=0):
    """Eksplicitno preračunava neocenjene uživo predloge (dugme „Preračunaj").

    Ocenjene prognoze sacuvaj_prognozu odbija — one su nepromenljive.
    """
    istorija = istorija_iz_conn(conn)
    cilj = ciljno_kolo(istorija)
    if cilj is None:
        return {"ciljno_kolo": None, "predlozi": []}
    for metod, broj in _predlozi_za(istorija, cilj, period).items():
        if broj is not None:
            baza.sacuvaj_prognozu(conn, cilj, metod, broj, period or None, "uzivo")
    return generisi_uzivo(conn, period)


def _predlozi_komb_za(istorija, cilj, period):
    per = period if period and period > 0 else None
    return {metod: fn(istorija, per, ciljno_kolo=cilj)
            for metod, (_naziv, fn, _opis) in PREDIKTORI_KOMB.items()}


def _komb_row_u_predlog(p):
    return {"metod": p["metod"], "naziv": PREDIKTORI_KOMB[p["metod"]][0],
            "opis": PREDIKTORI_KOMB[p["metod"]][2],
            "kombinacija": [int(x) for x in (p["kombinacija"] or "").split(",") if x.strip()],
            "period": p["period"], "kreirano": p["kreirano"]}


def generisi_uzivo_komb(conn, period=0):
    """Uživo kombinacijski predlozi za sledeće kolo; upisuje ih ako ne postoje (§5)."""
    istorija = istorija_iz_conn(conn)
    cilj = ciljno_kolo(istorija)
    if cilj is None:
        return {"ciljno_kolo": None, "predlozi": []}

    def komb_redovi():
        return {p["metod"]: p for p in baza.prognoze_za_kolo(conn, cilj, "uzivo")
                if p.get("vrsta") == "komb"}

    postojece = komb_redovi()
    if not postojece:
        for metod, komb in _predlozi_komb_za(istorija, cilj, period).items():
            if komb is not None:
                baza.sacuvaj_prognozu_komb(conn, cilj, metod, ",".join(map(str, komb)),
                                           period or None, "uzivo")
        postojece = komb_redovi()

    predlozi = [_komb_row_u_predlog(postojece[m]) for m in PREDIKTORI_KOMB if m in postojece]
    return {"ciljno_kolo": cilj, "predlozi": predlozi}


def preracunaj_uzivo_komb(conn, period=0):
    """Preračunava neocenjene uživo kombinacijske predloge (ocenjene sacuvaj odbija)."""
    istorija = istorija_iz_conn(conn)
    cilj = ciljno_kolo(istorija)
    if cilj is None:
        return {"ciljno_kolo": None, "predlozi": []}
    for metod, komb in _predlozi_komb_za(istorija, cilj, period).items():
        if komb is not None:
            baza.sacuvaj_prognozu_komb(conn, cilj, metod, ",".join(map(str, komb)),
                                       period or None, "uzivo")
    return generisi_uzivo_komb(conn, period)


def oceni_prognoze(conn, kolo, dobitni_set):
    """Ocenjuje sve otvorene prognoze za uneto kolo (hook iz čuvanja kola).

    Jednobrojne: pogodak 0/1 (PLAN_PROGNOZA §4.3). Kombinacijske: preklapanje 0..7
    preko zajedničkog bitmask modula (PLAN_PROGNOZA_KOMBINACIJE §5).
    """
    dob_maska = T.maska(dobitni_set)
    jed = conn.execute(
        "SELECT id, broj FROM prognoze WHERE kolo=? AND vrsta='broj' AND pogodak IS NULL",
        (kolo,)).fetchall()
    for r in jed:
        conn.execute("UPDATE prognoze SET pogodak=? WHERE id=?",
                     (1 if r["broj"] in dobitni_set else 0, r["id"]))
    komb = conn.execute(
        "SELECT id, kombinacija FROM prognoze WHERE kolo=? AND vrsta='komb' AND preklapanje IS NULL",
        (kolo,)).fetchall()
    for r in komb:
        brojevi = [int(x) for x in r["kombinacija"].split(",") if x.strip()]
        k = T.preklapanje(T.maska(brojevi), dob_maska)
        conn.execute("UPDATE prognoze SET preklapanje=? WHERE id=?", (k, r["id"]))
    conn.commit()
    return len(jed) + len(komb)


# ----------------------------------------------------------------------------
# Retro-bektest (walk-forward, inkrementalno stanje)
# ----------------------------------------------------------------------------

_LR = 0.005  # learning rate Bajesovog modela — isti kao rangiranje.bajes_verovanja


class _Stanje:
    """Inkrementalno stanje za retro petlju: klizni prozor od RETRO_PERIOD kola.

    Ekvivalencija sa čistim prediktorima nad isečkom istorije se dokazuje u
    tests/test_prognoza.py (uključujući Bajes: množenje/deljenje faktora čuva
    tačne odnose skorova, normalizacija ih izjednačava sa svežim računom).
    """

    def __init__(self, period):
        self.period = period
        self.prozor = deque()                                   # (globalni_indeks, kolo, brojevi)
        self.count = {b: 0 for b in range(1, MAX_BROJ + 1)}     # frekvencija u prozoru
        self.pojave = {b: deque() for b in range(1, MAX_BROJ + 1)}  # globalni indeksi u prozoru
        self.globalni_poslednji = {}                            # poslednja pojava u CELOJ istoriji
        self.bajes = {b: 1.0 / MAX_BROJ for b in range(1, MAX_BROJ + 1)}
        self.parovi = {}                                        # (a<b) -> broj zajedničkih kola u prozoru
        self.suma_par = {b: 0 for b in range(1, MAX_BROJ + 1)}  # zbir povezanosti po broju

    def dodaj(self, indeks, kolo, brojevi):
        self.prozor.append((indeks, kolo, brojevi))
        s = set(brojevi)
        for b in brojevi:
            self.count[b] += 1
            self.pojave[b].append(indeks)
            self.globalni_poslednji[b] = indeks
        for b in range(1, MAX_BROJ + 1):
            self.bajes[b] *= (1 + _LR) if b in s else (1 - _LR)
        self._normalizuj()
        bs = sorted(s)
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                k = (bs[i], bs[j])
                self.parovi[k] = self.parovi.get(k, 0) + 1
                self.suma_par[bs[i]] += 1
                self.suma_par[bs[j]] += 1
        if len(self.prozor) > self.period:
            self._ukloni_najstarije()

    def _ukloni_najstarije(self):
        indeks, _kolo, brojevi = self.prozor.popleft()
        s = set(brojevi)
        for b in brojevi:
            self.count[b] -= 1
            if self.pojave[b] and self.pojave[b][0] == indeks:
                self.pojave[b].popleft()
        for b in range(1, MAX_BROJ + 1):
            self.bajes[b] /= (1 + _LR) if b in s else (1 - _LR)
        self._normalizuj()
        bs = sorted(s)
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                k = (bs[i], bs[j])
                self.parovi[k] -= 1
                self.suma_par[bs[i]] -= 1
                self.suma_par[bs[j]] -= 1

    def _normalizuj(self):
        suma = sum(self.bajes.values())
        if suma > 0:
            for b in self.bajes:
                self.bajes[b] /= suma

    # --- predikcije iz stanja (ogledaju čiste funkcije iz prediktori.py) ---

    def hot(self):
        return max(range(1, MAX_BROJ + 1),
                   key=lambda b: (self.count[b],
                                  self.pojave[b][-1] if self.pojave[b] else -1, -b))

    def cold(self):
        return min(range(1, MAX_BROJ + 1),
                   key=lambda b: (self.count[b], self.globalni_poslednji.get(b, -1), b))

    def bayes(self):
        return max(range(1, MAX_BROJ + 1), key=lambda b: (self.bajes[b], -b))

    def hybrid(self):
        v = self.bajes
        top20 = [b for b, _ in sorted(v.items(), key=lambda x: (-x[1], x[0]))[:20]]
        maks = max(self.suma_par.values()) if self.suma_par else 0

        def finalni(b):
            bonus = sum(self.parovi.get((min(b, t), max(b, t)), 0) for t in top20 if t != b)
            norm = (bonus / maks) if maks > 0 else 0
            return v[b] * 0.8 + norm * 0.2

        return max(range(1, MAX_BROJ + 1), key=lambda b: (finalni(b), -b))

    def rhythm(self, korak):
        najbolji, najbolji_odnos = None, -1.0
        for b in range(1, MAX_BROJ + 1):
            p = self.pojave[b]
            if len(p) < 2:
                continue
            razmaci = [p[i + 1] - p[i] for i in range(len(p) - 1)]
            r = sum(razmaci) / len(razmaci)
            d = korak - p[-1]
            odnos = d / r if r > 0 else 0
            if odnos > najbolji_odnos or (odnos == najbolji_odnos and (najbolji is None or b < najbolji)):
                najbolji, najbolji_odnos = b, odnos
        return najbolji

    def fresh(self):
        return int(self.prozor[-1][2][0]) if self.prozor else None


def retro_bektest(conn, retro_period=RETRO_PERIOD, min_start=MIN_START):
    """Walk-forward bektest nad celom istorijom (PLAN §5). Briše stare retro redove.

    Za svako kolo N (posle min_start): predviđa iz stanja izgrađenog od kola < N,
    ocenjuje, PA TEK ONDA dodaje kolo N u stanje — to je garancija protiv curenja.
    """
    import random as _rnd

    pocetak = time.time()
    istorija = istorija_iz_conn(conn)
    baza.obrisi_retro_prognoze(conn)
    if len(istorija) <= min_start:
        return {"kola_ocenjeno": 0, "trajanje_s": 0.0, "poruka": "Premalo istorije za bektest."}

    stanje = _Stanje(retro_period)
    sada = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    redovi = []
    redovi_komb = []
    for korak, (kolo, brojevi) in enumerate(istorija):
        if korak >= min_start:
            dobitni = set(brojevi)
            predlozi = {
                "hot": stanje.hot(),
                "cold": stanje.cold(),
                "bayes": stanje.bayes(),
                "hybrid": stanje.hybrid(),
                "rhythm": stanje.rhythm(korak),
                "fresh": stanje.fresh(),
                "random": _rnd.Random(kolo).randint(1, MAX_BROJ),
            }
            for metod, broj in predlozi.items():
                if broj is None:
                    continue
                redovi.append((kolo, metod, broj, retro_period, "retro",
                               1 if broj in dobitni else 0, sada))
            # Kombinacijski prediktori: čiste funkcije nad istorijom STROGO PRE N
            # (isti obrazac kao test bez-curenja; garantuje da matrica ne vidi ciljno kolo).
            pre = istorija[:korak]
            dob_maska = T.maska(brojevi)
            for metod, (_n, fn, _o) in PREDIKTORI_KOMB.items():
                komb = fn(pre, retro_period, ciljno_kolo=kolo)
                if komb is None:
                    continue
                k = T.preklapanje(T.maska(komb), dob_maska)
                redovi_komb.append((kolo, metod, retro_period, "retro", sada,
                                    ",".join(map(str, komb)), k))
        stanje.dodaj(korak, kolo, brojevi)

    conn.executemany(
        "INSERT OR REPLACE INTO prognoze (kolo, metod, broj, period, izvor, pogodak, kreirano) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)", redovi)
    conn.executemany(
        "INSERT OR REPLACE INTO prognoze "
        "(kolo, metod, broj, period, izvor, pogodak, kreirano, vrsta, kombinacija, preklapanje) "
        "VALUES (?, ?, NULL, ?, ?, NULL, ?, 'komb', ?, ?)", redovi_komb)
    conn.commit()
    return {
        "kola_ocenjeno": len(istorija) - min_start,
        "redova": len(redovi) + len(redovi_komb),
        "redova_broj": len(redovi),
        "redova_komb": len(redovi_komb),
        "trajanje_s": round(time.time() - pocetak, 2),
    }


# ----------------------------------------------------------------------------
# Statistika i serije za grafikon
# ----------------------------------------------------------------------------

def statistika(conn, izvor="uzivo"):
    """Po metodu: n, k, uspešnost, dvostrani binomni test, zaključak (PLAN §7.3)."""
    rezultat = []
    for metod, (naziv, _fn, opis) in PREDIKTORI.items():
        redovi = conn.execute(
            "SELECT pogodak FROM prognoze WHERE metod=? AND izvor=? AND pogodak IS NOT NULL "
            "ORDER BY kolo ASC", (metod, izvor)).fetchall()
        n = len(redovi)
        k = sum(r["pogodak"] for r in redovi)
        stavka = {"metod": metod, "naziv": naziv, "opis": opis, "n": n, "k": k,
                  "uspesnost": round(100 * k / n, 2) if n else None,
                  "ocekivano": round(100 * BASELINE, 2), "p": None, "zakljucak": "—"}
        if n > 0:
            p = binomtest(k, n, BASELINE, alternative="two-sided").pvalue
            stavka["p"] = round(float(p), 5)
            if p < PRAG:
                stavka["zakljucak"] = "Odskače (proveriti!)"
            else:
                stavka["zakljucak"] = "Nerazlučivo od slučajnosti"
        rezultat.append(stavka)
    return {"metode": rezultat, "baseline": round(100 * BASELINE, 2),
            "prag": PRAG, "broj_metoda": len(PREDIKTORI)}


def serije(conn, izvor="uzivo"):
    """Kumulativna uspešnost (%) po metodu + pojas pouzdanosti oko baseline-a (PLAN §7.2)."""
    out = {}
    n_max = 0
    for metod in PREDIKTORI:
        redovi = conn.execute(
            "SELECT pogodak FROM prognoze WHERE metod=? AND izvor=? AND pogodak IS NOT NULL "
            "ORDER BY kolo ASC", (metod, izvor)).fetchall()
        kum, k = [], 0
        for i, r in enumerate(redovi, 1):
            k += r["pogodak"]
            kum.append(round(100 * k / i, 2))
        out[metod] = kum
        n_max = max(n_max, len(kum))

    # 95% pojas za binomnu raspodelu p=7/39 (normalna aproksimacija), sužava se sa n
    p = BASELINE
    donja, gornja = [], []
    for n in range(1, n_max + 1):
        margina = 1.96 * (p * (1 - p) / n) ** 0.5
        donja.append(round(100 * max(0.0, p - margina), 2))
        gornja.append(round(100 * min(1.0, p + margina), 2))

    return {"serije": out, "n_max": n_max, "baseline": round(100 * p, 2),
            "pojas_donja": donja, "pojas_gornja": gornja,
            "nazivi": {m: PREDIKTORI[m][0] for m in PREDIKTORI}}


# ----------------------------------------------------------------------------
# Kombinacijska statistika (PLAN_PROGNOZA_KOMBINACIJE §7)
# ----------------------------------------------------------------------------

def _komb_preklapanja(conn, metod, izvor):
    """(kolo, preklapanje) za ocenjene kombinacijske prognoze, hronološki."""
    redovi = conn.execute(
        "SELECT kolo, preklapanje FROM prognoze WHERE metod=? AND izvor=? AND vrsta='komb' "
        "AND preklapanje IS NOT NULL ORDER BY kolo ASC", (metod, izvor)).fetchall()
    return [(r["kolo"], r["preklapanje"]) for r in redovi]


def statistika_komb(conn, izvor="uzivo"):
    """Po kombinacijskom metodu: n, prosek preklapanja, maks, z-test, zaključak (§7.4)."""
    rezultat = []
    for metod, (naziv, _fn, opis) in PREDIKTORI_KOMB.items():
        parovi = _komb_preklapanja(conn, metod, izvor)
        n = len(parovi)
        vals = [k for _kolo, k in parovi]
        prosek = sum(vals) / n if n else None
        maks = max(vals) if n else None
        maks_kolo = parovi[vals.index(maks)][0] if n else None
        stavka = {"metod": metod, "naziv": naziv, "opis": opis, "n": n,
                  "prosek": round(prosek, 3) if prosek is not None else None,
                  "ocekivano": round(MU_PREKL, 3), "maks": maks, "maks_kolo": maks_kolo,
                  "p": None, "zakljucak": "—"}
        if n >= 30:
            _z, p = T.z_test_proseka(prosek, n)
            stavka["p"] = round(p, 5)
            stavka["zakljucak"] = "Odskače (proveriti!)" if p < PRAG_KOMB else "Nerazlučivo od slučajnosti"
        elif n > 0:
            stavka["zakljucak"] = "Premalo podataka (n < 30)"
        rezultat.append(stavka)
    return {"metode": rezultat, "ocekivano": round(MU_PREKL, 3),
            "sigma": round(T.sigma_preklapanja(), 4),
            "prag": PRAG_KOMB, "broj_metoda": len(PREDIKTORI) + len(PREDIKTORI_KOMB)}


def serije_komb(conn, izvor="uzivo"):
    """Kumulativni prosek preklapanja po metodu + pojas μ ± 1,96·σ/√n (§7.2)."""
    out = {}
    n_max = 0
    for metod in PREDIKTORI_KOMB:
        vals = [k for _kolo, k in _komb_preklapanja(conn, metod, izvor)]
        kum, s = [], 0
        for i, k in enumerate(vals, 1):
            s += k
            kum.append(round(s / i, 4))
        out[metod] = kum
        n_max = max(n_max, len(kum))

    donja, gornja = [], []
    for n in range(1, n_max + 1):
        d, g = T.pojas_proseka(n)
        donja.append(round(d, 4))
        gornja.append(round(g, 4))

    return {"serije": out, "n_max": n_max, "baseline": round(MU_PREKL, 4),
            "pojas_donja": donja, "pojas_gornja": gornja,
            "nazivi": {m: PREDIKTORI_KOMB[m][0] for m in PREDIKTORI_KOMB}}


def histogram_komb(conn, izvor="uzivo", metod=None):
    """Raspodela preklapanja (k=0..7) za jedan metod vs. teorija + hi-kvadrat (§7.3)."""
    if metod not in PREDIKTORI_KOMB:
        metod = next(iter(PREDIKTORI_KOMB))
    vals = [k for _kolo, k in _komb_preklapanja(conn, metod, izvor)]
    brojaci = [0] * (BROJEVA_U_KOMBINACIJI + 1)
    for k in vals:
        brojaci[k] += 1
    n = len(vals)
    posmatrano_udeo = [round(100 * brojaci[k] / n, 3) if n else 0.0
                       for k in range(BROJEVA_U_KOMBINACIJI + 1)]
    teorija_udeo = [round(100 * T.hipergeom_pmf(k), 3) for k in range(BROJEVA_U_KOMBINACIJI + 1)]
    return {
        "metod": metod, "naziv": PREDIKTORI_KOMB[metod][0], "n": n,
        "histogram": {"k": list(range(BROJEVA_U_KOMBINACIJI + 1)),
                      "posmatrano_broj": brojaci, "posmatrano_udeo": posmatrano_udeo,
                      "teorija_udeo": teorija_udeo},
        "test": T.hi_kvadrat_preklapanje(brojaci, n),
    }
