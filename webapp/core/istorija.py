"""Sloj „Istraži istoriju": sečenje istorije po granici/prozoru i prosleđivanje
postojećim analizama (UPUTSTVO_PROGRAMER_ISTORIJA.md, Faza 1).

Pojmovi (fiksni kroz ceo projekat):
  granica = poslednje kolo koje sistem „zna" (uključivo).
  cilj    = prvo stvarno kolo posle granice u bazi.
Dostupni podaci = sva kola ≤ granica; cilj i sve posle njega su nedostupni
sve do evaluacije.

Ovaj modul NE računa statistiku (frekvenciju, Bajesa, različitost, prognozu). On
samo pravi podskup „kola ≤ granica (poslednjih prozor)" i prosleđuje ga postojećim
modulima. Numeracija kola (godina*1000+redni) nije kontinualna, pa se navigacija
radi upitom nad poretkom u bazi, nikad aritmetikom kolo±1.
"""

from . import (konfig, analitika, rangiranje as rang, razlicitost as razl,
               prognoza as prog, razlicitost_teorija as teorija)
from .prediktori import PREDIKTORI
from .prediktori_komb import PREDIKTORI_KOMB

BROJEVA = konfig.BROJEVA_U_KOMBINACIJI
MAX_BROJ = konfig.MAX_BROJ


def _kolone_brojeva():
    return [f"b{i}" for i in range(1, BROJEVA + 1)]


def _red_u_kolo(r):
    """SQLite red -> dict kola sa brojevima u REDOSLEDU IZVLAČENJA (b1..b7)."""
    return {"kolo": int(r["kolo"]), "datum": r["datum"],
            "brojevi": [int(r[c]) for c in _kolone_brojeva()]}


# ----------------------------------------------------------------------------
# Navigacija (ORDER BY, bez kolo±1)
# ----------------------------------------------------------------------------

def prethodno_kolo(conn, kolo):
    """Najbliže kolo strogo pre datog (ili None ako je najstarije/ne postoji)."""
    r = conn.execute("SELECT kolo FROM istorijski_rezultati WHERE kolo < ? "
                     "ORDER BY kolo DESC LIMIT 1", (kolo,)).fetchone()
    return int(r[0]) if r else None


def sledece_kolo(conn, kolo):
    """Najbliže kolo strogo posle datog (ili None ako je najnovije/ne postoji)."""
    r = conn.execute("SELECT kolo FROM istorijski_rezultati WHERE kolo > ? "
                     "ORDER BY kolo ASC LIMIT 1", (kolo,)).fetchone()
    return int(r[0]) if r else None


def najnovije_kolo(conn):
    r = conn.execute("SELECT kolo FROM istorijski_rezultati ORDER BY kolo DESC LIMIT 1").fetchone()
    return int(r[0]) if r else None


def najstarije_kolo(conn):
    r = conn.execute("SELECT kolo FROM istorijski_rezultati ORDER BY kolo ASC LIMIT 1").fetchone()
    return int(r[0]) if r else None


def _kolo_sa_pomerajem(conn, kolo, pomeraj):
    """Kolo `pomeraj` mesta od datog u hronološkom poretku (id ASC).

    pomeraj < 0 = unazad, > 0 = unapred; rezultat je uklješten na granice baze
    (skok van opsega vraća najstarije/najnovije). Koristi se za dugmad <</>>.
    """
    kola = [int(r[0]) for r in conn.execute(
        "SELECT kolo FROM istorijski_rezultati ORDER BY id ASC")]
    if kolo not in kola:
        return None
    i = max(0, min(len(kola) - 1, kola.index(kolo) + pomeraj))
    return kola[i]


# ----------------------------------------------------------------------------
# Podskup podataka
# ----------------------------------------------------------------------------

def kola_do(conn, granica, prozor=None):
    """Kola ≤ granica, hronološki (id ASC); poslednjih `prozor` (None/0 = sva).

    Vraća listu dict-ova {kolo, datum, brojevi[7]} sa brojevima u redosledu izvlačenja.
    """
    redovi = conn.execute(
        "SELECT * FROM istorijski_rezultati WHERE kolo <= ? ORDER BY id ASC",
        (granica,)).fetchall()
    if prozor and prozor > 0:
        redovi = redovi[-prozor:]
    return [_red_u_kolo(r) for r in redovi]


def detalj_kola(conn, kolo):
    """Jedno kolo + prethodno/sledeće (za navigaciju iz tabele). None ako ne postoji."""
    r = conn.execute("SELECT * FROM istorijski_rezultati WHERE kolo = ?", (kolo,)).fetchone()
    if not r:
        return None
    return {"kolo": _red_u_kolo(r),
            "prethodno": prethodno_kolo(conn, kolo),
            "sledece": sledece_kolo(conn, kolo)}


def detalj_broja(conn, broj, granica, prozor=None):
    """Istorija jednog broja „kakva je bila na granici" (Faza 2).

    Seče DataFrame na kola ≤ granica i prosleđuje ga analitika.detalj_broja —
    nikakav račun se ne radi ovde (UPUTSTVO §1.3). Rezultati zavise isključivo od
    kola ≤ granica (test curenja u kasnijoj fazi).
    """
    df = analitika.ucitaj_df(conn)
    df = df[df["kolo"] <= granica]
    d = analitika.detalj_broja(df, broj, prozor if (prozor and prozor > 0) else None)
    d["granica"] = granica
    return d


# ----------------------------------------------------------------------------
# Objedinjeni kontekst za početni ekran (jedan poziv)
# ----------------------------------------------------------------------------

def granice(conn):
    """Meta bez analize: najstarije/najnovije kolo i ukupan broj (bootstrap UI-ja)."""
    return {"najstarije": najstarije_kolo(conn),
            "najnovije": najnovije_kolo(conn),
            "broj_kola": int(conn.execute(
                "SELECT COUNT(*) FROM istorijski_rezultati").fetchone()[0])}


def kontekst(conn, granica, prozor=None):
    """Sve što početni ekran „Istraži istoriju" treba u jednom pozivu.

    Sadrži prozor kola (tabela), cilj, granicu i mete za navigaciju (prethodna/
    sledeća granica + skok za ceo prozor). Sažetak statistike prozora dolazi u
    kasnijim fazama; ovde je samo raspon datuma.
    """
    kola = kola_do(conn, granica, prozor)
    ukupno = conn.execute(
        "SELECT COUNT(*) FROM istorijski_rezultati WHERE kolo <= ?", (granica,)).fetchone()[0]
    pom = prozor if (prozor and prozor > 0) else None
    df = analitika.ucitaj_df(conn)
    df = df[df["kolo"] <= granica]
    sazetak = analitika.sazetak_prozora(df, pom)      # „šta se dešavalo pre" (Faza 3)
    return {
        "granica": granica,
        "cilj": sledece_kolo(conn, granica),
        "prozor": prozor or 0,
        "kola": kola,                                      # tabela prethodnih kola
        "broj_u_prozoru": len(kola),
        "ukupno_do_granice": int(ukupno),
        "najstarije": najstarije_kolo(conn),
        "najnovije": najnovije_kolo(conn),
        "prethodna_granica": prethodno_kolo(conn, granica),
        "sledeca_granica": sledece_kolo(conn, granica),
        # dugmad << / >> : skok za ceo prozor (za „sva" -> na krajeve baze)
        "skok_nazad": (_kolo_sa_pomerajem(conn, granica, -pom) if pom else najstarije_kolo(conn)),
        "skok_napred": (_kolo_sa_pomerajem(conn, granica, pom) if pom else najnovije_kolo(conn)),
        "raspon_datuma": ([kola[0]["datum"], kola[-1]["datum"]] if kola else None),
        "sazetak": sazetak,
    }


# ----------------------------------------------------------------------------
# Kontekst kola: istorijska različitost i rangiranje „kakvo bi bilo na granici"
# (Faza 3). istorija.py samo seče podskup i prosleđuje postojećim modulima (§1.3).
# ----------------------------------------------------------------------------

def razlicitost_cilja(conn, cilj, prozor=None):
    """Preklapanje izvučene kombinacije `cilj` sa kolima PRE nje (< cilj).

    Prosleđuje isečak razlicitost.preklapanje_sa_istorijom; granica = kolo pre
    cilja. None ako `cilj` ne postoji u bazi.
    """
    r = conn.execute(
        "SELECT b1,b2,b3,b4,b5,b6,b7 FROM istorijski_rezultati WHERE kolo = ?",
        (cilj,)).fetchone()
    if not r:
        return None
    cilj_brojevi = [int(x) for x in r]
    pre = [(kolo, br) for kolo, br in razl.istorija_iz_conn(conn) if kolo < cilj]
    d = razl.preklapanje_sa_istorijom(pre, cilj_brojevi,
                                      prozor if (prozor and prozor > 0) else None)
    d["cilj"] = cilj
    d["granica"] = prethodno_kolo(conn, cilj)
    return d


def rangiranje(conn, granica, prozor=None):
    """Rangiranje brojeva (frekvencija / Bajes / hibrid) „kakvo bi bilo na granici".

    Seče df na kola ≤ granica (poslednjih prozor) i poziva rangiranje.py trima
    metodama; spaja ih u jednu tabelu po broju sa rangom i skorom svake metode.
    """
    df = analitika.ucitaj_df(conn)
    df = df[df["kolo"] <= granica]
    if prozor and prozor > 0:
        df = df.tail(prozor)
    df = df.reset_index(drop=True)

    def _rang_map(lista):
        return {r["broj"]: (poz + 1, float(r["skor"])) for poz, r in enumerate(lista)}

    rf = _rang_map(rang.frekvencija_rang(df))
    rb = _rang_map(rang.bajes_rang(df))
    rh = _rang_map(rang.hibrid_rang(df))
    tabela = [{
        "broj": b,
        "frekvencija": {"rang": rf[b][0], "skor": round(rf[b][1], 6)},
        "bajes": {"rang": rb[b][0], "skor": round(rb[b][1], 6)},
        "hibrid": {"rang": rh[b][0], "skor": round(rh[b][1], 6)},
    } for b in range(1, MAX_BROJ + 1)]
    return {"granica": granica, "prozor": prozor or 0,
            "broj_kola": int(len(df)), "tabela": tabela}


# ----------------------------------------------------------------------------
# Vremeplov prognoze (Faza 4): „šta bi sistem tada predvideo" + stvarni ishod.
# istorija.py samo poziva prognoza.prognoza_u_tacki/oceni_u_tacki i pakuje za UI.
# ----------------------------------------------------------------------------

def _obogati_prognozu(p):
    """Spakuj sirovu prognozu (metod→broj/komb) za UI: dodaj naziv/opis i teoriju."""
    jed = [{"metod": m, "naziv": PREDIKTORI[m][0], "opis": PREDIKTORI[m][2], "broj": b}
           for m, b in p["jedan_broj"].items() if b is not None]
    komb = [{"metod": m, "naziv": PREDIKTORI_KOMB[m][0], "opis": PREDIKTORI_KOMB[m][2],
             "kombinacija": list(k)} for m, k in p["kombinacija"].items() if k]
    return {
        "granica": p["granica"], "cilj": p["cilj"], "period": p["period"],
        "jedan_broj": jed, "kombinacija": komb,
        "teorija": {"ocekivano": round(teorija.ocekivano_preklapanje(), 4),
                    "sigma": round(teorija.sigma_preklapanja(), 4),
                    "baseline_udeo": round(100 * prog.BASELINE, 2)},
    }


def prognoza_u_tacki(conn, granica, metode=None):
    """Šta bi sistem predvideo na granici (za cilj). None ako nema kola ≤ granica."""
    p = prog.prognoza_u_tacki(prog.istorija_iz_conn(conn), granica, metode)
    return _obogati_prognozu(p) if p is not None else None


def prognoza_ishod(conn, granica, metode=None):
    """Prognoza na granici + stvarni ishod cilja + evaluacija (pogodak / preklapanje).

    `cilj_postoji=False` kad granica nema naredno kolo u bazi (tada nema ocene —
    ishod se namerno ne prikazuje dok se kolo ne odigra). None ako nema kola ≤ granica.
    """
    ist = prog.istorija_iz_conn(conn)
    p = prog.prognoza_u_tacki(ist, granica, metode)
    if p is None:
        return None
    out = _obogati_prognozu(p)
    stvarno = dict(ist).get(p["cilj"])          # brojevi cilja ako postoji u bazi
    if stvarno is None:
        out["cilj_postoji"] = False
        return out
    ocena = prog.oceni_u_tacki(p, stvarno)
    for red in out["jedan_broj"]:
        red["pogodak"] = ocena["jedan_broj"][red["metod"]]["pogodak"]
    for red in out["kombinacija"]:
        red["preklapanje"] = ocena["kombinacija"][red["metod"]]["preklapanje"]
    out["cilj_postoji"] = True
    out["stvarni"] = ocena["stvarni"]
    out["sazetak"] = ocena["sazetak"]
    return out
