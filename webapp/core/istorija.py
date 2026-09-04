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

from . import konfig

BROJEVA = konfig.BROJEVA_U_KOMBINACIJI


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
    }
