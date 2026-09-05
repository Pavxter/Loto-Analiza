"""Sloj pristupa bazi (SQLite).

Zamenjuje i proširuje logiku iz data_manager.py, ali bez oslanjanja na globalno
stanje ili Qt. Svaki poziv koristi kratkotrajnu konekciju (bezbedno za FastAPI koji
može da poziva iz više niti).

Bitna izmena u odnosu na original: bektest više NE čuva sve kombinacije kao ogroman
tekstualni blob. Umesto toga:
  - "lista"      -> ograničen skup kombinacija se čuva kao JSON niz (npr. Top 50),
  - "ceo_bazen"  -> čuva se samo bazen, a sve kombinacije se regenerišu pri proveri.
Time baza pada sa ~59 MB na ~2 MB.
"""

import json
import os
import shutil
import sqlite3
from datetime import datetime

from . import konfig


def napravi_backup(putanja=None):
    """Pravi vremenski označenu rezervnu kopiju baze. Vraća putanju kopije."""
    izvor = putanja or konfig.PUTANJA_BAZE
    if not os.path.exists(izvor):
        return None
    oznaka = datetime.now().strftime("%Y%m%d_%H%M%S")
    cilj = os.path.join(konfig.KOREN_PROJEKTA, f"loto_baza_backup_{oznaka}.db")
    shutil.copy2(izvor, cilj)
    return cilj


def obrisi_svu_istoriju(conn):
    """Briše sve istorijske rezultate i resetuje brojač id-a. Vraća broj obrisanih redova."""
    broj = conn.execute("SELECT COUNT(*) FROM istorijski_rezultati").fetchone()[0]
    conn.execute("DELETE FROM istorijski_rezultati")
    conn.execute("DELETE FROM sqlite_sequence WHERE name='istorijski_rezultati'")
    conn.commit()
    return broj


def konekcija(putanja=None):
    """Vraća novu SQLite konekciju sa row_factory za pristup po imenu kolone."""
    conn = sqlite3.connect(putanja or konfig.PUTANJA_BAZE)
    conn.row_factory = sqlite3.Row
    return conn


def postavi_bazu(putanja=None):
    """Kreira tabele ako ne postoje. Bezbedno je pozvati više puta."""
    conn = konekcija(putanja)
    try:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS istorijski_rezultati (
            id INTEGER PRIMARY KEY AUTOINCREMENT, kolo INTEGER, datum TEXT,
            b1 INTEGER, b2 INTEGER, b3 INTEGER, b4 INTEGER, b5 INTEGER, b6 INTEGER, b7 INTEGER,
            UNIQUE(kolo, datum))""")
        c.execute("""CREATE TABLE IF NOT EXISTS odigrani_tiketi (
            id INTEGER PRIMARY KEY AUTOINCREMENT, kombinacija TEXT UNIQUE,
            status TEXT DEFAULT 'aktivan', poslednji_rezultat INTEGER,
            datum_provere TEXT, dodatne_metrike TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS ai_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, datum_vreme TEXT,
            tip_zahteva TEXT, prompt TEXT, odgovor TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS virtualne_igre (
            id INTEGER PRIMARY KEY AUTOINCREMENT, kolo INTEGER, datum_kreiranja TEXT,
            filter_podesavanja TEXT, lista_kombinacija TEXT, broj_kombinacija INTEGER,
            rezultat TEXT, bazen_brojeva TEXT, indeks_promasaja INTEGER,
            indeks_iznenadjenja REAL, UNIQUE(kolo, filter_podesavanja))""")
        # broj je NULL za kombinacijske prognoze (PLAN_PROGNOZA_KOMBINACIJE §4) → nullable.
        c.execute("""CREATE TABLE IF NOT EXISTS prognoze (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            kolo     INTEGER NOT NULL,
            metod    TEXT    NOT NULL,
            broj     INTEGER,
            period   INTEGER,
            izvor    TEXT    NOT NULL,
            pogodak  INTEGER,
            kreirano TEXT    NOT NULL,
            UNIQUE (kolo, metod, izvor))""")
        conn.commit()
        _prognoze_broj_nullable(c)   # migracija starih baza gde je broj bio NOT NULL
        conn.commit()
        _dodaj_kolone_ako_nema(c, "virtualne_igre", {
            "prosek_preklapanja": "REAL",     # §8: prosek preklapanja kombinacija sa dobitnom
            "maks_preklapanje": "INTEGER",    # §8: najbolje preklapanje u setu
        })
        # PLAN_PROGNOZA_KOMBINACIJE §4: kombinacijske prognoze u istoj tabeli.
        # Napomena: kombinacijski metodi imaju zasebne id-jeve (k_*), pa postojeći
        # UNIQUE(kolo, metod, izvor) i dalje razdvaja 'broj' od 'komb' bez rekonstrukcije.
        _dodaj_kolone_ako_nema(c, "prognoze", {
            "vrsta": "TEXT NOT NULL DEFAULT 'broj'",   # 'broj' | 'komb'
            "kombinacija": "TEXT",                     # CSV 7 brojeva (sortirano) za 'komb'
            "preklapanje": "INTEGER",                  # 0..7 posle ocene; NULL za 'broj'
        })
        conn.commit()
    finally:
        conn.close()


def _dodaj_kolone_ako_nema(cursor, tabela, kolone):
    """Idempotentno dodaje kolone (SQLite ALTER TABLE ADD COLUMN) ako ne postoje."""
    postojece = {r[1] for r in cursor.execute(f"PRAGMA table_info({tabela})").fetchall()}
    for ime, tip in kolone.items():
        if ime not in postojece:
            cursor.execute(f"ALTER TABLE {tabela} ADD COLUMN {ime} {tip}")


def _prognoze_broj_nullable(cursor):
    """Rekonstruiše tabelu prognoze da 'broj' bude nullable (SQLite ne može ALTER-om).

    Potrebno jer stare baze imaju broj INTEGER NOT NULL, a kombinacijske prognoze ga
    ostavljaju NULL. Kopira samo originalnih 8 kolona; nove (vrsta/…) doda kasnije
    _dodaj_kolone_ako_nema. No-op ako je broj već nullable (nove/migrirane baze).
    """
    info = cursor.execute("PRAGMA table_info(prognoze)").fetchall()
    broj = [r for r in info if r[1] == "broj"]
    if not broj or broj[0][3] == 0:   # r[3] = notnull flag; 0 = već nullable
        return
    cursor.execute("ALTER TABLE prognoze RENAME TO _prognoze_staro")
    cursor.execute("""CREATE TABLE prognoze (
        id INTEGER PRIMARY KEY AUTOINCREMENT, kolo INTEGER NOT NULL, metod TEXT NOT NULL,
        broj INTEGER, period INTEGER, izvor TEXT NOT NULL, pogodak INTEGER,
        kreirano TEXT NOT NULL, UNIQUE (kolo, metod, izvor))""")
    cursor.execute("INSERT INTO prognoze (id, kolo, metod, broj, period, izvor, pogodak, kreirano) "
                   "SELECT id, kolo, metod, broj, period, izvor, pogodak, kreirano FROM _prognoze_staro")
    cursor.execute("DROP TABLE _prognoze_staro")


# ----------------------------------------------------------------------------
# Istorijski rezultati
# ----------------------------------------------------------------------------

def sva_kola(conn):
    """Vraća sve istorijske redove kao listu dict-ova, sortirano po id ASC."""
    redovi = conn.execute("SELECT * FROM istorijski_rezultati ORDER BY id ASC").fetchall()
    return [dict(r) for r in redovi]


def dodaj_kolo(conn, kolo, datum, brojevi):
    """Dodaje kolo. Vraća True ako je stvarno ubačeno (nije duplikat)."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO istorijski_rezultati (kolo, datum, b1, b2, b3, b4, b5, b6, b7) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (kolo, datum, *brojevi),
    )
    conn.commit()
    return cur.rowcount > 0


def izmeni_kolo(conn, unos_id, kolo, datum, brojevi):
    conn.execute(
        "UPDATE istorijski_rezultati SET kolo=?, datum=?, b1=?, b2=?, b3=?, b4=?, b5=?, b6=?, b7=? WHERE id=?",
        (kolo, datum, *brojevi, unos_id),
    )
    conn.commit()


def obrisi_kolo(conn, unos_id):
    conn.execute("DELETE FROM istorijski_rezultati WHERE id=?", (unos_id,))
    conn.commit()


# ----------------------------------------------------------------------------
# Odigrani tiketi
# ----------------------------------------------------------------------------

def svi_tiketi(conn):
    redovi = conn.execute("SELECT * FROM odigrani_tiketi ORDER BY id DESC").fetchall()
    return [dict(r) for r in redovi]


def dodaj_tiket(conn, kombinacija_str):
    """Dodaje tiket (string kombinacije, npr. '(1, 2, 3, 4, 5, 6, 7)'). Vraća True ako je nov."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO odigrani_tiketi (kombinacija, status) VALUES (?, 'aktivan')",
        (kombinacija_str,),
    )
    conn.commit()
    return cur.rowcount > 0


def izmeni_tiket(conn, tiket_id, kombinacija_str):
    conn.execute("UPDATE odigrani_tiketi SET kombinacija=? WHERE id=?", (kombinacija_str, tiket_id))
    conn.commit()


def promeni_status_tiketa(conn, tiket_id, status):
    conn.execute("UPDATE odigrani_tiketi SET status=? WHERE id=?", (status, tiket_id))
    conn.commit()


def obrisi_tiket(conn, tiket_id):
    conn.execute("DELETE FROM odigrani_tiketi WHERE id=?", (tiket_id,))
    conn.commit()


# ----------------------------------------------------------------------------
# Bektest (virtualne igre) — novi, vitki format
# ----------------------------------------------------------------------------

def svi_bektestovi(conn):
    redovi = conn.execute("SELECT * FROM virtualne_igre ORDER BY id DESC").fetchall()
    return [dict(r) for r in redovi]


def sacuvaj_bektest(conn, kolo, bazen, filter_opis, tip, kombinacije=None):
    """Čuva bektest u vitkom formatu.

    tip='lista'     -> `kombinacije` (lista lista int-ova) se čuva kao JSON.
    tip='ceo_bazen' -> čuva se samo bazen; kombinacije se regenerišu pri proveri.
    """
    podesavanja = {"tip": tip, "opis": filter_opis}
    bazen_str = ",".join(map(str, sorted(set(bazen)))) if bazen else ""

    if tip == "lista":
        lista_json = json.dumps([sorted(map(int, k)) for k in (kombinacije or [])])
        broj = len(kombinacije or [])
    else:  # ceo_bazen
        lista_json = ""
        from math import comb
        n = len(set(bazen))
        broj = comb(n, konfig.BROJEVA_U_KOMBINACIJI) if n >= konfig.BROJEVA_U_KOMBINACIJI else 0

    cur = conn.execute(
        "INSERT OR IGNORE INTO virtualne_igre "
        "(kolo, datum_kreiranja, filter_podesavanja, lista_kombinacija, broj_kombinacija, bazen_brojeva) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (kolo, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), json.dumps(podesavanja, ensure_ascii=False),
         lista_json, broj, bazen_str),
    )
    conn.commit()
    return cur.rowcount > 0


def obrisi_bektest(conn, unos_id):
    conn.execute("DELETE FROM virtualne_igre WHERE id=?", (unos_id,))
    conn.commit()


def azuriraj_rezultat_bektesta(conn, bektest_id, rezultat, indeks_promasaja, indeks_iznenadjenja,
                               prosek_preklapanja=None, maks_preklapanje=None):
    conn.execute(
        "UPDATE virtualne_igre SET rezultat=?, indeks_promasaja=?, indeks_iznenadjenja=?, "
        "prosek_preklapanja=?, maks_preklapanje=? WHERE id=?",
        (rezultat, indeks_promasaja, indeks_iznenadjenja, prosek_preklapanja, maks_preklapanje, bektest_id),
    )
    conn.commit()


# ----------------------------------------------------------------------------
# Prognoze (strana „Prognoza")
# ----------------------------------------------------------------------------

def sacuvaj_prognozu(conn, kolo, metod, broj, period, izvor):
    """Upisuje/menja prognozu. Ocenjenu prognozu je ZABRANJENO menjati — vraća False.

    UNIQUE(kolo, metod, izvor) + INSERT OR REPLACE: ponovni upis zamenjuje samo
    dok je pogodak IS NULL (poenta eksperimenta, PLAN_PROGNOZA.md §3).
    """
    red = conn.execute(
        "SELECT pogodak FROM prognoze WHERE kolo=? AND metod=? AND izvor=?",
        (kolo, metod, izvor)).fetchone()
    if red is not None and red["pogodak"] is not None:
        return False
    conn.execute(
        "INSERT OR REPLACE INTO prognoze (kolo, metod, broj, period, izvor, pogodak, kreirano) "
        "VALUES (?, ?, ?, ?, ?, NULL, ?)",
        (kolo, metod, broj, period, izvor, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    return True


def sacuvaj_prognozu_komb(conn, kolo, metod, kombinacija_csv, period, izvor):
    """Upisuje/menja kombinacijsku prognozu. Ocenjenu (preklapanje != NULL) ZABRANJENO menjati.

    UNIQUE(kolo, metod, izvor) i dalje razdvaja od jednobrojnih jer su metod id-jevi
    različiti (k_*). Vrsta='komb', broj=NULL, pogodak=NULL (PLAN_PROGNOZA_KOMBINACIJE §4).
    """
    red = conn.execute(
        "SELECT preklapanje FROM prognoze WHERE kolo=? AND metod=? AND izvor=?",
        (kolo, metod, izvor)).fetchone()
    if red is not None and red["preklapanje"] is not None:
        return False
    conn.execute(
        "INSERT OR REPLACE INTO prognoze "
        "(kolo, metod, broj, period, izvor, pogodak, kreirano, vrsta, kombinacija, preklapanje) "
        "VALUES (?, ?, NULL, ?, ?, NULL, ?, 'komb', ?, NULL)",
        (kolo, metod, period, izvor, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), kombinacija_csv))
    conn.commit()
    return True


def prognoze_za_kolo(conn, kolo, izvor=None):
    if izvor:
        redovi = conn.execute(
            "SELECT * FROM prognoze WHERE kolo=? AND izvor=? ORDER BY metod", (kolo, izvor)).fetchall()
    else:
        redovi = conn.execute(
            "SELECT * FROM prognoze WHERE kolo=? ORDER BY izvor, metod", (kolo,)).fetchall()
    return [dict(r) for r in redovi]


def prognoze_lista(conn, izvor=None, metod=None, limit=100, samo_ocenjene=False, vrsta=None):
    uslovi, parametri = [], []
    if izvor:
        uslovi.append("izvor=?"); parametri.append(izvor)
    if metod:
        uslovi.append("metod=?"); parametri.append(metod)
    if vrsta:
        uslovi.append("vrsta=?"); parametri.append(vrsta)
    if samo_ocenjene:
        uslovi.append(("preklapanje IS NOT NULL" if vrsta == "komb" else "pogodak IS NOT NULL"))
    where = ("WHERE " + " AND ".join(uslovi)) if uslovi else ""
    parametri.append(limit)
    redovi = conn.execute(
        f"SELECT * FROM prognoze {where} ORDER BY kolo DESC, metod LIMIT ?", parametri).fetchall()
    return [dict(r) for r in redovi]


def obrisi_retro_prognoze(conn):
    """Briše sve retro prognoze (pred ponovni bektest ili pri zameni istorije)."""
    cur = conn.execute("DELETE FROM prognoze WHERE izvor='retro'")
    conn.commit()
    return cur.rowcount


# ----------------------------------------------------------------------------
# AI log
# ----------------------------------------------------------------------------

def sacuvaj_ai_log(conn, tip_zahteva, prompt, odgovor):
    conn.execute(
        "INSERT INTO ai_log (datum_vreme, tip_zahteva, prompt, odgovor) VALUES (?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tip_zahteva, prompt, odgovor),
    )
    conn.commit()
