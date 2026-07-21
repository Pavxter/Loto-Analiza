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
        conn.commit()
    finally:
        conn.close()


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


def azuriraj_rezultat_bektesta(conn, bektest_id, rezultat, indeks_promasaja, indeks_iznenadjenja):
    conn.execute(
        "UPDATE virtualne_igre SET rezultat=?, indeks_promasaja=?, indeks_iznenadjenja=? WHERE id=?",
        (rezultat, indeks_promasaja, indeks_iznenadjenja, bektest_id),
    )
    conn.commit()


# ----------------------------------------------------------------------------
# AI log
# ----------------------------------------------------------------------------

def sacuvaj_ai_log(conn, tip_zahteva, prompt, odgovor):
    conn.execute(
        "INSERT INTO ai_log (datum_vreme, tip_zahteva, prompt, odgovor) VALUES (?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tip_zahteva, prompt, odgovor),
    )
    conn.commit()
