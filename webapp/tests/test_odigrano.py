"""Testovi dnevnika odigranih kombinacija — PLAN_TIKETI_GRADITELJ §3.

  - uvoz starih tiketa: jednom, kao „bez kola", neispravni se preskaču;
  - upis: jedinstven po (kolo, kombinacija), ista kombinacija u drugom kolu dozvoljena;
  - sličnost sa istorijom: tačno brojanje poklapanja i hipergeometrijsko očekivanje;
  - lista: status po kolu i pogoci za izvučena kola.

Radi nad privremenom bazom (ne dira loto_baza.db).

Pokretanje:  python -X utf8 -m webapp.tests.test_odigrano
"""

import os
import sqlite3
import sys
import tempfile
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from webapp.core import baza, odigrano, razlicitost  # noqa: E402
from webapp.tests.test_prognoza import nova_baza, sinteticka_istorija  # noqa: E402


def _ukloni(conn, putanja):
    conn.close()
    os.remove(putanja)


def test_uvoz_starih_tiketa():
    fd, putanja = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(putanja)
    # Stara baza: samo odigrani_tiketi, bez tabele odigrano.
    c = sqlite3.connect(putanja)
    c.execute("""CREATE TABLE odigrani_tiketi (
        id INTEGER PRIMARY KEY AUTOINCREMENT, kombinacija TEXT UNIQUE,
        status TEXT DEFAULT 'aktivan', poslednji_rezultat INTEGER,
        datum_provere TEXT, dodatne_metrike TEXT)""")
    for s in ["(7, 1, 2, 3, 4, 5, 6)", "(ML)(10, 11, 12, 13, 14, 15, 16)",
              "(1, 2, 3)", "(1, 2, 3, 4, 5, 6, 40)"]:
        c.execute("INSERT INTO odigrani_tiketi (kombinacija) VALUES (?)", (s,))
    c.commit()
    c.close()

    baza.postavi_bazu(putanja)
    baza.postavi_bazu(putanja)          # drugi poziv ne sme da duplira uvoz
    conn = baza.konekcija(putanja)
    try:
        redovi = conn.execute("SELECT kolo, kombinacija, izvor FROM odigrano ORDER BY id").fetchall()
        assert [tuple(r) for r in redovi] == [
            (None, "1,2,3,4,5,6,7", "uvoz"),
            (None, "10,11,12,13,14,15,16", "uvoz"),
        ], [tuple(r) for r in redovi]
        # stara tabela ostaje netaknuta (desktop je koristi)
        assert conn.execute("SELECT COUNT(*) FROM odigrani_tiketi").fetchone()[0] == 4
    finally:
        _ukloni(conn, putanja)
    print("test_uvoz_starih_tiketa: OK")


def test_upis_jedinstven_po_kolu():
    conn, putanja = nova_baza(sinteticka_istorija(10, seme=3))
    try:
        k = odigrano.u_csv(odigrano.normalizuj([9, 3, 1, 22, 15, 30, 39]))
        assert k == "1,3,9,15,22,30,39"
        assert baza.dodaj_odigrano(conn, 2026081, k) is not None
        assert baza.dodaj_odigrano(conn, 2026081, k) is None          # isto kolo → odbijeno
        assert baza.dodaj_odigrano(conn, 2026082, k) is not None      # drugo kolo → dozvoljeno
        assert baza.dodaj_odigrano(conn, 2026081, "1,2,3,4,5,6,7") is not None   # druga u istom kolu
    finally:
        _ukloni(conn, putanja)
    print("test_upis_jedinstven_po_kolu: OK")


def test_validacija():
    for losa in ([1, 2, 3], [1, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 40]):
        try:
            odigrano.normalizuj(losa)
        except ValueError:
            continue
        raise AssertionError(f"prihvaćeno: {losa}")
    assert odigrano.ispravno_kolo(2026081) and not odigrano.ispravno_kolo(2026000)
    assert not odigrano.ispravno_kolo(81)
    assert odigrano.ispravan_izvor("prognoza:k_hot7") and not odigrano.ispravan_izvor("nesto")
    print("test_validacija: OK")


def test_slicnost_brojanje():
    istorija = [
        (2020001, (1, 2, 3, 4, 5, 6, 7)),        # 7 istih (drugi redosled ne sme da smeta)
        (2020002, (7, 6, 5, 4, 3, 2, 39)),       # 6
        (2020003, (1, 2, 3, 4, 5, 38, 39)),      # 5
        (2020004, (1, 2, 3, 4, 36, 38, 39)),     # 4
        (2020005, (1, 2, 3, 35, 36, 38, 39)),    # 3 — ne ulazi u prikaz
    ]
    r = odigrano.slicnost(istorija, [7, 6, 5, 4, 3, 2, 1])
    assert r["maks"] == 7 and r["ista"] == [2020001] and r["upozorenje"]
    assert {x["k"]: x["broj"] for x in r["raspodela"]} == {7: 1, 6: 1, 5: 1, 4: 1}
    assert [n["kolo"] for n in r["najbliza"]] == [2020001, 2020002, 2020003, 2020004]

    # Očekivanje je hipergeometrijsko × broj kola.
    ukupno = comb(39, 7)
    for x in r["raspodela"]:
        tacno = comb(7, x["k"]) * comb(32, 7 - x["k"]) / ukupno * len(istorija)
        assert abs(x["ocekivano"] - tacno) < 1e-12

    bez = odigrano.slicnost(istorija, [10, 11, 12, 13, 14, 15, 16])
    assert bez["maks"] == 0 and not bez["upozorenje"] and not bez["ista"] and not bez["najbliza"]
    print("test_slicnost_brojanje: OK")


def test_slicnost_na_sintetici_blizu_ocekivanja():
    """Na 1.400 nasumičnih kola broj kola sa 4 poklapanja je reda 16 (§3.3)."""
    istorija = sinteticka_istorija(1400, seme=5)
    r = odigrano.slicnost(istorija, [3, 8, 12, 19, 22, 30, 37])
    k4 = next(x for x in r["raspodela"] if x["k"] == 4)
    assert abs(k4["ocekivano"] - 15.8) < 0.1, k4
    assert 4 <= k4["broj"] <= 32, k4                 # ~4σ oko očekivanja
    print(f"test_slicnost_na_sintetici_blizu_ocekivanja: OK (4/7: {k4['broj']} naspram {k4['ocekivano']:.1f})")


def test_lista_status_i_pogoci():
    istorija = [(2026001, (5, 1, 9, 13, 20, 33, 2)), (2026002, (4, 8, 15, 16, 23, 34, 39))]
    conn, putanja = nova_baza(istorija)
    try:
        assert odigrano.sledece_kolo(conn) == 2026003
        baza.dodaj_odigrano(conn, 2026002, "4,8,15,16,24,35,38", "rucno", "test")
        baza.dodaj_odigrano(conn, 2026003, "1,2,3,4,5,6,7", "generator")
        conn.execute("INSERT INTO odigrano (kolo, kombinacija, izvor, uneto) VALUES (NULL, '1,2,3,4,5,6,8', 'uvoz', 'x')")
        conn.commit()

        redovi = odigrano.lista(conn)
        assert [(r["kolo"], r["status"]) for r in redovi] == [
            (2026003, "ceka"), (2026002, "izvuceno"), (None, "bez_kola")]
        izv = redovi[1]
        assert izv["pogoci"] == 4 and izv["izvuceni"] == [4, 8, 15, 16, 23, 34, 39]
        assert redovi[0]["pogoci"] is None and redovi[0]["brojevi"] == [1, 2, 3, 4, 5, 6, 7]
    finally:
        _ukloni(conn, putanja)
    print("test_lista_status_i_pogoci: OK")


def test_isti_izvor_istorije_kao_mapa():
    """Sličnost čita istoriju istom funkcijom kao Mapa i Različitost."""
    conn, putanja = nova_baza(sinteticka_istorija(30, seme=2))
    try:
        ist = razlicitost.istorija_iz_conn(conn)
        r = odigrano.slicnost(ist, list(ist[-1][1]))
        assert r["ista"] == [ist[-1][0]] and r["broj_kola"] == 30
    finally:
        _ukloni(conn, putanja)
    print("test_isti_izvor_istorije_kao_mapa: OK")


def main():
    test_uvoz_starih_tiketa()
    test_upis_jedinstven_po_kolu()
    test_validacija()
    test_slicnost_brojanje()
    test_slicnost_na_sintetici_blizu_ocekivanja()
    test_lista_status_i_pogoci()
    test_isti_izvor_istorije_kao_mapa()
    print("\nSVI TESTOVI DNEVNIKA PROSLI [OK]")


if __name__ == "__main__":
    main()
