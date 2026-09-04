"""Testovi sloja „Istraži istoriju" (UPUTSTVO_PROGRAMER_ISTORIJA.md §3, Faza 1).

Pokriveni kriterijumi ove faze:
  - granica isključuje budućnost (nijedno kolo > granica u podskupu),
  - navigacija radi preko granice godine (ORDER BY, ne kolo±1),
  - prozor vraća tačan broj kola,
  - cilj = prvo stvarno kolo posle granice; None kad je granica najnovije.

Leakage-test nad kopijom baze i test jednakosti sa retro-bektestom dolaze u
kasnijim fazama (prognoza). Ovde se radi nad sintetičkom privremenom bazom.

Pokretanje:  python -X utf8 -m webapp.tests.test_istorija
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from webapp.core import baza, istorija  # noqa: E402


def _sinteticka_baza():
    """Privremena baza: dve godine sa prelazom (…2020-050 -> 2021-001…).

    Vraća (conn, putanja, lista_kola). Pozivalac zatvara conn i briše fajl.
    """
    fd, putanja = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    baza.postavi_bazu(putanja)
    conn = baza.konekcija(putanja)
    kola = [2020000 + i for i in range(1, 51)] + [2021000 + i for i in range(1, 51)]  # 100 kola
    for idx, kolo in enumerate(kola):
        # deterministički set od 7 jedinstvenih brojeva 1..39
        brojevi = [((kolo + p * 5 + idx) % 39) + 1 for p in range(7)]
        vidjeni, jedinstveni = set(), []
        b = 1
        for x in brojevi:                       # popuni duplikate deterministički
            while x in vidjeni:
                x = (x % 39) + 1
            vidjeni.add(x)
            jedinstveni.append(x)
        baza.dodaj_kolo(conn, kolo, f"2020-01-{idx+1:02d}"[:10], jedinstveni)
    conn.commit()
    return conn, putanja, kola


def _ocisti(conn, putanja):
    conn.close()
    try:
        os.remove(putanja)
    except OSError:
        pass


def test_granica_iskljucuje_buducnost():
    conn, putanja, kola = _sinteticka_baza()
    try:
        granica = 2021010
        for prozor in (0, 10, 50, 200):
            podskup = istorija.kola_do(conn, granica, prozor)
            assert all(k["kolo"] <= granica for k in podskup), f"curenje za prozor {prozor}"
        assert istorija.kontekst(conn, granica, 20)["cilj"] == 2021011
        print("test_granica_iskljucuje_buducnost: OK")
    finally:
        _ocisti(conn, putanja)


def test_prethodno_sledece_preko_godine():
    conn, putanja, kola = _sinteticka_baza()
    try:
        # prelaz: 2020050 -> 2021001, nikako 2020051 ili 2021000
        assert istorija.sledece_kolo(conn, 2020050) == 2021001
        assert istorija.prethodno_kolo(conn, 2021001) == 2020050
        assert istorija.prethodno_kolo(conn, 2020001) is None       # najstarije
        assert istorija.sledece_kolo(conn, 2021050) is None         # najnovije
        print("test_prethodno_sledece_preko_godine: OK")
    finally:
        _ocisti(conn, putanja)


def test_prozor_velicina():
    conn, putanja, kola = _sinteticka_baza()
    try:
        granica = 2021050  # najnovije -> puno kola dostupno (100)
        for p in (20, 50, 100, 200):
            n = len(istorija.kola_do(conn, granica, p))
            assert n == min(p, 100), f"prozor {p}: {n}"
        # na početku baze prozor je ograničen brojem dostupnih kola
        assert len(istorija.kola_do(conn, 2020005, 50)) == 5
        assert len(istorija.kola_do(conn, granica, 0)) == 100        # 0 = sva
        print("test_prozor_velicina: OK")
    finally:
        _ocisti(conn, putanja)


def test_skok_i_granice():
    conn, putanja, kola = _sinteticka_baza()
    try:
        k = istorija.kontekst(conn, 2021030, 20)
        assert k["skok_nazad"] == 2021010                  # 20 unazad
        assert k["skok_napred"] == 2021050                 # 20 unapred
        assert k["prethodna_granica"] == 2021029
        assert k["sledeca_granica"] == 2021031
        # uklještenje na krajeve
        assert istorija.kontekst(conn, 2020003, 20)["skok_nazad"] == 2020001
        assert istorija.granice(conn) == {"najstarije": 2020001, "najnovije": 2021050,
                                          "broj_kola": 100}
        print("test_skok_i_granice: OK")
    finally:
        _ocisti(conn, putanja)


def test_broj_zavisi_samo_od_prozora():
    """Detalj broja: prozor menja rezultate; nikad se ne gleda kolo > granica."""
    conn, putanja, kola = _sinteticka_baza()
    try:
        granica = 2021030
        d10 = istorija.detalj_broja(conn, 7, granica, 10)
        d50 = istorija.detalj_broja(conn, 7, granica, 50)
        assert d10["prozor_n"] == 10 and d50["prozor_n"] == 50
        assert d10["u_prozoru"] <= d50["u_prozoru"]          # manji prozor ≤ veći
        assert d10["ukupno"] == d50["ukupno"]                # ukupno nezavisno od prozora
        # nijedno kolo u timeline-u ne prelazi granicu
        assert all(t["kolo"] <= granica for t in d50["timeline"])
        if d50["poslednje_pojavljivanje"] is not None:
            assert d50["poslednje_pojavljivanje"] <= granica
        # zbir raspodele po pozicijama == broj pojavljivanja u prozoru
        assert sum(p["broj"] for p in d50["pozicije"]) == d50["u_prozoru"]
        print("test_broj_zavisi_samo_od_prozora: OK")
    finally:
        _ocisti(conn, putanja)


def test_broj_ne_vidi_buducnost():
    """Detalj na granici je identičan bez obzira na kola posle granice (anti-curenje)."""
    conn, putanja, kola = _sinteticka_baza()
    try:
        granica = 2021020
        pre = istorija.detalj_broja(conn, 13, granica, 30)
        # dodaj „budućnost" (nova kola posle granice) i ponovo izračunaj za istu granicu
        for kolo in (2021051, 2021052, 2021053):
            baza.dodaj_kolo(conn, kolo, "2099-01-01", [13, 1, 2, 3, 4, 5, 6])
        conn.commit()
        posle = istorija.detalj_broja(conn, 13, granica, 30)
        assert pre == posle, "detalj na granici se promenio zbog budućnosti!"
        print("test_broj_ne_vidi_buducnost: OK")
    finally:
        _ocisti(conn, putanja)


def main():
    test_granica_iskljucuje_buducnost()
    test_prethodno_sledece_preko_godine()
    test_prozor_velicina()
    test_skok_i_granice()
    test_broj_zavisi_samo_od_prozora()
    test_broj_ne_vidi_buducnost()
    print("\nSVI TESTOVI ISTORIJE PROSLI [OK]")


if __name__ == "__main__":
    main()
