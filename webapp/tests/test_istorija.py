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

from webapp.core import baza, istorija, prognoza  # noqa: E402


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


def test_sazetak_prozora():
    """Sažetak „šta se dešavalo pre": pokriva svih 39, zbir = prozor_n * 7."""
    conn, putanja, kola = _sinteticka_baza()
    try:
        granica = 2021030
        k = istorija.kontekst(conn, granica, 50)
        s = k["sazetak"]
        assert s["prozor_n"] == 50
        assert len(s["brojevi"]) == 39
        assert sum(b["pojavljivanja"] for b in s["brojevi"]) == 50 * 7
        for b in s["brojevi"]:
            if b["pojavljivanja"] == 0:
                assert b["poslednje"] is None and b["razmak"] is None
            else:
                assert b["poslednje"] <= granica and 0 <= b["razmak"] < 50
        print("test_sazetak_prozora: OK")
    finally:
        _ocisti(conn, putanja)


def test_razlicitost_cilja():
    """Preklapanje cilja sa prošlošću: samo kola < cilj, ne vidi budućnost."""
    conn, putanja, kola = _sinteticka_baza()
    try:
        cilj = 2021030
        d = istorija.razlicitost_cilja(conn, cilj, 20)
        assert d is not None and d["cilj"] == cilj
        assert d["granica"] == 2021029
        assert d["prozor_n"] == 20
        assert 0 <= d["sa_prethodnim"]["k"] <= 7
        assert d["maks"]["kolo"] < cilj                      # rekord je iz prošlosti
        assert all(p["puta"] > 0 for p in d["ponovljeni_parovi"])
        assert istorija.razlicitost_cilja(conn, 9999999, 20) is None
        # anti-curenje: kola posle cilja ne smeju menjati rezultat
        for kolo in (2021051, 2021052):
            baza.dodaj_kolo(conn, kolo, "2099-01-01", [1, 2, 3, 4, 5, 6, 7])
        conn.commit()
        assert istorija.razlicitost_cilja(conn, cilj, 20) == d
        print("test_razlicitost_cilja: OK")
    finally:
        _ocisti(conn, putanja)


def test_rangiranje_na_granici():
    """Rangiranje po tri metode: kompletno, i nezavisno od budućnosti."""
    conn, putanja, kola = _sinteticka_baza()
    try:
        granica = 2021030
        r = istorija.rangiranje(conn, granica, 50)
        assert len(r["tabela"]) == 39 and r["broj_kola"] == 50
        for metoda in ("frekvencija", "bajes", "hibrid"):
            rangovi = sorted(red[metoda]["rang"] for red in r["tabela"])
            assert rangovi == list(range(1, 40)), f"{metoda}: rangovi nisu 1..39"
        # anti-curenje
        for kolo in (2021051, 2021052, 2021053):
            baza.dodaj_kolo(conn, kolo, "2099-01-01", [1, 2, 3, 4, 5, 6, 7])
        conn.commit()
        assert istorija.rangiranje(conn, granica, 50) == r
        print("test_rangiranje_na_granici: OK")
    finally:
        _ocisti(conn, putanja)


def test_prognoza_jednaka_retro_bektestu():
    """Vremeplov mora dati IDENTIČNO što i red retro-bektesta za isto kolo (§1.4).

    Dva odvojena puta bi se garantovano razišla; ovo dokazuje da je put jedan.
    """
    conn, putanja, kola = _sinteticka_baza()
    try:
        prognoza.retro_bektest(conn)                 # min_start=50 → retro za 2021001..2021050
        ist = prognoza.istorija_iz_conn(conn)
        for cilj in (2021001, 2021025, 2021050):
            granica = istorija.prethodno_kolo(conn, cilj)
            p = prognoza.prognoza_u_tacki(ist, granica)
            assert p["cilj"] == cilj, (p["cilj"], cilj)
            redovi = baza.prognoze_za_kolo(conn, cilj, "retro")
            retro_jed = {r["metod"]: r["broj"] for r in redovi if r["vrsta"] != "komb"}
            retro_komb = {r["metod"]: r["kombinacija"] for r in redovi if r["vrsta"] == "komb"}
            for m, b in p["jedan_broj"].items():
                assert retro_jed[m] == b, f"{cilj}/{m}: retro={retro_jed[m]} tacka={b}"
            for m, k in p["kombinacija"].items():
                assert retro_komb[m] == ",".join(map(str, k)), f"{cilj}/{m} komb razlika"
        print("test_prognoza_jednaka_retro_bektestu: OK")
    finally:
        _ocisti(conn, putanja)


def test_prognoza_ishod_i_bez_curenja():
    """Ishod: pogodak/preklapanje tačni; nema cilja kad je granica najnovije;
    kola posle cilja ne menjaju ni prognozu ni ocenu (anti-curenje)."""
    conn, putanja, kola = _sinteticka_baza()
    try:
        granica = 2021029                            # cilj = 2021030
        ishod = istorija.prognoza_ishod(conn, granica)
        assert ishod["cilj_postoji"] and ishod["cilj"] == 2021030
        stvarni = set(ishod["stvarni"])
        for red in ishod["jedan_broj"]:
            assert red["pogodak"] == (1 if red["broj"] in stvarni else 0)
        for red in ishod["kombinacija"]:
            assert red["preklapanje"] == len(set(red["kombinacija"]) & stvarni)
        # granica = najnovije → nema narednog kola za ocenu
        naj = istorija.najnovije_kolo(conn)
        assert istorija.prognoza_ishod(conn, naj)["cilj_postoji"] is False
        # anti-curenje: dodaj budućnost posle cilja, rezultat isti
        for k in (2021051, 2021052):
            baza.dodaj_kolo(conn, k, "2099-01-01", [1, 2, 3, 4, 5, 6, 7])
        conn.commit()
        assert istorija.prognoza_ishod(conn, granica) == ishod
        print("test_prognoza_ishod_i_bez_curenja: OK")
    finally:
        _ocisti(conn, putanja)


def main():
    test_granica_iskljucuje_buducnost()
    test_prethodno_sledece_preko_godine()
    test_prozor_velicina()
    test_skok_i_granice()
    test_broj_zavisi_samo_od_prozora()
    test_broj_ne_vidi_buducnost()
    test_sazetak_prozora()
    test_razlicitost_cilja()
    test_rangiranje_na_granici()
    test_prognoza_jednaka_retro_bektestu()
    test_prognoza_ishod_i_bez_curenja()
    print("\nSVI TESTOVI ISTORIJE PROSLI [OK]")


if __name__ == "__main__":
    main()
