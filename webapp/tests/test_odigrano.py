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
        odigrano.oceni_sve(conn)

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


# ----------------------------------------------------------------------------
# Faza 2 — mere i ocenjivanje (§4)
# ----------------------------------------------------------------------------

def test_rastojanje_i_skoro():
    a, b = [1, 2, 3, 4, 5, 6, 7], [33, 34, 35, 36, 37, 38, 39]
    assert odigrano.rastojanje(a, a) == 0
    assert odigrano.rastojanje(a, b) == odigrano.rastojanje(b, a) == 224
    assert odigrano.rastojanje([7, 1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7, 8]) == 7   # redosled ne smeta
    # stara mera bi dala 1 (10 „pokriva" i 9 i 11); 1-na-1 uparivanje to ne dozvoljava
    assert odigrano.rastojanje([10, 20], [9, 11]) == 10

    izv = [5, 10, 20, 30]
    assert odigrano.skoro_pogoci([4, 10, 21, 31], izv) == 3      # 5, 20, 30 za ±1; 10 je pogodak
    assert odigrano.skoro_pogoci([10, 11], [10, 11]) == 0        # pogođeni se ne broje
    print("test_rastojanje_i_skoro: OK")


def test_raspodela_dp_tacna():
    """DP mora dati istu raspodelu kao gruba enumeracija (manji problem 7 od 15)."""
    from collections import Counter
    from itertools import combinations
    izv = [2, 3, 7, 8, 11, 13, 15]
    dp = odigrano.raspodela_rastojanja(izv, n=15)
    gruba = Counter(odigrano.rastojanje(x, izv) for x in combinations(range(1, 16), 7))
    assert {D: int(c) for D, c in enumerate(dp) if c} == dict(gruba)

    puna = odigrano.raspodela_rastojanja([3, 8, 12, 19, 22, 30, 37])
    assert int(puna.sum()) == comb(39, 7) and int(puna[0]) == 1
    print("test_raspodela_dp_tacna: OK")


def test_percentil_kalibrisan():
    """Srednji rang: očekivani percentil nasumičnog tiketa je TAČNO 0,5."""
    r = odigrano.raspodela_rastojanja([1, 9, 14, 20, 26, 33, 38])
    T = int(r.sum())
    ocekivano = sum(int(r[D]) * odigrano.percentil(D, r) for D in range(len(r))) / T
    assert abs(ocekivano - 0.5) < 1e-12, ocekivano
    # pun pogodak: svi ostali su dalji
    p0 = odigrano.percentil(0, r)
    assert abs(p0 - (T - 0.5) / T) < 1e-15
    # monotono: veće rastojanje → manji percentil
    assert odigrano.percentil(10, r) > odigrano.percentil(30, r) > odigrano.percentil(80, r)
    print("test_percentil_kalibrisan: OK")


def test_ocenjivanje_pri_unosu_kola():
    from webapp.core import bektest
    conn, putanja = nova_baza([(2026001, (5, 1, 9, 13, 20, 33, 2))])
    try:
        baza.dodaj_odigrano(conn, 2026002, "4,8,15,16,24,35,38")    # čeka
        baza.dodaj_odigrano(conn, 2026003, "1,2,3,4,5,6,7")         # drugo kolo, ostaje neocenjen
        conn.execute("INSERT INTO odigrano (kolo, kombinacija, izvor, uneto) "
                     "VALUES (NULL, '4,8,15,16,23,34,39', 'uvoz', 'x')")
        conn.commit()

        rez = bektest.dodaj_kolo_i_proveri(conn, 2026002, "2026-01-05", [4, 8, 15, 16, 23, 34, 39])
        assert rez["ocenjeno_odigranih"] == 1
        r = {x["kolo"]: x for x in odigrano.lista(conn)}
        oc = r[2026002]
        assert (oc["pogoci"], oc["skoro"], oc["rastojanje"]) == (4, 3, 3), oc
        assert 0.99 < oc["percentil"] < 1 and oc["ocenjeno"]
        assert r[2026003]["percentil"] is None
        assert r[None]["percentil"] is None                         # uvezeni se ne diraju
    finally:
        _ukloni(conn, putanja)
    print("test_ocenjivanje_pri_unosu_kola: OK")


def test_ocena_prati_izmenu_i_brisanje_kola():
    conn, putanja = nova_baza([(2026001, (1, 2, 3, 4, 5, 6, 7))])
    try:
        baza.dodaj_odigrano(conn, 2026001, "1,2,3,4,5,6,7")
        odigrano.oceni_sve(conn)
        assert odigrano.lista(conn)[0]["pogoci"] == 7

        unos_id = conn.execute("SELECT id FROM istorijski_rezultati").fetchone()[0]
        baza.izmeni_kolo(conn, unos_id, 2026001, "2020-01-01", [1, 2, 3, 30, 31, 32, 33])
        odigrano.oceni_sve(conn)
        assert odigrano.lista(conn)[0]["pogoci"] == 3

        baza.obrisi_kolo(conn, unos_id)
        odigrano.oceni_sve(conn)
        red = odigrano.lista(conn)[0]
        assert red["status"] == "ceka" and red["pogoci"] is None and red["percentil"] is None
    finally:
        _ukloni(conn, putanja)
    print("test_ocena_prati_izmenu_i_brisanje_kola: OK")


# ----------------------------------------------------------------------------
# Faza 3 — kumulativni pregled (§5)
# ----------------------------------------------------------------------------

def _baza_sa_tiketima(istorija, tiketi):
    """tiketi: lista (kolo, brojevi, izvor). Vraća conn, putanja sa ocenjenim dnevnikom."""
    conn, putanja = nova_baza(istorija)
    for kolo, br, izvor in tiketi:
        baza.dodaj_odigrano(conn, kolo, odigrano.u_csv(sorted(br)), izvor)
    odigrano.oceni_sve(conn)
    return conn, putanja


def test_ocekivanje_zavisi_od_tiketa():
    """Zašto se ne poredi sa 50%: kombinacija sa ivice ima nizak percentil i pod H₀."""
    ist = sinteticka_istorija(1400, seme=21)
    osnova = odigrano._null_osnova({k: frozenset(b) for k, b in ist})
    ivica = odigrano.null_vektori([1, 2, 3, 4, 5, 6, 7], osnova)
    sredina = odigrano.null_vektori([4, 10, 15, 20, 25, 30, 36], osnova)
    assert ivica["percentil"].mean() < 0.05, ivica["percentil"].mean()
    assert sredina["percentil"].mean() > 0.6, sredina["percentil"].mean()
    # pogoci ne zavise od položaja: hipergeometrijsko 49/39 za svaki tiket
    for v in (ivica, sredina):
        assert abs(v["pogoci"].mean() - 49 / 39) < 0.08, v["pogoci"].mean()
    print(f"test_ocekivanje_zavisi_od_tiketa: OK (ivica {ivica['percentil'].mean():.3f}, "
          f"sredina {sredina['percentil'].mean():.3f})")


def test_pregled_kalibrisan_na_slucaju():
    """Nasumični tiketi (2 po kolu, 40 kola) → nijedna mera ne sme da odstupa."""
    import random
    rng = random.Random(8)
    ist = sinteticka_istorija(1400, seme=22)
    tiketi = []
    for kolo, _b in ist[-40:]:
        for izvor in ("rucno", "generator"):
            tiketi.append((kolo, rng.sample(range(1, 40), 7), izvor))
    conn, putanja = _baza_sa_tiketima(ist, tiketi)
    try:
        p = odigrano.pregled(conn)
        assert p["n_tiketa"] == 80 and p["n_kola"] == 40
        for mera, t in p["ukupno"].items():
            assert abs(t["z"]) < 3.5, (mera, t)
        assert abs(p["prag"] - 0.025) < 1e-12 and len(p["po_izvoru"]) == 2
        assert not any(g["znacajno"] for g in p["po_izvoru"])
        assert len(p["tacke"]) == 80
    finally:
        _ukloni(conn, putanja)
    print("test_pregled_kalibrisan_na_slucaju: OK (z: " +
          ", ".join(f"{m} {t['z']:+.2f}" for m, t in p["ukupno"].items()) + ")")


def test_pregled_otkriva_signal():
    """Tiketi koji „znaju" 4 broja izvlačenja moraju dati veliki pozitivan z."""
    import random
    rng = random.Random(9)
    ist = sinteticka_istorija(1400, seme=23)
    tiketi = []
    for kolo, b in ist[-15:]:
        znani = rng.sample(list(b), 4)
        ostali = rng.sample([x for x in range(1, 40) if x not in b], 3)
        tiketi.append((kolo, znani + ostali, "prognoza:k_hot7"))
    conn, putanja = _baza_sa_tiketima(ist, tiketi)
    try:
        p = odigrano.pregled(conn)
        assert p["ukupno"]["pogoci"]["z"] > 6, p["ukupno"]["pogoci"]
        assert p["ukupno"]["percentil"]["z"] > 3, p["ukupno"]["percentil"]
        g = p["po_izvoru"]
        assert [x["izvor"] for x in g] == ["prognoza"] and g[0]["znacajno"]
    finally:
        _ukloni(conn, putanja)
    print(f"test_pregled_otkriva_signal: OK (percentil z {p['ukupno']['percentil']['z']:+.1f})")


def test_isti_tiket_dvaput_u_kolu_ne_dupla_dokaz():
    """Kovarijansa: dva ISTA tiketa u kolu nisu dva nezavisna dokaza — z ostaje isti."""
    import random
    rng = random.Random(10)
    ist = sinteticka_istorija(400, seme=24)
    osnova = odigrano._null_osnova({k: frozenset(b) for k, b in ist})
    redovi = []
    for kolo, b in ist[-12:]:
        t = sorted(rng.sample(range(1, 40), 7))
        m = odigrano.oceni(t, b)
        redovi.append({"kolo": kolo, "brojevi": t, **m})
    jednom = odigrano._test(redovi, osnova)
    dvaput = odigrano._test(redovi + [dict(r) for r in redovi], osnova)
    for mera in odigrano.MERE:
        assert abs(jednom[mera]["z"] - dvaput[mera]["z"]) < 1e-9, mera
    print("test_isti_tiket_dvaput_u_kolu_ne_dupla_dokaz: OK")


def main():
    test_uvoz_starih_tiketa()
    test_upis_jedinstven_po_kolu()
    test_validacija()
    test_slicnost_brojanje()
    test_slicnost_na_sintetici_blizu_ocekivanja()
    test_lista_status_i_pogoci()
    test_isti_izvor_istorije_kao_mapa()
    test_rastojanje_i_skoro()
    test_raspodela_dp_tacna()
    test_percentil_kalibrisan()
    test_ocenjivanje_pri_unosu_kola()
    test_ocena_prati_izmenu_i_brisanje_kola()
    test_ocekivanje_zavisi_od_tiketa()
    test_pregled_kalibrisan_na_slucaju()
    test_pregled_otkriva_signal()
    test_isti_tiket_dvaput_u_kolu_ne_dupla_dokaz()
    print("\nSVI TESTOVI DNEVNIKA PROSLI [OK]")


if __name__ == "__main__":
    main()
