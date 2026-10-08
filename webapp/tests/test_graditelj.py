"""Testovi Graditelja kombinacije — PLAN_TIKETI_GRADITELJ §7.

  - obrt: tačna formula se slaže sa simulacijom;
  - potencijal: Σz = 0 i ispravne vrednosti za broj koji nije izvučen;
  - bazen: tačno brojevi najvećeg potencijala;
  - pravila: vektorska maska = pojedinačne funkcije Generatora; istorija se isključuje;
  - sklapanje: predlozi prolaze pravila, alternative se razlikuju, skor je zbir
    komponenti, rezultat je reproducibilan.

Pokretanje:  python -X utf8 -m webapp.tests.test_graditelj
"""

import math
import os
import random
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from webapp.core import generator, graditelj as G  # noqa: E402
from webapp.tests.test_prognoza import sinteticka_istorija  # noqa: E402


def test_obrt_tacan():
    tacno = G.ocekivani_obrt()
    assert abs(tacno - 22.236) < 0.001, tacno
    # simulacija: 7 od 39 po kolu dok se ne pojave svi
    rng = random.Random(3)
    duzine = []
    for _ in range(20000):
        vidjeno, t = set(), 0
        while len(vidjeno) < 39:
            vidjeno.update(rng.sample(range(1, 40), 7))
            t += 1
        duzine.append(t)
    sim = sum(duzine) / len(duzine)
    assert abs(sim - tacno) < 0.1, (sim, tacno)
    # sitan slučaj ručno: 1 od 2 po kolu → E = 1 + 2 = 3
    assert abs(G.ocekivani_obrt(2, 1) - 3.0) < 1e-12
    print(f"test_obrt_tacan: OK ({tacno:.3f}, simulacija {sim:.3f})")


def test_potencijal():
    ist = sinteticka_istorija(200, seme=31)
    pot = G.potencijal(ist, 22)
    assert pot["w"] == 22
    assert abs(sum(pot["z"].values())) < 1e-9                  # Σ(E − O) = 7W − 7W = 0
    E, s = 22 * 7 / 39, math.sqrt(22 * 7 / 39 * 32 / 39)
    for b in range(1, 40):
        assert abs(pot["z"][b] - (E - pot["pojava"][b]) / s) < 1e-12
    sva = G.potencijal(ist, 0)
    assert sva["w"] == 200
    print("test_potencijal: OK")


def test_bazen_je_vrh_potencijala():
    ist = sinteticka_istorija(400, seme=32)
    r = G.sklopi(ist, w=50, bazen_vel=15, sa_slicnoscu=False)
    u = {x["broj"] for x in r["bazen"]}
    z = {x["broj"]: x["z"] for x in r["potencijal"]["brojevi"]}
    assert len(u) == 15
    assert min(z[b] for b in u) >= max(z[b] for b in range(1, 40) if b not in u)
    assert all(x["u_bazenu"] == (x["broj"] in u) for x in r["potencijal"]["brojevi"])
    print("test_bazen_je_vrh_potencijala: OK")


def test_pravila_maska_kao_generator():
    rng = random.Random(4)
    B = np.array(sorted({tuple(sorted(rng.sample(range(1, 40), 7))) for _ in range(5000)}))
    for pravila in ({"dekada_max": 3}, {"uzastopni_max": 1}, {"dekada_max": 2, "uzastopni_max": 0},
                    {"zbir_min": 120, "zbir_max": 160, "parni_min": 3, "parni_max": 4}):
        maska = G._pravila_maska(G.matrica_kombinacija(B - 1, 39), np.arange(1, 40), pravila)
        for red, ok in zip(B, maska):
            k = [int(x) for x in red]
            ocek = True
            if "dekada_max" in pravila:
                ocek &= generator.najvise_u_dekadi(k) <= pravila["dekada_max"]
            if "uzastopni_max" in pravila:
                ocek &= generator.broj_uzastopnih(k) <= pravila["uzastopni_max"]
            if "zbir_min" in pravila:
                ocek &= pravila["zbir_min"] <= sum(k) <= pravila["zbir_max"]
                p = sum(1 for x in k if x % 2 == 0)
                ocek &= pravila["parni_min"] <= p <= pravila["parni_max"]
            assert bool(ok) == ocek, (k, pravila)
    print("test_pravila_maska_kao_generator: OK")


def test_istorija_se_iskljucuje():
    istorija = [(2026001, (3, 9, 14, 20, 27, 33, 38)), (2026002, (1, 2, 5, 11, 17, 29, 30))]
    B = np.array([[3, 9, 14, 20, 27, 33, 38],     # izvučena: 7
                  [3, 9, 14, 20, 27, 33, 39],     # 6
                  [3, 9, 14, 20, 27, 32, 39],     # 5 — prolazi
                  [4, 8, 15, 21, 26, 34, 39]])    # 0
    H = G.brojaci(istorija).H()
    assert G._istorija_ok(B, H, 5).tolist() == [False, False, True, True]
    assert G._istorija_ok(B, H, 6).tolist() == [False, True, True, True]
    print("test_istorija_se_iskljucuje: OK")


def test_sklapanje():
    ist = sinteticka_istorija(1400, seme=33)
    r = G.sklopi(ist)
    assert r["ukupno_kombinacija"] == math.comb(15, 7)
    assert 0 < r["prolazi_pravila"] <= r["ukupno_kombinacija"]
    pred = r["predlozi"]
    assert len(pred) == 1 + G.ALTERNATIVA
    bazen = {x["broj"] for x in r["bazen"]}
    a, b, c = r["tezine"]
    for p in pred:
        k = p["brojevi"]
        assert set(k) <= bazen and k == sorted(k)
        assert p["osobine"]["u_dekadi"] <= 3 and p["osobine"]["uzastopni"] <= 1
        assert p["slicnost"]["maks"] <= 5
        assert abs(p["skor"] - sum(p["komponente"].values())) < 1e-9
    skorovi = [p["skor"] for p in pred]
    assert skorovi == sorted(skorovi, reverse=True)
    for x, y in combinations(pred, 2):
        assert len(set(x["brojevi"]) & set(y["brojevi"])) <= G.MAX_ZAJEDNICKIH_ALT

    # Najbolja je stvarno najbolja među kombinacijama koje prolaze pravila.
    sve = G.sklopi(ist, pravila={}, sa_slicnoscu=False)
    assert sve["prolazi_pravila"] == sve["ukupno_kombinacija"]
    assert sve["predlozi"][0]["skor"] >= pred[0]["skor"] - 1e-9

    # Reproducibilno.
    assert G.sklopi(ist)["predlozi"] == pred
    print(f"test_sklapanje: OK (prolazi {r['prolazi_pravila']}/{r['ukupno_kombinacija']})")


def test_samo_jedna_tezina():
    """a=1, b=c=0: najbolja kombinacija ima najveći mogući zbir normalizovanog potencijala."""
    ist = sinteticka_istorija(600, seme=34)
    r = G.sklopi(ist, w=100, tezine=(1, 0, 0), pravila={}, sa_slicnoscu=False)
    z = sorted((x["z"] for x in r["bazen"]), reverse=True)
    zmin, zmax = min(z), max(z)
    najbolje_moguce = sum((v - zmin) / (zmax - zmin) for v in z[:7])
    assert abs(r["predlozi"][0]["skor"] - najbolje_moguce) < 1e-6
    assert r["predlozi"][0]["komponente"]["ritam"] == 0 and r["predlozi"][0]["komponente"]["parovi"] == 0
    print("test_samo_jedna_tezina: OK")


def test_prestroga_pravila():
    ist = sinteticka_istorija(300, seme=35)
    r = G.sklopi(ist, pravila={"zbir_min": 300}, sa_slicnoscu=False)
    assert r["prolazi_pravila"] == 0 and r["predlozi"] == []
    print("test_prestroga_pravila: OK")


# ----------------------------------------------------------------------------
# Faza 6 — k_graditelj (§8)
# ----------------------------------------------------------------------------

def _isto_kao_primitivi(ist):
    from webapp.core import prediktori
    prim = prediktori._primitivi(ist)
    br = G.brojaci(ist)
    assert br.n == prim["n"]
    for b in range(1, 40):
        assert br.kasnjenje(b) == prim["kasnjenje"][b], b
        assert br.ritam(b) == prim["ritam"][b], (b, br.ritam(b), prim["ritam"][b])
    assert br.parovi == prim["parovi"]
    from webapp.core import razlicitost_teorija as T
    assert [int(x) for x in br.H()] == [T.maska(b) for _k, b in ist]


def test_brojaci_kao_primitivi():
    """Inkrementalni brojači = `prediktori._primitivi`: od nule, produženo i posle ispravke."""
    ist = sinteticka_istorija(300, seme=36)
    G._brojaci_kes.update(otisak=None, duzina=0, brojaci=None)
    _isto_kao_primitivi(ist[:120])
    for n in range(121, 140):                       # produžavanje po jedno kolo
        _isto_kao_primitivi(ist[:n])
    ispravljena = list(ist[:139])
    ispravljena[50] = (ispravljena[50][0], (1, 2, 3, 4, 5, 6, 7))   # ispravka starog kola
    _isto_kao_primitivi(ispravljena)
    _isto_kao_primitivi(ist[:30])                   # skraćena istorija
    print("test_brojaci_kao_primitivi: OK")


def test_maske_kao_teorija():
    from webapp.core import razlicitost_teorija as T
    rng = random.Random(5)
    B = np.array([sorted(rng.sample(range(1, 40), 7)) for _ in range(200)])
    assert [int(x) for x in G._maske(B)] == [T.maska(r) for r in B.tolist()]
    print("test_maske_kao_teorija: OK")


def test_predlog_za_je_najbolji_sa_strane():
    """k_graditelj (lenja provera istorije) = prvi predlog strane (puna provera)."""
    ist = sinteticka_istorija(800, seme=37)
    for n in (100, 400, 800):
        strana = G.sklopi(ist[:n], sa_slicnoscu=False)
        assert G.predlog_za(ist[:n]) == tuple(strana["predlozi"][0]["brojevi"]), n
    assert G.predlog_za(ist[:1]) is None
    print("test_predlog_za_je_najbolji_sa_strane: OK")


def test_k_graditelj_brzina():
    """Retro poziva k_graditelj ~1.350 puta; mora da stane u nekoliko sekundi."""
    import time
    from webapp.core.prediktori_komb import k_graditelj
    ist = sinteticka_istorija(1400, seme=38)
    t = time.perf_counter()
    for n in range(50, 1400):
        k_graditelj(ist[:n], 100)
    trajanje = time.perf_counter() - t
    assert trajanje < 4, trajanje
    print(f"test_k_graditelj_brzina: OK ({trajanje:.1f} s za 1.350 kola)")


def main():
    test_obrt_tacan()
    test_potencijal()
    test_bazen_je_vrh_potencijala()
    test_pravila_maska_kao_generator()
    test_istorija_se_iskljucuje()
    test_sklapanje()
    test_samo_jedna_tezina()
    test_prestroga_pravila()
    test_brojaci_kao_primitivi()
    test_maske_kao_teorija()
    test_predlog_za_je_najbolji_sa_strane()
    test_k_graditelj_brzina()
    print("\nSVI TESTOVI GRADITELJA PROSLI [OK]")


if __name__ == "__main__":
    main()
