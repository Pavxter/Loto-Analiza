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
        maska = G._pravila_maska(B, pravila, [])
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
    maska = G._pravila_maska(B, {"istorija_max": 5}, istorija)
    assert maska.tolist() == [False, False, True, True]
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


def main():
    test_obrt_tacan()
    test_potencijal()
    test_bazen_je_vrh_potencijala()
    test_pravila_maska_kao_generator()
    test_istorija_se_iskljucuje()
    test_sklapanje()
    test_samo_jedna_tezina()
    test_prestroga_pravila()
    print("\nSVI TESTOVI GRADITELJA PROSLI [OK]")


if __name__ == "__main__":
    main()
