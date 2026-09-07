"""Testovi strane „Sinteza" — PLAN_SINTEZA.md §7.

Tri grupe:
  - sakupljanje i korekcija: brojevi u Sintezi moraju biti isti kao na tabu
    Prognoza, a Bonferroni mora ići preko SVIH redova;
  - ansambl: težine se uče isključivo na ranijim kolima i na čistom šumu ne
    smeju da „nađu" signal;
  - rang: teorijska očekivanja se proveravaju na sintetici, gde se odgovor zna.

Radi nad privremenom bazom (ne dira loto_baza.db).

Pokretanje:  python -X utf8 -m webapp.tests.test_sinteza
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from webapp.core import konfig, mapa, prognoza, razlicitost, sinteza  # noqa: E402
from webapp.core import razlicitost_teorija as T  # noqa: E402
from webapp.core import prediktori as P  # noqa: E402
from webapp.tests.test_prognoza import nova_baza, sinteticka_istorija  # noqa: E402

MAX_BROJ = konfig.MAX_BROJ
K = konfig.BROJEVA_U_KOMBINACIJI


def _slucajne_kombinacije(broj, seme):
    rng = random.Random(seme)
    return [tuple(sorted(rng.sample(range(1, MAX_BROJ + 1), K))) for _ in range(broj)]


# ----------------------------------------------------------------------------
# Sakupljanje i korekcija
# ----------------------------------------------------------------------------

def test_isti_brojevi_kao_prognoza():
    """Red Sinteze mora nositi tačno one brojeve koje daje evaluacija u prognoza.py."""
    conn, putanja = nova_baza(sinteticka_istorija(300, seme=7))
    try:
        prognoza.retro_bektest(conn)
        rezime = sinteza.sakupi(conn, "retro")
        stat = {m["metod"]: m for m in prognoza.statistika(conn, "retro")["metode"]}
        stat_k = {m["metod"]: m for m in prognoza.statistika_komb(conn, "retro")["metode"]}

        for red in rezime["redovi"]["jedan_broj"]:
            izvor = stat[red["metod"]]
            assert red["n"] == izvor["n"], red["metod"]
            assert red["rezultat"] == izvor["k"], red["metod"]
            assert red["p_prikaz"] == izvor["p"], (red["metod"], red["p_prikaz"], izvor["p"])
        for red in rezime["redovi"]["kombinacija"]:
            izvor = stat_k[red["metod"]]
            assert red["n"] == izvor["n"], red["metod"]
            assert red["rezultat"] == izvor["prosek"], red["metod"]
            assert red["p_prikaz"] == izvor["p"], (red["metod"], red["p_prikaz"], izvor["p"])
        print(f"test_isti_brojevi_kao_prognoza: OK ({rezime['broj_redova']} redova)")
    finally:
        conn.close(); os.remove(putanja)


def test_bonferroni_preko_svih():
    """p_kor = min(1, p·N), gde je N broj SVIH redova — s kontrolom i testovima."""
    conn, putanja = nova_baza(sinteticka_istorija(300, seme=8))
    try:
        prognoza.retro_bektest(conn)
        rezime = sinteza.sakupi(conn, "retro")
        n = rezime["broj_redova"]
        svi = [r for tip in sinteza.TIPOVI for r in rezime["redovi"][tip]]
        assert len(svi) == n
        assert any(r["kontrola"] for r in svi), "kontrola mora biti u tabeli"
        assert any(r["tip"] == "test" for r in svi), "testovi moraju biti u tabeli"
        for r in svi:
            if r["p"] is None:
                assert r["p_korig"] is None and r["zakljucak"] == sinteza.BEZ_PODATAKA
                continue
            assert math.isclose(r["p_korig"], min(1.0, r["p"] * n), rel_tol=1e-12), r["metod"]
        assert rezime["ocekivano_laznih"] == round(n * sinteza.ALFA, 2)
        print(f"test_bonferroni_preko_svih: OK (N={n})")
    finally:
        conn.close(); os.remove(putanja)


def test_zakljucak_tri_ishoda():
    """Funkcija zaključka vraća tačno jedan od tri teksta, i ništa drugo."""
    ishodi = {
        sinteza.zakljucak(1.0, False),
        sinteza.zakljucak(0.05, False),
        sinteza.zakljucak(0.049, False),
        sinteza.zakljucak(0.0001, True),
    }
    assert ishodi == {sinteza.SLUCAJNOST, sinteza.ODSTUPA, sinteza.KONTROLA_ODSTUPA}
    assert sinteza.zakljucak(0.06, True) == sinteza.SLUCAJNOST
    try:
        sinteza.zakljucak(None, False)
        raise AssertionError("None mora da digne ValueError, ne da postane cetvrti ishod")
    except ValueError:
        pass
    print("test_zakljucak_tri_ishoda: OK")


# ----------------------------------------------------------------------------
# Ansambl
# ----------------------------------------------------------------------------

def test_ensemble_walk_forward():
    """Težine zavise SAMO od kola koja prethode delu koji se ocenjuje.

    Isti anti-leakage obrazac kao test_istorija: menjanje budućnosti ne sme da
    promeni ono što je model „znao" u prošlosti.
    """
    istorija = sinteticka_istorija(700, seme=21)
    period = prognoza.RETRO_PERIOD

    P._KES_TEZINA.clear()
    tezine_700 = P.nauci_tezine(istorija, period)

    # Zamena SVIH kola posle granice ucenja ne sme da pomeri tezine.
    izmenjena = istorija[:P.UCENJE_DO] + sinteticka_istorija(300, seme=99)
    P._KES_TEZINA.clear()
    assert P.nauci_tezine(izmenjena, period) == tezine_700, "tezine vide buducnost!"

    # Promena kola UNUTAR skupa za ucenje mora da ih pomeri (inace test ne meri nista).
    druga = sinteticka_istorija(700, seme=22)
    P._KES_TEZINA.clear()
    assert P.nauci_tezine(druga, period) != tezine_700, "tezine ne zavise ni od cega?"

    # Kratka istorija: nema sta da se uci, sve komponente jednake.
    P._KES_TEZINA.clear()
    kratke = P.nauci_tezine(istorija[:P.UCENJE_DO - 1], period)
    assert len(set(kratke.values())) == 1, kratke
    assert math.isclose(sum(kratke.values()), 1.0, rel_tol=1e-6)   # tezine su zaokruzene na 9 decimala
    assert math.isclose(sum(tezine_700.values()), 1.0, rel_tol=1e-6)   # tezine su zaokruzene na 9 decimala

    P._KES_TEZINA.clear()
    print(f"test_ensemble_walk_forward: OK (tezine={ {k: round(v, 3) for k, v in tezine_700.items()} })")


def test_ensemble_na_slucajnim_podacima():
    """Na čistom šumu ansambl ne sme da ispadne značajan posle korekcije."""
    conn, putanja = nova_baza(sinteticka_istorija(700, seme=31))
    try:
        prognoza.retro_bektest(conn)
        rezime = sinteza.sakupi(conn, "retro")
        svi = {r["metod"]: r for tip in sinteza.TIPOVI for r in rezime["redovi"][tip]}
        for metod in sinteza.ANSAMBLI:
            red = svi[metod]
            assert red["n"] > 0, metod
            assert red["p_korig"] is not None, metod
            assert red["p_korig"] > sinteza.ALFA, (metod, red["p_korig"])
            assert red["zakljucak"] == sinteza.SLUCAJNOST, (metod, red["zakljucak"])
        print("test_ensemble_na_slucajnim_podacima: OK "
              f"(ensemble p_kor={svi['ensemble']['p_korig']:.3f}, "
              f"k_ensemble p_kor={svi['k_ensemble']['p_korig']:.3f})")
    finally:
        conn.close(); os.remove(putanja)


# ----------------------------------------------------------------------------
# Rang kombinacije
# ----------------------------------------------------------------------------

def test_rang_uniformnost_sintetika():
    """10.000 slučajnih kombinacija: hi-kvadrat rangova ne odbacuje uniformnost."""
    rangovi = [mapa.rang(k) for k in _slucajne_kombinacije(10000, seme=101)]
    rez = razlicitost.test_rang_uniformnost(rangovi)
    assert rez["df"] == razlicitost.BROJ_KORPI_RANGA - 1
    assert rez["p"] is not None and rez["p"] > 0.01, rez
    print(f"test_rang_uniformnost_sintetika: OK (chi2={rez['chi2']}, df={rez['df']}, p={rez['p']})")


def test_rang_rastojanje_teorija():
    """Prosek |Δrang| na sintetici je unutar 2 standardne greške od M/3."""
    rangovi = [mapa.rang(k) for k in _slucajne_kombinacije(10000, seme=102)]
    rez = razlicitost.test_rang_rastojanja(rangovi)
    m = T.UKUPNO_KOMBINACIJA
    # Var|X−Y| = M²/6 − (M/3)² = M²/18 za dva nezavisna ravnomerna ranga.
    sigma = m / math.sqrt(18)
    granica = 2 * sigma / math.sqrt(rez["n"])
    odstupanje = abs(rez["prosek"] - m / 3)
    assert odstupanje < granica, (rez["prosek"], m / 3, granica)
    assert rez["p"] is not None and rez["p"] > 0.01, rez
    print(f"test_rang_rastojanje_teorija: OK (prosek={rez['prosek']:.0f}, "
          f"ocekivano={m / 3:.0f}, dozvoljeno ±{granica:.0f})")


def test_min_broj_raspodela():
    """P(min=k) sabira na 1 i slaže se sa empirijom na sintetici."""
    pmf = T.sve_p_min()
    assert len(pmf) == MAX_BROJ - K + 1
    assert math.isclose(sum(pmf), 1.0, rel_tol=1e-12), sum(pmf)
    assert math.isclose(pmf[0], K / MAX_BROJ, rel_tol=1e-12), pmf[0]

    istorija = [(2020001 + i, k) for i, k in enumerate(_slucajne_kombinacije(10000, seme=103))]
    rez = razlicitost.test_najmanji_broj(istorija)
    assert rez["p"] is not None and rez["p"] > 0.01, rez
    print(f"test_min_broj_raspodela: OK (chi2={rez['chi2']}, df={rez['df']}, p={rez['p']})")


def test_autokorelacija_na_sumu():
    """Slučajni rangovi nemaju autokorelaciju ni na jednom od pet pomaka."""
    rangovi = [mapa.rang(k) for k in _slucajne_kombinacije(5000, seme=104)]
    rez = razlicitost.test_rang_autokorelacija(rangovi)
    assert rez["df"] == razlicitost.POMACI_AUTOKORELACIJE
    assert rez["p"] is not None and rez["p"] > 0.01, rez
    assert all(abs(z) < 4 for z in rez["z"]), rez["z"]
    print(f"test_autokorelacija_na_sumu: OK (Q={rez['Q']}, p={rez['p']}, z={rez['z']})")


def main():
    test_isti_brojevi_kao_prognoza()
    test_bonferroni_preko_svih()
    test_zakljucak_tri_ishoda()
    test_ensemble_walk_forward()
    test_ensemble_na_slucajnim_podacima()
    test_rang_uniformnost_sintetika()
    test_rang_rastojanje_teorija()
    test_min_broj_raspodela()
    test_autokorelacija_na_sumu()
    print("\nSVI TESTOVI SINTEZE PROSLI [OK]")


if __name__ == "__main__":
    main()
