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


# ----------------------------------------------------------------------------
# Skan prozora
# ----------------------------------------------------------------------------
# Broj replika je u testovima namerno mali (300): dovoljno da p padne na svoj pod
# od 1/301 kad je signal jasan, a dovoljno brzo da se ceo fajl pusti u sekundi.

SKAN_REPLIKA_TEST = 300


def _istorija_sa_burstom(broj_kola=1400, favorit=7, od=600, do=800, udeo=0.22, seme=7):
    """Sintetika u kojoj je jedan broj pristrasan SAMO u prozoru [od, do).

    Višak je namerno odmeren tako da se u zbiru cele istorije izgubi: prozor od 200
    kola nosi z ≈ 6, a ista razlika razmazana na 1.400 kola daje z ≈ 2,3, što
    hi-kvadrat sa 38 stepeni slobode ne razaznaje. Zato ova istorija razdvaja skan
    od zbirne frekvencije — jedan test mora da vidi ono što drugi ne vidi.
    """
    rng = random.Random(seme)
    istorija = []
    for i in range(broj_kola):
        if od <= i < do and rng.random() < udeo:
            ostali = rng.sample([b for b in range(1, MAX_BROJ + 1) if b != favorit], K - 1)
            brojevi = tuple([favorit] + ostali)
        else:
            brojevi = tuple(rng.sample(range(1, MAX_BROJ + 1), K))
        istorija.append((2010001 + i, brojevi))
    return istorija


def test_skan_na_sumu():
    """Na čistoj slučajnosti najveći |z| ostaje u onome što slučajnost sama pravi."""
    istorija = sinteticka_istorija(1400, seme=91)
    rez = razlicitost.test_skan_prozora(istorija, replika=SKAN_REPLIKA_TEST)
    assert rez["p_tacno"] > 0.05, rez["p_tacno"]
    assert rez["statistika"] < rez["prag95"], (rez["statistika"], rez["prag95"])
    print(f"test_skan_na_sumu: OK (|z|={rez['statistika']}, prag95={rez['prag95']}, "
          f"p={rez['p_tacno']:.4f})")


def test_skan_vidi_sto_frekvencija_ne_vidi():
    """Pristrasnost koja traje 200 kola: skan je nalazi, zbirna frekvencija ne.

    Bez ovog para tvrdnji „skan ništa ne nalazi na pravim podacima" ne znači ništa —
    tek se ovde vidi da bi nalazio da ima šta.
    """
    istorija = _istorija_sa_burstom()
    frekv = razlicitost.test_frekvencija_brojeva(istorija)
    skan = razlicitost.test_skan_prozora(istorija, replika=SKAN_REPLIKA_TEST)

    assert frekv["p_tacno"] > 0.05, ("zbirna frekvencija ne sme videti burst", frekv["p_tacno"])
    assert skan["p_tacno"] < 0.01, ("skan mora videti burst", skan["p_tacno"])
    najbolji = skan["top"][0]
    assert najbolji["broj"] == 7, najbolji
    assert najbolji["duzina"] == 200, najbolji
    assert najbolji["od"] == 2010601 and najbolji["do"] == 2010800, najbolji
    print(f"test_skan_vidi_sto_frekvencija_ne_vidi: OK (frekvencija p={frekv['p_tacno']:.3f}, "
          f"skan p={skan['p_tacno']:.4f}, nasao broj {najbolji['broj']} "
          f"u prozoru {najbolji['od']}-{najbolji['do']}, z={najbolji['z']})")


def test_skan_determinizam():
    """Isti ulaz mora dati isti p — Monte Karlo je vezan fiksiranim semenom."""
    istorija = sinteticka_istorija(600, seme=92)
    a = razlicitost.test_skan_prozora(istorija, replika=SKAN_REPLIKA_TEST, seme=4242)
    razlicitost._SKAN_NULL_KES.clear()      # bez keša, da se meri sam račun a ne pamćenje
    b = razlicitost.test_skan_prozora(istorija, replika=SKAN_REPLIKA_TEST, seme=4242)
    assert a["statistika"] == b["statistika"] and a["p_tacno"] == b["p_tacno"], (a, b)
    assert a["prag95"] == b["prag95"], (a["prag95"], b["prag95"])
    print(f"test_skan_determinizam: OK (p={a['p_tacno']:.4f} dvaput, kes ociscen izmedju)")


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
    test_skan_na_sumu()
    test_skan_vidi_sto_frekvencija_ne_vidi()
    test_skan_determinizam()
    print("\nSVI TESTOVI SINTEZE PROSLI [OK]")


if __name__ == "__main__":
    main()
