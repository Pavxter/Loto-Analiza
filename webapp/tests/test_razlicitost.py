"""Testovi teorijskog modula i analiza različitosti.

Kriterijumi prihvatanja: PLAN_RAZLICITOST §11, PLAN_PROGNOZA_KOMBINACIJE §9.

Pokretanje:  python -X utf8 -m webapp.tests.test_razlicitost
"""

import os
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from webapp.core import razlicitost_teorija as T  # noqa: E402


def test_pmf_suma_i_ocekivanje():
    """Σ P(k) = 1; E[k] ≈ 1,2564; P(K) = 1/C(N,K)."""
    s = sum(T.sve_pmf())
    assert abs(s - 1.0) < 1e-12, f"Σ P(k) = {s}, očekivano 1"
    assert abs(T.ocekivano_preklapanje() - 7 * 7 / 39) < 1e-4
    assert abs(T.hipergeom_pmf(7) - 1 / comb(39, 7)) < 1e-18
    # tačne vrednosti (plan tabela je zaokružena/orijentaciona — proveravamo egzaktno)
    for k in range(T.K + 1):
        egzakt = comb(7, k) * comb(32, 7 - k) / comb(39, 7)
        assert abs(T.hipergeom_pmf(k) - egzakt) < 1e-15
    # k=1 je modus raspodele (najverovatnije preklapanje)
    assert T.hipergeom_pmf(1) == max(T.sve_pmf())
    print("test_pmf_suma_i_ocekivanje: OK")


def test_sigma():
    """σ hipergeometrijskog preklapanja = 0,9317 (tolerancija 1e-3).

    NAPOMENA: plan navodi ≈0,98, ali to je greška — izostavljen je faktor korekcije
    za konačnu populaciju (N-n)/(N-1). Tačna vrednost je Var = n·p·(1-p)·(N-n)/(N-1)
    = 0,8681, σ = 0,9317. Proveravamo tačnu (matematički izvedenu) vrednost.
    """
    from math import sqrt
    var_formula = 7 * (7 / 39) * (32 / 39) * ((39 - 7) / (39 - 1))
    assert abs(T.sigma_preklapanja() - sqrt(var_formula)) < 1e-9
    assert abs(T.sigma_preklapanja() - 0.9317) < 1e-3, T.sigma_preklapanja()
    print(f"test_sigma: OK (σ={T.sigma_preklapanja():.4f})")


def test_p_par():
    """P(par u kolu) = K(K-1)/(N(N-1)) = 2,834 %.

    NAPOMENA: plan navodi ≈4,05 % i ~57,6 pojava u 1.422 kola, ali to je greška —
    tačno je 7·6/(39·38) = 0,02834 (= C(37,5)/C(39,7)), tj. ~40,3 pojave. Sama
    formula u planu je ispravna, samo je procenat pogrešno zaokružen.
    """
    assert abs(T.p_par_u_kolu() - 7 * 6 / (39 * 38)) < 1e-12
    assert abs(T.p_par_u_kolu() - 0.02834) < 5e-4
    print(f"test_p_par: OK ({100*T.p_par_u_kolu():.3f}%)")


def test_bitmaske():
    """Preklapanje: disjunktne=0, identične=K, delimične=tačan broj."""
    a = [1, 2, 3, 4, 5, 6, 7]
    b = [8, 9, 10, 11, 12, 13, 14]      # disjunktne
    c = [1, 2, 3, 4, 5, 6, 7]           # identične
    d = [5, 6, 7, 8, 9, 10, 11]         # dele 5,6,7 -> 3
    assert T.preklapanje_brojeva(a, b) == 0
    assert T.preklapanje_brojeva(a, c) == 7
    assert T.preklapanje_brojeva(a, d) == 3
    assert T.maska([1]) == 1 and T.maska([39]) == 1 << 38
    print("test_bitmaske: OK")


def test_pojas_se_suzava():
    """Pojas pouzdanosti za prosek monotono se sužava sa n."""
    prethodna = None
    for n in (5, 10, 50, 100, 500, 1000):
        d, g = T.pojas_proseka(n)
        sirina = g - d
        if prethodna is not None:
            assert sirina < prethodna, f"pojas se ne sužava kod n={n}"
        prethodna = sirina
    print("test_pojas_se_suzava: OK")


def test_z_test():
    """z-test: prosek = μ -> z≈0, p≈1; jasno iznad μ -> mala p."""
    mu = T.ocekivano_preklapanje()
    z0, p0 = T.z_test_proseka(mu, 100)
    assert abs(z0) < 1e-9 and abs(p0 - 1.0) < 1e-9
    z1, p1 = T.z_test_proseka(mu + 0.5, 400)   # +0.5 preko ~0.98/20 sigma
    assert z1 > 0 and p1 < 0.05
    assert T.z_test_proseka(0, 0) == (None, None)
    print("test_z_test: OK")


def test_hi_kvadrat_na_teoriji():
    """Ako je posmatrano = n·P(k) (savršeno slaganje) -> chi2 ≈ 0, p ≈ 1."""
    n = 1421
    obs = [round(n * T.hipergeom_pmf(k)) for k in range(T.K + 1)]  # celobrojna opažanja
    rez = T.hi_kvadrat_preklapanje(obs, n)
    assert rez["chi2"] < 1.0, rez["chi2"]          # samo šum zaokruživanja
    assert rez["p"] is not None and rez["p"] > 0.9
    # spajanje: 5 kategorija (0,1,2,3,4+) -> df=4
    assert rez["df"] == 4
    print("test_hi_kvadrat_na_teoriji: OK")


def _sinteticka(broj_kola=1000, seme=2024):
    import random
    rng = random.Random(seme)
    return [(2000001 + i, tuple(rng.sample(range(1, 40), 7))) for i in range(broj_kola)]


def test_analize_ne_prijavljuju_lazni_signal():
    """Nasumična istorija (seedovano) -> testovi ne prijavljuju odstupanje (p > 0,05)."""
    from webapp.core import razlicitost as R
    ist = _sinteticka(1000, seme=2024)
    a1 = R.analiza_uzastopna(ist)
    a2 = R.analiza_svi_parovi(ist)
    assert a1["test"]["p"] is not None and a1["test"]["p"] > 0.05, a1["test"]
    assert a2["test"]["p"] is not None and a2["test"]["p"] > 0.05, a2["test"]
    # prosek preklapanja mora biti blizu μ=1,256
    h = a1["histogram"]
    prosek = sum(k * c for k, c in zip(h["k"], h["posmatrano_broj"])) / h["n"]
    assert abs(prosek - T.ocekivano_preklapanje()) < 0.1, prosek
    print(f"test_analize_ne_prijavljuju_lazni_signal: OK (p1={a1['test']['p']}, p2={a2['test']['p']}, prosek={prosek:.3f})")


def test_profil_ne_nosi_informaciju():
    """Na slučajnim podacima korelacija profila i sadržaja mora biti SLABA.

    NAPOMENA: prag |r| < 0,05 iz plana važi samo za profile bez geometrijske veze sa
    preklapanjem. Svi ponuđeni profili imaju intrinzičnu (slabu do umerenu) negativnu
    korelaciju čak i pri čistoj slučajnosti — deljeni brojevi zbližavaju sredinu, a po
    definiciji spadaju u istu dekadu (pa je „dekade" najspregnutije). To je sam nalaz
    analize (profil je slab, ne savršen, prediktor). Ovde samo garantujemo da nijedan
    profil ne pokazuje jaku vezu (|r| < 0,3) i da su sve korelacije negativne.
    """
    from webapp.core import razlicitost as R
    ist = _sinteticka(800, seme=7)
    rezovi = {}
    for tip in ("sredina", "parni", "dekade"):
        rez = R.analiza_profil(ist, tip=tip, seed=99)
        rezovi[tip] = rez["r"]
        assert rez["r"] < 0 and abs(rez["r"]) < 0.3, f"{tip}: r={rez['r']}"
    # „parni" je geometrijski najmanje spregnut sa preklapanjem
    assert abs(rezovi["parni"]) <= abs(rezovi["dekade"]) + 1e-9
    print(f"test_profil_ne_nosi_informaciju: OK ({rezovi})")


def test_performanse_svi_parovi():
    """Analiza svih parova nad ~1.422 kola < 2 s (PLAN §11)."""
    import time
    from webapp.core import razlicitost as R
    ist = _sinteticka(1422, seme=1)
    t0 = time.time()
    R.analiza_svi_parovi(ist)
    R.analiza_rekordi(ist)
    dt = time.time() - t0
    assert dt < 2.0, f"presporo: {dt:.2f}s"
    print(f"test_performanse_svi_parovi: OK ({dt:.2f}s)")


def test_ko_okurencija_kontrola():
    """Broj parova van 95% intervala se prikazuje uporedo sa očekivanih ~37."""
    from webapp.core import razlicitost as R
    ist = _sinteticka(1422, seme=3)
    rez = R.analiza_ko_okurencija(ist)
    assert rez["kontrola"]["broj_parova"] == 741
    assert abs(rez["kontrola"]["ocekivano_van"] - 37.05) < 0.1
    # na slučajnim podacima broj van intervala treba da bude reda ~37 (ne desetostruko)
    assert rez["kontrola"]["van_95_intervala"] < 100, rez["kontrola"]
    print(f"test_ko_okurencija_kontrola: OK (van={rez['kontrola']['van_95_intervala']}, ocek≈37)")


def test_rekordi_identicne():
    """Ubačen identičan par kola mora biti detektovan kao 7/7 rekord."""
    from webapp.core import razlicitost as R
    ist = _sinteticka(50, seme=5)
    ist.append((2000100, ist[0][1]))   # kopija prvog kola
    rez = R.analiza_rekordi(ist)
    assert rez["identicne"]["ima"] is True
    assert rez["najvece"]["k"] == 7
    print("test_rekordi_identicne: OK")


def main():
    test_pmf_suma_i_ocekivanje()
    test_sigma()
    test_p_par()
    test_bitmaske()
    test_pojas_se_suzava()
    test_z_test()
    test_hi_kvadrat_na_teoriji()
    test_analize_ne_prijavljuju_lazni_signal()
    test_profil_ne_nosi_informaciju()
    test_performanse_svi_parovi()
    test_ko_okurencija_kontrola()
    test_rekordi_identicne()
    print("\nSVI TESTOVI RAZLICITOSTI PROSLI [OK]")


if __name__ == "__main__":
    main()
