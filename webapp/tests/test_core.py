"""Smoke + invarijant testovi za core module na pravoj bazi.

Pokretanje:  python -m webapp.tests.test_core   (iz korena projekta)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from webapp.core import konfig, baza, analitika, rangiranje, generator, bektest  # noqa: E402


def main():
    baza.postavi_bazu()
    conn = baza.konekcija()
    df = analitika.ucitaj_df(conn)
    print(f"Ucitano kola: {len(df)}")
    assert len(df) > 0, "Baza je prazna"

    # --- Analiza ---
    a = analitika.Analiza(df, period_analize=0)
    dash = a.kao_dashboard()
    print("Dashboard:", {k: dash[k] for k in ("broj_kola", "globalni_prosek", "par_nepar")})
    assert dash["broj_kola"] == len(df)
    assert 1 <= dash["globalni_prosek"] <= 39
    assert len(a.vruci_brojevi) == konfig.BROJ_KATEGORIJA_FREKV
    assert len(a.hladni_brojevi) == konfig.BROJ_KATEGORIJA_FREKV
    # Svaki broj se javlja u tacno jednoj kategoriji
    assert a.vruci_brojevi.isdisjoint(a.hladni_brojevi)

    freq = a.kao_frekvencija()
    assert len(freq) == konfig.MAX_BROJ
    ukupno = sum(f["frekvencija"] for f in freq)
    assert ukupno == len(a.brojevi_po_kolima) * konfig.BROJEVA_U_KOMBINACIJI, \
        f"Zbir frekvencija {ukupno} != ocekivano"
    print("Frekvencija OK, ukupno izvlacenja:", ukupno)

    stat = a.kao_statistika()
    assert len(stat["poziciona"]["vrednosti"]) == konfig.MAX_BROJ
    print("Statistika keys:", sorted(stat.keys()))

    # --- Rangiranje ---
    for metoda in ("frekvencija", "bajes", "hibrid"):
        rang = rangiranje.rangiraj(df, metoda)
        assert len(rang) == konfig.MAX_BROJ
        # sortirano opadajuce
        skorovi = [r["skor"] for r in rang]
        assert skorovi == sorted(skorovi, reverse=True), f"{metoda} nije sortiran"
        print(f"Rang [{metoda}] top5:", [r['broj'] for r in rang[:5]])

    # Bajes verovanja se sumiraju na ~1
    b = rangiranje.bajes_verovanja(df)
    assert abs(sum(b.values()) - 1.0) < 1e-6, "Bajes nije normalizovan"

    # --- Generator ---
    bazen = dash["predlog_bazena"]
    rez = generator.generisi(a, bazen=bazen, filteri={
        "parni": 3, "strategija_svezine": "favorizuj", "primeni_pristrasnost": True,
    })
    print(f"Generator: bazen={len(bazen)} -> validnih={rez['ukupno_validnih']}, prikaz top1={rez['kombinacije'][:1]}")
    for k in rez["kombinacije"]:
        assert len(k["brojevi"]) == konfig.BROJEVA_U_KOMBINACIJI
        assert sum(1 for x in k["brojevi"] if x % 2 == 0) == 3, "Filter parnih ne radi"
    # Skorovi opadajuci
    sk = [k["skor"] for k in rez["kombinacije"]]
    assert sk == sorted(sk, reverse=True)

    # --- Bektest metrike (bez upisa u bazu) ---
    model = bektest.model_verovatnoce(df, 0)
    assert abs(sum(model.values()) - 1.0) < 1e-6
    komb = sorted(bazen)[:7]
    p = bektest.promasaj_kombinacije(komb, set(df.iloc[-1][konfig.KOLONE_ZA_BROJEVE].astype(int)))
    iz = bektest.indeks_iznenadjenja(komb, model)
    print(f"Bektest metrike: promasaj={p}, iznenadjenje={iz:.2f}")
    assert p is not None and iz is not None

    conn.close()
    print("\nSVE PROVERE PROSLE [OK]")


if __name__ == "__main__":
    main()
