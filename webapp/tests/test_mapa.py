"""Testovi indeksiranja i rasporeda mape (plan_mapa_kombinacija.md, Faze 1 i 2).

Pokriveni kriterijumi ove faze:
  - rang je zaista leksikografski, tj. jednak indeksu u itertools.combinations,
  - rang <-> unrang je bijekcija (uzorak + obe ivice + odbijanje neispravnog),
  - Hilbert je bijekcija i svi rangovi padaju unutar kvadrata 4096x4096,
  - susedni rangovi su susedne ćelije (osobina zbog koje je kriva izabrana),
  - vektorske osobine daju isto što i osobine jedne kombinacije,
  - detalj kombinacije i detalj ćelije opisuju istu stvar iz dva pravca,
  - ako su pločice generisane, slažu se sa trenutnim konstantama.

Pokretanje:  python -X utf8 -m webapp.tests.test_mapa
"""

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from webapp.core import mapa  # noqa: E402

UZORAK = 5000


def _uzorak_rangova(n=UZORAK, seed=20260905):
    """Nasumični rangovi po celom prostoru, uvek isti (fiksan seed)."""
    r = np.random.default_rng(seed)
    return r.integers(0, mapa.UKUPNO_KOMBINACIJA, size=n, dtype=np.int64)


def test_rang_je_leksikografski():
    """Rang mora biti tačno indeks u itertools.combinations (na tome stoji generator)."""
    for i, komb in enumerate(itertools.islice(
            itertools.combinations(range(1, mapa.MAX_BROJ + 1), mapa.BROJEVA), UZORAK)):
        assert mapa.rang(komb) == i, f"rang{komb} = {mapa.rang(komb)}, očekivano {i}"
        assert mapa.unrang(i) == komb, f"unrang({i}) = {mapa.unrang(i)}, očekivano {komb}"
    print("test_rang_je_leksikografski: OK")


def test_rang_unrang_bijekcija():
    """rang(unrang(r)) == r na uzorku i na obe ivice prostora."""
    prva = tuple(range(1, mapa.BROJEVA + 1))
    poslednja = tuple(range(mapa.MAX_BROJ - mapa.BROJEVA + 1, mapa.MAX_BROJ + 1))
    assert mapa.rang(prva) == 0
    assert mapa.rang(poslednja) == mapa.UKUPNO_KOMBINACIJA - 1
    assert mapa.unrang(0) == prva
    assert mapa.unrang(mapa.UKUPNO_KOMBINACIJA - 1) == poslednja

    for r in _uzorak_rangova():
        komb = mapa.unrang(int(r))
        assert list(komb) == sorted(komb) and len(set(komb)) == mapa.BROJEVA
        assert mapa.rang(komb) == int(r)

    # redosled unetih brojeva ne menja rang (kombinacija, ne varijacija)
    izvlacenje = [23, 5, 39, 12, 7, 31, 18]
    assert mapa.rang(izvlacenje) == mapa.rang(sorted(izvlacenje))
    print("test_rang_unrang_bijekcija: OK")


def test_odbija_neispravno():
    """Loša kombinacija i rang van opsega dižu ValueError, ne vraćaju besmislicu."""
    losi = [
        [1, 2, 3, 4, 5, 6],                 # premalo brojeva
        [1, 2, 3, 4, 5, 6, 7, 8],           # previše brojeva
        [1, 2, 3, 4, 5, 6, 6],              # duplikat
        [0, 1, 2, 3, 4, 5, 6],              # broj ispod 1
        [1, 2, 3, 4, 5, 6, 40],             # broj iznad 39
    ]
    for komb in losi:
        try:
            mapa.rang(komb)
        except ValueError:
            continue
        raise AssertionError(f"prihvaćena neispravna kombinacija: {komb}")

    for r in (-1, mapa.UKUPNO_KOMBINACIJA):
        try:
            mapa.unrang(r)
        except ValueError:
            continue
        raise AssertionError(f"prihvaćen rang van opsega: {r}")
    print("test_odbija_neispravno: OK")


def test_hilbert_bijekcija_i_granice():
    """Svi rangovi padaju u kvadrat 4096x4096 i inverz vraća isti rang."""
    d = np.arange(mapa.UKUPNO_KOMBINACIJA, dtype=np.int64)
    x, y = mapa.hilbert_xy(d)
    assert x.min() >= 0 and y.min() >= 0
    assert x.max() < mapa.DIMENZIJA and y.max() < mapa.DIMENZIJA
    # inverz nad svim rangovima ujedno dokazuje i da su ćelije jedinstvene:
    # preslikavanje koje ima levi inverz ne može dve kombinacije da spoji u jednu ćeliju
    assert np.array_equal(mapa.hilbert_d(x, y), d), "inverz Hilberta nije vratio isti rang"
    print("test_hilbert_bijekcija_i_granice: OK")


def test_hilbert_susedi():
    """Susedni rangovi su susedne ćelije — zbog toga je kriva i izabrana."""
    d = np.arange(mapa.DIMENZIJA * mapa.DIMENZIJA, dtype=np.int64)
    x, y = mapa.hilbert_xy(d)
    korak = np.abs(np.diff(x)) + np.abs(np.diff(y))
    assert korak.max() == 1 and korak.min() == 1, "kriva pravi skok veći od jedne ćelije"
    print("test_hilbert_susedi: OK")


def test_prazan_deo_krive():
    """Ćelije iza poslednje kombinacije nemaju rang (prazan deo mape)."""
    prazno = mapa.UKUPNO_KOMBINACIJA
    x, y = mapa.hilbert_xy(prazno)
    assert mapa.rang_iz_koordinata(x, y) is None
    assert mapa.kombinacija_na_koordinati(x, y) is None

    ukupno_celija = mapa.DIMENZIJA * mapa.DIMENZIJA
    assert ukupno_celija - mapa.UKUPNO_KOMBINACIJA == 1396279  # 4096^2 - C(39,7)

    for r in _uzorak_rangova(200):
        x, y = mapa.koordinate(int(r))
        assert mapa.rang_iz_koordinata(x, y) == int(r)
        assert mapa.kombinacija_na_koordinati(x, y) == mapa.unrang(int(r))
    print("test_prazan_deo_krive: OK")


def test_osobine_vektorski_isto():
    """osobina_niz nad nizom daje isto što i osobine nad pojedinačnom kombinacijom."""
    rangovi = _uzorak_rangova(2000)
    kombinacije = np.array([mapa.unrang(int(r)) for r in rangovi], dtype=np.uint8)
    pojedinacno = [mapa.osobine(k) for k in kombinacije]
    for naziv, opis in mapa.OSOBINE.items():
        niz = mapa.osobina_niz(naziv, kombinacije)
        ocekivano = np.array([o[naziv] for o in pojedinacno], dtype=np.int16)
        assert np.array_equal(niz, ocekivano), f"osobina {naziv} se ne poklapa"
        vmin, vmax = opis["opseg"]
        assert niz.min() >= vmin and niz.max() <= vmax, f"osobina {naziv} van opsega"

    # ivice opsega: najmanja i najveća kombinacija
    assert mapa.osobine(mapa.unrang(0)) == {"zbir": 28, "raspon": 6, "parni": 3, "dekade": 1}
    assert mapa.osobine(mapa.unrang(mapa.UKUPNO_KOMBINACIJA - 1)) == {
        "zbir": 252, "raspon": 6, "parni": 3, "dekade": 1}
    print("test_osobine_vektorski_isto: OK")


def test_detalj_kombinacije_i_celije():
    """Detalj kombinacije i detalj ćelije opisuju istu stvar iz dva pravca (Faza 2)."""
    izvlacenje = [23, 5, 39, 12, 7, 31, 18]          # redosled izvlačenja, ne sortirano
    d = mapa.detalj_kombinacije(izvlacenje)
    assert d["brojevi"] == sorted(izvlacenje)
    assert d["rang"] == mapa.rang(izvlacenje)
    assert (d["x"], d["y"]) == mapa.koordinate(d["rang"])
    assert d["osobine"] == mapa.osobine(izvlacenje)
    assert mapa.detalj_celije(d["x"], d["y"]) == d

    x, y = mapa.hilbert_xy(mapa.UKUPNO_KOMBINACIJA)  # prva ćelija iza poslednje kombinacije
    prazna = mapa.detalj_celije(x, y)
    assert prazna["brojevi"] is None and prazna["rang"] is None and prazna["osobine"] is None

    for xy in [(-1, 0), (0, mapa.DIMENZIJA)]:
        try:
            mapa.detalj_celije(*xy)
        except ValueError:
            continue
        raise AssertionError(f"prihvaćene koordinate van kvadrata: {xy}")
    print("test_detalj_kombinacije_i_celije: OK")


def test_plocice_ako_postoje():
    """Ako su pločice generisane, moraju odgovarati trenutnim konstantama."""
    koren = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "static", "mapa")
    slojevi = [s for s in mapa.OSOBINE
               if os.path.isfile(os.path.join(koren, s, "meta.json"))]
    if not slojevi:
        print("test_plocice_ako_postoje: preskočeno (pločice nisu generisane)")
        return

    for sloj in slojevi:
        with open(os.path.join(koren, sloj, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["red_krive"] == mapa.RED_KRIVE, f"{sloj}: pločice su za drugi red krive"
        assert meta["dimenzija"] == mapa.DIMENZIJA, f"{sloj}: pločice su za drugu dimenziju"
        assert meta["broj_kombinacija"] == mapa.UKUPNO_KOMBINACIJA
        ocekivano = sum(4 ** z for z in range(mapa.MAX_ZOOM + 1))   # 341 za zumove 0..4
        assert meta["broj_plocica"] == ocekivano, f"{sloj}: {meta['broj_plocica']} pločica"
        for z in range(mapa.MAX_ZOOM + 1):
            n = 2 ** z
            put = os.path.join(koren, sloj, str(z), str(n - 1), f"{n - 1}.png")
            assert os.path.isfile(put), f"nedostaje pločica {put}"
        _proveri_piksele(koren, sloj, meta)
    print(f"test_plocice_ako_postoje: OK ({', '.join(slojevi)})")


def _proveri_piksele(koren, sloj, meta):
    """Piksel na najvišem zumu mora nositi osobinu baš one kombinacije koja je tu.

    Ovim se zatvara krug generator -> pločica -> klik na mapu: da su x i y
    zamenjeni ili da je red pločica obrnut, ova provera bi pala.
    """
    from PIL import Image

    for r in _uzorak_rangova(20, seed=4096):
        r = int(r)
        x, y = mapa.koordinate(r)
        slika = Image.open(os.path.join(koren, sloj, str(mapa.MAX_ZOOM),
                                        str(x // mapa.VELICINA_PLOCICE),
                                        f"{y // mapa.VELICINA_PLOCICE}.png"))
        indeks = slika.getpixel((x % mapa.VELICINA_PLOCICE, y % mapa.VELICINA_PLOCICE))
        vrednost = mapa.osobine(mapa.unrang(r))[sloj]
        ocekivano = 1 + round((vrednost - meta["min"]) / (meta["max"] - meta["min"]) * 254)
        assert indeks == ocekivano, (f"{sloj}: piksel ({x},{y}) ima indeks {indeks}, "
                                     f"a kombinacija {mapa.unrang(r)} traži {ocekivano}")


def main():
    test_rang_je_leksikografski()
    test_rang_unrang_bijekcija()
    test_odbija_neispravno()
    test_hilbert_bijekcija_i_granice()
    test_hilbert_susedi()
    test_prazan_deo_krive()
    test_osobine_vektorski_isto()
    test_detalj_kombinacije_i_celije()
    test_plocice_ako_postoje()
    print("\nSVI TESTOVI MAPE PROSLI [OK]")


if __name__ == "__main__":
    main()
