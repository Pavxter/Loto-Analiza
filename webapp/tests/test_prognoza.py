"""Testovi strane „Prognoza" — PLAN_PROGNOZA.md §6 i §9.

Najvažniji: zaštita od curenja budućnosti i determinizam retro-bektesta.
Radi nad privremenom bazom (ne dira loto_baza.db).

Pokretanje:  python -X utf8 -m webapp.tests.test_prognoza
"""

import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from webapp.core import baza, bektest, konfig, prognoza  # noqa: E402
from webapp.core import razlicitost_teorija as T  # noqa: E402
from webapp.core.prediktori import PREDIKTORI  # noqa: E402
from webapp.core.prediktori_komb import PREDIKTORI_KOMB  # noqa: E402


def sinteticka_istorija(broj_kola=300, seme=42):
    """Deterministička sintetička istorija: (kolo, 7 brojeva u 'redosledu izvlačenja')."""
    rng = random.Random(seme)
    istorija = []
    for i in range(broj_kola):
        brojevi = tuple(rng.sample(range(1, konfig.MAX_BROJ + 1), konfig.BROJEVA_U_KOMBINACIJI))
        istorija.append((2020001 + i, brojevi))
    return istorija


def nova_baza(istorija):
    """Privremena baza napunjena datom istorijom."""
    fd, putanja = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(putanja)  # postavi_bazu će je kreirati
    baza.postavi_bazu(putanja)
    conn = baza.konekcija(putanja)
    for kolo, brojevi in istorija:
        baza.dodaj_kolo(conn, kolo, f"2020-01-{1 + kolo % 28:02d}", list(brojevi))
    return conn, putanja


def test_ekvivalencija_i_bez_curenja():
    """Retro motor za kolo N mora dati ISTO što i čisti prediktor nad istorijom < N.

    Time se istovremeno dokazuje: (a) inkrementalno stanje == direktan račun
    (uključujući klizni Bajes), (b) motor ne vidi ciljno kolo (čista funkcija
    fizički ne prima kola >= N).
    """
    istorija = sinteticka_istorija(220)
    conn, putanja = nova_baza(istorija)
    try:
        prognoza.retro_bektest(conn)
        for n_index in (60, 100, 150, 219):  # nasumični preseci kroz istoriju
            kolo_n = istorija[n_index][0]
            redovi = {r["metod"]: r["broj"]
                      for r in baza.prognoze_za_kolo(conn, kolo_n, "retro")}
            pre_n = istorija[:n_index]
            for metod, (_naziv, fn, _opis) in PREDIKTORI.items():
                ocekivano = fn(pre_n, prognoza.RETRO_PERIOD, ciljno_kolo=kolo_n)
                assert redovi[metod] == ocekivano, (
                    f"kolo {kolo_n}, metod {metod}: motor={redovi[metod]}, cist={ocekivano}")
        print("test_ekvivalencija_i_bez_curenja: OK")
    finally:
        conn.close(); os.remove(putanja)


def test_prosirenje_istorije_ne_menja_proslost():
    """Dodavanje novog kola NE SME da promeni retro predikcije za ranija kola."""
    ist_kratka = sinteticka_istorija(200)
    ist_duga = ist_kratka + [(2020201, (39, 1, 2, 3, 4, 5, 6))]

    def retro_redovi(istorija):
        conn, putanja = nova_baza(istorija)
        try:
            prognoza.retro_bektest(conn)
            redovi = conn.execute(
                "SELECT kolo, metod, broj, pogodak FROM prognoze WHERE izvor='retro' "
                "ORDER BY kolo, metod").fetchall()
            return [tuple(r) for r in redovi]
        finally:
            conn.close(); os.remove(putanja)

    kratki = retro_redovi(ist_kratka)
    dugi = retro_redovi(ist_duga)
    # svi redovi kratke istorije moraju biti identični u dugoj (novo kolo samo dodaje red)
    assert dugi[:len(kratki)] == kratki, "Predikcije za prošla kola su se promenile!"
    print("test_prosirenje_istorije_ne_menja_proslost: OK")


def test_determinizam_retro():
    """Dva pokretanja retro-bektesta -> identičan skup redova (uklj. random sa seedom)."""
    istorija = sinteticka_istorija(200)
    conn, putanja = nova_baza(istorija)
    try:
        prognoza.retro_bektest(conn)
        prvi = conn.execute("SELECT kolo, metod, broj, pogodak FROM prognoze "
                            "WHERE izvor='retro' ORDER BY kolo, metod").fetchall()
        prognoza.retro_bektest(conn)
        drugi = conn.execute("SELECT kolo, metod, broj, pogodak FROM prognoze "
                             "WHERE izvor='retro' ORDER BY kolo, metod").fetchall()
        assert [tuple(r) for r in prvi] == [tuple(r) for r in drugi], "Retro nije determinističан!"
        assert len(prvi) > 0
        print(f"test_determinizam_retro: OK ({len(prvi)} redova)")
    finally:
        conn.close(); os.remove(putanja)


def test_evaluacija_pri_unosu_kola():
    """Pri unosu kola sve otvorene prognoze dobijaju pogodak 0/1 bez akcije korisnika."""
    istorija = sinteticka_istorija(100)
    conn, putanja = nova_baza(istorija)
    try:
        sledece = prognoza.ciljno_kolo(istorija)
        # dve ručne prognoze: jedna će pogoditi, druga ne
        dobitna = [10, 20, 30, 1, 2, 3, 4]
        baza.sacuvaj_prognozu(conn, sledece, "hot", 10, None, "uzivo")     # pogodak
        baza.sacuvaj_prognozu(conn, sledece, "cold", 39, None, "uzivo")    # promašaj
        rezime = bektest.dodaj_kolo_i_proveri(conn, sledece, "2026-01-01", dobitna)
        assert rezime["ocenjeno_prognoza"] == 2
        redovi = {r["metod"]: r["pogodak"] for r in baza.prognoze_za_kolo(conn, sledece)}
        assert redovi["hot"] == 1 and redovi["cold"] == 0
        print("test_evaluacija_pri_unosu_kola: OK")
    finally:
        conn.close(); os.remove(putanja)


def test_ocenjena_prognoza_nepromenljiva():
    """Pokušaj izmene ocenjene prognoze mora biti odbijen."""
    istorija = sinteticka_istorija(60)
    conn, putanja = nova_baza(istorija)
    try:
        kolo = istorija[-1][0]
        conn.execute("INSERT INTO prognoze (kolo, metod, broj, period, izvor, pogodak, kreirano) "
                     "VALUES (?, 'hot', 5, NULL, 'uzivo', 1, '2026-01-01')", (kolo,))
        conn.commit()
        uspeh = baza.sacuvaj_prognozu(conn, kolo, "hot", 7, None, "uzivo")
        assert uspeh is False, "Izmena ocenjene prognoze je prošla!"
        red = conn.execute("SELECT broj FROM prognoze WHERE kolo=? AND metod='hot'", (kolo,)).fetchone()
        assert red["broj"] == 5, "Broj je promenjen uprkos oceni!"
        print("test_ocenjena_prognoza_nepromenljiva: OK")
    finally:
        conn.close(); os.remove(putanja)


def test_random_u_pojasu_pouzdanosti():
    """Sanity ceo sistem: kontrolna grupa mora završiti unutar 95% pojasa (PLAN §9)."""
    istorija = sinteticka_istorija(400, seme=7)
    conn, putanja = nova_baza(istorija)
    try:
        prognoza.retro_bektest(conn)
        stat = prognoza.statistika(conn, "retro")
        rnd = next(m for m in stat["metode"] if m["metod"] == "random")
        n, k = rnd["n"], rnd["k"]
        p = prognoza.BASELINE
        margina = 1.96 * (p * (1 - p) / n) ** 0.5
        uspesnost = k / n
        assert abs(uspesnost - p) <= margina, (
            f"Kontrolna grupa van pojasa: {uspesnost:.3f} vs {p:.3f} ± {margina:.3f} — bag u evaluaciji!")
        print(f"test_random_u_pojasu_pouzdanosti: OK ({100*uspesnost:.1f}% vs {100*p:.1f}%, n={n})")
    finally:
        conn.close(); os.remove(putanja)


def test_uzivo_tok():
    """Generisanje uživo predloga: kreiraju se jednom, ne preračunavaju se sami."""
    istorija = sinteticka_istorija(120)
    conn, putanja = nova_baza(istorija)
    try:
        prvi = prognoza.generisi_uzivo(conn, period=100)
        assert prvi["ciljno_kolo"] == istorija[-1][0] + 1
        assert len(prvi["predlozi"]) == len(PREDIKTORI)
        drugi = prognoza.generisi_uzivo(conn, period=50)  # drugi period — ali NE preračunava
        assert [p["broj"] for p in prvi["predlozi"]] == [p["broj"] for p in drugi["predlozi"]], \
            "Predlozi su se preračunali sami od sebe!"
        print("test_uzivo_tok: OK")
    finally:
        conn.close(); os.remove(putanja)


def test_komb_prediktori_validni():
    """Svaki kombinacijski prediktor vraća tačno 7 jedinstvenih brojeva 1–39, sortirano."""
    istorija = sinteticka_istorija(200)
    for metod, (_naziv, fn, _opis) in PREDIKTORI_KOMB.items():
        komb = fn(istorija, prognoza.RETRO_PERIOD, ciljno_kolo=2020500)
        assert komb is not None and len(komb) == 7, f"{metod}: {komb}"
        assert len(set(komb)) == 7, f"{metod}: nisu jedinstveni {komb}"
        assert all(1 <= b <= konfig.MAX_BROJ for b in komb), f"{metod}: van opsega {komb}"
        assert list(komb) == sorted(komb), f"{metod}: nije sortirano {komb}"
    print("test_komb_prediktori_validni: OK")


def test_komb_bez_curenja_i_determinizam():
    """Retro kombinacijski red za kolo N == čista funkcija nad istorijom < N (bez curenja),
    uključujući k_cooc (matrica ne sme da vidi ciljno kolo)."""
    istorija = sinteticka_istorija(220)
    conn, putanja = nova_baza(istorija)
    try:
        prognoza.retro_bektest(conn)
        for n_index in (60, 130, 219):
            kolo_n = istorija[n_index][0]
            redovi = {r["metod"]: r["kombinacija"]
                      for r in baza.prognoze_za_kolo(conn, kolo_n, "retro") if r["vrsta"] == "komb"}
            pre_n = istorija[:n_index]
            for metod, (_naziv, fn, _opis) in PREDIKTORI_KOMB.items():
                ocek = fn(pre_n, prognoza.RETRO_PERIOD, ciljno_kolo=kolo_n)
                assert redovi[metod] == ",".join(map(str, ocek)), \
                    f"kolo {kolo_n}, {metod}: motor={redovi[metod]} cist={ocek}"
        print("test_komb_bez_curenja_i_determinizam: OK")
    finally:
        conn.close(); os.remove(putanja)


def test_komb_evaluacija_pri_unosu():
    """Pri unosu kola kombinacijske prognoze dobijaju preklapanje 0..7 (ne pogodak)."""
    istorija = sinteticka_istorija(100)
    conn, putanja = nova_baza(istorija)
    try:
        sledece = prognoza.ciljno_kolo(istorija)
        baza.sacuvaj_prognozu_komb(conn, sledece, "k_hot7", "1,2,3,4,5,6,7", None, "uzivo")
        dobitna = [1, 2, 3, 20, 21, 22, 23]   # deli 3 sa predlogom
        rezime = bektest.dodaj_kolo_i_proveri(conn, sledece, "2026-01-01", dobitna)
        assert rezime["ocenjeno_prognoza"] == 1
        red = baza.prognoze_za_kolo(conn, sledece)[0]
        assert red["preklapanje"] == 3, red["preklapanje"]
        assert red["pogodak"] is None   # kombinacije ne koriste pogodak
        print("test_komb_evaluacija_pri_unosu: OK")
    finally:
        conn.close(); os.remove(putanja)


def test_komb_random_u_pojasu():
    """Kontrolna grupa (k_random): prosek preklapanja unutar pojasa oko μ=1,256 (§9)."""
    istorija = sinteticka_istorija(400, seme=11)
    conn, putanja = nova_baza(istorija)
    try:
        prognoza.retro_bektest(conn)
        stat = prognoza.statistika_komb(conn, "retro")
        rnd = next(m for m in stat["metode"] if m["metod"] == "k_random")
        n, prosek = rnd["n"], rnd["prosek"]
        d, g = T.pojas_proseka(n)
        assert d <= prosek <= g, f"k_random van pojasa: {prosek} nije u [{d:.3f}, {g:.3f}]"
        print(f"test_komb_random_u_pojasu: OK (prosek={prosek:.3f}, μ=1.256, n={n})")
    finally:
        conn.close(); os.remove(putanja)


def test_retro_brzina_oba():
    """Ceo retro (jednobrojni + kombinacijski) nad ~1.400 kola < 15 s (§6)."""
    istorija = sinteticka_istorija(1400, seme=1)
    conn, putanja = nova_baza(istorija)
    try:
        rez = prognoza.retro_bektest(conn)
        assert rez["trajanje_s"] < 15.0, f"presporo: {rez['trajanje_s']}s"
        assert rez["redova_komb"] > 0 and rez["redova_broj"] > 0
        print(f"test_retro_brzina_oba: OK ({rez['trajanje_s']}s, broj={rez['redova_broj']}, komb={rez['redova_komb']})")
    finally:
        conn.close(); os.remove(putanja)


def main():
    test_ekvivalencija_i_bez_curenja()
    test_prosirenje_istorije_ne_menja_proslost()
    test_determinizam_retro()
    test_evaluacija_pri_unosu_kola()
    test_ocenjena_prognoza_nepromenljiva()
    test_random_u_pojasu_pouzdanosti()
    test_uzivo_tok()
    test_komb_prediktori_validni()
    test_komb_bez_curenja_i_determinizam()
    test_komb_evaluacija_pri_unosu()
    test_komb_random_u_pojasu()
    test_retro_brzina_oba()
    print("\nSVI TESTOVI PROGNOZE PROSLI [OK]")


if __name__ == "__main__":
    main()
