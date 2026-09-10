"""Testovi sekvencijalnog prediktora — PLAN_SEKVENCIJALNI_PREDIKTOR §7.

Tri grupe:
  - raspodele: svaki ekspert daje ispravnu raspodelu (p ≥ 0, Σp = 7, p < 1);
  - koeficijent: K = 1 tačno kad je u mešavini samo uniformni; K unutar pojasa na
    slučajnoj sintetici; K ispod pojasa na PRISTRASNOJ sintetici — bez tog para
    testova K ≈ 1 na pravim podacima ne znači ništa;
  - registar i Sinteza: `k_sekv` prolazi kroz retro-bektest kao svaki drugi metod;
  - trajno stanje: serijalizacija, inkrementalni korak na unos kola i pogledi za API;
  - garancije motora: bez curenja budućnosti, determinizam, brzina.

Radi nad privremenom bazom (ne dira loto_baza.db).

Pokretanje:  python -X utf8 -m webapp.tests.test_sekv
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from webapp.core import (analitika, baza, generator, konfig, prediktori, prelazi,  # noqa: E402
                         prognoza, razlicitost_teorija, sinteza, sekvencijalni as S)
from webapp.tests.test_prognoza import nova_baza, sinteticka_istorija  # noqa: E402

MAX_BROJ = konfig.MAX_BROJ
K = konfig.BROJEVA_U_KOMBINACIJI


def pristrasna_istorija(broj_kola=1500, favorit=7, udeo=0.30, seme=11):
    """Sintetika sa ugrađenim signalom: `favorit` je nasilno u kolu u `udeo` slučajeva.

    P(favorit) ≈ 0,30 + 0,70 · 7/39 ≈ 0,43 naspram 0,18 pod slučajnošću. Ovo je
    jedina istorija u testovima na kojoj model SME nešto da nauči.
    """
    rng = random.Random(seme)
    istorija = []
    for i in range(broj_kola):
        if rng.random() < udeo:
            ostali = rng.sample([b for b in range(1, MAX_BROJ + 1) if b != favorit], K - 1)
            brojevi = tuple([favorit] + ostali)
        else:
            brojevi = tuple(rng.sample(range(1, MAX_BROJ + 1), K))
        istorija.append((2020001 + i, brojevi))
    return istorija


# ----------------------------------------------------------------------------
# Raspodele (§2.1)
# ----------------------------------------------------------------------------

def test_raspodela_suma_7():
    """Svaki ekspert u svakoj tački: p ≥ 0, Σp = 7, i p < 1 (da ln(1 − p) postoji)."""
    istorija = sinteticka_istorija(400, seme=3)
    tacaka = 0
    for granica in (60, 120, 250, 399):
        prozor = istorija[max(0, granica - S.PERIOD):granica]
        for e, p in S.raspodele(prozor).items():
            assert len(p) == MAX_BROJ, e
            assert all(v > 0 for v in p.values()), e
            assert all(v < 1 for v in p.values()), (e, max(p.values()))
            assert abs(sum(p.values()) - K) < 1e-9, (e, sum(p.values()))
            tacaka += 1
    # i mešavina mora da zadrži Σp = 7
    m = S.Mesavina()
    p, _po_ekspertu, predlog = m.predvidi(istorija[100:200])
    assert abs(sum(p.values()) - K) < 1e-9
    assert len(set(predlog)) == K and list(predlog) == sorted(predlog)
    print(f"test_raspodela_suma_7: OK ({tacaka} raspodela, {len(S.EKSPERTI)} eksperata)")


def test_period_isti_kao_retro():
    """Sekvencijalni model mora da vidi isti prozor kao retro-bektest (§2.6)."""
    assert konfig.SEKV_PERIOD == prognoza.RETRO_PERIOD
    assert konfig.SEKV_MIN_START == prognoza.MIN_START
    print("test_period_isti_kao_retro: OK (prozor 100, start 50)")


# ----------------------------------------------------------------------------
# Koeficijent nepredvidivosti (§2.5)
# ----------------------------------------------------------------------------

def test_uniformni_K_jednak_1():
    """Mešavina samo sa uniformnim ekspertom → K = 1,000 tačno, σ = 0."""
    istorija = sinteticka_istorija(200, seme=5)
    # gubitak uniformnog eksperta ne zavisi od izvučenog kola — to je imenilac K
    for _kolo, brojevi in istorija[:5]:
        g = S.log_gubitak(S.uniformna(), set(brojevi))
        assert abs(g - S.GUBITAK_UNIFORMNOG) < 1e-12, g

    m = S.Mesavina(eksperti=(S.UNIFORMNI,))
    list(S.prodji(istorija, mesavina=m))
    assert m.n == 200 - S.MIN_START
    assert m.koeficijent == 1.0, m.koeficijent
    # E[ℓ] se računa drugom formulom (C − 7·ā) pa sme da odstupi za jedan ulp
    assert abs(m.ocekivano - 1.0) < 1e-12, m.ocekivano
    assert m.sigma == 0.0, m.sigma
    assert m.tezine[S.UNIFORMNI] == 1.0
    print(f"test_uniformni_K_jednak_1: OK (K = {m.koeficijent}, σ = {m.sigma})")


def test_K_na_sintetici():
    """5.000 slučajnih kola: K unutar pojasa i nijedan ekspert se ne izdvaja.

    Ovo je očekivani ishod iz §1, zapisan kao test: kod nezavisnih izvlačenja model
    ne sme ništa da nauči, i sam to mora da prijavi.
    """
    istorija = sinteticka_istorija(5000, seme=17)
    m, koraci = S.prodji_do_kraja(istorija)
    s = m.stanje()
    assert s["pojas_donja"] <= s["k"] <= s["pojas_gornja"], s
    assert len(koraci) == 5000 - S.MIN_START
    # svaki ekspert osim uniformnog na šumu gubi VIŠE od uniformnog
    for e in S.EKSPERTI:
        if e != S.UNIFORMNI:
            assert s["k_eksperti"][e] > 1.0, (e, s["k_eksperti"][e])
    # Nijedan ekspert se ne izdvaja iz grupe. Fixed-share čini da težine prate
    # skorašnji učinak, a na šumu skorašnji učinak nikoga trajno ne izdvaja — zato
    # se ne traži da baš uniformni bude na vrhu (na šumu je vrh stvar slučaja),
    # nego da niko ne pobegne od 1/n.
    n = len(S.EKSPERTI)
    najveca, najmanja = max(s["tezine"].values()), min(s["tezine"].values())
    assert najveca < 2.0 / n, s["tezine"]
    assert najmanja > 0.3 / n, s["tezine"]
    print(f"test_K_na_sintetici: OK (K = {s['k']}, pojas {s['pojas_donja']}–{s['pojas_gornja']}, "
          f"težine {najmanja:.4f}–{najveca:.4f} oko 1/n = {1 / n:.4f})")


def test_K_na_pristrasnoj_sintetici():
    """Sintetika gde broj 7 izlazi znatno češće: K mora pasti ISPOD pojasa.

    Dokaz da bi model prepoznao signal kad bi ga bilo — bez njega K ≈ 1 na pravim
    podacima ne bi bio osobina podataka nego moguća mana modela.

    Odstupanje od plana (§7): plan očekuje da `hot` dobije najveću težinu. U praksi
    pobeđuje `bayes`, koji meri istu stvar (frekvenciju) preko glatkijeg prozora.
    Test zato traži da vrh bude iz frekvencijske porodice, a ne baš `hot`.
    """
    istorija = pristrasna_istorija(1500, favorit=7)
    m, koraci = S.prodji_do_kraja(istorija)
    s = m.stanje()
    assert s["k"] < s["pojas_donja"], s
    assert S.zakljucak(s["k"], s["pojas_donja"], s["pojas_gornja"]).startswith("Ispod pojasa")
    najveci = max(s["tezine"], key=lambda e: s["tezine"][e])
    assert najveci in ("hot", "bayes", "hybrid"), s["tezine"]
    assert s["tezine"]["hot"] > s["tezine"][S.UNIFORMNI], s["tezine"]
    assert s["k_eksperti"]["hot"] < 1.0, s["k_eksperti"]
    # signal se vidi i na predlogu: favorit je skoro uvek u top-7
    udeo = sum(1 for x in koraci if 7 in x["predlog"]) / len(koraci)
    assert udeo > 0.9, udeo
    print(f"test_K_na_pristrasnoj_sintetici: OK (K = {s['k']} < {s['pojas_donja']}, "
          f"vrh = {najveci}, broj 7 u {100 * udeo:.0f}% predloga)")


# ----------------------------------------------------------------------------
# Ravnoća raspodele i prag (PLAN_KORAK_IZBORA §2.3, §2.4)
# ----------------------------------------------------------------------------

# Šta je izmereno na pravoj bazi u kolu 2026072 i pokrenulo ceo plan: raspon p_mix
# je bio 5,7% od baseline-a 7/39, a razlika 7. i 8. kandidata 0,24%. Pitanje testa
# je da li su te vrednosti signal ili ono što čist šum ionako proizvodi.
IZMERENO_RASPON = 0.057
IZMERENO_ZAZOR = 0.0024

# Semena i dužina od kojih je izveden konfig.PRAG_RASPONA. Menjati samo uz ponovno
# izvođenje praga i novi datum u konfig.py — inače test i konstanta govore o
# različitim merenjima.
PRAG_SEMENA = (17, 23, 31, 47, 59)
PRAG_KOLA = 1500


def _ravnoca_kroz_istoriju(istorija):
    """Mera ravnoće za svaki ocenjeni korak — isti prolaz kao `prodji`, bez baze."""
    m = S.Mesavina()
    izlaz = []
    for i in range(len(istorija)):
        if i < S.MIN_START:
            m.posmatraj(istorija[i][1])
            continue
        p, po_ekspertu, _predlog = m.predvidi(S._prozor_pre(istorija, i, S.PERIOD))
        izlaz.append(S.izracunaj_ravnocu(p))
        m.uci(p, po_ekspertu, {int(b) for b in istorija[i][1]})
    return izlaz


def _percentil(vrednosti, q):
    a = sorted(vrednosti)
    return a[min(len(a) - 1, int(q * len(a)))]


def test_raspon_p_mix_na_sintetici():
    """KLJUČNI test plana: raspon p_mix na čistom šumu, i odatle PRAG_RASPONA.

    Model se pušta na uniformne sintetičke istorije — nezavisna izvlačenja, ništa
    za naučiti. Raspodela `raspon_udeo` koju tamo pravi JESTE raspon koji šum sam
    proizvodi. Ako izmerenih 5,7% sa prave baze upada u tu raspodelu, onda raspon
    nije nalaz nego pozadina, i predlog modela je izbor iz gotovo ravne raspodele.

    Ispis ovog testa je postupak izvođenja praga: 95. percentil je vrednost koja
    stoji u konfig.PRAG_RASPONA. Test je zato i regresioni — svaka izmena eksperata
    ili mešavine pomera raspodelu, pa prag mora da se izvede iznova i da dobije nov
    datum u konfig.py (§7: prag se nikad ne bira po osećaju).
    """
    svi = []
    for seme in PRAG_SEMENA:
        svi.extend(_ravnoca_kroz_istoriju(sinteticka_istorija(PRAG_KOLA, seme=seme)))
    raspon = [x["raspon_udeo"] for x in svi]
    zazor = [x["zazor_udeo"] for x in svi]
    p05, p50, p95 = (_percentil(raspon, q) for q in (0.05, 0.50, 0.95))

    # 1. Izmereno na pravoj bazi je unutar šuma — i to ispod medijane šuma.
    assert p05 < IZMERENO_RASPON < p95, (p05, IZMERENO_RASPON, p95)
    assert IZMERENO_RASPON < p50, (IZMERENO_RASPON, p50)
    assert IZMERENO_ZAZOR < _percentil(zazor, 0.95), (IZMERENO_ZAZOR, _percentil(zazor, 0.95))

    # 2. Prag u konfigu je tačno taj 95. percentil (na 4 decimale).
    assert round(p95, 4) == konfig.PRAG_RASPONA, (
        f"prag se razišao sa merenjem: izmereno {p95:.6f}, u konfigu {konfig.PRAG_RASPONA}. "
        "Ako je model menjan, upiši novu vrednost i nov datum u konfig.PRAG_RASPONA.")

    # 3. Ni sama sintetika ne sme da pređe prag češće nego u 5% koraka.
    preko = sum(1 for x in raspon if x > konfig.PRAG_RASPONA) / len(raspon)
    assert preko <= 0.06, preko
    assert all(not S.bez_preferencije(x) for x in raspon if x > konfig.PRAG_RASPONA)
    print(f"test_raspon_p_mix_na_sintetici: OK ({len(raspon)} koraka šuma — "
          f"raspon_udeo p05={p05:.4f} p50={p50:.4f} p95={p95:.4f}; "
          f"izmereno na kolu 2026072 = {IZMERENO_RASPON} → unutar šuma, "
          f"PRAG_RASPONA = {konfig.PRAG_RASPONA})")


def test_ravnoca_u_stanju():
    """`raspon_udeo` i `zazor_udeo` se upisuju uz svako kolo i stižu do API-ja."""
    istorija = sinteticka_istorija(200, seme=77)
    conn, putanja = nova_baza(istorija)
    try:
        S.rekonstruisi(conn)
        redovi = baza.sekv_lista(conn)
        assert len(redovi) == 200 - S.MIN_START
        for r in redovi:
            for kolona in ("p_min", "p_max", "raspon_udeo", "zazor_udeo"):
                assert r[kolona] is not None, (r["kolo"], kolona)
            assert 0 < r["p_min"] <= r["p_max"] < 1, r["kolo"]
            assert abs(r["raspon_udeo"] - (r["p_max"] - r["p_min"]) / S.BASELINE) < 1e-9
            assert r["zazor_udeo"] >= 0

        # ista mera stiže do sva tri pogleda, sa pragom i zaključkom
        st = S.stanje_api(conn)
        for cvor in (st["ravnoca"], st["ravnoca_poslednjeg"],
                     S.korak_api(conn, istorija[120][0])["ravnoca"]):
            assert cvor is not None
            assert cvor["prag_raspona"] == konfig.PRAG_RASPONA
            assert cvor["bez_preferencije"] == (cvor["raspon_udeo"] <= konfig.PRAG_RASPONA)
        # ravnoća ciljnog kola se računa iz žive mešavine, ne prepisuje iz poslednjeg reda
        assert st["ravnoca"]["raspon_udeo"] != st["ravnoca_poslednjeg"]["raspon_udeo"]
        print(f"test_ravnoca_u_stanju: OK ({len(redovi)} redova, "
              f"raspon ciljnog kola {st['ravnoca']['raspon_udeo']:.4f})")
    finally:
        conn.close(); os.remove(putanja)


def test_stari_redovi_bez_ravnoce():
    """Baza od pre ove izmene: kolone su NULL, API vraća None umesto izmišljenog broja."""
    istorija = sinteticka_istorija(120, seme=79)
    conn, putanja = nova_baza(istorija)
    try:
        S.rekonstruisi(conn)
        conn.execute("UPDATE sekv_stanje SET p_min=NULL, p_max=NULL, "
                     "raspon_udeo=NULL, zazor_udeo=NULL")
        conn.commit()
        assert S.rezime(conn)["ravnoca"] is None
        assert S.stanje_api(conn)["ravnoca_poslednjeg"] is None
        assert S.korak_api(conn, istorija[80][0])["ravnoca"] is None
        print("test_stari_redovi_bez_ravnoce: OK (NULL → None, bez izuzetka)")
    finally:
        conn.close(); os.remove(putanja)


# ----------------------------------------------------------------------------
# Tie-break i nepromenjen K (PLAN_KORAK_IZBORA §2.2, Faza 2)
# ----------------------------------------------------------------------------

# K zabeležen PRE izmene tie-breaka, na fiksnim sintetičkim istorijama. Izbor
# sedmorke i merenje znanja su odvojeni: K se računa iz raspodele, a tie-break bira
# među brojevima kad je raspodela već data. Ako se ovi brojevi ikad pomere zbog
# izmene u koraku izbora, to je greška, ne poboljšanje. (Referenca sa prave baze:
# K = 1,000068 posle 1.372 kola zaključno sa 2026071.)
K_PRE_IZMENE = {(17, 1500): 1.000036, (42, 300): 1.00003, (95, 200): 1.000235}


def _stari_izbor(p):
    """Tie-break kakav je bio pre Faze 2: kod jednakih verovatnoća manji broj."""
    return tuple(sorted(sorted(range(1, MAX_BROJ + 1), key=lambda b: (-p[b], b))[:K]))


def test_tiebreak_reproducibilan():
    """Isto seme → isti predlog, koliko god puta se pozvalo; različito seme → sme drugi."""
    p = S.uniformna()          # sve verovatnoće tačno jednake: odlučuje samo tie-break
    assert S.izaberi_top7(p, 2026071) == S.izaberi_top7(p, 2026071)
    assert len({S.izaberi_top7(p, 2026071) for _ in range(50)}) == 1
    assert S.izaberi_top7(p, 2026071) != S.izaberi_top7(p, 2026072)

    # seme se izvodi iz prozora, pa ga nijedan pozivalac ne bira ručno
    istorija = sinteticka_istorija(300, seme=61)
    prozor = istorija[100:200]
    assert S.seme_izbora(prozor) == prozor[-1][0] == istorija[199][0]

    # ceo prolaz dva puta → identičan niz predloga (retro-bektest ostaje determinističan)
    prvi = [x["predlog"] for x in S.prodji_do_kraja(istorija)[1]]
    drugi = [x["predlog"] for x in S.prodji_do_kraja(istorija)[1]]
    assert prvi == drugi
    # i predlog iz keširanog prefiksa je isti kao iz punog prolaza
    S.zaboravi_kes()
    assert S.predlog_za(istorija[:250]) == prvi[250 - S.MIN_START]
    print(f"test_tiebreak_reproducibilan: OK ({len(prvi)} koraka, seme = poslednje viđeno kolo)")


def test_tiebreak_bez_pristrasnosti():
    """Kod jednakih verovatnoća staro pravilo uvek bira 1–7; novo bira po celom opsegu.

    Test se radi na uniformnoj raspodeli, jer je to jedini slučaj u kom tie-break
    uopšte odlučuje. Na pravim raspodelama tačnih veza nema (drugi deo testa), pa bi
    merenje proseka izabranog broja kroz istoriju merilo šum eksperata, ne ovo pravilo.
    """
    p = S.uniformna()
    assert _stari_izbor(p) == tuple(range(1, K + 1)), "staro pravilo nije birlo 1–7"

    semena = [2020001 + i for i in range(2000)]
    izabrani = [b for seme in semena for b in S.izaberi_top7(p, seme)]
    prosek = sum(izabrani) / len(izabrani)
    assert len(set(izabrani)) == MAX_BROJ, "novo pravilo ne dohvata sve brojeve"
    assert abs(prosek - (MAX_BROJ + 1) / 2) < 0.25, prosek
    # nijedan broj se ne bira bitno češće od 7/39 udela
    for b in range(1, MAX_BROJ + 1):
        udeo = izabrani.count(b) / len(semena)
        assert 0.12 < udeo < 0.24, (b, udeo)

    # Na pravim raspodelama tačnih veza nema, pa se izbor ne menja — izmereno, ne
    # pretpostavljeno. Zato ova izmena ni ne pomera nijedan postojeći rezultat.
    istorija = sinteticka_istorija(400, seme=63)
    m, veza, isto = S.Mesavina(), 0, 0
    for i in range(len(istorija)):
        if i < S.MIN_START:
            m.posmatraj(istorija[i][1])
            continue
        p_mix, po_ekspertu, predlog = m.predvidi(S._prozor_pre(istorija, i, S.PERIOD))
        vrednosti = sorted(p_mix.values())
        veza += len(vrednosti) - len(set(vrednosti))
        isto += (predlog == _stari_izbor(p_mix))
        m.uci(p_mix, po_ekspertu, {int(b) for b in istorija[i][1]})
    koraka = len(istorija) - S.MIN_START
    assert veza == 0, f"pojavile su se tačne veze ({veza}) — tie-break sada zaista odlučuje"
    assert isto == koraka, (isto, koraka)
    print(f"test_tiebreak_bez_pristrasnosti: OK (na jednakim p prosek {prosek:.2f} ≈ 20 "
          f"i svih 39 brojeva; na {koraka} pravih koraka 0 veza, izbor nepromenjen)")


def test_K_nepromenjen():
    """Regresioni: izmena koraka izbora ne sme da pomeri K ni na jednoj decimali.

    K se računa iz raspodele i stvarnog kola, a tie-break bira među brojevima tek
    pošto je raspodela data. Ta dva se ne mešaju, i ovaj test je jedino mesto gde
    to piše kao broj.
    """
    for (seme, kola), ocekivano in K_PRE_IZMENE.items():
        m, _koraci = S.prodji_do_kraja(sinteticka_istorija(kola, seme=seme))
        assert m.stanje()["k"] == ocekivano, (seme, kola, m.stanje()["k"], ocekivano)
    print(f"test_K_nepromenjen: OK ({len(K_PRE_IZMENE)} istorije, "
          f"K = {', '.join(str(v) for v in K_PRE_IZMENE.values())})")


# ----------------------------------------------------------------------------
# Dva izlaza: predlog modela i tiket Generatora (PLAN_KORAK_IZBORA §2.1, Faza 3)
# ----------------------------------------------------------------------------

# Filteri koji menjaju izbor dovoljno da se razlika vidi: tražena parnost i
# zabranjeni uzastopni brojevi. Ako bi predlog modela ikad reagovao na njih, prestao
# bi da svedoči o modelu — što je i razlog zašto ne prolazi kroz Generator.
OSTRI_FILTERI = {"parni": 3, "uzastopni": 0, "min_sv": 15, "max_sv": 25}


def _analiza_iz(conn):
    return analitika.Analiza(analitika.ucitaj_df(conn), period_analize=0)


def test_predlog_bez_filtera():
    """Predlog modela ne zavisi ni od jednog podešavanja Generatora (§2.1, §7)."""
    istorija = sinteticka_istorija(220, seme=81)
    conn, putanja = nova_baza(istorija)
    try:
        S.rekonstruisi(conn)
        a = _analiza_iz(conn)
        bez = S.stanje_api(conn, analiza=a)
        sa = S.stanje_api(conn, analiza=a, filteri=OSTRI_FILTERI)
        prazno = S.stanje_api(conn)                      # bez analitike uopšte

        assert bez["predlog"] == sa["predlog"] == prazno["predlog"]
        assert bez["bazen"] == sa["bazen"] == prazno["bazen"]
        assert bez["ravnoca"] == sa["ravnoca"]
        # a tiket se od filtera menja — inače test ne bi ništa dokazivao
        assert sa["tiket"]["kombinacija"] != bez["tiket"]["kombinacija"], sa["tiket"]
        assert prazno["tiket"] is None                   # nema analitike → nema tiketa
        print(f"test_predlog_bez_filtera: OK (predlog {bez['predlog']} isti uz "
              f"tiket {bez['tiket']['kombinacija']} → {sa['tiket']['kombinacija']})")
    finally:
        conn.close(); os.remove(putanja)


def test_tiket_iz_bazena():
    """Tiket je podskup bazena od SEKV_BAZEN i zadovoljava aktivne filtere (§7)."""
    istorija = sinteticka_istorija(220, seme=83)
    conn, putanja = nova_baza(istorija)
    try:
        S.rekonstruisi(conn)
        a = _analiza_iz(conn)
        st = S.stanje_api(conn, analiza=a, filteri=OSTRI_FILTERI)
        bazen, tiket = st["bazen"], st["tiket"]

        assert len(bazen) == konfig.SEKV_BAZEN == st["sekv_bazen"]
        assert set(st["predlog"]) <= set(bazen), "predlog mora biti podskup bazena"
        assert len(tiket["kombinacija"]) == K
        assert set(tiket["kombinacija"]) <= set(bazen), tiket
        assert tiket["bira"] == "generator" and tiket["ista_sansa"] is True
        assert st["predlog_izlaz"]["bira"] == "model"

        # aktivni filteri zaista važe za tiket
        o = tiket["osobine"]
        assert o["parni"] == OSTRI_FILTERI["parni"], o
        assert o["uzastopni"] == OSTRI_FILTERI["uzastopni"], o
        assert OSTRI_FILTERI["min_sv"] <= o["zbir"] / K <= OSTRI_FILTERI["max_sv"], o
        assert o == generator.osobine_kombinacije(tiket["kombinacija"])

        # bazen je zapisan uz svaki korak, pa se tiket računa i za staro kolo
        korak = S.korak_api(conn, istorija[150][0], analiza=a, filteri=OSTRI_FILTERI)
        assert len(korak["bazen"]) == konfig.SEKV_BAZEN
        assert set(korak["predlog"]) <= set(korak["bazen"])
        assert set(korak["tiket"]["kombinacija"]) <= set(korak["bazen"])

        # preuski filteri: nema kombinacije, ali ni greške
        prazan = S.stanje_api(conn, analiza=a, filteri={"parni": 7, "uzastopni": 6})
        assert prazan["tiket"]["kombinacija"] is None
        assert prazan["tiket"]["ukupno_validnih"] == 0
        print(f"test_tiket_iz_bazena: OK (bazen {len(bazen)}, tiket "
              f"{tiket['kombinacija']}, parnih {o['parni']}, zbir {o['zbir']})")
    finally:
        conn.close(); os.remove(putanja)


def test_bazen_sadrzi_predlog_kroz_istoriju():
    """Kroz ceo prolaz: predlog je prefiks istog poretka, dakle uvek u bazenu.

    Bazen i predlog koriste isti tie-break sa istim semenom, pa je podskup zagarantovan
    konstrukcijom, a ne srećom. Bez toga bi „tiket iz bazena" mogao da ne sadrži nijedan
    broj koji je model zaista favorizovao.
    """
    istorija = sinteticka_istorija(400, seme=85)
    m = S.Mesavina()
    provereno = 0
    for i in range(len(istorija)):
        if i < S.MIN_START:
            m.posmatraj(istorija[i][1])
            continue
        prozor = S._prozor_pre(istorija, i, S.PERIOD)
        p, po_ekspertu, predlog = m.predvidi(prozor)
        bazen = S.bazen_iz(p, konfig.SEKV_BAZEN, S.seme_izbora(prozor))
        assert set(predlog) <= set(bazen), (istorija[i][0], predlog, bazen)
        assert len(bazen) == konfig.SEKV_BAZEN
        provereno += 1
        m.uci(p, po_ekspertu, {int(b) for b in istorija[i][1]})
    print(f"test_bazen_sadrzi_predlog_kroz_istoriju: OK ({provereno} koraka)")


# ----------------------------------------------------------------------------
# Objašnjenje klase i ravnoća kroz vreme (PLAN_KORAK_IZBORA §4.3, Faza 4)
# ----------------------------------------------------------------------------

def test_klase_pokrivaju_sve_kombinacije():
    """Veličine klasa su tačne: zbir po parnosti i po uzastopnim daje tačno C(39,7).

    Bez ove provere bi tekst „takve kombinacije su ređe kao klasa" tvrdio broj koji
    nikad nije proveren. Zbir preko svih klasa mora da bude ceo prostor, inače je
    formula pogrešna.
    """
    ukupno = razlicitost_teorija.UKUPNO_KOMBINACIJA
    assert ukupno == 15380937
    assert sum(razlicitost_teorija.broj_sa_parnih(i) for i in range(K + 1)) == ukupno
    assert sum(razlicitost_teorija.broj_sa_uzastopnih(i) for i in range(K)) == ukupno
    # van opsega nema kombinacija, a ne izuzetak
    assert razlicitost_teorija.broj_sa_parnih(-1) == 0
    assert razlicitost_teorija.broj_sa_uzastopnih(K) == 0

    # provera grubom silom na manjem prostoru: iste formule za N=12, K=4
    from itertools import combinations
    from math import comb as _comb
    n, k = 12, 4
    po_parnosti, po_uzastopnim = {}, {}
    for komb in combinations(range(1, n + 1), k):
        pa = sum(1 for b in komb if b % 2 == 0)
        uz = sum(1 for i in range(k - 1) if komb[i + 1] == komb[i] + 1)
        po_parnosti[pa] = po_parnosti.get(pa, 0) + 1
        po_uzastopnim[uz] = po_uzastopnim.get(uz, 0) + 1
    for pa, broj in po_parnosti.items():
        assert _comb(n // 2, pa) * _comb(n - n // 2, k - pa) == broj, pa
    for uz, broj in po_uzastopnim.items():
        r = k - uz
        assert _comb(n - k + 1, r) * _comb(k - 1, r - 1) == broj, uz
    print(f"test_klase_pokrivaju_sve_kombinacije: OK (obe klase sumiraju na {ukupno}, "
          f"formule proverene grubom silom na C(12,4))")


def test_klasa_uz_oba_izlaza():
    """Uz predlog i uz tiket ide veličina njihove klase, izvedena iz njihovih osobina."""
    istorija = sinteticka_istorija(220, seme=87)
    conn, putanja = nova_baza(istorija)
    try:
        S.rekonstruisi(conn)
        st = S.stanje_api(conn, analiza=_analiza_iz(conn))
        for izlaz in (st["predlog_izlaz"], st["tiket"]):
            o, klasa = izlaz["osobine"], izlaz["klasa"]
            assert klasa["ukupno"] == razlicitost_teorija.UKUPNO_KOMBINACIJA
            assert klasa["parnost_broj"] == razlicitost_teorija.broj_sa_parnih(o["parni"])
            assert klasa["uzastopni_broj"] == razlicitost_teorija.broj_sa_uzastopnih(o["uzastopni"])
            assert 0 < klasa["parnost_udeo"] <= 1 and 0 < klasa["uzastopni_udeo"] <= 1
        print(f"test_klasa_uz_oba_izlaza: OK (predlog {st['predlog']}, "
              f"{st['predlog_izlaz']['osobine']['parni']} parnih → "
              f"{100 * st['predlog_izlaz']['klasa']['parnost_udeo']:.2f}% svih kombinacija)")
    finally:
        conn.close(); os.remove(putanja)


def test_ravnoca_kroz_vreme():
    """Krivulja ravnoće i njen sažetak: raspodela je ravna kroz CELU istoriju."""
    istorija = sinteticka_istorija(300, seme=89)
    conn, putanja = nova_baza(istorija)
    try:
        S.rekonstruisi(conn)
        h = S.istorija_api(conn)
        n = h["n"]
        assert len(h["raspon_udeo"]) == len(h["zazor_udeo"]) == n
        assert all(v is not None for v in h["raspon_udeo"])
        assert h["prag_raspona"] == konfig.PRAG_RASPONA

        sz = h["ravnoca_sazetak"]
        assert sz["n"] == n and sz["prag_raspona"] == konfig.PRAG_RASPONA
        assert sz["medijana"] <= sz["najveci"]
        assert sz["preko_praga"] == sum(1 for v in h["raspon_udeo"] if v > konfig.PRAG_RASPONA)
        assert abs(sz["preko_praga_udeo"] - sz["preko_praga"] / n) < 1e-12
        # na sintetici prag po definiciji prelazi oko 5% koraka
        assert sz["preko_praga_udeo"] < 0.15, sz

        # baza od pre Faze 1: kolone su NULL → sažetak izostaje, ali bez izuzetka
        conn.execute("UPDATE sekv_stanje SET raspon_udeo=NULL, zazor_udeo=NULL")
        conn.commit()
        prazna = S.istorija_api(conn)
        assert prazna["ravnoca_sazetak"] is None
        assert all(v is None for v in prazna["raspon_udeo"])
        print(f"test_ravnoca_kroz_vreme: OK ({n} kola, medijana "
              f"{sz['medijana']:.4f}, preko praga {sz['preko_praga']} = "
              f"{100 * sz['preko_praga_udeo']:.1f}%)")
    finally:
        conn.close(); os.remove(putanja)


# ----------------------------------------------------------------------------
# Poštena mešavina: ista oštrina, težina bez smrti
# ----------------------------------------------------------------------------

def test_ista_ostrina_za_sve():
    """Svaki ekspert govori istim tonom: odstupanje od 7/39 je tačno λ puta sirovo.

    Sirove raspodele šest omotanih eksperata već imaju identičan odnos najveće i
    najmanje verovatnoće (ocene su min-max normalizovane, pa softmaks daje tačno
    e^(1/τ)); λ zatim svima jednako smanjuje amplitudu. Time koliko brzo ekspert
    gubi težinu zavisi od toga ŠTA tvrdi, a ne od toga koliko glasno.
    """
    from math import e as E
    istorija = sinteticka_istorija(300, seme=101)
    prozor = istorija[150:250]

    skorovi = prediktori.skorovi_za_prozor(prozor)
    for komponenta, obrni in S._KOMPONENTA.values():
        s = skorovi[komponenta]
        sirovo = S.u_raspodelu({b: (1 - s[b]) if obrni else s[b] for b in range(1, MAX_BROJ + 1)})
        odnos = max(sirovo.values()) / min(sirovo.values())
        assert abs(odnos - E ** (1 / S.TEMPERATURA)) < 1e-9, (komponenta, odnos)

    # posle izjednačavanja je odstupanje svakog eksperta tačno λ puta manje
    for e, p in S.raspodele(prozor).items():
        assert abs(sum(p.values()) - K) < 1e-9, e
        assert max(p.values()) < 1 and min(p.values()) > 0, e
    sirovo = S.u_raspodelu({b: 1.0 if b == 7 else 0.0 for b in range(1, MAX_BROJ + 1)})
    posle = S.izjednaci_ostrinu(sirovo)
    for b in range(1, MAX_BROJ + 1):
        ocekivano = S.BASELINE + S.LAMBDA * (sirovo[b] - S.BASELINE)
        assert abs(posle[b] - ocekivano) < 1e-12, b
    print(f"test_ista_ostrina_za_sve: OK (sirovi odnos e^(1/τ) za svih 6, λ = {S.LAMBDA})")


def test_nijedan_ekspert_ne_umire():
    """Fixed-share drži pod na α/n; bez njega bi ekspert pao ka nuli i ostao tamo."""
    istorija = pristrasna_istorija(800, favorit=7, udeo=0.35, seme=5)
    n = len(S.EKSPERTI)
    pod = S.ALFA / n

    m, _ = S.prodji_do_kraja(istorija)
    assert min(m.tezine.values()) > pod, m.tezine
    assert all(w > 0 for w in m.tezine.values())
    assert abs(sum(m.tezine.values()) - 1.0) < 1e-12

    bez = S.Mesavina(alfa=0.0)
    list(S.prodji(istorija, mesavina=bez))
    assert min(bez.tezine.values()) < pod, "bez fixed-share ekspert nije ni pao — test ne meri ništa"
    print(f"test_nijedan_ekspert_ne_umire: OK (min {min(m.tezine.values()):.5f} > pod {pod:.6f}; "
          f"bez deljenja {min(bez.tezine.values()):.2e})")


def test_tezina_se_vraca():
    """Učenje nije jednosmerno: ekspert koji je potonuo mora da se digne kad prestane
    da greši. Faza 1 nosi frekvencijski signal, faza 2 je čist šum."""
    faza1 = pristrasna_istorija(800, favorit=7, udeo=0.35, seme=5)
    faza2 = [(2020801 + i, br) for i, (_k, br) in enumerate(sinteticka_istorija(800, seme=6))]

    m1 = S.Mesavina()
    list(S.prodji(faza1, mesavina=m1))
    m2 = S.Mesavina()
    list(S.prodji(faza1 + faza2, mesavina=m2))
    pre, posle = m1.tezine, m2.tezine

    assert pre["cold"] < pre["hot"], pre          # signal je u fazi 1 protiv hladnih
    assert posle["cold"] > pre["cold"], (pre["cold"], posle["cold"])
    assert posle["hot"] < pre["hot"], (pre["hot"], posle["hot"])
    print(f"test_tezina_se_vraca: OK (cold {pre['cold']:.4f} → {posle['cold']:.4f}, "
          f"hot {pre['hot']:.4f} → {posle['hot']:.4f})")


def test_pomak_zbira_ne_odlucuje_sam():
    """Monotoni ekspert ne sme sam da određuje predlog.

    Kad su ostali eksperti bili mrtvi, jedini koji je brojeve uređivao po veličini
    bio je `pomak_zbira`, pa je predlog ispadao kao niz susednih brojeva sa jednog
    kraja opsega. Sa fiksnim podom težine ostali eksperti ostaju u igri i nose
    najveći deo raspona mešavine.
    """
    istorija = sinteticka_istorija(600, seme=103)
    m, _ = S.prodji_do_kraja(istorija[:-1])
    p, po_ekspertu, predlog = m.predvidi(S._prozor_pre(istorija, len(istorija) - 1, S.PERIOD))
    w = m.tezine

    def doprinos(e):
        return w[e] * (max(po_ekspertu[e].values()) - min(po_ekspertu[e].values()))

    zbir = doprinos("pomak_zbira")
    najveci = max(doprinos(e) for e in S.EKSPERTI)
    assert zbir < 0.2 * najveci, (zbir, najveci)

    najnizi = tuple(range(1, K + 1))
    najvisi = tuple(range(MAX_BROJ - K + 1, MAX_BROJ + 1))
    assert predlog != najnizi and predlog != najvisi, predlog
    assert max(predlog) - min(predlog) > K, predlog   # nije niz susednih brojeva
    print(f"test_pomak_zbira_ne_odlucuje_sam: OK (predlog {list(predlog)}, "
          f"doprinos zbira {zbir:.6f} vs najveći {najveci:.6f})")


# ----------------------------------------------------------------------------
# Eksperti prelaza (§2.2, Faza 2)
# ----------------------------------------------------------------------------

def preklapajuca_istorija(broj_kola=2000, zadrzi=3, udeo=0.6, seme=13):
    """Sintetika sa signalom U PRELAZU, a ne u frekvenciji.

    U `udeo` slučajeva novo kolo zadržava tačno `zadrzi` brojeva iz prethodnog, a
    ostatak izvlači nasumično. Po simetriji svi brojevi ostaju jednako verovatni,
    pa frekvencijski eksperti tu nemaju šta da nađu — signal vide samo eksperti
    prelaza. Očekivano preklapanje: 0,6·3 + 0,4·1,256 ≈ 2,30.
    """
    rng = random.Random(seme)
    istorija = [(2020001, tuple(rng.sample(range(1, MAX_BROJ + 1), K)))]
    for i in range(1, broj_kola):
        prethodno = list(istorija[-1][1])
        if rng.random() < udeo:
            ostaju = rng.sample(prethodno, zadrzi)
            bazen = [b for b in range(1, MAX_BROJ + 1) if b not in ostaju]
            brojevi = tuple(ostaju + rng.sample(bazen, K - zadrzi))
        else:
            brojevi = tuple(rng.sample(range(1, MAX_BROJ + 1), K))
        istorija.append((2020001 + i, brojevi))
    return istorija


def test_prelazi_pocinju_na_teoriji():
    """Prazno stanje mora dati tačno uniformne raspodele — teorija je početna vrednost."""
    st = prelazi.Prelazi()
    assert abs(st.stopa_povratka() - prelazi.BASELINE) < 1e-12
    assert abs(st.stopa_zadrzavanja() - prelazi.BASELINE) < 1e-12
    assert abs(st.ocekivano_preklapanje(3) - prelazi.MU_PREKL) < 1e-12
    assert abs(st.ciljni_zbir(200) - prelazi.SREDINA_ZBIRA) < 1e-12
    for e, p in st.raspodele().items():
        assert all(abs(v - prelazi.BASELINE) < 1e-12 for v in p.values()), e
    # ni posle jednog kola nema razlike iz koje bi se učilo
    st.azuriraj((1, 2, 3, 4, 5, 6, 7))
    for e, p in st.raspodele().items():
        assert all(abs(v - prelazi.BASELINE) < 1e-12 for v in p.values()), e
    print(f"test_prelazi_pocinju_na_teoriji: OK ({len(prelazi.EKSPERTI)} eksperata prelaza)")


def test_prelazi_konvergiraju():
    """Na uniformnoj sintetici svaki ekspert prelaza konvergira ka svojoj teoriji.

    Tolerancija je 4 standardne greške za dati broj posmatranja (blago proširena
    jer brojevi unutar istog kola nisu nezavisni).
    """
    istorija = sinteticka_istorija(5000, seme=23)
    st = prelazi.Prelazi()
    for _kolo, brojevi in istorija:
        st.azuriraj(brojevi)
    d = st.dijagnostika()

    for e in ("povratak", "zadrzavanje"):
        n, procena, teorija = d[e]["n"], d[e]["procena"], d[e]["teorija"]
        dozvoljeno = 4 * (teorija * (1 - teorija) / n) ** 0.5
        assert abs(procena - teorija) < dozvoljeno, (e, procena, teorija, dozvoljeno)

    n = d["prelaz_prekl"]["n"]
    from webapp.core import razlicitost_teorija as T
    dozvoljeno = 4 * T.sigma_preklapanja() / n ** 0.5
    assert abs(d["prelaz_prekl"]["procena"] - d["prelaz_prekl"]["teorija"]) < dozvoljeno, d["prelaz_prekl"]

    for korpa in d["pomak_zbira"]["korpe"]:
        assert korpa["n"] > 0, korpa
        dozvoljeno = 4 * prelazi.SD_ZBIRA / korpa["n"] ** 0.5
        assert abs(korpa["procena"] - prelazi.SREDINA_ZBIRA) < dozvoljeno, korpa

    print(f"test_prelazi_konvergiraju: OK (povratak {d['povratak']['procena']:.4f}, "
          f"zadržavanje {d['zadrzavanje']['procena']:.4f}, teorija {prelazi.BASELINE:.4f}; "
          f"preklapanje {d['prelaz_prekl']['procena']:.3f} vs {prelazi.MU_PREKL:.3f})")


def test_prelazi_uce_signal_prelaza():
    """Kad je signal u prelazu a ne u frekvenciji, nalaze ga isključivo eksperti prelaza.

    Parnjak testa na pristrasnoj sintetici: dokazuje da ova četiri eksperta nisu
    ukras nego da vide ono što frekvencijski eksperti po konstrukciji ne mogu.
    """
    istorija = preklapajuca_istorija(2000, zadrzi=3, udeo=0.6)
    m, _koraci = S.prodji_do_kraja(istorija)
    s, d = m.stanje(), m.prelazi.dijagnostika()

    assert d["prelaz_prekl"]["procena"] > 2.0, d["prelaz_prekl"]
    assert d["zadrzavanje"]["procena"] > 0.28, d["zadrzavanje"]
    assert d["povratak"]["procena"] < prelazi.BASELINE, d["povratak"]

    assert s["k"] < s["pojas_donja"], s
    najveci = max(s["tezine"], key=lambda e: s["tezine"][e])
    assert najveci in prelazi.EKSPERTI, s["tezine"]
    assert s["k_eksperti"]["prelaz_prekl"] < 1.0, s["k_eksperti"]
    print(f"test_prelazi_uce_signal_prelaza: OK (preklapanje {d['prelaz_prekl']['procena']:.3f}, "
          f"K = {s['k']}, vrh = {najveci})")


def test_jedanaest_eksperata():
    """Mešavina mora imati tačno 11 eksperata: uniformni + 6 omotanih + 4 prelaza."""
    assert len(S.EKSPERTI) == 11, list(S.EKSPERTI)
    assert S.UNIFORMNI in S.EKSPERTI
    assert set(prelazi.EKSPERTI) <= set(S.EKSPERTI)
    assert len(set(S.EKSPERTI)) == len(S.EKSPERTI)
    m = S.Mesavina()
    assert set(m.tezine) == set(S.EKSPERTI)
    assert abs(sum(m.tezine.values()) - 1.0) < 1e-12
    print(f"test_jedanaest_eksperata: OK ({', '.join(S.EKSPERTI)})")


# ----------------------------------------------------------------------------
# Registar i Sinteza (Faza 3)
# ----------------------------------------------------------------------------

def test_kes_daje_isto_sto_i_racun_od_nule():
    """Keširani prefiks mora dati identičan predlog kao račun od nule.

    Keš pamti sadržaj prefiksa upravo zato što dve različite istorije nose ISTA
    kola: drugi deo testa pušta obe kroz isti keš i traži da se predlozi razlikuju.
    """
    a = sinteticka_istorija(300, seme=51)
    b = a[:200] + sinteticka_istorija(300, seme=52)[200:]

    S.zaboravi_kes()
    postupno = [S.predlog_za(a[:n]) for n in (120, 180, 240, 300)]   # keš se produžava
    od_nule = []
    for n in (120, 180, 240, 300):
        S.zaboravi_kes()
        od_nule.append(S.predlog_za(a[:n]))
    assert postupno == od_nule, (postupno, od_nule)

    # ista kola, drugi brojevi: keš ne sme da ih pomeša
    pa, pb = S.predlog_za(a), S.predlog_za(b)
    S.zaboravi_kes()
    assert pb == S.predlog_za(b), "keš je vratio predlog druge istorije"
    assert pa != pb, "test ne meri ništa ako se istorije poklapaju"
    print(f"test_kes_daje_isto_sto_i_racun_od_nule: OK ({len(postupno)} tačaka)")


def test_k_sekv_u_registru():
    """`k_sekv` prolazi kroz retro-bektest kao svaki drugi kombinacijski prediktor."""
    from webapp.core.prediktori_komb import PREDIKTORI_KOMB
    assert "k_sekv" in PREDIKTORI_KOMB

    istorija = sinteticka_istorija(400, seme=61)
    conn, putanja = nova_baza(istorija)
    try:
        prognoza.retro_bektest(conn)
        redovi = baza.prognoze_lista(conn, izvor="retro", metod="k_sekv", limit=10000)
        assert len(redovi) == 400 - prognoza.MIN_START
        for r in redovi:
            komb = [int(x) for x in r["kombinacija"].split(",")]
            assert len(set(komb)) == K and komb == sorted(komb)
            assert r["preklapanje"] is not None

        # predlog za kolo N mora zavisiti samo od kola pre N — isti test kao za
        # ostale metode, samo što ovaj metod nosi stanje
        po_kolu = {r["kolo"]: r["kombinacija"] for r in redovi}
        for indeks in (60, 150, 399):
            kolo = istorija[indeks][0]
            S.zaboravi_kes()
            ocekivano = S.predlog_za(istorija[:indeks])
            assert po_kolu[kolo] == ",".join(map(str, ocekivano)), kolo

        # i mora se poklopiti sa predlogom koji je zapisala rekonstrukcija
        S.rekonstruisi(conn)
        for red in baza.sekv_lista(conn):
            assert po_kolu[red["kolo"]] == red["predlog"], red["kolo"]
        print(f"test_k_sekv_u_registru: OK ({len(redovi)} kola, isti predlog u retru i u sekv_stanje)")
    finally:
        conn.close(); os.remove(putanja)


def test_sinteza_ima_oba_reda():
    """Sinteza mora prikazati `k_sekv` i red koeficijenta, oba u istoj korekciji."""
    conn, putanja = nova_baza(sinteticka_istorija(400, seme=71))
    try:
        prognoza.retro_bektest(conn)
        S.rekonstruisi(conn)
        rezime = sinteza.sakupi(conn, "retro")
        komb = {r["metod"]: r for r in rezime["redovi"]["kombinacija"]}
        testovi = {r["metod"]: r for r in rezime["redovi"]["test"]}
        assert "k_sekv" in komb, list(komb)
        assert komb["k_sekv"]["n"] == 400 - prognoza.MIN_START
        assert komb["k_sekv"]["p"] is not None

        k_red = testovi["sekv_koeficijent"]
        assert k_red["n"] > 0 and k_red["rezultat"] is not None
        assert k_red["p_korig"] == min(1.0, k_red["p"] * rezime["broj_redova"])
        assert k_red["zakljucak"] == sinteza.SLUCAJNOST

        detalj = sinteza.detalj_metoda(conn, "sekv_koeficijent", "retro")
        assert len(detalj["serija"]) == k_red["n"]
        assert detalj["baseline"] == 1.0
        print(f"test_sinteza_ima_oba_reda: OK ({rezime['broj_redova']} redova, "
              f"K = {k_red['rezultat']}, p_kor = {k_red['p_korig_prikaz']})")
    finally:
        conn.close(); os.remove(putanja)


def test_sinteza_bez_rekonstrukcije():
    """Bez rekonstruisanog stanja red koeficijenta postoji, ali kaže da nema podataka."""
    conn, putanja = nova_baza(sinteticka_istorija(120, seme=73))
    try:
        rezime = sinteza.sakupi(conn, "retro")
        red = {r["metod"]: r for r in rezime["redovi"]["test"]}["sekv_koeficijent"]
        assert red["n"] == 0 and red["p"] is None
        assert red["zakljucak"] == sinteza.BEZ_PODATAKA
        assert red["napomena"]
        assert sinteza.detalj_metoda(conn, "sekv_koeficijent", "retro") is None
        print("test_sinteza_bez_rekonstrukcije: OK (red postoji, bez p-vrednosti)")
    finally:
        conn.close(); os.remove(putanja)


# ----------------------------------------------------------------------------
# Trajno stanje i inkrementalni korak (Faza 4)
# ----------------------------------------------------------------------------

def test_serijalizacija_stanja():
    """Stanje kroz JSON mora biti identično stanju u memoriji — do poslednjeg bita.

    Bez toga bi inkrementalni korak posle restarta servera tiho krenuo od drugog
    broja, a K bi zavisio od toga kada je aplikacija podignuta.
    """
    istorija = sinteticka_istorija(160, seme=81)
    m, _ = S.prodji_do_kraja(istorija)
    kopija = S.Mesavina.iz_json(json.loads(json.dumps(m.u_json())))

    assert kopija.tezine == m.tezine
    assert kopija.gubitak == m.gubitak and kopija.gubitak_unif == m.gubitak_unif
    assert kopija.ocekivani_gubitak == m.ocekivani_gubitak
    assert kopija.varijansa == m.varijansa and kopija.n == m.n
    assert kopija.gubitak_eksperta == m.gubitak_eksperta
    assert kopija.prelazi.u_json() == m.prelazi.u_json()
    assert list(kopija.prelazi.poslednja) == list(m.prelazi.poslednja)
    assert kopija.stanje() == m.stanje()

    # i dalje mora da nastavi isto: jedan korak nad kopijom == korak nad originalom
    prosireno = sinteticka_istorija(161, seme=81)
    a = S._korak(m, prosireno, 160, S.PERIOD, S.MIN_START)
    b = S._korak(kopija, prosireno, 160, S.PERIOD, S.MIN_START)
    assert a == b, "kopija se razišla posle jednog koraka"
    print(f"test_serijalizacija_stanja: OK (K = {m.stanje()['k']})")


def test_inkrementalno_jednako_rekonstrukciji():
    """Dodavanje jednog kola korakom mora dati isto što i pun prolaz sa tim kolom."""
    istorija = sinteticka_istorija(200, seme=83)
    conn, putanja = nova_baza(istorija[:-1])
    try:
        S.rekonstruisi(conn)
        kolo, brojevi = istorija[-1]
        baza.dodaj_kolo(conn, kolo, "2020-01-05", list(brojevi))
        rez = S.azuriraj_posle_kola(conn, kolo)
        assert rez["nacin"] == "korak", rez
        korakom = _redovi_bez_vremena(conn)

        S.rekonstruisi(conn)
        punim = _redovi_bez_vremena(conn)
        assert korakom == punim, "inkrementalni korak se razlikuje od rekonstrukcije"
        assert len(korakom) == 200 - S.MIN_START
        print(f"test_inkrementalno_jednako_rekonstrukciji: OK ({len(korakom)} redova, K = {rez['k']})")
    finally:
        conn.close(); os.remove(putanja)


def test_izmenjena_proslost_vodi_na_rekonstrukciju():
    """Izmena starog kola mora poništiti sačuvano stanje — otisak zavisi od brojeva."""
    istorija = sinteticka_istorija(200, seme=85)
    conn, putanja = nova_baza(istorija)
    try:
        S.rekonstruisi(conn)
        assert S.ucitaj_model(conn, S.istorija_iz_conn(conn)) is not None

        red = conn.execute("SELECT id, kolo, datum FROM istorijski_rezultati ORDER BY id LIMIT 1 OFFSET 20").fetchone()
        baza.izmeni_kolo(conn, red["id"], red["kolo"], red["datum"], [1, 2, 3, 4, 5, 6, 7])
        assert S.ucitaj_model(conn, S.istorija_iz_conn(conn)) is None, "zastarelo stanje nije prepoznato"
        assert S.stanje_api(conn)["zastarelo"] is True

        novo = (istorija[-1][0] + 1, (8, 9, 10, 11, 12, 13, 14))
        baza.dodaj_kolo(conn, novo[0], "2020-02-02", list(novo[1]))
        rez = S.azuriraj_posle_kola(conn, novo[0])
        assert rez["nacin"] == "rekonstrukcija", rez
        assert S.stanje_api(conn)["zastarelo"] is False
        print("test_izmenjena_proslost_vodi_na_rekonstrukciju: OK (otisak uhvatio izmenu)")
    finally:
        conn.close(); os.remove(putanja)


def test_unos_kola_pomera_K():
    """Kriterijum završetka Faze 4: unos kola menja K bez ijedne ručne akcije.

    Ide kroz `bektest.dodaj_kolo_i_proveri` — istu funkciju koju zove POST /api/istorija.
    """
    from webapp.core import bektest
    istorija = sinteticka_istorija(220, seme=87)
    conn, putanja = nova_baza(istorija[:-1])
    try:
        S.rekonstruisi(conn)
        pre = S.stanje_api(conn)
        gubitak_pre = S.ucitaj_model(conn, S.istorija_iz_conn(conn)).gubitak
        kolo, brojevi = istorija[-1]
        rezime = bektest.dodaj_kolo_i_proveri(conn, kolo, "2020-02-03", list(brojevi))
        assert rezime["sekv"]["nacin"] == "korak", rezime["sekv"]

        posle = S.stanje_api(conn)
        assert posle["n"] == pre["n"] + 1
        # Kumulativni gubitak je pravo merilo da je model učio; prikazani K se na
        # dugoj istoriji pomeri tek u osmoj decimali, pa se na njega ne oslanjamo.
        assert S.ucitaj_model(conn, S.istorija_iz_conn(conn)).gubitak > gubitak_pre
        assert posle["zastarelo"] is False
        assert posle["ciljno_kolo"] == kolo + 1
        assert len(posle["predlog"]) == K
        assert posle["poslednji_korak"]["cilj"] == kolo
        print(f"test_unos_kola_pomera_K: OK (n {pre['n']} → {posle['n']}, K {pre['k']} → {posle['k']})")
    finally:
        conn.close(); os.remove(putanja)


def test_predlog_iz_baze_i_iz_registra_jednaki():
    """Predlog iz sačuvanog stanja mora biti isti kao predlog metoda `k_sekv`.

    Dva različita puta do istog broja: API čita stanje iz baze, registar produžava
    keširani prefiks. Ako se raziđu, korisnik bi na dva mesta video dve kombinacije.
    """
    from webapp.core.prediktori_komb import k_sekv
    istorija = sinteticka_istorija(180, seme=89)
    conn, putanja = nova_baza(istorija)
    try:
        S.rekonstruisi(conn)
        iz_baze = S.stanje_api(conn)["predlog"]
        S.zaboravi_kes()
        iz_registra = list(k_sekv(istorija, prognoza.RETRO_PERIOD, ciljno_kolo=istorija[-1][0] + 1))
        assert iz_baze == iz_registra, (iz_baze, iz_registra)
        print(f"test_predlog_iz_baze_i_iz_registra_jednaki: OK ({iz_baze})")
    finally:
        conn.close(); os.remove(putanja)


def test_pogledi_za_api():
    """Tri pogleda koja hrani UI: stanje, serija kroz vreme, korak u tački."""
    istorija = sinteticka_istorija(200, seme=95)
    conn, putanja = nova_baza(istorija)
    try:
        S.rekonstruisi(conn)
        st = S.stanje_api(conn)
        assert st["n"] == 200 - S.MIN_START and st["kola_u_bazi"] == 200
        assert set(st["eksperti"]) == set(S.EKSPERTI)
        assert set(st["poslednji_korak"]["gubitak_eksperta"]) == set(S.EKSPERTI)

        h = S.istorija_api(conn)
        assert h["n"] == st["n"]
        for e in S.EKSPERTI:
            assert len(h["tezine"][e]) == h["n"] and len(h["k_eksperti"][e]) == h["n"]
        assert len(h["pojas_donja"]) == h["n"] and len(h["ocekivano"]) == h["n"]

        granica = istorija[120][0]
        korak = S.korak_api(conn, granica)
        assert korak["cilj"] == istorija[121][0]
        assert korak["stvarni"] == sorted(istorija[121][1])
        assert korak["preklapanje"] == len(set(korak["predlog"]) & set(korak["stvarni"]))
        # težine koje su proizvele predlog su stanje PRE tog kola
        assert korak["tezine"] != korak["tezine_posle"]
        assert abs(sum(korak["tezine"].values()) - 1.0) < 1e-8

        # granica na poslednjem kolu: nema cilja, ali ni greške
        assert S.korak_api(conn, istorija[-1][0])["cilj"] is None
        print(f"test_pogledi_za_api: OK (stanje, {h['n']} tačaka serije, korak na {korak['cilj']})")
    finally:
        conn.close(); os.remove(putanja)


# ----------------------------------------------------------------------------
# Garancije motora
# ----------------------------------------------------------------------------

def _redovi_bez_vremena(conn):
    return [{k: v for k, v in r.items() if k != "kreirano"} for r in baza.sekv_lista(conn)]


def test_bez_curenja():
    """Stanje u kolu g zavisi isključivo od kola ≤ g — dokazano mutacijom budućnosti."""
    granica = 200
    a = sinteticka_istorija(300, seme=21)
    b = a[:granica] + [(kolo, brojevi) for kolo, brojevi in sinteticka_istorija(300, seme=22)[granica:]]
    assert a[granica:] != b[granica:], "mutacija nije promenila budućnost"

    conn_a, put_a = nova_baza(a)
    conn_b, put_b = nova_baza(b)
    try:
        S.rekonstruisi(conn_a)
        S.rekonstruisi(conn_b)
        ra = [r for r in _redovi_bez_vremena(conn_a) if r["redni"] < granica]
        rb = [r for r in _redovi_bez_vremena(conn_b) if r["redni"] < granica]
        assert ra and ra == rb, "budućnost je uticala na prošlost"
        # a posle granice se stanja moraju razići (inače test ništa ne meri)
        posle_a = [r for r in _redovi_bez_vremena(conn_a) if r["redni"] >= granica]
        posle_b = [r for r in _redovi_bez_vremena(conn_b) if r["redni"] >= granica]
        assert posle_a != posle_b
        print(f"test_bez_curenja: OK ({len(ra)} identičnih redova pre granice)")
    finally:
        conn_a.close(); os.remove(put_a)
        conn_b.close(); os.remove(put_b)


def test_determinizam():
    """Dve rekonstrukcije nad istom istorijom daju identične K_t i težine."""
    conn, putanja = nova_baza(sinteticka_istorija(300, seme=31))
    try:
        prva_rez = S.rekonstruisi(conn)
        prva = _redovi_bez_vremena(conn)
        druga_rez = S.rekonstruisi(conn)
        druga = _redovi_bez_vremena(conn)
        assert prva == druga, "rekonstrukcija nije deterministička"
        assert prva_rez["k"] == druga_rez["k"] and prva_rez["tezine"] == druga_rez["tezine"]
        assert len(prva) == 300 - S.MIN_START
        # težine u bazi su ispravan JSON i sabiraju se na 1
        tezine = json.loads(prva[-1]["tezine"])
        # u bazi su zaokružene na 9 decimala, pa zbir sme da odstupi za par ulp-ova
        assert abs(sum(tezine.values()) - 1.0) < 1e-8, tezine
        assert set(tezine) == set(S.EKSPERTI)
        print(f"test_determinizam: OK ({len(prva)} redova, K = {prva_rez['k']})")
    finally:
        conn.close(); os.remove(putanja)


def test_rekonstrukcija_brzina():
    """Ceo prolaz kroz ~1.400 kola mora stati u 20 s (Faza 1, kriterijum završetka)."""
    conn, putanja = nova_baza(sinteticka_istorija(1400, seme=41))
    try:
        rez = S.rekonstruisi(conn)
        assert rez["trajanje_s"] < 20.0, f"presporo: {rez['trajanje_s']}s"
        assert rez["kola"] == 1400 - S.MIN_START
        poslednje = baza.sekv_poslednje(conn)
        assert poslednje["k"] == rez["k"]
        assert all(r["k"] is not None for r in baza.sekv_lista(conn)), "K nije izračunat za svako kolo"
        print(f"test_rekonstrukcija_brzina: OK ({rez['trajanje_s']}s, {rez['kola']} kola, "
              f"K = {rez['k']}, {rez['zakljucak']})")
    finally:
        conn.close(); os.remove(putanja)


# ----------------------------------------------------------------------------
# Klizni K (§3.2 C) — „je li K ikad odstupio u NEKOM periodu"
# ----------------------------------------------------------------------------

def istorija_sa_pristrasnim_blokom(broj_kola=1250, favorit=7, od=650, do=850,
                                   udeo=0.30, seme=13):
    """Sintetika koja je čist šum SVUDA osim u jednom bloku od 200 kola.

    Granice bloka su namerno poravnate sa podelom na disjunktne blokove (prvih
    `SEKV_MIN_START` kola samo puni prozor, pa je red i = kolo i + 50). Signal je
    dovoljno jak da ga model u tom bloku nauči, a dovoljno kratak da se u zbiru
    preko 1.200 ocenjenih kola izgubi.
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


def test_klizni_prozor_preko_svega_jednak_kumulativnom():
    """Prozor koji obuhvata SVE korake mora dati tačno kumulativni K, E i σ.

    Kumulativne vrednosti se računaju sabiranjem u toku prolaza, a prozorske
    sabiranjem zapisanih članova po koraku. Ako se te dve staze raziđu, momenti po
    koraku nisu isti oni od kojih je napravljen kumulativni pojas.
    """
    conn, putanja = nova_baza(sinteticka_istorija(400, seme=55))
    try:
        S.rekonstruisi(conn)
        rez = S.rezime(conn)
        b = S.blokovi(conn, prozor=rez["n"])
        assert b["broj_blokova"] == 1, b
        ceo = b["blokovi"][0]
        assert abs(ceo["k"] - rez["k"]) < 1e-6, (ceo["k"], rez["k"])
        assert abs(ceo["ocekivano"] - rez["ocekivano"]) < 1e-6, (ceo["ocekivano"], rez["ocekivano"])
        assert abs(ceo["sigma"] - rez["sigma"]) < 1e-8, (ceo["sigma"], rez["sigma"])
        print(f"test_klizni_prozor_preko_svega_jednak_kumulativnom: OK "
              f"(K={ceo['k']}, kumulativni {rez['k']}, n={rez['n']})")
    finally:
        conn.close(); os.remove(putanja)


def test_klizni_na_sumu():
    """Na čistom šumu nijedan blok ne odstupa posle Šidákove korekcije."""
    conn, putanja = nova_baza(sinteticka_istorija(1250, seme=56))
    try:
        S.rekonstruisi(conn)
        kl = S.klizni(conn)
        b = S.blokovi(conn)
        assert kl["n_max"] > 0 and not kl["napomena"], kl["napomena"]
        assert len(kl["serija"]) == len(kl["pojas_donja"]) == len(kl["kola"])
        assert b["p_zajedno"] > 0.05, b["p_zajedno"]
        print(f"test_klizni_na_sumu: OK ({b['broj_blokova']} blokova, "
              f"p_zajedno={b['p_zajedno']:.4f}, klizna kriva {kl['n_max']} tacaka)")
    finally:
        conn.close(); os.remove(putanja)


def test_klizni_nalazi_blok_koji_kumulativni_ne_vidi():
    """Signal koji traje 200 kola: blokovi ga nalaze, kumulativni K ga ne vidi.

    Ovo je razlog postojanja reda: `sekv_koeficijent` meri celu istoriju odjednom i
    kratak period mu se utopi u proseku. Bez ovog testa „K ≈ 1 na pravim podacima"
    ne bi značilo da signala nema, nego samo da ga zbir ne bi ni video.
    """
    conn, putanja = nova_baza(istorija_sa_pristrasnim_blokom())
    try:
        S.rekonstruisi(conn)
        rez = S.rezime(conn)
        b = S.blokovi(conn)
        assert rez["p"] > 0.05, ("kumulativni K ne sme videti blok", rez["p"])
        assert b["p_zajedno"] < 0.01, ("blokovi moraju videti blok", b["p_zajedno"])
        naj = b["najekstremniji"]
        assert naj["z"] < 0, naj          # model je NAUCIO -> K pada ispod ocekivanog
        assert (naj["od"], naj["do"]) == (2010651, 2010850), naj
        print(f"test_klizni_nalazi_blok_koji_kumulativni_ne_vidi: OK "
              f"(kumulativni p={rez['p']:.3f}, blokovi p={b['p_zajedno']:.6f}, "
              f"blok {naj['od']}-{naj['do']} z={naj['z']})")
    finally:
        conn.close(); os.remove(putanja)


def test_sinteza_ima_red_kliznog():
    """Red kliznog K stoji u tabeli i njegov panel crta kliznu krivu."""
    conn, putanja = nova_baza(sinteticka_istorija(1250, seme=57))
    try:
        S.rekonstruisi(conn)
        redovi = {r["metod"]: r for r in sinteza.sakupi(conn, "retro")["redovi"]["test"]}
        red = redovi["sekv_klizni_k"]
        assert red["p"] is not None and red["jedinica"] == "K", red
        assert red["n"] % S.PROZOR_K == 0, red["n"]
        detalj = sinteza.detalj_metoda(conn, "sekv_klizni_k")
        assert detalj["tip"] == "koeficijent" and detalj["serija"], detalj
        assert len(detalj["serija"]) == len(detalj["pojas_gornja"])
        print(f"test_sinteza_ima_red_kliznog: OK (n={red['n']}, p={red['p']:.4f}, "
              f"panel {len(detalj['serija'])} tacaka)")
    finally:
        conn.close(); os.remove(putanja)


def test_klizni_bez_momenata():
    """Stari redovi bez momenata po koraku: red kaze da fali prolaz, ne puca."""
    conn, putanja = nova_baza(sinteticka_istorija(1250, seme=58))
    try:
        S.rekonstruisi(conn)
        conn.execute("UPDATE sekv_stanje SET ocekivano_korak=NULL, varijansa_korak=NULL")
        conn.commit()
        assert S.klizni(conn)["serija"] == []
        assert S.blokovi(conn)["blokovi"] == []
        redovi = {r["metod"]: r for r in sinteza.sakupi(conn, "retro")["redovi"]["test"]}
        red = redovi["sekv_klizni_k"]
        assert red["p"] is None and red["zakljucak"] == sinteza.BEZ_PODATAKA, red
        assert red["napomena"] == S.NEDOSTAJU_MOMENTI, red["napomena"]
        assert sinteza.detalj_metoda(conn, "sekv_klizni_k") is None
        print("test_klizni_bez_momenata: OK (red bez p-vrednosti, uz napomenu)")
    finally:
        conn.close(); os.remove(putanja)


def main():
    test_raspodela_suma_7()
    test_period_isti_kao_retro()
    test_uniformni_K_jednak_1()
    test_K_na_sintetici()
    test_K_na_pristrasnoj_sintetici()
    test_raspon_p_mix_na_sintetici()
    test_ravnoca_u_stanju()
    test_stari_redovi_bez_ravnoce()
    test_tiebreak_reproducibilan()
    test_tiebreak_bez_pristrasnosti()
    test_K_nepromenjen()
    test_predlog_bez_filtera()
    test_tiket_iz_bazena()
    test_bazen_sadrzi_predlog_kroz_istoriju()
    test_klase_pokrivaju_sve_kombinacije()
    test_klasa_uz_oba_izlaza()
    test_ravnoca_kroz_vreme()
    test_ista_ostrina_za_sve()
    test_nijedan_ekspert_ne_umire()
    test_tezina_se_vraca()
    test_pomak_zbira_ne_odlucuje_sam()
    test_prelazi_pocinju_na_teoriji()
    test_prelazi_konvergiraju()
    test_prelazi_uce_signal_prelaza()
    test_jedanaest_eksperata()
    test_kes_daje_isto_sto_i_racun_od_nule()
    test_k_sekv_u_registru()
    test_sinteza_ima_oba_reda()
    test_sinteza_bez_rekonstrukcije()
    test_serijalizacija_stanja()
    test_inkrementalno_jednako_rekonstrukciji()
    test_izmenjena_proslost_vodi_na_rekonstrukciju()
    test_unos_kola_pomera_K()
    test_predlog_iz_baze_i_iz_registra_jednaki()
    test_pogledi_za_api()
    test_bez_curenja()
    test_determinizam()
    test_rekonstrukcija_brzina()
    test_klizni_prozor_preko_svega_jednak_kumulativnom()
    test_klizni_na_sumu()
    test_klizni_nalazi_blok_koji_kumulativni_ne_vidi()
    test_sinteza_ima_red_kliznog()
    test_klizni_bez_momenata()
    print("\nSVI TESTOVI SEKVENCIJALNOG PREDIKTORA PROSLI [OK]")


if __name__ == "__main__":
    main()
