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

from webapp.core import (baza, konfig, prediktori, prelazi, prognoza, sinteza,  # noqa: E402
                         sekvencijalni as S)
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


def main():
    test_raspodela_suma_7()
    test_period_isti_kao_retro()
    test_uniformni_K_jednak_1()
    test_K_na_sintetici()
    test_K_na_pristrasnoj_sintetici()
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
    print("\nSVI TESTOVI SEKVENCIJALNOG PREDIKTORA PROSLI [OK]")


if __name__ == "__main__":
    main()
