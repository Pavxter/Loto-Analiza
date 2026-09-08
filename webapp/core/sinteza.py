"""Strana „Sinteza": svi metodi i testovi u istoj tabeli, mereni istim alatom.

PLAN_SINTEZA.md. Modul ne uvodi ni nov način predviđanja ni novu evaluaciju:
`rezultat`, `z` i `p` dolaze iz `prognoza.statistika` / `prognoza.statistika_komb`
(isti kod koji hrani tab Prognoza) i iz testova u `razlicitost`. Sinteza samo:

  1. sakuplja sve redove u jedan oblik (`Eksperiment`),
  2. primenjuje JEDNU Bonferroni korekciju preko SVIH redova — uključujući
     kontrolu (`random`, `k_random`) i testove slučajnosti,
  3. iz korigovane p-vrednosti izvodi jednu od tri rečenice zaključka.

Kontrola nije fusnota nego red kao svaki drugi: ulazi u korekciju i prikazuje se
istim stilom. Ako kontrola „odstupi", to je očekivan lažno pozitivan nalaz i
zaključak to kaže eksplicitno.
"""

from dataclasses import asdict, dataclass, field

from . import konfig, prognoza, razlicitost, sekvencijalni
from .prediktori import PREDIKTORI
from .prediktori_komb import PREDIKTORI_KOMB

ALFA = 0.05

# Metodi koji su po konstrukciji čista slučajnost — referenca, ne kandidat.
KONTROLE = {"random", "k_random"}

# Ansambl je običan red u tabeli; izdvojen je samo zato što od njegovog ishoda
# zavisi koji se objašnjavajući tekst prikazuje (PLAN §5.4).
ANSAMBLI = {"ensemble", "k_ensemble"}

# Tri ishoda zaključka (PLAN §5.3) i jedna oznaka za red bez p-vrednosti.
SLUCAJNOST = "≈ slučajnost"
ODSTUPA = "odstupa — proveriti"
KONTROLA_ODSTUPA = "kontrola odstupa — lažno pozitivan"
BEZ_PODATAKA = "—"

TIPOVI = ("jedan_broj", "kombinacija", "test")


@dataclass
class Eksperiment:
    """Jedan red Sinteze — isti oblik za prediktor, kontrolu i test slučajnosti."""

    metod: str                      # id iz registra ili id testa
    naziv: str
    tip: str                        # "jedan_broj" | "kombinacija" | "test"
    n: int = 0                      # broj evaluiranih kola (ili opažanja u testu)
    rezultat: float | None = None   # pogodaka / prosečno preklapanje / statistika testa
    ocekivano: float | None = None  # iz teorije, nikad hardkodovano
    jedinica: str = ""              # "pogodaka" | "preklapanja" | "χ²" …
    z: float | None = None
    p: float | None = None          # nezaokružena p-vrednost iz postojeće evaluacije
    p_korig: float | None = None    # Bonferroni preko svih redova tabele
    zakljucak: str = BEZ_PODATAKA
    kontrola: bool = False
    granica_od: int | None = None
    granica_do: int | None = None
    opis: str = ""
    napomena: str = ""              # npr. „premalo podataka (n < 30)"
    detalj: dict = field(default_factory=dict)   # histogram i sl. za panel reda

    def kao_dict(self):
        d = asdict(self)
        d["p_prikaz"] = None if self.p is None else round(self.p, 5)
        d["p_korig_prikaz"] = None if self.p_korig is None else round(self.p_korig, 5)
        return d


# ----------------------------------------------------------------------------
# Zaključak: tačno tri ishoda (PLAN §5.3)
# ----------------------------------------------------------------------------

def zakljucak(p_korig, je_kontrola, alfa=ALFA):
    """Jedna od tri rečenice. Za p_korig=None pozivalac koristi BEZ_PODATAKA."""
    if p_korig is None:
        raise ValueError("zakljucak traži korigovanu p-vrednost; red bez nje nije ishod.")
    if p_korig >= alfa:
        return SLUCAJNOST
    return KONTROLA_ODSTUPA if je_kontrola else ODSTUPA


def _primeni_korekciju(redovi, alfa=ALFA):
    """Bonferroni preko SVIH redova: p_kor = min(1, p·N). Vraća broj značajnih.

    N je ukupan broj redova u tabeli, uključujući kontrolu i testove — jedan
    eksperiment, jedna korekcija (PLAN §2.4).
    """
    n_redova = len(redovi)
    znacajnih = 0
    for r in redovi:
        if r.p is None:
            r.p_korig = None
            r.zakljucak = BEZ_PODATAKA
            continue
        r.p_korig = min(1.0, r.p * n_redova)
        r.zakljucak = zakljucak(r.p_korig, r.kontrola, alfa)
        if r.p_korig < alfa:
            znacajnih += 1
    return znacajnih


# ----------------------------------------------------------------------------
# Sakupljanje redova iz postojećih evaluacija
# ----------------------------------------------------------------------------

def _opseg(conn, izvor, vrsta):
    """(prvo kolo, poslednje kolo) evaluiranih prognoza datog izvora i vrste."""
    kolona = "pogodak" if vrsta == "broj" else "preklapanje"
    r = conn.execute(
        f"SELECT MIN(kolo), MAX(kolo) FROM prognoze WHERE izvor=? AND vrsta=? "
        f"AND {kolona} IS NOT NULL", (izvor, vrsta)).fetchone()
    if not r or r[0] is None:
        return (None, None)
    return (int(r[0]), int(r[1]))


def redovi_jedan_broj(conn, izvor="retro"):
    """Jednobrojni prediktori — rezultat je broj pogodaka, očekivanje n·7/39."""
    stat = prognoza.statistika(conn, izvor)
    od, do = _opseg(conn, izvor, "broj")
    redovi = []
    for s in stat["metode"]:
        n = s["n"]
        redovi.append(Eksperiment(
            metod=s["metod"], naziv=s["naziv"], tip="jedan_broj", n=n,
            rezultat=s["k"], ocekivano=round(n * prognoza.BASELINE, 1) if n else None,
            jedinica="pogodaka", z=s["z"], p=s["p_tacno"],
            kontrola=s["metod"] in KONTROLE, granica_od=od, granica_do=do,
            opis=s["opis"],
            napomena="" if n else "nema evaluiranih kola",
            detalj={"uspesnost": s["uspesnost"], "ocekivani_udeo": s["ocekivano"]},
        ))
    return redovi


def redovi_kombinacija(conn, izvor="retro"):
    """Kombinacijski prediktori — rezultat je prosečno preklapanje, očekivanje μ."""
    stat = prognoza.statistika_komb(conn, izvor)
    od, do = _opseg(conn, izvor, "komb")
    redovi = []
    for s in stat["metode"]:
        n = s["n"]
        redovi.append(Eksperiment(
            metod=s["metod"], naziv=s["naziv"], tip="kombinacija", n=n,
            rezultat=s["prosek"], ocekivano=stat["ocekivano"],
            jedinica="preklapanja", z=s["z"], p=s["p_tacno"],
            kontrola=s["metod"] in KONTROLE, granica_od=od, granica_do=do,
            opis=s["opis"],
            napomena=("premalo podataka (n < 30)" if 0 < n < 30 else
                      ("" if n else "nema evaluiranih kola")),
            detalj={"maks": s["maks"], "maks_kolo": s["maks_kolo"], "sigma": stat["sigma"]},
        ))
    return redovi


# Testovi slučajnosti: id -> (naziv, ključ u analize_ranga, jedinica, opis).
# Svi mere samu istoriju, ne prognozu, pa ne zavise od izvora evaluacije.
TESTOVI = {
    "frekvencija_brojeva": (
        "Frekvencija brojeva", "frekvencija", "χ²",
        "Izlazi li svih 39 brojeva podjednako često (očekivano n·7/39 po broju)."),
    "rang_uniformnost": (
        "Rang — uniformnost", "uniformnost", "χ²",
        "Gomilaju li se izvučene kombinacije u nekom delu prostora rangova (50 korpi)."),
    "rang_rastojanja": (
        "Rang — rastojanja", "rastojanja", "χ²",
        "Rastojanje uzastopnih rangova protiv trougaone raspodele (prosek M/3)."),
    "rang_autokorelacija": (
        "Rang — autokorelacija", "autokorelacija", "Q",
        "Pamti li rang prethodnih pet kola (Ljung–Box preko pomaka 1–5)."),
    "najmanji_broj": (
        "Najmanji izvučeni broj", "najmanji_broj", "χ²",
        "P(min=k) = C(39−k,6)/C(39,7) — objašnjava zašto su rangovi mahom mali."),
}


def redovi_testovi(conn, period=0):
    """Testovi slučajnosti nad samom istorijom (PLAN §2.3).

    Uvek mere celu istoriju: `period` sužava prognozu, ne pitanje da li su
    izvlačenja slučajna. Rezultat svakog testa je statistika sa poznatim brojem
    stepeni slobode, pa je „očekivano" upravo df (srednja vrednost hi-kvadrata).
    """
    istorija = razlicitost.istorija_iz_conn(conn)
    if len(istorija) < 2:
        return []
    analize = razlicitost.analize_ranga(istorija)
    uzastopna = razlicitost.analiza_uzastopna(istorija, 0)["test"]

    redovi = []
    for metod, (naziv, kljuc, jedinica, opis) in TESTOVI.items():
        t = analize[kljuc]
        statistika = t.get("chi2", t.get("Q"))
        redovi.append(Eksperiment(
            metod=metod, naziv=naziv, tip="test", n=t.get("n", len(istorija)),
            rezultat=statistika, ocekivano=t["df"], jedinica=jedinica,
            p=t.get("p_tacno"), opis=opis,
            napomena="" if t.get("p_tacno") is not None else "premalo podataka za test",
            detalj=t,
        ))

    redovi.append(Eksperiment(
        metod="preklapanje_uzastopnih", naziv="Preklapanje uzastopnih kola",
        tip="test", n=uzastopna["n"], rezultat=uzastopna["chi2"],
        ocekivano=uzastopna["df"], jedinica="χ²", p=uzastopna.get("p_tacno"),
        opis="Koliko brojeva kolo deli sa prethodnim, protiv hipergeometrijske raspodele.",
        napomena="" if uzastopna.get("p_tacno") is not None else "premalo podataka za test",
        detalj=uzastopna,
    ))
    redovi.append(_red_koeficijenta(conn))
    return redovi


def _red_koeficijenta(conn):
    """Koeficijent nepredvidivosti kao red tipa `test` (PLAN_SEKVENCIJALNI §5.4).

    Pitanje reda je „odstupa li K od vrednosti koju bi model imao na slučajnim
    podacima". Plan ga naziva „K ≠ 1?"; centar je E[K | H₀], a ne tačno 1, jer
    mešavina koja hedžuje pod slučajnošću gubi nešto više od uniformnog modela
    (obrazloženje u sekvencijalni.moment_gubitka). Razlika je u petoj decimali i
    vidi se u koloni „očekivano".
    """
    opis = ("Odnos gubitka sekvencijalnog modela i uniformnog kroz celu istoriju. "
            "Ispod očekivanog: model je nešto naučio. Iznad: preučen je.")
    s = sekvencijalni.rezime(conn)
    if not s:
        return Eksperiment(
            metod="sekv_koeficijent", naziv="Koeficijent nepredvidivosti",
            tip="test", n=0, jedinica="K", opis=opis,
            napomena="sekvencijalno stanje nije rekonstruisano")
    return Eksperiment(
        metod="sekv_koeficijent", naziv="Koeficijent nepredvidivosti",
        tip="test", n=s["n"], rezultat=round(s["k"], 6),
        ocekivano=round(s["ocekivano"], 6), jedinica="K",
        z=round(s["z"], 4) if s["z"] is not None else None, p=s["p"],
        opis=opis, detalj=s)


# ----------------------------------------------------------------------------
# Detalj jednog reda (PLAN §5.2) — ništa se ne crta dvaput
# ----------------------------------------------------------------------------
# Prediktorski red pokazuje krivulju kroz vreme sa pojasom oko očekivanja; oba
# dolaze iz `prognoza.serije` / `prognoza.serije_komb`, istih serija koje crta tab
# Prognoza. Test pokazuje svoj histogram iz `razlicitost.analize_ranga`. Panel
# nigde ne računa novu statistiku — samo bira šta da prikaže i kuda vodi dalje.


def _histogram(oznake, posmatrano, ocekivano):
    return {"oznake": [str(o) for o in oznake],
            "posmatrano": [float(x) for x in posmatrano],
            "ocekivano": [round(float(x), 3) for x in ocekivano]}


def _detalj_testa(metod, analize, uzastopna):
    """Histogram testa u zajedničkom obliku (oznake / posmatrano / očekivano)."""
    if metod == "frekvencija_brojeva":
        t = analize["frekvencija"]
        ocek = t["n"] * konfig.BROJEVA_U_KOMBINACIJI / konfig.MAX_BROJ
        return _histogram(range(1, konfig.MAX_BROJ + 1), t["brojaci"],
                          [ocek] * konfig.MAX_BROJ), "Broj"
    if metod == "rang_uniformnost":
        t = analize["uniformnost"]
        n_korpi = len(t["brojaci"])
        return _histogram(range(1, n_korpi + 1), t["brojaci"],
                          [t["ocekivano_po_korpi"]] * n_korpi), "Korpa ranga"
    if metod == "rang_rastojanja":
        t = analize["rastojanja"]
        n_korpi = len(t["brojaci"])
        ocek = t["n"] / n_korpi if n_korpi else 0
        return _histogram(range(1, n_korpi + 1), t["brojaci"],
                          [ocek] * n_korpi), "Korpa rastojanja (jednako verovatne)"
    if metod == "rang_autokorelacija":
        t = analize["autokorelacija"]
        return _histogram(range(1, len(t["r"]) + 1), t["r"],
                          [0.0] * len(t["r"])), "Pomak"
    if metod == "najmanji_broj":
        t = analize["najmanji_broj"]
        return _histogram([c["oznaka"] for c in t["kategorije"]],
                          [c["posmatrano"] for c in t["kategorije"]],
                          [c["ocekivano"] for c in t["kategorije"]]), "Najmanji broj"
    if metod == "preklapanje_uzastopnih":
        return _histogram([c["oznaka"] for c in uzastopna["kategorije"]],
                          [c["posmatrano"] for c in uzastopna["kategorije"]],
                          [c["ocekivano"] for c in uzastopna["kategorije"]]), "Preklapanje"
    return None, ""


def detalj_metoda(conn, metod, izvor="retro"):
    """Podaci za panel jednog reda: krivulja ili histogram, plus kuda vodi dalje."""
    if metod in PREDIKTORI:
        serije = prognoza.serije(conn, izvor)
        return {"metod": metod, "naziv": PREDIKTORI[metod][0], "tip": "jedan_broj",
                "opis": PREDIKTORI[metod][2],
                "serija": serije["serije"].get(metod, []),
                "pojas_donja": serije["pojas_donja"], "pojas_gornja": serije["pojas_gornja"],
                "baseline": serije["baseline"], "jedinica": "% pogodaka",
                "vodi_na": "prognoza"}
    if metod in PREDIKTORI_KOMB:
        serije = prognoza.serije_komb(conn, izvor)
        return {"metod": metod, "naziv": PREDIKTORI_KOMB[metod][0], "tip": "kombinacija",
                "opis": PREDIKTORI_KOMB[metod][2],
                "serija": serije["serije"].get(metod, []),
                "pojas_donja": serije["pojas_donja"], "pojas_gornja": serije["pojas_gornja"],
                "baseline": serije["baseline"], "jedinica": "prosečno preklapanje",
                "vodi_na": "prognoza"}

    if metod == "sekv_koeficijent":
        s = sekvencijalni.serija(conn)
        if not s["serija"]:
            return None
        # Isti oblik kao krivulje prediktora, pa ga postojeći panel crta bez izmene:
        # serija + pojas + referentna linija. Tip nije „test" jer to nije histogram.
        return {"metod": metod, "naziv": "Koeficijent nepredvidivosti", "tip": "koeficijent",
                "opis": "K kroz vreme sa pojasom ±2σ oko vrednosti očekivane pod slučajnošću.",
                "serija": s["serija"], "pojas_donja": s["pojas_donja"],
                "pojas_gornja": s["pojas_gornja"], "baseline": s["baseline"],
                "jedinica": "K", "vodi_na": "prognoza"}

    istorija = razlicitost.istorija_iz_conn(conn)
    if len(istorija) < 2:
        return None
    analize = razlicitost.analize_ranga(istorija)
    uzastopna = razlicitost.analiza_uzastopna(istorija, 0)["test"]
    histogram, osa = _detalj_testa(metod, analize, uzastopna)
    if histogram is None:
        return None
    naziv = (TESTOVI[metod][0] if metod in TESTOVI else "Preklapanje uzastopnih kola")
    detalj = {"metod": metod, "naziv": naziv, "tip": "test", "histogram": histogram,
              "osa": osa, "vodi_na": "razlicitost"}
    if metod == "rang_autokorelacija":
        # Pojedinačni z po pomaku ostaju vidljivi iako je test jedan (Ljung–Box).
        detalj["z_po_pomaku"] = analize["autokorelacija"]["z"]
    return detalj


# ----------------------------------------------------------------------------
# Objedinjeni izlaz strane
# ----------------------------------------------------------------------------

def _globalna_recenica(n_redova, znacajnih, ocekivano_laznih):
    """Prva rečenica strane — izvedena isključivo iz brojanja ishoda."""
    if n_redova == 0:
        return "Nema nijednog eksperimenta — pokreni retro-bektest."
    if znacajnih == 0:
        koliko = "nijedan ne odstupa"
    elif znacajnih == 1:
        koliko = "1 odstupa"
    else:
        koliko = f"{znacajnih} odstupaju"
    laznih = f"{ocekivano_laznih:.2f}".replace(".", ",")   # decimalni zarez, kao u ostatku UI-ja
    return (f"Od {n_redova} metoda i testova, {koliko} od slučajnosti na 5% posle "
            f"Bonferroni korekcije. Očekivano lažno pozitivnih bez korekcije: {laznih}.")


def sakupi(conn, izvor="retro", period=0):
    """Svi redovi Sinteze + jedna Bonferroni korekcija + globalna rečenica.

    `izvor` bira skup evaluacija ("retro" = walk-forward bektest nad celom
    istorijom, "uzivo" = stvarno odigrane prognoze). Testovi slučajnosti ne
    zavise od izvora — oni mere samu istoriju.
    """
    redovi = (redovi_jedan_broj(conn, izvor)
              + redovi_kombinacija(conn, izvor)
              + redovi_testovi(conn, period))
    znacajnih = _primeni_korekciju(redovi)
    n_redova = len(redovi)
    ocekivano_laznih = round(n_redova * ALFA, 2)

    kontrole = [r for r in redovi if r.kontrola and r.p_korig is not None]
    ansambli = [r for r in redovi if r.metod in ANSAMBLI and r.p_korig is not None]
    return {
        "izvor": izvor,
        "broj_redova": n_redova,
        "znacajnih": znacajnih,
        "alfa": ALFA,
        "ocekivano_laznih": ocekivano_laznih,
        "globalno": _globalna_recenica(n_redova, znacajnih, ocekivano_laznih),
        "kontrola_odstupa": any(r.p_korig < ALFA for r in kontrole),
        # Ako ansambl ikad odstupi, tekst „zašto nije bolji" se ne prikazuje nego
        # se traži provera curenja — plan to izričito zahteva (§5.4).
        "ansambl_ima_red": bool(ansambli),
        "ansambl_odstupa": any(r.p_korig < ALFA for r in ansambli),
        "redovi": {t: [r.kao_dict() for r in redovi if r.tip == t] for t in TIPOVI},
        "n_jedan_broj": max((r.n for r in redovi if r.tip == "jedan_broj"), default=0),
        "n_kombinacija": max((r.n for r in redovi if r.tip == "kombinacija"), default=0),
        "baseline_udeo": round(100 * prognoza.BASELINE, 2),
        "ocekivano_preklapanje": round(prognoza.MU_PREKL, 4),
        "broj_metoda": len(PREDIKTORI) + len(PREDIKTORI_KOMB),
    }
