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

from . import prognoza
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


def redovi_testovi(conn, period=0):
    """Testovi slučajnosti nad samom istorijom (Faza 3: rang i najmanji broj)."""
    return []


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
