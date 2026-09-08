"""Sekvencijalni prediktor i koeficijent nepredvidivosti (PLAN_SEKVENCIJALNI_PREDIKTOR §2).

Model prolazi istoriju kolo po kolo. U koraku t vidi isključivo kola ≤ t, predlaže
raspodelu za kolo t+1, pa tek onda dobija t+1 i iz sopstvene greške pomera težine.
Isti anti-curenje obrazac kao retro-bektest (prognoza.py) — samo što ovde stanje
nije predlog nego mešavina raspodela.

Tri fiksne odluke iz plana:
  §2.1  izlaz eksperta je raspodela p ∈ ℝ³⁹, p ≥ 0, Σp = 7 (očekivan broj pogodaka);
        postojeći prediktori se OMOTAVAJU (softmax nad njihovim ocenama), ne menjaju;
  §2.3  mešanje su eksponencijalne težine (Hedge) nad Bernulijevim log-gubitkom;
        uniformni ekspert je uvek u mešavini — ugrađena bezbednost protiv preučavanja;
  §2.5  K = L_mešavine / L_uniformnog (kumulativno). K = 1 → model ne zna više od
        slučajnosti, K < 1 → izvlači informaciju, K > 1 → preučen. Pojas ±2σ je
        centriran na E[K | H₀] umesto na 1,00 — razlog i dokaz u moment_gubitka.

Šest omotanih eksperata su ISTE ocene koje već koristi ansambl
(prediktori.skorovi_komponenti) — nema druge implementacije istih veličina.
Kao i kod ansambla (vidi komentar u prediktori.py), „pozicija" iz plana nije deo
interfejsa prediktora, pa je zamenjena „svežinom" — osom koju testiraju fresh i cold.
"""

import hashlib
import json
import random
import threading
import time
from datetime import datetime
from math import exp, log, sqrt

from scipy.stats import norm

from . import baza, konfig, prelazi
from . import prediktori as P
from . import razlicitost_teorija as T

MAX_BROJ = konfig.MAX_BROJ
K_BROJEVA = konfig.BROJEVA_U_KOMBINACIJI
BASELINE = K_BROJEVA / MAX_BROJ            # 7/39 ≈ 0,1795 — uniformna verovatnoća po broju

ETA = konfig.ETA_HEDGE
TEMPERATURA = konfig.TEMPERATURA_SOFTMAX
ALFA = konfig.ALFA_DELJENJA
LAMBDA = konfig.LAMBDA_OSTRINE
FORMAT_STANJA = 2            # verzija serijalizacije; starije stanje se odbacuje
PERIOD = konfig.SEKV_PERIOD                # isti prozor kao retro-bektest
MIN_START = konfig.SEKV_MIN_START          # pre ovoga nema dovoljno istorije
PRAG_RASPONA = konfig.PRAG_RASPONA         # granica šuma za raspon p_mix (§2.4)

UNIFORMNI = "uniformni"
BROJEVI = tuple(range(1, MAX_BROJ + 1))

# Gubitak uniformnog eksperta je konstanta (ne zavisi od toga koje je kolo izvučeno):
# −7·ln(7/39) − 32·ln(32/39). To je imenilac koeficijenta K.
GUBITAK_UNIFORMNOG = (-K_BROJEVA * log(BASELINE)
                      - (MAX_BROJ - K_BROJEVA) * log(1 - BASELINE))

# Registar eksperata: id -> (naziv, opis). Redosled je fiksan i određuje redosled
# svuda dalje (težine, JSON, UI): uniformni, šest omotanih, četiri prelaza = 11.
EKSPERTI = {
    UNIFORMNI: ("Uniformni", "Svaki broj ima šansu 7/39. Referenca i osigurač: ako nijedan drugi ekspert ne radi, težina se sliva ovde."),
    "hot":     ("Vrući", "Raspodela iz frekvencije u periodu — što češći broj, veća verovatnoća."),
    "cold":    ("Hladni", "Obrnuta frekvencija: retki brojevi dobijaju veću verovatnoću (due hipoteza)."),
    "bayes":   ("Bajesovski", "Raspodela iz skorova Bajesovog modela (isti kao na strani Rangiranje)."),
    "hybrid":  ("Hibridni", "80% Bajes + 20% povezanost sa top-20 brojeva."),
    "rhythm":  ("Ritam", "Koliko broj kasni u odnosu na svoj prosečan razmak ponavljanja (D/R)."),
    "fresh":   ("Sveži", "Skoro izvučeni brojevi dobijaju veću verovatnoću (hipoteza da vrući ostaju vrući)."),
    **prelazi.EKSPERTI,   # povratak, zadrzavanje, prelaz_prekl, pomak_zbira (§2.2)
}

# Preslikavanje omotanih eksperata na komponente ansambla; True = obrnuta komponenta.
_KOMPONENTA = {
    "hot":    ("frekvencija", False),
    "cold":   ("frekvencija", True),
    "bayes":  ("bajes", False),
    "hybrid": ("hibrid", False),
    "rhythm": ("ritam", False),
    "fresh":  ("svezina", False),
}


# ----------------------------------------------------------------------------
# Raspodele (§2.1)
# ----------------------------------------------------------------------------

def uniformna():
    """p_i = 7/39 za svaki broj."""
    return {b: BASELINE for b in BROJEVI}


def u_raspodelu(skor, temperatura=TEMPERATURA):
    """Softmax nad ocenama, normalizovan na Σp = 7 (§2.1).

    Ocene su min-max normalizovane na [0, 1] (skorovi_komponenti), pa je odnos
    najveće i najmanje verovatnoće ograničen na e^(1/τ). Za τ = 1 najveće p ostaje
    ispod 0,47 — nikad ne dodiruje 1, pa je ln(1 − p) uvek konačan.
    """
    t = temperatura if temperatura and temperatura > 0 else 1.0
    najveci = max(skor.values())
    tezine = {b: exp((skor[b] - najveci) / t) for b in BROJEVI}   # −max radi stabilnosti
    ukupno = sum(tezine.values())
    return {b: K_BROJEVA * tezine[b] / ukupno for b in BROJEVI}


def izjednaci_ostrinu(p, lam=LAMBDA):
    """(1 − λ)·uniformna + λ·p — isti λ za SVAKOG eksperta.

    Bez ovoga koliko brzo ekspert gubi težinu zavisi od toga koliko je „glasan",
    a glasnoća omotanih eksperata dolazi iz temperature softmaksa, koja je
    proizvoljno izabrana. Sa istim λ za sve, najveće moguće odstupanje od 7/39 je
    isto za svakog, pa težine mere sadržaj tvrdnje, ne njen ton.

    Šta se time NE izjednačava: oblik raspodele. Šest omotanih eksperata već ima
    identičan odnos najveće i najmanje verovatnoće (ocene su min-max normalizovane
    pa softmaks daje tačno e), a razlikuju se po tome KOJI brojevi su gore — a to
    je sadržaj. Eksperti prelaza ovim postaju namerno tiši nego što bi im brojači
    dozvoljavali; to je cena iste mere za sve.
    """
    return {b: (1.0 - lam) * BASELINE + lam * p[b] for b in BROJEVI}


def raspodele(prozor, temperatura=TEMPERATURA, eksperti=None, stanje_prelaza=None):
    """Raspodela svakog eksperta nad datim prozorom kola (bez ciljnog kola).

    `eksperti` sužava skup (testovi mešaju samo uniformnog); podrazumevano su svi.
    Eksperti prelaza ne čitaju prozor nego online stanje: bez njega (`stanje_prelaza`
    = None) vraćaju svoje početne, teorijske — dakle uniformne — raspodele.
    """
    trazeni = tuple(eksperti) if eksperti else tuple(EKSPERTI)
    izlaz = {}
    if UNIFORMNI in trazeni:
        izlaz[UNIFORMNI] = uniformna()
    potrebne = [e for e in trazeni if e in _KOMPONENTA]
    if potrebne:
        skorovi = P.skorovi_za_prozor(prozor)   # deli račun sa ansamblom u istom koraku
        for e in potrebne:
            komponenta, obrni = _KOMPONENTA[e]
            s = skorovi[komponenta]
            izlaz[e] = u_raspodelu({b: (1.0 - s[b]) if obrni else s[b] for b in BROJEVI},
                                   temperatura)
    if any(e in prelazi.EKSPERTI for e in trazeni):
        stanje = stanje_prelaza if stanje_prelaza is not None else prelazi.Prelazi()
        izlaz.update(stanje.raspodele())
    # izjednačavanje oštrine ide na SVE, uključujući uniformnog (kod njega je no-op)
    return {e: izjednaci_ostrinu(izlaz[e]) for e in trazeni}


def seme_izbora(prozor):
    """Seme tie-breaka: broj poslednjeg kola koje je model video (PLAN_KORAK_IZBORA §2.2).

    ODSTUPANJE OD PLANA. Plan traži `seed = kolo`, dakle broj kola koje se predviđa.
    Taj broj se ne zna pre nego što je kolo izvučeno: numeracija je godina·1000 + broj,
    pa „poslednje + 1" pogađa pogrešno na svakoj granici godine, a u ovoj bazi i na
    preskočenom kolu 2025054 — ukupno 14 mesta. Panel bi tada pokazao predlog sa
    jednim semenom, a zapis posle unosa kola sa drugim.

    Poslednje VIĐENO kolo obe strane znaju tačno i isto: prozor za korak koji
    predviđa kolo na poziciji i završava se kolom i−1, a isti taj prozor dobija i
    `predlog_za` kad računa predlog za sledeće kolo. Svaki cilj i dalje ima svoje
    seme, jer svaki ima svog prethodnika, a nijedno seme ne zavisi od budućnosti.
    """
    return prozor[-1][0] if prozor else 0


def izaberi_top7(p, seme):
    """7 brojeva sa najvećim p; kod TAČNO jednakih verovatnoća bira slučaj, ne redosled.

    Staro pravilo je kod jednakih p uzimalo manji broj, što je kroz vreme sistematska
    pristrasnost ka niskim brojevima. Novo je pseudoslučajno, ali sa semenom iz
    `seme_izbora`, pa ostaje reproducibilno — retro-bektest i vremeplov moraju da daju
    isti predlog pri svakom pokretanju.

    Koliko ovo menja u praksi: ništa, i to je izmereno. Na 1.372 koraka prave baze
    nema nijedne tačne veze između dve verovatnoće (razlike su reda 10⁻⁴, ne 0), pa
    tie-break nikad ne odlučuje. Izmereni prosek izabranog broja je 21,2, ali isti
    prolaz na čistoj sintetici daje 18,5–21,2 zavisno od semena — dakle nagib dolazi
    od šuma, ne od ovog pravila. Pravilo je osigurač: kad do veze dođe, ne sme da je
    razreši redosled brojeva.

    `rnd.random()` se poziva tačno jednom po broju, u rastućem redosledu brojeva,
    pa je izlaz određen isključivo semenom.
    """
    rnd = random.Random(seme)
    rang = sorted(BROJEVI, key=lambda b: (-p[b], rnd.random()))
    return tuple(sorted(rang[:K_BROJEVA]))


def izracunaj_ravnocu(p):
    """Koliko je raspodela ravna — mere koje idu uz svaki predlog (PLAN_KORAK_IZBORA §2.3).

    `raspon` je razlika najveće i najmanje verovatnoće, `zazor` razlika 7. i 8.
    kandidata — koliko je izbor sedmorke uopšte bio izbor. Obe se izražavaju i kao
    udeo baseline-a 7/39, jer je apsolutna vrednost od 0,0004 nečitljiva sama za
    sebe, a 0,24% od baseline-a odmah kaže da razlike praktično nema.

    Merenje, ne ocena: prag iznad kog raspon prestaje da bude šum je PRAG_RASPONA
    u konfig.py, izveden iz sintetike (§2.4).
    """
    opadajuce = sorted((p[b] for b in BROJEVI), reverse=True)
    p_max, p_min = opadajuce[0], opadajuce[-1]
    raspon = p_max - p_min
    zazor = opadajuce[K_BROJEVA - 1] - opadajuce[K_BROJEVA]
    return {"p_min": p_min, "p_max": p_max,
            "raspon": raspon, "raspon_udeo": raspon / BASELINE,
            "zazor": zazor, "zazor_udeo": zazor / BASELINE}


def bez_preferencije(raspon_udeo, prag=PRAG_RASPONA):
    """True kad je raspon unutar onoga što daje čist šum — model nema preferenciju."""
    return raspon_udeo is None or raspon_udeo <= prag


# ----------------------------------------------------------------------------
# Gubitak i varijansa (§2.3, §2.5)
# ----------------------------------------------------------------------------

def _logsumexp(vrednosti):
    """Stabilan log(Σ exp(x)) — normalizacija težina u log-prostoru."""
    v = list(vrednosti)
    najveci = max(v)
    return najveci + log(sum(exp(x - najveci) for x in v))


def log_gubitak(p, dobitni):
    """Bernulijev log-gubitak po broju: −Σ_{i∈S} ln p_i − Σ_{i∉S} ln(1 − p_i)."""
    ukupno = 0.0
    for b in BROJEVI:
        ukupno -= log(p[b]) if b in dobitni else log(1.0 - p[b])
    return ukupno


def moment_gubitka(p):
    """(E[ℓ], Var[ℓ]) za jedno kolo pod uniformnom hipotezom — analitički, bez bootstrap-a.

    Uz a_i = ln(p_i / (1 − p_i)) važi ℓ = C − Σ_{i∈S} a_i, gde je C = −Σ ln(1 − p_i)
    konstanta datog koraka. Pod H₀ je S slučajan podskup od 7 brojeva bez vraćanja, pa je
    E[Σ_{i∈S} a_i] = 7·ā i Var(Σ_{i∈S} a_i) = 7·σ²·(N − 7)/(N − 1), sa populacionom
    varijansom σ² od a. Za uniformnu raspodelu su svi a_i jednaki → E[ℓ] je tačno
    gubitak uniformnog eksperta, a varijansa tačno 0.

    ODSTUPANJE OD PLANA (§2.5). Plan traži pojas ±2σ oko 1,00 i računa samo varijansu.
    Ali mešavina koja deli težinu na više eksperata pod H₀ u proseku gubi VIŠE od
    uniformnog (Gibsova nejednakost), pa je njeno očekivano K strogo veće od 1. Na
    čistom šumu to je oko 2,6σ iznad 1,00 — dakle van pojasa iz plana, i to trajno,
    jer i višak i σ opadaju kao 1/T. Pojas oko 1,00 bi zato lažno prijavljivao
    preučavanje na svakoj slučajnoj istoriji. Zato je pojas centriran na E[K | H₀],
    izvedeno iz ISTE raspodele gubitka koju plan i propisuje — samo se koristi i
    njena sredina, ne samo varijansa. Rastojanje E[K | H₀] od 1,00 ostaje vidljivo
    kao zaseban broj (`ocekivano`), pa se ništa ne gubi.
    """
    a = [log(p[b] / (1.0 - p[b])) for b in BROJEVI]
    c = -sum(log(1.0 - p[b]) for b in BROJEVI)
    sredina = sum(a) / len(a)
    var_pop = sum((x - sredina) ** 2 for x in a) / len(a)
    ocekivano = c - K_BROJEVA * sredina
    varijansa = K_BROJEVA * var_pop * (MAX_BROJ - K_BROJEVA) / (MAX_BROJ - 1)
    return ocekivano, varijansa


# ----------------------------------------------------------------------------
# Hedge mešavina (§2.3) + koeficijent (§2.5)
# ----------------------------------------------------------------------------

class Mesavina:
    """Stanje modela: težine eksperata i kumulativni gubici. Serijalizabilno.

    Ne drži istoriju kola — samo brojače. Zato je jedan korak jeftin i inkrementalni
    režim (Faza 4) po konstrukciji daje isto što i rekonstrukcija cele istorije.
    """

    def __init__(self, eta=ETA, temperatura=TEMPERATURA, eksperti=None, alfa=ALFA):
        self.eta = eta
        self.temperatura = temperatura
        self.alfa = alfa
        self.eksperti = tuple(eksperti) if eksperti else tuple(EKSPERTI)
        n = len(self.eksperti)
        # Težine se drže u log-prostoru: Hedge je tamo sabiranje, pa nema
        # potkoračenja ni na hiljadama kola.
        self.log_tezine = {e: -log(n) for e in self.eksperti}
        self.gubitak_eksperta = {e: 0.0 for e in self.eksperti}
        self.gubitak = 0.0            # kumulativni gubitak mešavine
        self.gubitak_unif = 0.0       # kumulativni gubitak uniformnog (imenilac K)
        self.ocekivani_gubitak = 0.0  # zbir E[ℓ | H₀] po koraku — centar pojasa
        self.varijansa = 0.0          # zbir Var[ℓ | H₀] po koraku — širina pojasa
        self.n = 0
        self.prelazi = prelazi.Prelazi()   # online brojači eksperata prelaza (§2.2)

    @property
    def tezine(self):
        """Težine u običnom prostoru; izvedene iz log-težina, ne čuvaju se posebno."""
        return {e: exp(self.log_tezine[e]) for e in self.eksperti}

    # --- predikcija ---

    def predvidi(self, prozor):
        """(p_mešavine, raspodele po ekspertu, predlog) za sledeće kolo."""
        po_ekspertu = raspodele(prozor, self.temperatura, self.eksperti, self.prelazi)
        w = self.tezine
        p = {b: sum(w[e] * po_ekspertu[e][b] for e in self.eksperti) for b in BROJEVI}
        # seme se izvodi iz samog prozora, pa nijedan pozivalac ne može da ga promaši
        return p, po_ekspertu, izaberi_top7(p, seme_izbora(prozor))

    def posmatraj(self, brojevi):
        """Kolo ulazi u brojače prelaza, ali se ne meri (zagrevanje pre min_start).

        Isto što i prozor radi za omotane eksperte: stanje sme da vidi rana kola,
        samo se na njima još ništa ne ocenjuje.
        """
        self.prelazi.azuriraj(brojevi)

    # --- učenje iz stvarnog kola ---

    def uci(self, p, po_ekspertu, dobitni):
        """Ažurira težine stvarnim kolom i vraća detalje koraka (šta je ko izgubio).

        Hedge: w ← w · exp(−η·ℓ), pa normalizacija. Od gubitaka se oduzima najmanji
        — normalizacija to poništava, ali čuva eksponent od prelivanja.
        """
        gubici = {e: log_gubitak(po_ekspertu[e], dobitni) for e in self.eksperti}
        gubitak_mesavine = log_gubitak(p, dobitni)

        ocekivano, varijansa = moment_gubitka(p)
        self.gubitak += gubitak_mesavine
        self.gubitak_unif += gubici[UNIFORMNI]
        self.ocekivani_gubitak += ocekivano
        self.varijansa += varijansa
        for e, g in gubici.items():
            self.gubitak_eksperta[e] += g
        self.n += 1

        self._pomeri_tezine(gubici)
        # tek sada kolo ulazi u brojače prelaza — nikad pre nego što je ocenjeno
        self.posmatraj(dobitni)
        return {"gubitak": gubitak_mesavine, "gubitak_eksperta": gubici}

    def _pomeri_tezine(self, gubici):
        """Hedge u log-prostoru, pa fixed-share korak.

        Hedge sam po sebi tera težinu izgubljenog eksperta ka nuli i tamo je
        ostavlja: učenje bi bilo jednosmerno, pa ekspert koji počne da pogađa ne bi
        mogao da se vrati. Fixed-share posle svakog kola vraća deo `alfa` ukupne
        težine ravnomerno svima, čime nastaje pod od alfa/n (za 11 eksperata i
        alfa = 0,01 to je 0,09%). Ispod tog poda niko ne pada, a ko ponovo počne da
        pogađa penje se odatle.
        """
        nove = {e: self.log_tezine[e] - self.eta * gubici[e] for e in self.eksperti}
        norma = _logsumexp(nove.values())
        nove = {e: v - norma for e, v in nove.items()}
        if self.alfa > 0:
            n = len(self.eksperti)
            linearne = {e: (1.0 - self.alfa) * exp(v) + self.alfa / n for e, v in nove.items()}
            ukupno = sum(linearne.values())
            nove = {e: log(w / ukupno) for e, w in linearne.items()}
        self.log_tezine = nove

    # --- koeficijent i pojas ---

    @property
    def koeficijent(self):
        """K = L_mešavine / L_uniformnog. Bez ijednog kola vraća 1,0 po definiciji."""
        if self.gubitak_unif <= 0:
            return 1.0
        return self.gubitak / self.gubitak_unif

    def koeficijent_eksperta(self, e):
        if self.gubitak_unif <= 0:
            return 1.0
        return self.gubitak_eksperta[e] / self.gubitak_unif

    @property
    def ocekivano(self):
        """E[K | H₀]: koliko bi K bio da su kola zaista slučajna — centar pojasa.

        Za mešavinu koja se svela na uniformnog eksperta je tačno 1,0; svako
        rasipanje težine ga podiže iznad 1 (vidi moment_gubitka).
        """
        if self.gubitak_unif <= 0:
            return 1.0
        return self.ocekivani_gubitak / self.gubitak_unif

    @property
    def sigma(self):
        """σ koeficijenta K pod H₀ (koraci su nezavisni → varijanse se sabiraju)."""
        if self.gubitak_unif <= 0:
            return 0.0
        return sqrt(self.varijansa) / self.gubitak_unif

    def pojas(self, k_sigma=2.0):
        """Pojas ±2σ oko E[K | H₀] (§2.5, uz odstupanje opisano u moment_gubitka)."""
        c, s = self.ocekivano, self.sigma
        return (c - k_sigma * s, c + k_sigma * s)

    def stanje(self):
        """Sažetak za upis i API."""
        d, g = self.pojas()
        return {
            "n": self.n,
            "k": round(self.koeficijent, 6),
            "ocekivano": round(self.ocekivano, 6),
            "pojas_donja": round(d, 6),
            "pojas_gornja": round(g, 6),
            "sigma": round(self.sigma, 8),
            "tezine": {e: round(self.tezine[e], 9) for e in self.eksperti},
            "k_eksperti": {e: round(self.koeficijent_eksperta(e), 6) for e in self.eksperti},
        }


    # --- serijalizacija (§3: stanje modela je serijalizabilno) ---

    def u_json(self):
        """Celo stanje kao obični tipovi. JSON čuva float-ove tačno (repr), pa je
        obilazak kroz bazu identičan držanju objekta u memoriji."""
        return {
            "verzija": FORMAT_STANJA,
            "eta": self.eta, "temperatura": self.temperatura, "alfa": self.alfa,
            "eksperti": list(self.eksperti), "log_tezine": dict(self.log_tezine),
            "gubitak_eksperta": dict(self.gubitak_eksperta),
            "gubitak": self.gubitak, "gubitak_unif": self.gubitak_unif,
            "ocekivani_gubitak": self.ocekivani_gubitak, "varijansa": self.varijansa,
            "n": self.n, "prelazi": self.prelazi.u_json(),
        }

    @classmethod
    def iz_json(cls, d):
        m = cls(eta=d["eta"], temperatura=d["temperatura"],
                eksperti=tuple(d["eksperti"]), alfa=d["alfa"])
        m.log_tezine = dict(d["log_tezine"])
        m.gubitak_eksperta = dict(d["gubitak_eksperta"])
        m.gubitak = d["gubitak"]
        m.gubitak_unif = d["gubitak_unif"]
        m.ocekivani_gubitak = d["ocekivani_gubitak"]
        m.varijansa = d["varijansa"]
        m.n = d["n"]
        m.prelazi = prelazi.Prelazi.iz_json(d["prelazi"])
        return m


def zakljucak(k, donja, gornja):
    """Tri ishoda, ista logika kao u Sintezi (§5.1)."""
    if k > gornja:
        return "Iznad pojasa — model je preučen (proveriti!)"
    if k < donja:
        return "Ispod pojasa — model izvlači informaciju (proveriti curenje!)"
    return "Unutar pojasa — nerazlučivo od slučajnosti"


# ----------------------------------------------------------------------------
# Prolaz kroz istoriju (§2.6) — jedini put do K_t
# ----------------------------------------------------------------------------

def _prozor_pre(istorija, i, period):
    """Kola koja model sme da vidi kad predviđa kolo na poziciji i (strogo pre njega)."""
    return istorija[max(0, i - period):i] if period else istorija[:i]


def _korak(m, istorija, i, period, min_start):
    """Obradi kolo na poziciji i: zagrevanje, ili predikcija pa učenje.

    Jedina implementacija jednog koraka. I prolaz kroz istoriju i inkrementalni keš
    idu kroz nju, pa ne mogu da se raziđu.
    """
    kolo, brojevi = istorija[i]
    if i < min_start:
        m.posmatraj(brojevi)
        return None
    p, po_ekspertu, predlog = m.predvidi(_prozor_pre(istorija, i, period))
    dobitni = {int(b) for b in brojevi}
    ravnoca = izracunaj_ravnocu(p)
    korak = m.uci(p, po_ekspertu, dobitni)
    stanje = m.stanje()
    return {"kolo": kolo, "redni": i, "predlog": predlog,
            "preklapanje": T.preklapanje_brojeva(predlog, dobitni),
            "ravnoca": ravnoca,
            "gubitak": korak["gubitak"],
            "gubitak_unif": korak["gubitak_eksperta"][UNIFORMNI],
            "gubitak_eksperta": korak["gubitak_eksperta"],
            **stanje}


def prodji(istorija, period=PERIOD, min_start=MIN_START, mesavina=None):
    """Generator koraka: predvidi za kolo N iz kola < N, pa uči iz kola N.

    Kola pre `min_start` samo pune prozor: nema ni predloga ni ažuriranja težina,
    pa K_t kreće od istog kola od kog kreće i učenje. Težine se nikad ne
    inicijalizuju iz cele istorije (§2.6, §8).
    """
    m = mesavina if mesavina is not None else Mesavina()
    for i in range(len(istorija)):
        zapis = _korak(m, istorija, i, period, min_start)
        if zapis is not None:
            yield zapis


def prodji_do_kraja(istorija, period=PERIOD, min_start=MIN_START):
    """(mešavina, lista koraka) — pogodno za testove i rekonstrukciju."""
    m = Mesavina()
    koraci = list(prodji(istorija, period, min_start, m))
    return m, koraci


# ----------------------------------------------------------------------------
# Predlog za sledeće kolo (registar prediktora, §5.4)
# ----------------------------------------------------------------------------
# Sekvencijalni model je jedini metod u registrima koji nosi stanje: predlog za
# kolo N zavisi od celog niza kola pre N. Retro-bektest ga zove ~1.400 puta sa
# rastućim prefiksom, pa bi svaki poziv od nule bio kvadratan. Zato se pamti
# poslednja mešavina i njen prefiks: ako novi poziv PRODUŽAVA taj prefiks, dodaju
# se samo kola koja nedostaju, kroz isti `_korak`. Ako ne produžava (druga
# istorija, izmenjeno kolo, kraći prefiks), sve se računa iznova.
#
# Keš pamti sadržaj prefiksa, ne njegov otisak: dve različite istorije nose ista
# kola, pa bi otisak (dužina + prvo i poslednje kolo) lažno pogađao.

_KES_BRAVA = threading.Lock()
_KES = {"kljuc": None, "duzina": 0, "sadrzaj": None, "mesavina": None}


def _nastavi_kes(istorija, period, min_start):
    """Mešavina koja je videla tačno `istorija`. Poziva se pod bravom."""
    kljuc = (period, min_start)
    duzina = _KES["duzina"]
    nastavlja = (_KES["mesavina"] is not None and _KES["kljuc"] == kljuc
                 and duzina <= len(istorija)
                 and _KES["sadrzaj"] == tuple(istorija[:duzina]))
    m = _KES["mesavina"] if nastavlja else Mesavina()
    if not nastavlja:
        duzina = 0
    for i in range(duzina, len(istorija)):
        _korak(m, istorija, i, period, min_start)
    _zapamti(istorija, period, min_start, m)
    return m


def zaboravi_kes():
    """Prazni keš prefiksa. Za testove i za slučaj izmene istorije u bazi."""
    with _KES_BRAVA:
        _KES.update(kljuc=None, duzina=0, sadrzaj=None, mesavina=None)


def _zapamti(istorija, period, min_start, mesavina):
    _KES.update(kljuc=(period, min_start), duzina=len(istorija),
                sadrzaj=tuple(istorija), mesavina=mesavina)


def predlog_za(istorija, period=PERIOD, min_start=MIN_START):
    """7 brojeva sa najvećim p_mix za prvo kolo POSLE date istorije.

    Vraća None dok istorija ne dosegne `min_start` — dotle model još ništa nije
    ocenio. Brava štiti deljeno stanje keša od paralelnih zahteva servera.
    """
    if len(istorija) < min_start:
        return None
    with _KES_BRAVA:
        m = _nastavi_kes(istorija, period, min_start)
        prozor = _prozor_pre(istorija, len(istorija), period)
        return m.predvidi(prozor)[2]


# ----------------------------------------------------------------------------
# Rekonstrukcija u bazu
# ----------------------------------------------------------------------------

def _red_za_upis(korak, sada):
    r = korak["ravnoca"]
    return (korak["kolo"], korak["redni"], ",".join(map(str, korak["predlog"])),
            korak["preklapanje"], korak["gubitak"], korak["gubitak_unif"],
            korak["k"], korak["ocekivano"], korak["pojas_donja"], korak["pojas_gornja"],
            korak["sigma"], json.dumps(korak["tezine"]), json.dumps(korak["k_eksperti"]),
            json.dumps({e: round(g, 6) for e, g in korak["gubitak_eksperta"].items()}),
            r["p_min"], r["p_max"], r["raspon_udeo"], r["zazor_udeo"], sada)


def istorija_iz_conn(conn):
    """Istorija kao lista (kolo, brojevi), hronološki — isti oblik kao u prognoza.py."""
    return [(int(r[0]), tuple(int(x) for x in r[1:8])) for r in conn.execute(
        "SELECT kolo, b1, b2, b3, b4, b5, b6, b7 FROM istorijski_rezultati ORDER BY id ASC")]


def otisak_istorije(istorija):
    """Sažetak sadržaja cele istorije — po njemu se prepoznaje zastarelo stanje.

    Zavisi od izvučenih brojeva, ne samo od numeracije kola: izmena starog kola
    menja otisak, pa se sačuvani model odbacuje i računa iznova.
    """
    h = hashlib.blake2b(digest_size=16)
    for kolo, brojevi in istorija:
        h.update(f"{kolo}:{','.join(map(str, brojevi))};".encode())
    return h.hexdigest()


def rekonstruisi(conn, period=PERIOD, min_start=MIN_START):
    """Prolazi celu istoriju od nule i upisuje K_t u tabelu sekv_stanje.

    Deterministički: ista ulazna kola daju identične težine i K_t (test_determinizam).
    """
    pocetak = time.time()
    istorija = istorija_iz_conn(conn)
    baza.sekv_obrisi(conn)
    baza.sekv_model_obrisi(conn)
    if len(istorija) <= min_start:
        return {"kola": 0, "trajanje_s": 0.0, "poruka": "Premalo istorije za sekvencijalni prolaz."}

    sada = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    m = Mesavina()
    redovi = [_red_za_upis(k, sada) for k in prodji(istorija, period, min_start, m)]
    baza.sekv_upisi(conn, redovi)
    _sacuvaj_model(conn, istorija, m)
    with _KES_BRAVA:      # gotovo stanje je tačno ono što `predlog_za` traži sledeće
        _zapamti(istorija, period, min_start, m)
    s = m.stanje()
    return {"kola": len(redovi), "trajanje_s": round(time.time() - pocetak, 2),
            "k": s["k"], "ocekivano": s["ocekivano"],
            "pojas_donja": s["pojas_donja"], "pojas_gornja": s["pojas_gornja"],
            "zakljucak": zakljucak(s["k"], s["pojas_donja"], s["pojas_gornja"]),
            "tezine": s["tezine"]}


# ----------------------------------------------------------------------------
# Čitanje zapisanog stanja (red u Sintezi, §5.4)
# ----------------------------------------------------------------------------

def rezime(conn):
    """Poslednje zapisano stanje + z i p za pitanje „odstupa li K od slučajnosti".

    Pod nultom hipotezom je K asimptotski normalan sa sredinom E[K | H₀] i
    devijacijom σ — obe veličine su izračunate analitički u toku prolaza i
    zapisane uz svako kolo, pa test ne uvodi nijednu novu pretpostavku.
    Vraća None ako stanje nije rekonstruisano.
    """
    red = baza.sekv_poslednje(conn)
    if not red:
        return None
    k, ocekivano, sigma = red["k"], red["ocekivano"], red["sigma"]
    z = (k - ocekivano) / sigma if sigma else None
    p = float(2 * norm.sf(abs(z))) if z is not None else None
    return {
        "kolo": red["kolo"], "n": baza.sekv_broj(conn),
        "k": k, "ocekivano": ocekivano, "sigma": sigma,
        "pojas_donja": red["pojas_donja"], "pojas_gornja": red["pojas_gornja"],
        "z": z, "p": p,
        "predlog": [int(x) for x in red["predlog"].split(",") if x.strip()],
        "tezine": json.loads(red["tezine"]), "k_eksperti": json.loads(red["k_eksperti"]),
        "ravnoca": _ravnoca_izlaz(red),
        "zakljucak": zakljucak(k, red["pojas_donja"], red["pojas_gornja"]),
    }


def serija(conn):
    """K_t kroz vreme sa pojasom — isti oblik kao serije na tabu Prognoza."""
    redovi = baza.sekv_lista(conn)
    return {
        "kola": [r["kolo"] for r in redovi],
        "serija": [r["k"] for r in redovi],
        "pojas_donja": [r["pojas_donja"] for r in redovi],
        "pojas_gornja": [r["pojas_gornja"] for r in redovi],
        "baseline": 1.0,
        "n_max": len(redovi),
    }


# ----------------------------------------------------------------------------
# Inkrementalni korak na unos kola (§6, Faza 4.4)
# ----------------------------------------------------------------------------

def _sacuvaj_model(conn, istorija, mesavina):
    baza.sekv_model_sacuvaj(conn, len(istorija), otisak_istorije(istorija),
                            json.dumps(mesavina.u_json()))


def ucitaj_model(conn, istorija):
    """Sačuvana mešavina ako tačno odgovara datoj istoriji, inače None."""
    red = baza.sekv_model_ucitaj(conn)
    if not red or red["duzina"] != len(istorija):
        return None
    if red["otisak"] != otisak_istorije(istorija):
        return None
    d = json.loads(red["stanje"])
    if d.get("verzija") != FORMAT_STANJA:
        return None      # stariji zapis: format se promenio, računa se iznova
    return Mesavina.iz_json(d)


def azuriraj_posle_kola(conn, kolo, period=PERIOD, min_start=MIN_START):
    """Hook za unos kola: jedan korak modela umesto ponovnog prolaza kroz istoriju.

    Radi korak samo ako sačuvano stanje odgovara istoriji BEZ tog kola — dakle ako
    je kolo prosto dopisano na kraj. U svakom drugom slučaju (izmenjena prošlost,
    prazno stanje, prvi put) radi punu rekonstrukciju, pa se zastarelost sama leči.
    Rezultat je po konstrukciji isti u oba slučaja (test_inkrementalno_jednako_rekonstrukciji).
    """
    istorija = istorija_iz_conn(conn)
    if len(istorija) <= min_start:
        return {"nacin": "preskoceno", "razlog": "premalo istorije"}
    i = len(istorija) - 1
    if istorija[i][0] != kolo:
        return rekonstruisi(conn, period, min_start) | {"nacin": "rekonstrukcija"}

    m = ucitaj_model(conn, istorija[:i])
    if m is None:
        return rekonstruisi(conn, period, min_start) | {"nacin": "rekonstrukcija"}

    zapis = _korak(m, istorija, i, period, min_start)
    if zapis is not None:
        baza.sekv_upisi(conn, [_red_za_upis(zapis, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))])
    _sacuvaj_model(conn, istorija, m)
    with _KES_BRAVA:
        _zapamti(istorija, period, min_start, m)
    s = m.stanje()
    return {"nacin": "korak", "kolo": kolo, "kola": s["n"], "k": s["k"],
            "ocekivano": s["ocekivano"], "pojas_donja": s["pojas_donja"],
            "pojas_gornja": s["pojas_gornja"],
            "zakljucak": zakljucak(s["k"], s["pojas_donja"], s["pojas_gornja"])}


# ----------------------------------------------------------------------------
# Pogledi za API (§4)
# ----------------------------------------------------------------------------

def _eksperti_opis():
    return {e: {"naziv": naziv, "opis": opis} for e, (naziv, opis) in EKSPERTI.items()}


def _ravnoca_izlaz(izvor):
    """Mere ravnoće za API, sa pragom i zaključkom (PLAN_KORAK_IZBORA §2.3, §2.4).

    `izvor` je ili dict iz `izracunaj_ravnocu` ili red iz `sekv_stanje` — u redovima
    upisanim pre ove izmene su kolone NULL, pa se vraća `None` i UI prikazuje „—"
    umesto da izmišlja broj. Prag putuje uz vrednost da ga UI ne bi duplirao.
    """
    raspon_udeo = izvor.get("raspon_udeo") if izvor else None
    if raspon_udeo is None:
        return None
    zazor_udeo = izvor.get("zazor_udeo")
    # apsolutne vrednosti se izvode ovde, da UI ne bi računao ni sa 7/39
    return {"p_min": izvor.get("p_min"), "p_max": izvor.get("p_max"),
            "raspon": raspon_udeo * BASELINE, "raspon_udeo": raspon_udeo,
            "zazor": None if zazor_udeo is None else zazor_udeo * BASELINE,
            "zazor_udeo": zazor_udeo,
            "baseline": BASELINE, "prag_raspona": PRAG_RASPONA,
            "bez_preferencije": bez_preferencije(raspon_udeo)}


def stanje_api(conn, period=PERIOD, min_start=MIN_START):
    """/api/sekv/stanje: težine, K, pojas, predlog za sledeće kolo, broj kola.

    `zastarelo` je True kad sačuvano stanje ne odgovara trenutnoj istoriji — tada
    se brojevi i dalje prikazuju, ali uz jasnu oznaku da traže rekonstrukciju.
    """
    istorija = istorija_iz_conn(conn)
    izlaz = {"n": baza.sekv_broj(conn), "kola_u_bazi": len(istorija),
             "eksperti": _eksperti_opis(), "zastarelo": True,
             "predlog": None, "ciljno_kolo": None, "ravnoca": None}
    s = rezime(conn)
    if s:
        izlaz.update({k: v for k, v in s.items() if k not in ("predlog", "ravnoca")})
        izlaz["predlog_poslednjeg"] = s["predlog"]
        izlaz["ravnoca_poslednjeg"] = s["ravnoca"]
    redovi = baza.sekv_lista(conn)
    if redovi:
        izlaz["poslednji_korak"] = _korak_iz_redova(conn, redovi, len(redovi) - 1)
    m = ucitaj_model(conn, istorija)
    if m is not None:
        izlaz["zastarelo"] = False
        p, _po_ekspertu, predlog = m.predvidi(_prozor_pre(istorija, len(istorija), period))
        izlaz["predlog"] = list(predlog)
        izlaz["ravnoca"] = _ravnoca_izlaz(izracunaj_ravnocu(p))
        izlaz["ciljno_kolo"] = istorija[-1][0] + 1
    return izlaz


def istorija_api(conn):
    """/api/sekv/istorija: K_t, pojas, K po ekspertu i težine kroz vreme."""
    redovi = baza.sekv_lista(conn)
    tezine = {e: [] for e in EKSPERTI}
    k_eksperti = {e: [] for e in EKSPERTI}
    for r in redovi:
        w, ke = json.loads(r["tezine"]), json.loads(r["k_eksperti"])
        for e in EKSPERTI:
            tezine[e].append(w.get(e))
            k_eksperti[e].append(ke.get(e))
    return {"kola": [r["kolo"] for r in redovi], "k": [r["k"] for r in redovi],
            "ocekivano": [r["ocekivano"] for r in redovi],
            "pojas_donja": [r["pojas_donja"] for r in redovi],
            "pojas_gornja": [r["pojas_gornja"] for r in redovi],
            "tezine": tezine, "k_eksperti": k_eksperti,
            "eksperti": _eksperti_opis(), "n": len(redovi)}


def _korak_iz_redova(conn, redovi, i):
    """Detalj jednog koraka: predlog, ishod, gubitak po ekspertu, težine pre i posle.

    Težine koje su PROIZVELE predlog su one iz prethodnog reda — red kola N nosi
    stanje posle učenja iz kola N. Za prvi red to su jednake početne težine.
    """
    red = redovi[i]
    pre = redovi[i - 1] if i > 0 else None
    tezine_pre = json.loads(pre["tezine"]) if pre else {e: 1.0 / len(EKSPERTI) for e in EKSPERTI}
    stvarni = conn.execute(
        "SELECT b1, b2, b3, b4, b5, b6, b7 FROM istorijski_rezultati WHERE kolo=?",
        (red["kolo"],)).fetchone()
    return {
        "cilj": red["kolo"], "redni": red["redni"],
        "predlog": [int(x) for x in red["predlog"].split(",") if x.strip()],
        "preklapanje": red["preklapanje"],
        "stvarni": sorted(int(x) for x in stvarni) if stvarni else None,
        "gubitak": round(red["gubitak"], 4), "gubitak_unif": round(red["gubitak_unif"], 4),
        "gubitak_eksperta": json.loads(red["gubitak_eksperta"] or "{}"),
        "k": red["k"], "ocekivano": red["ocekivano"],
        "ravnoca": _ravnoca_izlaz(red),
        "pojas_donja": red["pojas_donja"], "pojas_gornja": red["pojas_gornja"],
        "tezine": tezine_pre, "tezine_posle": json.loads(red["tezine"]),
        "k_eksperti": json.loads(red["k_eksperti"]),
        "zakljucak": zakljucak(red["k"], red["pojas_donja"], red["pojas_gornja"]),
    }


def korak_api(conn, granica):
    """/api/sekv/korak: stanje modela u tački — predlog koji je tada dao i ishod.

    Cilj je prvo kolo POSLE granice, isto pravilo kao `prognoza_u_tacki`.
    """
    redovi = baza.sekv_lista(conn)
    if not redovi:
        return None
    sledeci = [i for i, r in enumerate(redovi) if r["kolo"] > granica]
    if not sledeci:
        return {"granica": granica, "cilj": None, "poruka": "Nema kola posle granice."}
    return {"granica": granica, "eksperti": _eksperti_opis(),
            **_korak_iz_redova(conn, redovi, sledeci[0])}
