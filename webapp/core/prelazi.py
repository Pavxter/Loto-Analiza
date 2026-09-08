"""Eksperti prelaza — ono što se uči iz razlike dva uzastopna kola (§2.2).

Četiri eksperta, nijedan model: sve su brojači koji se ažuriraju online.

    povratak       broj koji je upravo OTIŠAO — kolika mu je šansa u sledećem kolu
    zadrzavanje    broj koji je OSTAO dva kola zaredom — ostaje li i dalje
    prelaz_prekl   matrica prelaza preklapanja 8×8 → očekivano preklapanje sa
                   poslednjim kolom → pojačava ili slabi brojeve iz njega
    pomak_zbira    kuda ide zbir posle datog zbira → težina ka nižim ili višim brojevima

Svaki ima teorijsko očekivanje pod nezavisnim izvlačenjima i to očekivanje mu je
POČETNA vrednost: brojači kreću sa `SNAGA_PRIORA_PRELAZA` pseudo-posmatranja
raspoređenih po teoriji (7/39 za stope, hipergeometrijska za preklapanje, 140 za
zbir). Zato prazan model daje tačno uniformnu raspodelu i ne može da „nađe" signal
pre nego što ga vidi.

Anti-curenje: `azuriraj` se poziva tek POŠTO je kolo prošlo i broji isključivo
trojke kola koje su cele u prošlosti. Stanje nikad ne vidi kolo koje predviđa.
"""

from collections import deque
from math import exp, sqrt

from . import konfig
from . import razlicitost_teorija as T

MAX_BROJ = konfig.MAX_BROJ
K = konfig.BROJEVA_U_KOMBINACIJI
BROJEVI = tuple(range(1, MAX_BROJ + 1))
BASELINE = K / MAX_BROJ                      # 7/39 — teorijska stopa za povratak i zadržavanje
SNAGA = konfig.SNAGA_PRIORA_PRELAZA

MU_PREKL = T.ocekivano_preklapanje()         # 1,256 — teorijsko preklapanje dva kola
SREDINA_ZBIRA = K * (MAX_BROJ + 1) / 2       # 140 — očekivan zbir kola
# σ zbira pri izvlačenju bez vraćanja: n·σ²·(N−n)/(N−1), σ² = (N²−1)/12
SD_ZBIRA = sqrt(K * (MAX_BROJ ** 2 - 1) / 12 * (MAX_BROJ - K) / (MAX_BROJ - 1))
GRANICE_Z = (-1.0, -1.0 / 3, 1.0 / 3, 1.0)   # korpe zbira u jedinicama σ
BROJ_KORPI = len(GRANICE_Z) + 1

# Numerički zaštitnici: verovatnoća po broju ne sme da dodirne 0 ni 1, inače
# log-gubitak beži u beskonačno. Uz prior se procene nikad i ne približe ovome.
_Q_MIN, _Q_MAX = 0.01, 0.90
_NAGIB_MAX = 0.06                            # najveći nagib tiltovanja zbira

EKSPERTI = {
    "povratak":     ("Povratak", "Šansa da se broj koji je upravo otišao odmah vrati (teorija: 7/39)."),
    "zadrzavanje":  ("Zadržavanje", "Šansa da broj koji je ostao dva kola zaredom ostane i dalje (teorija: 7/39)."),
    "prelaz_prekl": ("Prelaz preklapanja", "Matrica prelaza preklapanja 8×8: koliko se brojeva iz poslednjeg kola očekuje ponovo (teorija: 1,256)."),
    "pomak_zbira":  ("Pomak zbira", "Kuda ide zbir kola posle datog zbira; pomera težinu ka nižim ili višim brojevima (teorija: 140)."),
}


# ----------------------------------------------------------------------------
# Raspodele iz jedne procene
# ----------------------------------------------------------------------------

def uniformna():
    return {b: BASELINE for b in BROJEVI}


def raspodela_skupa(skup, q):
    """p_i = q za brojeve iz skupa, ostatak mase ravnomerno na ostale; Σp = 7.

    Ovim oblikom rade tri od četiri eksperta: svaki izdvaja jedan skup brojeva
    (otišli / ostali / poslednje kolo) i tvrdi samo koliku šansu taj skup ima.
    """
    s = set(skup)
    if not s or len(s) >= MAX_BROJ:
        return uniformna()
    q = min(max(q, _Q_MIN), _Q_MAX)
    ostatak = min(max((K - q * len(s)) / (MAX_BROJ - len(s)), _Q_MIN), _Q_MAX)
    p = {b: (q if b in s else ostatak) for b in BROJEVI}
    ukupno = sum(p.values())          # posle odsecanja vrati zbir na tačno 7
    return {b: v * K / ukupno for b, v in p.items()}


def _ocekivan_zbir(nagib):
    """Očekivan zbir kola pod eksponencijalno tiltovanom raspodelom p ∝ e^(λ·b)."""
    sredina = (MAX_BROJ + 1) / 2
    w = [exp(nagib * (b - sredina)) for b in BROJEVI]
    ukupno = sum(w)
    return sum(b * K * x / ukupno for b, x in zip(BROJEVI, w))


_KES_NAGIB = {}
_KES_NAGIB_MAX = 4096


def _nagib_za_zbir(cilj):
    """λ takav da tiltovana raspodela ima očekivan zbir `cilj` (bisekcija, monotono).

    Van dosega λ ∈ [−0,06, 0,06] vraća granicu — time je i najveća verovatnoća po
    broju ograničena, pa raspodela ostaje daleko od 0 i 1.

    Cilj se pre traženja zaokružuje na tri decimale i pamti: procena zbira po korpi
    se između kola pomera za hiljaditi deo, pa bi se ista bisekcija ponavljala.
    Zaokruživanje je determinističko, a greška od 0,001 u zbiru je daleko ispod
    šuma same procene.
    """
    cilj = round(cilj, 3)
    postojece = _KES_NAGIB.get(cilj)
    if postojece is not None:
        return postojece
    nagib = _bisekcija_nagiba(cilj)
    if len(_KES_NAGIB) >= _KES_NAGIB_MAX:
        _KES_NAGIB.clear()
    _KES_NAGIB[cilj] = nagib
    return nagib


def _bisekcija_nagiba(cilj):
    lo, hi = -_NAGIB_MAX, _NAGIB_MAX
    if cilj <= _ocekivan_zbir(lo):
        return lo
    if cilj >= _ocekivan_zbir(hi):
        return hi
    for _ in range(40):
        sred = (lo + hi) / 2
        if _ocekivan_zbir(sred) < cilj:
            lo = sred
        else:
            hi = sred
    return (lo + hi) / 2


def raspodela_zbira(cilj):
    """Raspodela najveće entropije čiji je očekivan zbir jednak `cilj`; Σp = 7.

    Za cilj = 140 (teorijska vrednost) izlazi tačno uniformna raspodela.
    """
    nagib = _nagib_za_zbir(cilj)
    sredina = (MAX_BROJ + 1) / 2
    w = {b: exp(nagib * (b - sredina)) for b in BROJEVI}
    ukupno = sum(w.values())
    return {b: K * x / ukupno for b, x in w.items()}


def korpa_zbira(zbir):
    """Indeks korpe kojoj pripada dati zbir (granice u jedinicama σ oko 140)."""
    z = (zbir - SREDINA_ZBIRA) / SD_ZBIRA
    indeks = 0
    for granica in GRANICE_Z:
        if z >= granica:
            indeks += 1
    return indeks


# ----------------------------------------------------------------------------
# Online stanje
# ----------------------------------------------------------------------------

class Prelazi:
    """Brojači nad nizom razlika. Serijalizabilni; ne drže istoriju kola, samo tri poslednja."""

    def __init__(self):
        self.poslednja = deque(maxlen=3)                       # skupovi poslednjih kola
        self.povratak = [0.0, 0.0]                             # [vratilo se, otišlo ukupno]
        self.zadrzavanje = [0.0, 0.0]                          # [ostalo i dalje, ostalo ukupno]
        self.matrica = [[0.0] * (K + 1) for _ in range(K + 1)]  # prelaz preklapanja 8×8
        self.korpe = [[0.0, 0.0] for _ in range(BROJ_KORPI)]   # [n, zbir narednog kola]
        self.n = 0

    # --- procene (uvek sa priorom, pa su definisane i na praznom stanju) ---

    def stopa_povratka(self):
        return (self.povratak[0] + SNAGA * BASELINE) / (self.povratak[1] + SNAGA)

    def stopa_zadrzavanja(self):
        return (self.zadrzavanje[0] + SNAGA * BASELINE) / (self.zadrzavanje[1] + SNAGA)

    def ocekivano_preklapanje(self, k_prethodno):
        """E[preklapanje sledećeg kola sa poslednjim] iz reda matrice prelaza."""
        red = self.matrica[k_prethodno]
        tezine = [red[k] + SNAGA * T.hipergeom_pmf(k) for k in range(K + 1)]
        ukupno = sum(tezine)
        return sum(k * t for k, t in enumerate(tezine)) / ukupno

    def ciljni_zbir(self, zbir_poslednjeg):
        """E[zbir sledećeg kola] za korpu u kojoj je poslednji zbir.

        Prati se zbir NAREDNOG kola, a ne sam Δ: to je ista veličina (Δ = naredni −
        poslednji, a poslednji je poznat), ali joj je teorijska vrednost ista za sve
        korpe — 140 — pa je prior trivijalno tačan. Δ se prikazuje kao cilj minus
        poslednji zbir.
        """
        n, zbir = self.korpe[korpa_zbira(zbir_poslednjeg)]
        return (zbir + SNAGA * SREDINA_ZBIRA) / (n + SNAGA)

    # --- predikcija ---

    def raspodele(self):
        """Raspodela svakog od četiri eksperta iz trenutnog stanja.

        Dok se ne vide bar dva kola, nema razlike iz koje bi se učilo — sva četiri
        vraćaju uniformnu raspodelu.
        """
        if len(self.poslednja) < 2:
            return {e: uniformna() for e in EKSPERTI}
        prethodno, tekuce = self.poslednja[-2], self.poslednja[-1]
        otisli = prethodno - tekuce
        ostali = prethodno & tekuce
        return {
            "povratak": raspodela_skupa(otisli, self.stopa_povratka()),
            "zadrzavanje": raspodela_skupa(ostali, self.stopa_zadrzavanja()),
            "prelaz_prekl": raspodela_skupa(tekuce, self.ocekivano_preklapanje(len(ostali)) / K),
            "pomak_zbira": raspodela_zbira(self.ciljni_zbir(sum(tekuce))),
        }

    # --- učenje ---

    def azuriraj(self, brojevi):
        """Prima kolo koje je upravo prošlo i pomera brojače.

        Broji se samo ono što je celo u prošlosti: trojka (t−2, t−1, t) za stope i
        matricu, par (t−1, t) za zbir. Zato pozivalac sme da zove ovo posle svakog
        kola, i pre i posle početka merenja.
        """
        s = {int(b) for b in brojevi}
        if len(self.poslednja) >= 2:
            prethodno, tekuce = self.poslednja[-2], self.poslednja[-1]
            otisli = prethodno - tekuce
            if otisli:
                self.povratak[0] += len(otisli & s)
                self.povratak[1] += len(otisli)
            ostali = prethodno & tekuce
            if ostali:
                self.zadrzavanje[0] += len(ostali & s)
                self.zadrzavanje[1] += len(ostali)
            self.matrica[len(ostali)][len(tekuce & s)] += 1
        if self.poslednja:
            tekuce = self.poslednja[-1]
            korpa = self.korpe[korpa_zbira(sum(tekuce))]
            korpa[0] += 1
            korpa[1] += sum(s)
        self.poslednja.append(s)
        self.n += 1

    # --- serijalizacija (stanje preživljava restart servera) ---

    def u_json(self):
        """Ceo sadržaj brojača kao obični tipovi. Skupovi idu kao sortirane liste —
        sve dalje operacije nad njima su skupovne, pa redosled ništa ne nosi."""
        return {"poslednja": [sorted(s) for s in self.poslednja],
                "povratak": list(self.povratak), "zadrzavanje": list(self.zadrzavanje),
                "matrica": [list(red) for red in self.matrica],
                "korpe": [list(k) for k in self.korpe], "n": self.n}

    @classmethod
    def iz_json(cls, d):
        st = cls()
        st.poslednja = deque((set(s) for s in d["poslednja"]), maxlen=3)
        st.povratak = list(d["povratak"])
        st.zadrzavanje = list(d["zadrzavanje"])
        st.matrica = [list(red) for red in d["matrica"]]
        st.korpe = [list(k) for k in d["korpe"]]
        st.n = d["n"]
        return st

    # --- dijagnostika (test konvergencije i UI detalj) ---

    def dijagnostika(self):
        """Trenutna procena svakog eksperta naspram njegove teorijske vrednosti."""
        redovi = [sum(self.matrica[k]) for k in range(K + 1)]
        ukupno_prelaza = sum(redovi)
        prosek_prekl = (sum(k * self.matrica[i][k] for i in range(K + 1) for k in range(K + 1))
                        / ukupno_prelaza) if ukupno_prelaza else None
        korpe = []
        for i, (n, zbir) in enumerate(self.korpe):
            korpe.append({"korpa": i, "n": n,
                          "prosek": (zbir / n) if n else None,
                          "procena": (zbir + SNAGA * SREDINA_ZBIRA) / (n + SNAGA)})
        return {
            "n": self.n,
            "povratak": {"procena": self.stopa_povratka(), "teorija": BASELINE,
                         "n": self.povratak[1],
                         "sirovo": (self.povratak[0] / self.povratak[1]) if self.povratak[1] else None},
            "zadrzavanje": {"procena": self.stopa_zadrzavanja(), "teorija": BASELINE,
                            "n": self.zadrzavanje[1],
                            "sirovo": (self.zadrzavanje[0] / self.zadrzavanje[1]) if self.zadrzavanje[1] else None},
            "prelaz_prekl": {"procena": prosek_prekl, "teorija": MU_PREKL, "n": ukupno_prelaza,
                             "po_redu": [{"k": k, "n": redovi[k],
                                          "procena": self.ocekivano_preklapanje(k)}
                                         for k in range(K + 1)]},
            "pomak_zbira": {"teorija": SREDINA_ZBIRA, "korpe": korpe,
                            "n": sum(int(n) for n, _z in self.korpe)},
        }
