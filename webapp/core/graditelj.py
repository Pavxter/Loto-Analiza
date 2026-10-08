"""Graditelj kombinacije (PLAN_TIKETI_GRADITELJ §7).

Tri koraka na jednoj strani, svaki vidljiv:
  1. potencijal po broju u prozoru „obrta" → bazen najvećeg potencijala;
  2. signali za bazen na celoj istoriji: ritam (D/R) i kookurencija parova (lift);
  3. sve C(bazen, 7) kombinacije: težinski skor signala, pa pravila oblika isključuju.

Signali (potencijal, ritam, parovi) tvrde nešto o budućnosti i zato idu u bektest
(`k_graditelj`, faza 6). Pravila oblika (dekade, uzastopni, zbir, parnost, istorija)
ne menjaju verovatnoću — svaka kombinacija je jednako verovatna — pa samo isključuju,
ne boduju.

Postojeće funkcije se uvoze, ne kopiraju: prozor i frekvencije iz `prediktori`, min-max
iz `prediktori._normalizuj`, tie-break iz `sekvencijalni`, osobine iz `generator`,
sličnost sa istorijom iz `odigrano`. Brojači cele istorije (ritam, parovi) su ovde
inkrementalni, jer retro-bektest zove Graditelja za svako kolo; test proverava da
daju isto što i `prediktori._primitivi`.
"""

import math
from fractions import Fraction
from functools import lru_cache
from itertools import combinations

import numpy as np

from . import generator, konfig, odigrano, prediktori, sekvencijalni

N = konfig.MAX_BROJ
K = konfig.BROJEVA_U_KOMBINACIJI
P = K / N                                  # verovatnoća da je broj u jednom kolu
PAR = K * (K - 1) / (N * (N - 1))          # verovatnoća da je PAR u jednom kolu
PROZORI = (22, 50, 100, 500, 0)            # 0 = sva kola
BAZEN_OPSEG = (10, 20)                     # C(20,7) = 77.520 — još uvek trenutno
ALTERNATIVA = 3
MAX_ZAJEDNICKIH_ALT = 5                    # alternativa mora da se razlikuje u ≥2 broja


@lru_cache(maxsize=None)
def ocekivani_obrt(n=N, k=K):
    """Očekivan broj kola dok se svih n brojeva ne pojavi bar jednom (7 od 39 po kolu).

    E[T] = Σ_t P(T > t) = Σ_{j≥1} (−1)^{j+1} C(n,j) / (1 − q_j), gde je
    q_j = C(n−j,k)/C(n,k) verovatnoća da kolo promaši ceo zadati skup od j brojeva.
    Računato u razlomcima, jer naizmenični zbir velikih binomnih koeficijenata u
    float-u gubi preciznost. Za 7/39 daje 22,24.
    """
    ukupno = math.comb(n, k)
    e = Fraction(0)
    for j in range(1, n + 1):
        q = Fraction(math.comb(n - j, k), ukupno)
        e += (-1) ** (j + 1) * math.comb(n, j) / (1 - q)
    return float(e)


# ----------------------------------------------------------------------------
# 1. Potencijal
# ----------------------------------------------------------------------------

def potencijal(istorija, w):
    """z po broju u poslednjih `w` kola: (očekivano − stvarno) / σ. Pozitivno = kasni.

    Broj je u svakom kolu sa verovatnoćom 7/39, nezavisno od kola do kola, pa je broj
    pojava binomni: E = W·p, σ = √(W·p·(1−p)). z je opisna mera odstupanja od
    ravnomerne raspodele, ne verovatnoća za sledeće kolo.
    """
    prozor = prediktori._prozor(istorija, w)
    W = len(prozor)
    count, _ = prediktori._frekvencija_i_poslednji(prozor)
    E = W * P
    sigma = math.sqrt(W * P * (1 - P)) if W else 0.0
    z = {b: ((E - count[b]) / sigma if sigma else 0.0) for b in range(1, N + 1)}
    return {"w": W, "ocekivano": E, "sigma": sigma, "pojava": count, "z": z}


# ----------------------------------------------------------------------------
# 2. Signali bazena
# ----------------------------------------------------------------------------

class _Brojaci:
    """Brojači cele istorije koje signali traže: prva/poslednja pojava, broj pojava, parovi.

    Isti pojmovi kao `prediktori._primitivi` (test to proverava), ali se produžavaju za
    jedno kolo u O(21) — retro-bektest zove Graditelja za svako kolo sa istorijom dužom
    za tačno jedno, pa bi pun prolaz svaki put bio kvadratan.
    """

    def __init__(self):
        self.n = 0
        self.prva, self.poslednja = {}, {}
        self.pojava = dict.fromkeys(range(1, N + 1), 0)
        self.parovi = {}
        self.maske = []              # bitmaska svakog kola, za pravilo istorije
        self._H = None

    def dodaj(self, brojevi):
        i = self.n
        bs = sorted(set(brojevi))
        for b in bs:
            self.prva.setdefault(b, i)
            self.poslednja[b] = i
            self.pojava[b] += 1
        for x, y in combinations(bs, 2):
            self.parovi[(x, y)] = self.parovi.get((x, y), 0) + 1
        self.maske.append(sum(1 << (b - 1) for b in bs))
        self._H = None
        self.n += 1

    def H(self):
        """Maske svih kola kao uint64 niz (keširan dok se ne doda kolo)."""
        if self._H is None:
            self._H = np.array(self.maske, dtype=np.uint64)
        return self._H

    def kasnjenje(self, b):
        """Kola od poslednje pojave (poslednje kolo → 1); nikad izvučen → n + 1."""
        return self.n - self.poslednja[b] if b in self.poslednja else self.n + 1

    def ritam(self, b):
        """Prosečan razmak između pojava = (poslednja − prva) / (pojava − 1); < 2 pojave → None."""
        k = self.pojava[b]
        return (self.poslednja[b] - self.prva[b]) / (k - 1) if k >= 2 else None


_brojaci_kes = {"otisak": None, "duzina": 0, "brojaci": None}


def _otisak(istorija):
    return hash(tuple(istorija))


def brojaci(istorija):
    """Brojači za istoriju; produžava keširane kad je istorija prethodna + jedno kolo.

    Ključ je otisak CELE istorije, pa ispravka starog kola ne može da vrati zastarele
    brojače — tada se računa iznova.
    """
    k = _brojaci_kes
    if k["brojaci"] is not None:
        if k["duzina"] == len(istorija) and k["otisak"] == _otisak(istorija):
            return k["brojaci"]
        if k["duzina"] == len(istorija) - 1 and k["otisak"] == _otisak(istorija[:-1]):
            k["brojaci"].dodaj(istorija[-1][1])
            k.update(duzina=len(istorija), otisak=_otisak(istorija))
            return k["brojaci"]
    br = _Brojaci()
    for _kolo, brojevi in istorija:
        br.dodaj(brojevi)
    k.update(brojaci=br, duzina=len(istorija), otisak=_otisak(istorija))
    return br


def signali(istorija, bazen):
    """Ritam i lift parova za brojeve bazena, uvek na CELOJ istoriji (§7.2).

    Par u proseku izađe zajedno jednom u ~35 kola, pa bi u prozoru od 22 kola lift
    bio čist šum — zato ovde ne važi prozor potencijala.
    """
    br = brojaci(istorija)
    n = br.n
    odnos = {}
    for b in bazen:
        r = br.ritam(b)
        odnos[b] = br.kasnjenje(b) / r if r else 0.0
    ocek_par = n * PAR
    lift = {}
    for a, b in combinations(sorted(bazen), 2):
        lift[(a, b)] = br.parovi.get((a, b), 0) / ocek_par if ocek_par else 0.0
    return {"n": n, "kasnjenje": {b: br.kasnjenje(b) for b in bazen},
            "ritam": {b: br.ritam(b) for b in bazen}, "odnos": odnos,
            "lift": lift, "ocekivano_par": ocek_par}


# ----------------------------------------------------------------------------
# 3. Sklapanje
# ----------------------------------------------------------------------------

def _pravila_maska(XT, brojevi, pravila):
    """Bool maska kombinacija koje prolaze pravila oblika.

    XT je matrica 0/1 oblika (broj × kombinacija), brojevi rastuće. Raspored je
    namerno transponovan: zbirovi idu po dugoj, neprekidnoj osi, što je za bektest
    (poziv po kolu) nekoliko puta brže. Pravilo istorije je odvojeno (`_istorija_ok`),
    jer se u bektestu proverava samo nad kandidatima redom po skoru.
    """
    ok = np.ones(XT.shape[1], dtype=bool)
    p = pravila or {}
    b = np.asarray(brojevi)
    if p.get("dekada_max") is not None:                      # = generator.najvise_u_dekadi
        D = np.eye(4, dtype=XT.dtype)[np.minimum(b // 10, 3)]
        ok &= (D.T @ XT).max(axis=0) <= p["dekada_max"]
    if p.get("uzastopni_max") is not None:                   # = generator.broj_uzastopnih
        uz = np.zeros(XT.shape[1], dtype=XT.dtype)
        for i in np.flatnonzero(np.diff(b) == 1):            # samo susedni parovi brojeva
            uz += XT[i] * XT[i + 1]
        ok &= uz <= p["uzastopni_max"]
    if p.get("zbir_min") is not None or p.get("zbir_max") is not None:
        zbir = b.astype(np.float64) @ XT
        if p.get("zbir_min") is not None:
            ok &= zbir >= p["zbir_min"]
        if p.get("zbir_max") is not None:
            ok &= zbir <= p["zbir_max"]
    if p.get("parni_min") is not None or p.get("parni_max") is not None:
        parni = (b % 2 == 0).astype(np.float64) @ XT
        if p.get("parni_min") is not None:
            ok &= parni >= p["parni_min"]
        if p.get("parni_max") is not None:
            ok &= parni <= p["parni_max"]
    return ok


def matrica_kombinacija(redovi, m):
    """Matrica 0/1 oblika (m × broj redova): kolona = jedna kombinacija pozicija 0..m−1."""
    XT = np.zeros((m, len(redovi)), dtype=np.float32)
    XT[redovi, np.arange(len(redovi))[:, None]] = 1.0
    return XT


def _maske(B):
    """Bitmaske redova B (bit b−1 za broj b), isti zapis kao razlicitost_teorija.maska."""
    return np.left_shift(np.uint64(1), (B - 1).astype(np.uint64)).sum(axis=1, dtype=np.uint64)


def _istorija_ok(B, H, istorija_max):
    """Bool po redu B: najveće poklapanje sa bilo kojim kolom (maske H) ≤ istorija_max."""
    out = np.ones(len(B), dtype=bool)
    M = _maske(B)
    for start in range(0, len(M), 2048):           # 2.048 × 1.431 popcount-a po delu
        deo = M[start:start + 2048]
        out[start:start + 2048] = np.bitwise_count(deo[:, None] & H[None, :]).max(axis=1) <= istorija_max
    return out


@lru_cache(maxsize=None)
def _indeksi(m):
    """Sve 7-kombinacije pozicija 0..m−1, kao indeksi i kao matrica 0/1 (m × kombinacija)."""
    idx = np.array(list(combinations(range(m), K)))
    return idx, matrica_kombinacija(idx, m)


def sklopi(istorija, w=konfig.GRADITELJ_W, bazen_vel=konfig.GRADITELJ_BAZEN,
           tezine=konfig.GRADITELJ_TEZINE, pravila=None, sa_slicnoscu=True,
           predloga=1 + ALTERNATIVA, prebroj=True):
    """Ceo tok Graditelja nad istorijom (lista (kolo, brojevi), hronološki).

    prebroj=False (bektest): pravilo istorije se proverava samo nad kandidatima redom
    po skoru dok se ne nađe `predloga` predloga, umesto nad svim kombinacijama —
    ista funkcija, manji skup; `prolazi_pravila` je tada None.
    """
    if pravila is None:
        pravila = konfig.GRADITELJ_PRAVILA
    bazen_vel = max(BAZEN_OPSEG[0], min(BAZEN_OPSEG[1], int(bazen_vel)))
    a, b_, c = (float(x) for x in tezine)

    pot = potencijal(istorija, w)
    seme = sekvencijalni.seme_izbora(istorija)
    bazen = list(sekvencijalni.bazen_iz(pot["z"], bazen_vel, seme))    # sortiran
    sig = signali(istorija, bazen)

    # Min-max preko BAZENA (ne svih 39): težine treba da biraju između kandidata.
    z_n = prediktori._normalizuj({x: pot["z"][x] for x in bazen})
    r_n = prediktori._normalizuj(sig["odnos"])
    l_n = prediktori._normalizuj(sig["lift"])

    m = len(bazen)
    pos = {x: i for i, x in enumerate(bazen)}
    Z = np.array([z_n[x] for x in bazen])
    R = np.array([r_n[x] for x in bazen])
    L = np.zeros((m, m))
    for (x, y), v in l_n.items():
        L[pos[x], pos[y]] = L[pos[y], pos[x]] = v

    idx, XT = _indeksi(m)
    bz = np.array(bazen)
    s_pot = Z @ XT
    s_rit = R @ XT
    s_par = ((L.astype(np.float32) @ XT) * XT).sum(axis=0) / 2   # L simetrična, 0 na dijagonali → svaki par jednom
    # 21 par → /3 da bude na skali od 7 brojeva, kao druge dve komponente
    skor = a * s_pot + b_ * s_rit + c * s_par / 3

    ok = _pravila_maska(XT, bz, pravila)
    ist_max = pravila.get("istorija_max")
    H = brojaci(istorija).H() if (ist_max is not None and istorija) else None
    if H is not None and prebroj and ok.any():
        ok[ok] = _istorija_ok(bz[idx[ok]], H, ist_max)
    # Reproducibilan slučajan tie-break, isto seme kao bazen (PLAN_KORAK_IZBORA §2.2).
    tie = np.random.default_rng(seme).random(len(idx))

    # Redom po (skor ↓, tie ↑), ali bez sortiranja svih: najčešće je potreban samo vrh.
    preostalo = np.where(ok, np.round(skor, 6), -np.inf)
    izabrane = []
    while len(izabrane) < predloga:
        vrh = preostalo.max()
        if vrh == -np.inf:
            break
        jednaki = np.flatnonzero(preostalo == vrh)
        i = jednaki[np.argmin(tie[jednaki])]
        preostalo[i] = -np.inf
        if H is not None and not prebroj and not _istorija_ok(bz[idx[i:i + 1]], H, ist_max)[0]:
            continue
        if all(len(set(idx[i]) & set(idx[j])) <= MAX_ZAJEDNICKIH_ALT for j in izabrane):
            izabrane.append(i)

    def opis(i):
        komb = [int(x) for x in bz[idx[i]]]
        parovi = sorted(((sig["lift"][(x, y)], x, y) for x, y in combinations(komb, 2)), reverse=True)
        d = {
            "brojevi": komb,
            "skor": float(skor[i]),
            "komponente": {"potencijal": float(a * s_pot[i]), "ritam": float(b_ * s_rit[i]),
                           "parovi": float(c * s_par[i] / 3)},
            "po_broju": [{"broj": x, "z": pot["z"][x], "z_n": z_n[x],
                          "odnos": sig["odnos"][x], "odnos_n": r_n[x]} for x in komb],
            "najjaci_parovi": [{"a": x, "b": y, "lift": l} for l, x, y in parovi[:3]],
            "osobine": generator.osobine_kombinacije(komb) | {"u_dekadi": generator.najvise_u_dekadi(komb)},
        }
        if sa_slicnoscu:
            sl = odigrano.slicnost(istorija, komb)
            d["slicnost"] = {"maks": sl["maks"], "ista": sl["ista"], "upozorenje": sl["upozorenje"],
                             "najbliza": sl["najbliza"][:1]}
        return d

    return {
        "w": pot["w"], "w_trazeno": w, "broj_kola": len(istorija),
        "w_obrt": ocekivani_obrt(),
        "potencijal": {"ocekivano": pot["ocekivano"], "sigma": pot["sigma"],
                       "brojevi": [{"broj": x, "pojava": pot["pojava"][x], "z": pot["z"][x],
                                    "u_bazenu": x in bazen} for x in range(1, N + 1)]},
        "bazen": [{"broj": x, "pojava": pot["pojava"][x], "z": pot["z"][x],
                   "kasnjenje": sig["kasnjenje"][x], "ritam": sig["ritam"][x],
                   "odnos": sig["odnos"][x]} for x in bazen],
        "lift": [[(sig["lift"].get((min(x, y), max(x, y))) if x != y else None) for y in bazen]
                 for x in bazen],
        "ocekivano_par": sig["ocekivano_par"],
        "tezine": [a, b_, c], "pravila": pravila,
        "ukupno_kombinacija": int(len(idx)),
        "prolazi_pravila": int(ok.sum()) if prebroj else None,
        "predlozi": [opis(i) for i in izabrane],
    }


def predlog_za(istorija):
    """Najbolja kombinacija sa ZAKLJUČANIM podešavanjima (§2.5) — ulaz za `k_graditelj`.

    Ne prima nikakva podešavanja: bektest uvek ocenjuje isto što i podrazumevana strana.
    """
    if len(istorija) < 2:
        return None
    r = sklopi(istorija, sa_slicnoscu=False, predloga=1, prebroj=False)
    return tuple(r["predlozi"][0]["brojevi"]) if r["predlozi"] else None
