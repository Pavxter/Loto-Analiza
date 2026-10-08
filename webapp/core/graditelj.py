"""Graditelj kombinacije (PLAN_TIKETI_GRADITELJ §7).

Tri koraka na jednoj strani, svaki vidljiv:
  1. potencijal po broju u prozoru „obrta" → bazen najvećeg potencijala;
  2. signali za bazen na celoj istoriji: ritam (D/R) i kookurencija parova (lift);
  3. sve C(bazen, 7) kombinacije: težinski skor signala, pa pravila oblika isključuju.

Signali (potencijal, ritam, parovi) tvrde nešto o budućnosti i zato idu u bektest
(faza 6). Pravila oblika (dekade, uzastopni, zbir, parnost, istorija) ne menjaju
verovatnoću — svaka kombinacija je jednako verovatna — pa samo isključuju, ne boduju.

Postojeće funkcije se uvoze, ne kopiraju: brojači i ritam iz `prediktori._primitivi`,
min-max iz `prediktori._normalizuj`, tie-break iz `sekvencijalni`, pravila iz
`generator`, sličnost sa istorijom iz `odigrano`.
"""

import math
import random
from fractions import Fraction
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

def signali(istorija, bazen):
    """Ritam i lift parova za brojeve bazena, uvek na CELOJ istoriji (§7.2).

    Par u proseku izađe zajedno jednom u ~35 kola, pa bi u prozoru od 22 kola lift
    bio čist šum — zato ovde ne važi prozor potencijala.
    """
    prim = prediktori._primitivi(istorija)
    n = prim["n"]
    odnos = {}
    for b in bazen:
        r = prim["ritam"][b]
        odnos[b] = prim["kasnjenje"][b] / r if r else 0.0
    ocek_par = n * PAR
    lift = {}
    for a, b in combinations(sorted(bazen), 2):
        lift[(a, b)] = prim["parovi"].get((a, b), 0) / ocek_par if ocek_par else 0.0
    return {"n": n, "kasnjenje": {b: prim["kasnjenje"][b] for b in bazen},
            "ritam": {b: prim["ritam"][b] for b in bazen}, "odnos": odnos,
            "lift": lift, "ocekivano_par": ocek_par}


# ----------------------------------------------------------------------------
# 3. Sklapanje
# ----------------------------------------------------------------------------

def _pravila_maska(B, pravila, istorija):
    """Bool maska kombinacija (redovi B, sortirani brojevi) koje prolaze pravila oblika."""
    ok = np.ones(len(B), dtype=bool)
    p = pravila or {}
    if p.get("dekada_max") is not None:
        dek = np.minimum(B // 10, 3)
        po_dekadi = np.stack([(dek == d).sum(axis=1) for d in range(4)], axis=1)
        ok &= po_dekadi.max(axis=1) <= p["dekada_max"]       # = generator.najvise_u_dekadi
    if p.get("uzastopni_max") is not None:
        ok &= (np.diff(B, axis=1) == 1).sum(axis=1) <= p["uzastopni_max"]   # = broj_uzastopnih
    zbir = B.sum(axis=1)
    if p.get("zbir_min") is not None:
        ok &= zbir >= p["zbir_min"]
    if p.get("zbir_max") is not None:
        ok &= zbir <= p["zbir_max"]
    parni = (B % 2 == 0).sum(axis=1)
    if p.get("parni_min") is not None:
        ok &= parni >= p["parni_min"]
    if p.get("parni_max") is not None:
        ok &= parni <= p["parni_max"]
    if p.get("istorija_max") is not None and istorija and ok.any():
        # Najveće poklapanje sa bilo kojim izvučenim kolom: binarne matrice, pa proizvod.
        H = np.zeros((N + 1, len(istorija)), dtype=np.float32)
        for j, (_kolo, br) in enumerate(istorija):
            H[list(br), j] = 1
        idx = np.flatnonzero(ok)
        for start in range(0, len(idx), 4096):       # u delovima: 77.520 × 1.431 ne staje odjednom
            deo = idx[start:start + 4096]
            X = np.zeros((len(deo), N + 1), dtype=np.float32)
            np.put_along_axis(X, B[deo], 1, axis=1)
            ok[deo] &= (X @ H).max(axis=1) <= p["istorija_max"]
    return ok


def sklopi(istorija, w=konfig.GRADITELJ_W, bazen_vel=konfig.GRADITELJ_BAZEN,
           tezine=konfig.GRADITELJ_TEZINE, pravila=None, sa_slicnoscu=True):
    """Ceo tok Graditelja nad istorijom (lista (kolo, brojevi), hronološki)."""
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

    idx = np.array(list(combinations(range(m), K)))
    B = np.array(bazen)[idx]
    s_pot = Z[idx].sum(axis=1)
    s_rit = R[idx].sum(axis=1)
    s_par = sum(L[idx[:, i], idx[:, j]] for i, j in combinations(range(K), 2))
    # 21 par → /3 da bude na skali od 7 brojeva, kao druge dve komponente
    skor = a * s_pot + b_ * s_rit + c * s_par / 3

    ok = _pravila_maska(B, pravila, istorija)
    # Reproducibilan slučajan tie-break, isto seme kao bazen (PLAN_KORAK_IZBORA §2.2).
    rnd = random.Random(seme)
    tie = np.array([rnd.random() for _ in range(len(B))])
    red =[i for i in np.lexsort((tie, -np.round(skor, 9))) if ok[i]]

    izabrane = []
    for i in red:
        if len(izabrane) > ALTERNATIVA:
            break
        if all(len(set(B[i]) & set(B[j])) <= MAX_ZAJEDNICKIH_ALT for j in izabrane):
            izabrane.append(i)

    def opis(i):
        komb = [int(x) for x in B[i]]
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
        "ukupno_kombinacija": int(len(B)), "prolazi_pravila": int(ok.sum()),
        "predlozi": [opis(i) for i in izabrane],
    }
