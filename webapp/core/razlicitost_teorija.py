"""Teorijska osnova za analizu različitosti kombinacija (PLAN_RAZLICITOST §1–§2).

Zajednički modul za dve strane:
  - „Različitost" — meri preklapanje izvučenih kombinacija i poredi sa slučajnošću;
  - „Prognoza / Kombinacija" — meri preklapanje predložene kombinacije sa dobitnom.

Sve funkcije su čiste i deterministički izvedene iz pravila igre (konfig.MAX_BROJ,
konfig.BROJEVA_U_KOMBINACIJI) — ništa nije hardkodovano, sve se izračunava iz N i K.

Ključni pojmovi:
  N = MAX_BROJ (39), K = BROJEVA_U_KOMBINACIJI (7).
  Preklapanje dve kombinacije = broj zajedničkih brojeva (0..K).
  Pod nultom hipotezom (dve nezavisne slučajne kombinacije) preklapanje prati
  hipergeometrijsku raspodelu P(k) = C(K,k)·C(N-K,K-k)/C(N,K).
"""

from math import comb, sqrt

from scipy.stats import chi2 as _chi2, norm as _norm

from . import konfig

N = konfig.MAX_BROJ                     # ukupno brojeva u igri (39)
K = konfig.BROJEVA_U_KOMBINACIJI        # brojeva po kombinaciji (7)


# ----------------------------------------------------------------------------
# Bitmaske (PLAN_RAZLICITOST §2) — jedina implementacija preklapanja u aplikaciji
# ----------------------------------------------------------------------------

def maska(brojevi) -> int:
    """39-bitni integer: bit (b-1) je 1 ako je broj b prisutan."""
    m = 0
    for b in brojevi:
        m |= 1 << (int(b) - 1)
    return m


def preklapanje(m1: int, m2: int) -> int:
    """Broj zajedničkih brojeva dve maske (popcount preseka)."""
    return (m1 & m2).bit_count()


def preklapanje_brojeva(a, b) -> int:
    """Preklapanje dva skupa brojeva (pogodno gde maske nisu preračunate)."""
    return preklapanje(maska(a), maska(b))


# ----------------------------------------------------------------------------
# Hipergeometrijska raspodela preklapanja (PLAN_RAZLICITOST §1)
# ----------------------------------------------------------------------------

_UKUPNO = comb(N, K)   # C(39,7)


def hipergeom_pmf(k: int) -> float:
    """P(dve nezavisne slučajne kombinacije dele tačno k brojeva)."""
    if k < 0 or k > K or (K - k) > (N - K):
        return 0.0
    return comb(K, k) * comb(N - K, K - k) / _UKUPNO


def sve_pmf() -> list[float]:
    """[P(0), P(1), ..., P(K)] — cela raspodela preklapanja."""
    return [hipergeom_pmf(k) for k in range(K + 1)]


def p_preklapanje_bar(k_min: int) -> float:
    """P(preklapanje ≥ k_min) — rep raspodele (npr. za '5+ zajedničkih')."""
    return sum(hipergeom_pmf(k) for k in range(k_min, K + 1))


def ocekivano_preklapanje() -> float:
    """E[k] = K·K/N ≈ 1,2564 (hipergeometrijsko očekivanje)."""
    return K * K / N


def sigma_preklapanja() -> float:
    """Standardna devijacija preklapanja: sqrt(Σ k²·P(k) − μ²) ≈ 0,9797."""
    mu = ocekivano_preklapanje()
    drugi_moment = sum(k * k * hipergeom_pmf(k) for k in range(K + 1))
    return sqrt(max(0.0, drugi_moment - mu * mu))


def p_par_u_kolu() -> float:
    """Verovatnoća da se konkretan par brojeva nađe zajedno u jednom kolu.

    C(N-2, K-2)/C(N,K) = K·(K-1)/(N·(N-1)) ≈ 4,05 %.
    """
    return comb(N - 2, K - 2) / _UKUPNO


# ----------------------------------------------------------------------------
# Statistički testovi (PLAN_RAZLICITOST §3, PLAN_PROGNOZA_KOMBINACIJE §1)
# ----------------------------------------------------------------------------

def binomni_interval(p: float, n: int, z: float = 1.96):
    """Simetričan interval pouzdanosti za udeo (normalna aproksimacija).

    Vraća (donja, gornja) granicu za verovatnoću p posle n opažanja.
    """
    if n <= 0:
        return (0.0, 1.0)
    margina = z * sqrt(p * (1 - p) / n)
    return (max(0.0, p - margina), min(1.0, p + margina))


def pojas_proseka(n: int, z: float = 1.96):
    """Interval pouzdanosti za kumulativni prosek preklapanja posle n kola:
    μ ± z·σ/√n (PLAN_PROGNOZA_KOMBINACIJE §1). Vraća (donja, gornja)."""
    mu = ocekivano_preklapanje()
    if n <= 0:
        return (mu, mu)
    margina = z * sigma_preklapanja() / sqrt(n)
    return (mu - margina, mu + margina)


def z_test_proseka(prosek: float, n: int):
    """Dvostrani z-test da je prosek preklapanja jednak μ (PLAN §1).

    Vraća (z, p_vrednost). Za n ≤ 0 vraća (None, None). Pozivalac odlučuje da li
    prikazuje p-vrednost (plan preporučuje 'premalo podataka' za n < 30).
    """
    if n <= 0:
        return (None, None)
    sigma = sigma_preklapanja()
    if sigma == 0:
        return (None, None)
    z = (prosek - ocekivano_preklapanje()) / (sigma / sqrt(n))
    p = 2.0 * _norm.sf(abs(z))
    return (float(z), float(p))


def hi_kvadrat_preklapanje(posmatrano, n: int, spoji_od: int = 4):
    """Hi-kvadrat test podudarnosti raspodele preklapanja sa hipergeometrijskom.

    posmatrano: sekvenca dužine K+1 (broj slučajeva za k = 0..K) ILI dict {k: broj}.
    n:          ukupan broj opažanja (par kola). Ako je None, uzima se Σ posmatrano.
    spoji_od:   kategorije k ≥ spoji_od spajaju se u jednu ćeliju (pravilo očekivane
                frekvencije ≥ 5; PLAN_RAZLICITOST §3).

    Vraća dict: {chi2, df, p, kategorije: [{k, oznaka, posmatrano, ocekivano}]}.
    Ako je n premalo (neka očekivana frekvencija < 5 i posle spajanja), vraća p=None
    uz upozorenje.
    """
    if isinstance(posmatrano, dict):
        obs = [int(posmatrano.get(k, 0)) for k in range(K + 1)]
    else:
        obs = [int(x) for x in posmatrano] + [0] * (K + 1 - len(posmatrano))
        obs = obs[:K + 1]
    if n is None:
        n = sum(obs)

    kategorije = []
    for k in range(spoji_od):
        kategorije.append({"k": k, "oznaka": str(k),
                           "posmatrano": obs[k], "ocekivano": n * hipergeom_pmf(k)})
    spojeno_obs = sum(obs[spoji_od:])
    spojeno_exp = n * p_preklapanje_bar(spoji_od)
    kategorije.append({"k": spoji_od, "oznaka": f"{spoji_od}+",
                       "posmatrano": spojeno_obs, "ocekivano": spojeno_exp})

    chi2 = 0.0
    for c in kategorije:
        e = c["ocekivano"]
        if e > 0:
            chi2 += (c["posmatrano"] - e) ** 2 / e
    df = len(kategorije) - 1
    min_exp = min(c["ocekivano"] for c in kategorije)
    p = float(_chi2.sf(chi2, df)) if (df > 0 and min_exp >= 5) else None

    return {"chi2": round(chi2, 4), "df": df,
            "p": (round(p, 5) if p is not None else None),
            "n": n, "min_ocekivano": round(min_exp, 3),
            "kategorije": [{"k": c["k"], "oznaka": c["oznaka"],
                            "posmatrano": c["posmatrano"],
                            "ocekivano": round(c["ocekivano"], 3)} for c in kategorije]}
