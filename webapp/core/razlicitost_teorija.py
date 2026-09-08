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
            "p_tacno": p,
            "n": n, "min_ocekivano": round(min_exp, 3),
            "kategorije": [{"k": c["k"], "oznaka": c["oznaka"],
                            "posmatrano": c["posmatrano"],
                            "ocekivano": round(c["ocekivano"], 3)} for c in kategorije]}


# ----------------------------------------------------------------------------
# Rang kombinacije (PLAN_SINTEZA §2.3) — teorijska očekivanja
# ----------------------------------------------------------------------------
# Rang je leksikografski redni broj kombinacije (mapa.rang), 0 .. C(N,K)-1. Pod
# nultom hipotezom je izvučena kombinacija ravnomerno izabrana iz svih C(N,K),
# pa je i rang ravnomeran na tom opsegu. Sve što sledi izvedeno je iz te jedne
# rečenice — ništa nije podešeno prema podacima.

UKUPNO_KOMBINACIJA = _UKUPNO


# --- Koliko je neka KLASA kombinacija česta (PLAN_KORAK_IZBORA §4.3) ---
# Predlog sa šest parnih izgleda neobično, i jeste redak kao klasa — ali svaka
# pojedinačna kombinacija u toj klasi ima istu šansu kao bilo koja druga. Ove dve
# funkcije daju tačan broj, da se to ne bi tvrdilo napamet.

_PARNIH = N // 2                        # 2, 4, …, 38 → 19 parnih brojeva
_NEPARNIH = N - _PARNIH                 # 20 neparnih


def broj_sa_parnih(parnih: int) -> int:
    """Koliko od C(39,7) kombinacija ima tačno `parnih` parnih brojeva."""
    if not 0 <= parnih <= K:
        return 0
    return comb(_PARNIH, parnih) * comb(_NEPARNIH, K - parnih)


def broj_sa_uzastopnih(parova: int) -> int:
    """Koliko kombinacija ima tačno `parova` susednih parova (npr. 3,4 broji jedan).

    K brojeva raspoređenih u `r` neprekinutih nizova daju tačno K − r susednih
    parova. Broj K-podskupova od N sa tačno r nizova je C(N − K + 1, r)·C(K − 1, r − 1),
    pa je odgovor ta vrednost za r = K − parova.
    """
    if not 0 <= parova <= K - 1:
        return 0
    r = K - parova
    return comb(N - K + 1, r) * comb(K - 1, r - 1)


def ocekivano_po_korpi(n: int, broj_korpi: int) -> float:
    """Očekivan broj kola po korpi ako je rang ravnomeran."""
    if broj_korpi <= 0:
        raise ValueError("Broj korpi mora biti pozitivan.")
    return n / broj_korpi


def cdf_rastojanja(d: float, maks: int = _UKUPNO) -> float:
    """P(|X − Y| ≤ d) za dva nezavisna ravnomerna ranga na [0, maks].

    Razlika dva ravnomerna broja ima trougaonu raspodelu: f(d) = 2(M−d)/M², pa je
    F(d) = 2d/M − (d/M)². Odatle i prosek M/3.
    """
    if maks <= 0:
        return 0.0
    x = min(max(d / maks, 0.0), 1.0)
    return 2 * x - x * x


def kvantil_rastojanja(q: float, maks: int = _UKUPNO) -> float:
    """Inverz `cdf_rastojanja`: d takvo da je F(d) = q. Daje ivice jednako
    verovatnih korpi, pa nijedna ćelija hi-kvadrata nema malo očekivanje."""
    q = min(max(q, 0.0), 1.0)
    return maks * (1 - sqrt(1 - q))


def ocekivano_rastojanje(maks: int = _UKUPNO) -> float:
    """E|X − Y| = M/3 za dva nezavisna ravnomerna ranga na [0, M]."""
    return maks / 3.0


def p_min_broj(k: int) -> float:
    """P(najmanji izvučen broj je tačno k) = C(N−k, K−1) / C(N,K).

    Kombinacija sa minimumom k bira preostalih K−1 brojeva iz {k+1..N}, kojih je
    N−k. Zbir po k od 1 do N−K+1 daje tačno 1 (hokej-štap identitet).
    """
    if k < 1 or k > N - K + 1:
        return 0.0
    return comb(N - k, K - 1) / _UKUPNO


def sve_p_min() -> list[float]:
    """[P(min=1), …, P(min=N−K+1)] — cela raspodela najmanjeg broja."""
    return [p_min_broj(k) for k in range(1, N - K + 2)]


def hi_kvadrat_opsti(posmatrano, ocekivano):
    """Hi-kvadrat podudarnosti za proizvoljne korpe: Σ (O−E)²/E, df = korpe − 1.

    Vraća p=None ako je najmanje očekivanje manje od 5 (pravilo primenljivosti);
    pozivalac je dužan da korpe spoji pre poziva.
    """
    obs = [float(x) for x in posmatrano]
    exp = [float(x) for x in ocekivano]
    if len(obs) != len(exp) or not obs:
        raise ValueError("Posmatrano i očekivano moraju biti iste, nenulte dužine.")
    chi2 = sum((o - e) ** 2 / e for o, e in zip(obs, exp) if e > 0)
    df = len(obs) - 1
    min_exp = min(exp)
    p = float(_chi2.sf(chi2, df)) if (df > 0 and min_exp >= 5) else None
    return {"chi2": round(chi2, 4), "df": df,
            "p": (round(p, 6) if p is not None else None),
            "p_tacno": p, "min_ocekivano": round(min_exp, 3)}


def autokorelacija(niz, pomak: int) -> float:
    """Pearsonova autokorelacija niza sa samim sobom pomerenim za `pomak`."""
    n = len(niz)
    if pomak <= 0 or n - pomak < 2:
        return 0.0
    a = niz[:-pomak]
    b = niz[pomak:]
    m_a = sum(a) / len(a)
    m_b = sum(b) / len(b)
    kov = sum((x - m_a) * (y - m_b) for x, y in zip(a, b))
    va = sum((x - m_a) ** 2 for x in a)
    vb = sum((y - m_b) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    return kov / sqrt(va * vb)


def ljung_box(korelacije, n: int):
    """Ljung–Box Q za pomake 1..m: Q = n(n+2) Σ r_k²/(n−k), hi-kvadrat sa df=m.

    Plan predviđa z = r·√n po pomaku. Pet pomaka bi bilo pet testova, pa bi ušlo
    pet puta u korekciju; Ljung–Box ih sabira u jedan test iste nulte hipoteze
    („nema autokorelacije ni na jednom pomaku"). Pojedinačni z ostaju u detalju.
    """
    m = len(korelacije)
    if n <= m + 1 or m == 0:
        return {"Q": None, "df": m, "p": None, "p_tacno": None, "z": []}
    q = n * (n + 2) * sum(r * r / (n - k) for k, r in enumerate(korelacije, start=1))
    p = float(_chi2.sf(q, m))
    return {"Q": round(q, 4), "df": m, "p": round(p, 6), "p_tacno": p,
            "z": [round(r * sqrt(n), 3) for r in korelacije]}


def hi_kvadrat_frekvencije(brojaci, n_kola: int):
    """Ravnomernost 39 brojeva: očekivano n·K/N po broju, df = N−1."""
    ocek = n_kola * K / N
    return hi_kvadrat_opsti(list(brojaci), [ocek] * N)
