"""Analize strane „Različitost" (PLAN_RAZLICITOST §3–§7).

Sve analize se svode na preklapanje skupova preko bitmaski (razlicitost_teorija).
Pet analiza:
  1. preklapanje uzastopnih kola   -> histogram + hi-kvadrat
  2. preklapanje svih parova kola  -> histogram + hi-kvadrat (orijentacioni test)
  3. najbliži sused i rekordi      -> kartice + stepenasta linija rekorda
  4. profil ne predviđa sadržaj    -> scatter + Pearson korelacija
  5. ko-okurencija parova brojeva  -> matrica z-skorova + kontrola lažnih alarma

Analize 1, 2, 4, 5 poštuju globalni period; Analiza 3 (rekordi) uvek radi nad
celom istorijom (svaki rekord se pripisuje trenutku kada se desio).
"""

import random

from . import konfig, razlicitost_teorija as T

MAX_BROJ = konfig.MAX_BROJ
K = konfig.BROJEVA_U_KOMBINACIJI


# ----------------------------------------------------------------------------
# Priprema: istorija + maske (izračunate jednom)
# ----------------------------------------------------------------------------

def istorija_iz_conn(conn):
    """Lista (kolo, brojevi_tuple), hronološki (isti izvor kao prognoza)."""
    redovi = conn.execute(
        "SELECT kolo, b1, b2, b3, b4, b5, b6, b7 FROM istorijski_rezultati ORDER BY id ASC"
    ).fetchall()
    return [(int(r[0]), tuple(int(x) for x in r[1:8])) for r in redovi]


def _isecak(istorija, period):
    return istorija[-period:] if period and period > 0 else istorija


def _maske(istorija):
    return [T.maska(brojevi) for _kolo, brojevi in istorija]


def _histogram_udeli(brojaci, n):
    """Za sekvencu brojača po k (0..K): udeli (%) i teorijski udeli P(k) (%)."""
    posmatrano_udeo = [round(100 * brojaci[k] / n, 3) if n else 0.0 for k in range(K + 1)]
    teorija_udeo = [round(100 * T.hipergeom_pmf(k), 3) for k in range(K + 1)]
    return {
        "k": list(range(K + 1)),
        "posmatrano_broj": [brojaci[k] for k in range(K + 1)],
        "posmatrano_udeo": posmatrano_udeo,
        "teorija_udeo": teorija_udeo,
        "n": n,
    }


# ----------------------------------------------------------------------------
# Analiza 1: preklapanje uzastopnih kola (§3)
# ----------------------------------------------------------------------------

def analiza_uzastopna(istorija, period=0):
    ise = _isecak(istorija, period)
    maske = _maske(ise)
    brojaci = [0] * (K + 1)
    for i in range(len(maske) - 1):
        brojaci[T.preklapanje(maske[i], maske[i + 1])] += 1
    n = max(0, len(maske) - 1)
    return {
        "histogram": _histogram_udeli(brojaci, n),
        "test": T.hi_kvadrat_preklapanje(brojaci, n),
        "period": len(ise),
    }


# ----------------------------------------------------------------------------
# Analiza 2: preklapanje svih parova kola (§4)
# ----------------------------------------------------------------------------

def analiza_svi_parovi(istorija, period=0):
    ise = _isecak(istorija, period)
    maske = _maske(ise)
    m = len(maske)
    brojaci = [0] * (K + 1)
    for i in range(m):
        mi = maske[i]
        for j in range(i + 1, m):
            brojaci[(mi & maske[j]).bit_count()] += 1
    n = m * (m - 1) // 2
    return {
        "histogram": _histogram_udeli(brojaci, n),
        "test": T.hi_kvadrat_preklapanje(brojaci, n),
        "napomena": "Parovi nisu nezavisni (svako kolo je u m−1 parova) → test je "
                    "orijentacioni; merodavna je vizuelna podudarnost sa krivom i Analiza 1.",
        "period": m,
    }


# ----------------------------------------------------------------------------
# Analiza 3: najbliži sused i rekordi (§5) — uvek cela istorija
# ----------------------------------------------------------------------------

def analiza_rekordi(istorija):
    maske = _maske(istorija)
    m = len(maske)
    broj_parova = m * (m - 1) // 2

    najvece_k = -1
    najvece_par = None            # (kolo_i, kolo_j)
    identicne = []                # [(kolo_i, kolo_j)] gde je preklapanje == K
    parova_5plus = 0
    rekord_linija = []            # (redni_broj_kola, tekuci_rekord) — stepenasto
    tekuci_rekord = 0

    for j in range(m):
        mj = maske[j]
        najbolji_za_j = 0
        for i in range(j):
            k = (mj & maske[i]).bit_count()
            if k >= 5:
                parova_5plus += 1
            if k > najbolji_za_j:
                najbolji_za_j = k
            if k > najvece_k:
                najvece_k = k
                najvece_par = (istorija[i][0], istorija[j][0])
            if k == K:
                identicne.append((istorija[i][0], istorija[j][0]))
        if najbolji_za_j > tekuci_rekord:
            tekuci_rekord = najbolji_za_j
        rekord_linija.append([j + 1, tekuci_rekord])

    zajednicki = []
    if najvece_par is not None:
        ma = dict(istorija)[najvece_par[0]]
        mb = dict(istorija)[najvece_par[1]]
        zajednicki = sorted(set(ma) & set(mb))

    ocek_5plus = broj_parova * T.p_preklapanje_bar(5)
    return {
        "broj_kola": m,
        "broj_parova": broj_parova,
        "identicne": {"ima": len(identicne) > 0, "parovi": identicne[:20]},
        "najvece": {"k": max(0, najvece_k), "par": najvece_par, "zajednicki": zajednicki},
        "parova_5plus": {"posmatrano": parova_5plus, "ocekivano": round(ocek_5plus, 2)},
        "rekord_linija": rekord_linija,
    }


# ----------------------------------------------------------------------------
# Analiza 4: profil ne predviđa sadržaj (§6)
# ----------------------------------------------------------------------------

_PROFILI = {
    "sredina": "razlika srednjih vrednosti",
    "parni":   "razlika u broju parnih",
    "dekade":  "L1 razlika rasporeda po dekadama",
}


def _dekade_vektor(brojevi):
    v = [0, 0, 0, 0]
    for b in brojevi:
        v[min((b - 1) // 10, 3)] += 1
    return v


def _profil_razlika(a, b, tip):
    if tip == "parni":
        return abs(sum(1 for x in a if x % 2 == 0) - sum(1 for x in b if x % 2 == 0))
    if tip == "dekade":
        va, vb = _dekade_vektor(a), _dekade_vektor(b)
        return sum(abs(va[i] - vb[i]) for i in range(4))
    # sredina (podrazumevano)
    return abs(sum(a) / len(a) - sum(b) / len(b))


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / (sxx * syy) ** 0.5


def analiza_profil(istorija, period=0, tip="sredina", uzorak=20000, seed=12345,
                   max_tacaka=1500):
    """Scatter (razlika profila vs. preklapanje) + Pearson r + linija proseka po binu."""
    if tip not in _PROFILI:
        tip = "sredina"
    ise = _isecak(istorija, period)
    komb = [brojevi for _kolo, brojevi in ise]
    maske = _maske(ise)
    m = len(maske)
    ukupno_parova = m * (m - 1) // 2
    rng = random.Random(seed)

    parovi = []
    if ukupno_parova <= uzorak:
        for i in range(m):
            for j in range(i + 1, m):
                parovi.append((i, j))
    else:
        vidjeni = set()
        while len(parovi) < uzorak:
            i = rng.randrange(m)
            j = rng.randrange(m)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if (i, j) in vidjeni:
                continue
            vidjeni.add((i, j))
            parovi.append((i, j))

    xs, ys = [], []
    bin_sum, bin_cnt = {}, {}
    for i, j in parovi:
        x = _profil_razlika(komb[i], komb[j], tip)
        y = (maske[i] & maske[j]).bit_count()
        xs.append(x)
        ys.append(y)
        b = round(x)   # bin širine 1
        bin_sum[b] = bin_sum.get(b, 0.0) + y
        bin_cnt[b] = bin_cnt.get(b, 0) + 1

    r = _pearson(xs, ys)

    # scatter tačke (podskup radi veličine odgovora) sa malim jitterom po Y
    korak = max(1, len(parovi) // max_tacaka)
    tacke = []
    jit = random.Random(seed + 1)
    for idx in range(0, len(parovi), korak):
        tacke.append([round(xs[idx], 3), round(ys[idx] + (jit.random() - 0.5) * 0.5, 3)])

    binovi = sorted(bin_sum)
    linija = [[b, round(bin_sum[b] / bin_cnt[b], 3)] for b in binovi]

    if abs(r) < 0.05:
        tumac = "Profil ne nosi informaciju o sadržaju (|r| < 0,05)."
    elif abs(r) < 0.15:
        tumac = "Vrlo slaba veza — praktično zanemarljiva."
    else:
        tumac = "Postoji merljiva veza (proveriti — neočekivano za slučajnost)."

    return {
        "tip": tip,
        "profili": _PROFILI,
        "r": round(r, 4),
        "tumacenje": tumac,
        "tacke": tacke,
        "linija_proseka": linija,
        "referenca": round(T.ocekivano_preklapanje(), 4),
        "uzorak": len(parovi),
        "period": m,
    }


# ----------------------------------------------------------------------------
# Analiza 5: ko-okurencija parova brojeva (§7)
# ----------------------------------------------------------------------------

def _matrica_ko_okurencije(ise):
    """Vraća (matrica[1..N][1..N] broj zajedničkih kola, n_kola)."""
    mat = [[0] * (MAX_BROJ + 1) for _ in range(MAX_BROJ + 1)]
    for _kolo, brojevi in ise:
        bs = sorted(set(brojevi))
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                a, b = bs[i], bs[j]
                mat[a][b] += 1
                mat[b][a] += 1
    return mat, len(ise)


def analiza_ko_okurencija(istorija, period=0):
    ise = _isecak(istorija, period)
    mat, n = _matrica_ko_okurencije(ise)
    p = T.p_par_u_kolu()
    ocek = n * p
    sd = (n * p * (1 - p)) ** 0.5 if n > 0 else 0.0

    z_matrica = [[0.0] * (MAX_BROJ + 1) for _ in range(MAX_BROJ + 1)]
    van_intervala = 0
    najveci_z = []   # (z, a, b, posmatrano)
    for a in range(1, MAX_BROJ + 1):
        for b in range(a + 1, MAX_BROJ + 1):
            obs = mat[a][b]
            z = (obs - ocek) / sd if sd > 0 else 0.0
            z_matrica[a][b] = round(z, 3)
            z_matrica[b][a] = round(z, 3)
            if abs(z) > 1.96:
                van_intervala += 1
            najveci_z.append((z, a, b, obs))

    najveci_z.sort(key=lambda t: -abs(t[0]))
    broj_parova = MAX_BROJ * (MAX_BROJ - 1) // 2
    ocek_van = round(0.05 * broj_parova, 1)

    return {
        "n_kola": n,
        "matrica_z": z_matrica,           # [a][b], indeksi 1..N; 0-ti red/kolona nule
        "ocekivano_po_paru": round(ocek, 2),
        "p_par": round(p, 5),
        "kontrola": {
            "van_95_intervala": van_intervala,
            "ocekivano_van": ocek_van,
            "broj_parova": broj_parova,
            "tekst": (f"Van 95% intervala je {van_intervala} parova; slučajnost bi dala "
                      f"≈{ocek_van}. Ako je posmatrano ≈ očekivano, „vrući parovi“ su šum, "
                      f"ne signal."),
        },
        "top_odstupanja": [
            {"a": a, "b": b, "z": round(z, 3), "posmatrano": obs,
             "ocekivano": round(ocek, 2)}
            for z, a, b, obs in najveci_z[:15]
        ],
        "period": len(ise),
    }


def ko_okurencija_par(istorija, a, b, period=0, poslednjih=5):
    """Detalj za jedan par: posmatrano, očekivano, z, i poslednjih N kola gde su zajedno."""
    ise = _isecak(istorija, period)
    n = len(ise)
    p = T.p_par_u_kolu()
    ocek = n * p
    sd = (n * p * (1 - p)) ** 0.5 if n > 0 else 0.0
    kola_zajedno = [kolo for kolo, brojevi in ise if a in brojevi and b in brojevi]
    obs = len(kola_zajedno)
    z = (obs - ocek) / sd if sd > 0 else 0.0
    return {
        "a": a, "b": b, "posmatrano": obs, "ocekivano": round(ocek, 2),
        "z": round(z, 3), "poslednja_kola": kola_zajedno[-poslednjih:],
    }


# ----------------------------------------------------------------------------
# Mere za Generator i Bektest (§8) — reuse istog bitmask preklapanja
# ----------------------------------------------------------------------------

def razlicitost_seta(kombinacije):
    """Unutrašnja različitost i pokrivenost skupa generisanih kombinacija (§8).

    kombinacije: lista listi/tuple-ova brojeva. Vraća prosečno i maksimalno
    preklapanje po svim parovima + koliko različitih brojeva set pokriva.
    """
    maske = [T.maska(k) for k in kombinacije]
    m = len(maske)
    prosek, maks = None, 0
    if m >= 2:
        ukupno, parova = 0, 0
        for i in range(m):
            mi = maske[i]
            for j in range(i + 1, m):
                k = (mi & maske[j]).bit_count()
                ukupno += k
                if k > maks:
                    maks = k
                parova += 1
        prosek = ukupno / parova if parova else None
    pokriveni = sorted({int(b) for k in kombinacije for b in k})
    return {
        "broj_kombinacija": m,
        "prosek_preklapanja": round(prosek, 3) if prosek is not None else None,
        "maks_preklapanje": maks,
        "referenca": round(T.ocekivano_preklapanje(), 3),
        "pokrivenost": len(pokriveni),
        "pokriveni_brojevi": pokriveni,
        "ukupno_brojeva": MAX_BROJ,
    }


# ----------------------------------------------------------------------------
# Objedinjeni izlaz za stranu
# ----------------------------------------------------------------------------

def sve_analize(conn, period=0):
    istorija = istorija_iz_conn(conn)
    if len(istorija) < 2:
        return {"dovoljno_podataka": False, "broj_kola": len(istorija)}
    return {
        "dovoljno_podataka": True,
        "broj_kola": len(istorija),
        "period": period,
        "teorija": {
            "ocekivano": round(T.ocekivano_preklapanje(), 4),
            "sigma": round(T.sigma_preklapanja(), 4),
            "pmf": [round(T.hipergeom_pmf(k), 6) for k in range(K + 1)],
        },
        "uzastopna": analiza_uzastopna(istorija, period),
        "svi_parovi": analiza_svi_parovi(istorija, period),
        "rekordi": analiza_rekordi(istorija),
        "profil": analiza_profil(istorija, period),
        "ko_okurencija": analiza_ko_okurencija(istorija, period),
    }
