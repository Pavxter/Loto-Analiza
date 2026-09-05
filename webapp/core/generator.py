"""Generator kombinacija: filtriranje, bodovanje i filter diverziteta.

Verno portovano iz analiza.py:
  - pokreni_generisanje (filteri, linije 1217-1316)
  - izracunaj_skor (1497-1534)
  - primeni_filter_diverziteta (1536-1554)
"""

import itertools

from . import konfig

MAX_BROJ = konfig.MAX_BROJ
BROJEVA_U_KOMBINACIJI = konfig.BROJEVA_U_KOMBINACIJI


def izracunaj_skor(kombinacija, analiza, strategija_svezine="favorizuj", primeni_pristrasnost=False):
    """Bodovanje jedne kombinacije. `analiza` je core.analitika.Analiza instanca."""
    skor = 0.0

    sr = sum(kombinacija) / float(BROJEVA_U_KOMBINACIJI)
    udaljenost = abs(sr - analiza.globalni_prosek)
    if analiza.globalna_std_dev > 0:
        skor += max(0, 100 * (1 - udaljenost / (2 * analiza.globalna_std_dev)))

    broj_svezih = sum(1 for b in kombinacija if b in analiza.svezi_brojevi)
    if strategija_svezine == "favorizuj":
        skor += broj_svezih * 10
    elif strategija_svezine == "kaznjavaj":
        skor -= broj_svezih * 10
    # "ignorisi" -> bez uticaja

    ap = analiza.analiza_ponavljanja
    pozitivni = ap[ap > 0]
    if len(ap) > 0 and len(pozitivni) > 0:
        prosek_ritma = pozitivni.mean()
        for b in kombinacija:
            ritam = ap.get(b, prosek_ritma)
            if abs(ritam - prosek_ritma) < 3:
                skor += 5

    if primeni_pristrasnost:
        for i, b in enumerate(kombinacija):
            pozicija = i + 1
            bonus = analiza.model_pristrasnosti.get((b, pozicija), 1.0)
            skor += (bonus - 1) * 10

    return round(skor, 2)


def primeni_filter_diverziteta(kandidati, max_slicnost, broj_kola_za_izbegavanje, loto_df):
    """Zadržava kombinacije koje se ne preklapaju previše međusobno i sa skorašnjim kolima."""
    if not kandidati:
        return []
    zona_izbegavanja = []
    if broj_kola_za_izbegavanje > 0 and not loto_df.empty:
        poslednja = loto_df[konfig.KOLONE_ZA_BROJEVE].tail(broj_kola_za_izbegavanje)
        zona_izbegavanja = [set(row) for row in poslednja.values]

    finalne = []
    zadrzane = []
    for skor, komb in kandidati:
        s = set(komb)
        previse_slican = any(len(s & post) > max_slicnost for post in finalne + zona_izbegavanja)
        if not previse_slican:
            finalne.append(s)
            zadrzane.append((skor, komb))
    return zadrzane


def generisi(analiza, bazen=None, filteri=None):
    """Generiše, filtrira i rangira kombinacije.

    filteri: dict sa ključevima (svi opcioni):
      min_sv, max_sv, parni, vruci, hladni, uzastopni, dekada_max,
      filtriraj_unikate, strategija_svezine, primeni_pristrasnost,
      diverzitet (bool), max_slicnost (int)
    """
    f = filteri or {}
    izvor = sorted(set(bazen)) if bazen else list(range(1, MAX_BROJ + 1))

    min_sv = f.get("min_sv", 1)
    max_sv = f.get("max_sv", 39)
    parni = f.get("parni")           # int ili None (bez filtera)
    vruci = f.get("vruci")
    hladni = f.get("hladni")
    uzastopni = f.get("uzastopni")
    dekada_max = f.get("dekada_max")
    filtriraj_unikate = f.get("filtriraj_unikate", False)

    validne = []
    for komb in itertools.combinations(izvor, BROJEVA_U_KOMBINACIJI):
        if filtriraj_unikate and komb in analiza.set_istorijskih_kombinacija:
            continue
        sr = sum(komb) / float(BROJEVA_U_KOMBINACIJI)
        if not (min_sv <= sr <= max_sv):
            continue
        if parni is not None and sum(1 for b in komb if b % 2 == 0) != parni:
            continue
        if vruci is not None and sum(1 for b in komb if b in analiza.vruci_brojevi) != vruci:
            continue
        if hladni is not None and sum(1 for b in komb if b in analiza.hladni_brojevi) != hladni:
            continue
        if uzastopni is not None and sum(1 for i in range(len(komb) - 1) if komb[i + 1] == komb[i] + 1) != uzastopni:
            continue
        if dekada_max is not None:
            dekade = {"1-9": 0, "10-19": 0, "20-29": 0, "30-39": 0}
            for b in komb:
                if b <= 9:
                    dekade["1-9"] += 1
                elif b <= 19:
                    dekade["10-19"] += 1
                elif b <= 29:
                    dekade["20-29"] += 1
                else:
                    dekade["30-39"] += 1
            if max(dekade.values()) > dekada_max:
                continue
        validne.append(komb)

    strategija = f.get("strategija_svezine", "favorizuj")
    primeni = f.get("primeni_pristrasnost", False)
    sa_skorom = [(izracunaj_skor(k, analiza, strategija, primeni), k) for k in validne]
    sa_skorom.sort(key=lambda x: x[0], reverse=True)

    ukupno_pre_diverziteta = len(sa_skorom)
    if f.get("diverzitet"):
        sa_skorom = primeni_filter_diverziteta(sa_skorom, f.get("max_slicnost", 4), 1, analiza.loto_df)

    return {
        "ukupno_validnih": ukupno_pre_diverziteta,
        "posle_diverziteta": len(sa_skorom),
        "kombinacije": [{"skor": s, "brojevi": list(k)} for s, k in sa_skorom],
    }
