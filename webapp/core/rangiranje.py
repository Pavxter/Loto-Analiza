"""Rangiranje brojeva trima metodama: Frekvencija, Bajes, Hibrid.

Sve tri proizvode rangiranu listu svih brojeva. Verno portovano iz analiza.py:
  - Bajes:  pokreni_bajesovsku_analizu (linije 2005-2062)
  - Hibrid: pokreni_hibridnu_analizu (2225-2321) + izracunaj_matricu_povezanosti (2202-2223)
"""

import itertools

import pandas as pd

from . import konfig

KOLONE = konfig.KOLONE_ZA_BROJEVE
MAX_BROJ = konfig.MAX_BROJ
BROJEVA_U_KOMBINACIJI = konfig.BROJEVA_U_KOMBINACIJI
LEARNING_RATE = 0.005


def _izvuceni_po_kolima(loto_df):
    for _, kolo in loto_df.iterrows():
        brojevi = set(kolo[KOLONE].dropna().astype(int))
        if len(brojevi) == BROJEVA_U_KOMBINACIJI:
            yield brojevi


def frekvencija_rang(loto_df):
    """Rangiranje po ukupnoj frekvenciji pojavljivanja."""
    if loto_df.empty:
        return [{"broj": b, "skor": 0.0} for b in range(1, MAX_BROJ + 1)]
    svi = pd.concat([loto_df[c] for c in KOLONE]).dropna().astype(int)
    freq = svi.value_counts()
    rezultat = [{"broj": b, "skor": float(freq.get(b, 0))} for b in range(1, MAX_BROJ + 1)]
    rezultat.sort(key=lambda x: x["skor"], reverse=True)
    return rezultat


def bajes_verovanja(loto_df):
    """Iterativno Bajesovsko 'verovanje' po broju (deterministički, po kolu)."""
    verovanja = {b: 1.0 / MAX_BROJ for b in range(1, MAX_BROJ + 1)}
    df = loto_df.sort_values(by="kolo") if not loto_df.empty else loto_df
    for izvuceni in _izvuceni_po_kolima(df):
        for b in range(1, MAX_BROJ + 1):
            verovanja[b] *= (1 + LEARNING_RATE) if b in izvuceni else (1 - LEARNING_RATE)
        suma = sum(verovanja.values())
        if suma > 0:
            verovanja = {b: v / suma for b, v in verovanja.items()}
    return verovanja


def bajes_rang(loto_df):
    verovanja = bajes_verovanja(loto_df)
    rezultat = [{"broj": b, "skor": v} for b, v in verovanja.items()]
    rezultat.sort(key=lambda x: x["skor"], reverse=True)
    return rezultat


def matrica_povezanosti(loto_df):
    """39x39 matrica: koliko puta je par brojeva izvučen zajedno."""
    matrica = pd.DataFrame(0, index=range(1, MAX_BROJ + 1), columns=range(1, MAX_BROJ + 1))
    if loto_df.empty:
        return matrica
    for kombinacija in loto_df[KOLONE].dropna().astype(int).values:
        for a, b in itertools.combinations(kombinacija, 2):
            matrica.loc[a, b] += 1
            matrica.loc[b, a] += 1
    return matrica


def hibrid_rang(loto_df):
    """Fuzija: 80% Bajes + 20% normalizovan bonus povezanosti sa top-20 Bajes brojevima."""
    bajes = bajes_verovanja(loto_df)
    matrica = matrica_povezanosti(loto_df)

    top_20 = [b for b, _ in sorted(bajes.items(), key=lambda x: x[1], reverse=True)[:20]]
    max_moguci_bonus = matrica.sum().max() if not matrica.empty else 0

    rezultat = []
    for broj in range(1, MAX_BROJ + 1):
        bonus = sum(matrica.loc[broj, t] for t in top_20 if t != broj)
        norm_bonus = (bonus / max_moguci_bonus) if max_moguci_bonus > 0 else 0
        final = bajes.get(broj, 0) * 0.8 + norm_bonus * 0.2
        rezultat.append({
            "broj": broj,
            "skor": final,
            "bajes": bajes.get(broj, 0),
            "povezanost": int(bonus),
        })
    rezultat.sort(key=lambda x: x["skor"], reverse=True)
    return rezultat


def rangiraj(loto_df, metoda="frekvencija"):
    if metoda == "bajes":
        return bajes_rang(loto_df)
    if metoda == "hibrid":
        return hibrid_rang(loto_df)
    return frekvencija_rang(loto_df)
