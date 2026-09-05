"""Priprema sirovog CSV-a sa rezultatima za uvoz u Loto Analizator.

Sirovi fajlovi (npr. Lotto.csv) dolaze u formatu koji se razlikuje od onoga što
aplikacija očekuje:

    sirovo:     BrKola, datum (DD.MM.YYYY), br1..br7   — BrKola se resetuje svake godine
    aplikacija: kolo,   datum (YYYY-MM-DD), b1..b7     — kolo mora biti jedinstveno i rastuće

Skripta radi konverziju, ispravlja poznate greške i validira rezultat.

Numeracija kola:  kolo = godina * 1000 + BrKola   (npr. 56/2026 -> 2026056)
  - jedinstveno je, hronološki raste i čuva zvanični broj kola.

Upotreba:
    python pripremi_csv.py "C:\\Users\\pavks\\Desktop\\Lotto.csv"
    python pripremi_csv.py ulaz.csv --izlaz spremno.csv
"""

import argparse
import os
import re
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MAX_BROJ = 39
BROJEVA = 7

# Ručne ispravke poznatih grešaka u izvornim podacima: (datum, BrKola) -> ispravni brojevi
ISPRAVKE_BROJEVA = {
    ("2024-11-26", 94): [23, 16, 6, 39, 32, 12, 34],
}


def popravi_datum(d):
    """DD.MM.YYYY -> YYYY-MM-DD.

    Ispravlja i godine kojima fali cifra: kod '226' je ispuštena nula na drugom
    mestu (2_26), pa je ispravno 2026. Rezultat se proverava na razuman opseg.
    """
    d = str(d).strip()
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{3,4})$", d)
    if not m:
        return None
    dan, mesec, god = m.groups()
    if len(god) == 3:               # '226' -> '2026', '213' -> '2013'
        god = god[0] + "0" + god[1:]
    g = int(god)
    if not (2000 <= g <= 2100):     # zaštita od besmislenih godina
        return None
    return f"{g:04d}-{int(mesec):02d}-{int(dan):02d}"


def main():
    p = argparse.ArgumentParser(description="Priprema CSV za uvoz u Loto Analizator")
    p.add_argument("ulaz", help="Putanja do sirovog CSV fajla")
    p.add_argument("--izlaz", default=None, help="Izlazni CSV (podrazumevano <ulaz>_za_uvoz.csv)")
    args = p.parse_args()

    if not os.path.exists(args.ulaz):
        print(f"GREŠKA: fajl ne postoji: {args.ulaz}")
        return 1

    izlaz = args.izlaz or os.path.splitext(args.ulaz)[0] + "_za_uvoz.csv"
    df = pd.read_csv(args.ulaz)
    print(f"Učitano redova: {len(df)}  |  kolone: {list(df.columns)}")

    # 1) Mapiranje kolona
    mapa = {"BrKola": "kolo", "brKola": "kolo"}
    for i in range(1, BROJEVA + 1):
        mapa[f"br{i}"] = f"b{i}"
    df = df.rename(columns=mapa)

    obavezne = ["kolo", "datum"] + [f"b{i}" for i in range(1, BROJEVA + 1)]
    nedostaju = [c for c in obavezne if c not in df.columns]
    if nedostaju:
        print(f"GREŠKA: nedostaju kolone posle mapiranja: {nedostaju}")
        return 1

    # 2) Datumi
    df["datum_novi"] = df["datum"].apply(popravi_datum)
    losi = df[df["datum_novi"].isna()]
    if len(losi):
        print(f"GREŠKA: {len(losi)} redova sa neprepoznatim datumom:")
        print(losi[["kolo", "datum"]].to_string())
        return 1
    popravljeni = df[df["datum"].astype(str).str.match(r"^\d{1,2}\.\d{1,2}\.\d{3}$")]
    if len(popravljeni):
        print(f"Ispravljeno skraćenih godina: {len(popravljeni)}")
        for _, r in popravljeni.iterrows():
            print(f"   {r['datum']}  ->  {r['datum_novi']}")
    df["datum"] = df["datum_novi"]
    df = df.drop(columns=["datum_novi"])

    kolone_b = [f"b{i}" for i in range(1, BROJEVA + 1)]

    # 3) Ručne ispravke brojeva
    for (datum, brkola), tacni in ISPRAVKE_BROJEVA.items():
        maska = (df["datum"] == datum) & (df["kolo"] == brkola)
        if maska.any():
            staro = df.loc[maska, kolone_b].values[0].tolist()
            df.loc[maska, kolone_b] = tacni
            print(f"Ispravljeni brojevi za {datum} (kolo {brkola}): {staro} -> {tacni}")

    # 4) Globalno jedinstven broj kola: godina*1000 + BrKola
    godina = df["datum"].str[:4].astype(int)
    df["kolo"] = godina * 1000 + df["kolo"].astype(int)

    # 5) Hronološko sortiranje
    df = df.sort_values(by=["datum", "kolo"]).reset_index(drop=True)

    # 6) Validacija
    greske = []
    for idx, r in df.iterrows():
        brojevi = [int(r[c]) for c in kolone_b]
        if any(not (1 <= b <= MAX_BROJ) for b in brojevi):
            greske.append(f"  red {idx}: {r['datum']} broj van opsega 1-{MAX_BROJ}: {brojevi}")
        if len(set(brojevi)) != BROJEVA:
            greske.append(f"  red {idx}: {r['datum']} ponovljeni brojevi: {brojevi}")
    dupli = df[df["kolo"].duplicated(keep=False)]
    if len(dupli):
        greske.append(f"  ponovljen broj kola kod {len(dupli)} redova: {sorted(dupli['kolo'].unique())[:10]}")

    if greske:
        print(f"\nNEISPRAVNI REDOVI ({len(greske)}) — ispravi ih pa ponovi:")
        for g in greske:
            print(g)
        return 1

    # 7) Upis
    df[obavezne].to_csv(izlaz, index=False, encoding="utf-8")
    sortiranih = sum(1 for _, r in df.iterrows() if list(r[kolone_b]) == sorted(r[kolone_b]))
    print("\n--- Sve provere prošle ---")
    print(f"Kola: {len(df)}   period: {df['datum'].iloc[0]} .. {df['datum'].iloc[-1]}")
    print(f"Broj kola: {df['kolo'].iloc[0]} .. {df['kolo'].iloc[-1]} (svi jedinstveni)")
    print(f"Sortiranih kola: {sortiranih} ({100*sortiranih/len(df):.2f}%) -> "
          f"{'redosled izvlačenja (dobro)' if sortiranih/len(df) < 0.05 else 'PAŽNJA: deluje sortirano'}")
    print(f"\nSpreman fajl za uvoz: {izlaz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
