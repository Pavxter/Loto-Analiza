"""Bektest metrike i provera kola.

Verno portovano iz analiza.py:
  - izracunaj_promasaj_za_kombinaciju (1556-1568)
  - izracunaj_indeks_promasaja (1570-1592)
  - izracunaj_model_verovatnoce (1594-1622)
  - izracunaj_indeks_iznenadjenja (1624-1637)
  - proveri_i_dodaj_kolo (1639-1767) -> dodaj_kolo_i_proveri

Za razliku od originala, kombinacije se čitaju iz JSON-a ili se regenerišu iz bazena
(nema više eval() i nema ogromnih tekstualnih blobova).
"""

import itertools
import json
import math
import re

import pandas as pd

from . import baza, konfig

KOLONE = konfig.KOLONE_ZA_BROJEVE
MAX_BROJ = konfig.MAX_BROJ
BROJEVA_U_KOMBINACIJI = konfig.BROJEVA_U_KOMBINACIJI


def promasaj_kombinacije(kombinacija, dobitni_set):
    """Zbir najmanjih rastojanja svakog dobitnog broja do najbližeg u kombinaciji."""
    if not kombinacija or not dobitni_set:
        return None
    try:
        return sum(min(abs(d - b) for b in kombinacija) for d in dobitni_set)
    except Exception:
        return None


def model_verovatnoce(df_istorija, period_analize=0):
    """Verovatnoća po broju uz Laplace (add-one) smoothing."""
    if period_analize > 0 and period_analize <= len(df_istorija):
        analizirani = df_istorija.tail(period_analize)
    else:
        analizirani = df_istorija
    if analizirani.empty:
        v = 1 / MAX_BROJ
        return {b: v for b in range(1, MAX_BROJ + 1)}
    svi = pd.concat([analizirani[c] for c in KOLONE]).dropna().astype(int)
    freq = {b: 1 for b in range(1, MAX_BROJ + 1)}
    for b in svi:
        if b in freq:
            freq[b] += 1
    ukupno = sum(freq.values())
    return {b: f / ukupno for b, f in freq.items()}


def indeks_iznenadjenja(kombinacija, model):
    """-log verovatnoća kombinacije (veći = ređa/iznenađujuća)."""
    try:
        log_v = sum(math.log(model.get(b, 1e-9)) for b in kombinacija)
        return -1 * log_v
    except (ValueError, TypeError):
        return None


def _kombinacije_bektesta(red):
    """Vraća listu kombinacija (lista int-ova) za dati red virtualne_igre.

    Novi format: filter_podesavanja je JSON sa 'tip'. Stari format: lista_kombinacija
    je 'komb;komb' sa tuple stringovima -> parsiramo tolerantno radi kompatibilnosti.
    """
    lista_str = red.get("lista_kombinacija") or ""
    podesavanja = red.get("filter_podesavanja") or ""
    tip = None
    try:
        tip = json.loads(podesavanja).get("tip")
    except Exception:
        tip = None

    if tip == "ceo_bazen":
        bazen = _parsiraj_bazen(red.get("bazen_brojeva"))
        if len(bazen) < BROJEVA_U_KOMBINACIJI:
            return []
        return [list(k) for k in itertools.combinations(sorted(bazen), BROJEVA_U_KOMBINACIJI)]

    if tip == "lista" or lista_str.strip().startswith("["):
        try:
            return [list(map(int, k)) for k in json.loads(lista_str)]
        except Exception:
            pass

    # Stari format: "(1, 2, ...);(...)"
    rezultat = []
    for komb in lista_str.split(";"):
        komb = komb.strip()
        if not komb:
            continue
        try:
            rezultat.append([int(x) for x in re.findall(r"\d+", komb)])
        except Exception:
            continue
    return rezultat


def _parsiraj_bazen(bazen_str):
    if not bazen_str:
        return []
    try:
        return sorted({int(x.strip()) for x in bazen_str.split(",") if x.strip()})
    except Exception:
        return []


def _parsiraj_tiket(kombinacija_str):
    """Parsira string tiketa uz uklanjanje prefiksa tipa (ML)/(GEN)/(POOL)."""
    bez_prefiksa = re.sub(r"^\(\w+\)", "", kombinacija_str or "")
    return [int(x) for x in re.findall(r"\d+", bez_prefiksa)]


def dodaj_kolo_i_proveri(conn, kolo, datum, brojevi):
    """Dodaje kolo i automatski proverava sve tikete i bektestove za to kolo.

    Vraća dict sa rezimeom. Replicira proveri_i_dodaj_kolo iz originala.
    """
    dobitni_lista = list(brojevi)
    dobitni_set = set(dobitni_lista)

    uspeh = baza.dodaj_kolo(conn, kolo, datum, dobitni_lista)

    loto_df = pd.read_sql_query("SELECT * FROM istorijski_rezultati ORDER BY id ASC", conn)
    istorija_pre = loto_df[loto_df["kolo"] < kolo]
    model = model_verovatnoce(istorija_pre, 0)

    # 1) Provera tiketa
    provereno_tiketa = 0
    for t in baza.svi_tiketi(conn):
        try:
            tiket_lista = _parsiraj_tiket(t["kombinacija"])
            if not tiket_lista:
                continue
            promasaj = promasaj_kombinacije(tiket_lista, dobitni_set)
            iznen = indeks_iznenadjenja(tiket_lista, model)
            metrike = (json.dumps({"promasaj": promasaj, "iznenadjenje": iznen})
                       if promasaj is not None and iznen is not None else None)
            pogodaka = len(dobitni_set & set(tiket_lista))
            conn.execute(
                "UPDATE odigrani_tiketi SET poslednji_rezultat=?, datum_provere=?, dodatne_metrike=? WHERE id=?",
                (pogodaka, datum, metrike, t["id"]))
            provereno_tiketa += 1
        except Exception:
            continue
    conn.commit()

    # 2) Provera bektestova za ovo kolo
    bektestovi = conn.execute("SELECT * FROM virtualne_igre WHERE kolo=?", (kolo,)).fetchall()
    provereno_bektestova = 0
    for red in [dict(r) for r in bektestovi]:
        kombinacije = _kombinacije_bektesta(red)

        # a) uspešnost bazena
        bazen_rez = ""
        bazen = _parsiraj_bazen(red.get("bazen_brojeva"))
        if bazen:
            pog = len(dobitni_set & set(bazen))
            bazen_rez = f"Bazen: {pog}/{len(bazen)} | "

        # b) uspešnost kombinacija
        pogoci = {7: 0, 6: 0, 5: 0, 4: 0, 3: 0}
        for komb in kombinacije:
            p = len(dobitni_set & set(komb))
            if p in pogoci:
                pogoci[p] += 1
        komb_rez = f"Komb: 7:{pogoci[7]}, 6:{pogoci[6]}, 5:{pogoci[5]}, 4:{pogoci[4]}"
        rezultat = bazen_rez + komb_rez

        # c) indeks promašaja (najmanji u setu)
        ip = None
        for komb in kombinacije:
            p = promasaj_kombinacije(komb, dobitni_set)
            if p is not None:
                ip = p if ip is None else min(ip, p)

        # d) indeks iznenađenja (najmanji u setu, model iz perioda)
        period = 0
        try:
            m = re.search(r"Period: Posl\. (\d+)", red.get("filter_podesavanja") or "")
            period = int(m.group(1)) if m else 0
        except Exception:
            period = 0
        model_b = model_verovatnoce(istorija_pre, period)
        iznen_min = None
        for komb in kombinacije:
            v = indeks_iznenadjenja(komb, model_b)
            if v is not None:
                iznen_min = v if iznen_min is None else min(iznen_min, v)

        baza.azuriraj_rezultat_bektesta(conn, red["id"], rezultat, ip, iznen_min)
        provereno_bektestova += 1

    return {
        "dodato": uspeh,
        "kolo": kolo,
        "provereno_tiketa": provereno_tiketa,
        "provereno_bektestova": provereno_bektestova,
    }
