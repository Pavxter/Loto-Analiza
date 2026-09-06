"""Jednokratno generisanje pločica za stranu „Mapa kombinacija"
(plan_mapa_kombinacija.md, §2.4 i Faza 1).

Pokretanje iz korena projekta:

    python -X utf8 generisi_mapu.py                 # sloj „zbir"
    python -X utf8 generisi_mapu.py --sloj sve      # svi implementirani slojevi
    python -X utf8 generisi_mapu.py --sloj parni,dekade
    python -X utf8 generisi_mapu.py --sloj ocena    # skor Generatora (čita bazu)

Šta radi: za svaku od 15.380.937 kombinacija računa izabranu osobinu, upisuje je
u ćeliju koju Hilbertova kriva dodeljuje njenom rangu i peče PNG pločice 256x256
za zumove 0..4 u `webapp/static/mapa/{sloj}/{z}/{x}/{y}.png`.

Na zumu 4 jedan piksel je jedna kombinacija; na nižim zumovima piksel je prosek
osobine za blok kombinacija. Prazne ćelije (rang >= 15.380.937) su providne.

Sloj `ocena` je jedini koji čita bazu: skor Generatora zavisi od svežih brojeva i
ritma, pa se peče za stanje baze u trenutku pokretanja. To kolo se upisuje u
meta.json i piše u aplikaciji; nove pločice se prave tek kad se skripta ponovi.

Skripta nije deo servera — pokreće se posle instalacije ili kad se promeni
raspored (RED_KRIVE / DIMENZIJA u webapp/core/mapa.py).
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from webapp.core import mapa  # noqa: E402

PODRAZUMEVANI_IZLAZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "webapp", "static", "mapa")
KOMAD = 2_000_000            # koliko rangova odjednom ide kroz Hilberta

# Skala je u webapp/core/mapa.py da bi legenda u browseru i ispečene boje bile
# iz istog izvora. Indeks 0 palete je rezervisan za prazne ćelije i providan je,
# pa boje idu na indekse 1..255.
KONTROLNE_BOJE = mapa.SKALA_BOJA


def paleta():
    """768 bajtova palete: indeks 0 = prazno, 1..255 = sekvencijalna skala."""
    t = np.linspace(0, len(KONTROLNE_BOJE) - 1, 255)
    donji = np.floor(t).astype(int)
    gornji = np.minimum(donji + 1, len(KONTROLNE_BOJE) - 1)
    udeo = (t - donji)[:, None]
    boje = np.array(KONTROLNE_BOJE, dtype=np.float64)
    skala = boje[donji] * (1 - udeo) + boje[gornji] * udeo
    return [0, 0, 0] + np.round(skala).astype(int).ravel().tolist()


def parametri_ocene():
    """Parametri skora Generatora za trenutno stanje baze (samo za sloj `ocena`).

    Vraća (parametri, opis_stanja). Vektorizacija se odmah proverava na uzorku
    protiv `generator.izracunaj_skor`: ako se ikad raziđu, pločice ne smeju nastati.
    """
    from webapp.core import analitika, baza, generator

    conn = baza.konekcija()
    try:
        df = analitika.ucitaj_df(conn)
    finally:
        conn.close()
    if df.empty:
        raise SystemExit("Baza je prazna — sloj „ocena“ nema od čega da se izračuna.")

    analiza = analitika.Analiza(df, period_analize=0)
    parametri = generator.parametri_skora(analiza)
    granica = int(df["kolo"].iloc[-1])

    uzorak = [tuple(int(b) for b in mapa.unrang(int(r)))
              for r in mapa.slucajni_rangovi(300, seed=1)]
    vektorski = mapa.ocena_niz(np.array(uzorak, dtype=np.uint8), parametri)
    for komb, dobijeno in zip(uzorak, vektorski):
        ocekivano = generator.izracunaj_skor(komb, analiza)
        if abs(float(dobijeno) - ocekivano) >= 0.01:
            raise SystemExit(f"Vektorski skor se ne slaže sa Generatorom za {komb}: "
                             f"{dobijeno} umesto {ocekivano}. Pločice nisu napravljene.")

    stanje = {
        "granica": granica,
        "broj_kola": int(len(df)),
        "strategija_svezine": parametri["strategija_svezine"],
        # parametri idu u meta.json da bi pločice bile samodovoljne: test može da
        # proveri piksel ocene bez baze, isto kao za ostale slojeve
        "parametri": parametri,
    }
    print(f"  parametri ocene: prosek {parametri['prosek']:.2f}, "
          f"std {parametri['std']:.2f}, baza do kola {granica} "
          f"({len(df)} kola); uzorak od 300 se poklapa sa izracunaj_skor")
    return parametri, stanje


def vrednosti_osobine(naziv, kombinacije, parametri):
    """Vrednost osobine za blok kombinacija, kao float32."""
    if naziv == "ocena":
        return mapa.ocena_niz(kombinacije, parametri)
    return mapa.osobina_niz(naziv, kombinacije).astype(np.float32)


def mreza_osobine(naziv, parametri=None):
    """Mreža 4096x4096 sa vrednošću osobine po ćeliji (NaN = prazna ćelija)."""
    t0 = time.time()
    kombinacije = mapa.sve_kombinacije()
    # u komadima: ocena pravi float međurezultate, pa bi ceo prostor odjednom
    # tražio skoro gigabajt bez ikakve potrebe
    vrednosti = np.empty(mapa.UKUPNO_KOMBINACIJA, dtype=np.float32)
    for pocetak in range(0, mapa.UKUPNO_KOMBINACIJA, KOMAD):
        kraj = min(pocetak + KOMAD, mapa.UKUPNO_KOMBINACIJA)
        vrednosti[pocetak:kraj] = vrednosti_osobine(naziv, kombinacije[pocetak:kraj], parametri)
    del kombinacije
    print(f"  osobina „{naziv}“ izračunata za {len(vrednosti):,} kombinacija "
          f"({time.time() - t0:.1f}s)".replace(",", "."))

    t0 = time.time()
    mreza = np.full(mapa.DIMENZIJA * mapa.DIMENZIJA, np.nan, dtype=np.float32)
    for pocetak in range(0, mapa.UKUPNO_KOMBINACIJA, KOMAD):
        kraj = min(pocetak + KOMAD, mapa.UKUPNO_KOMBINACIJA)
        x, y = mapa.hilbert_xy(np.arange(pocetak, kraj, dtype=np.int64))
        mreza[y * mapa.DIMENZIJA + x] = vrednosti[pocetak:kraj]
    print(f"  raspored po Hilbertovoj krivoj ({time.time() - t0:.1f}s)")
    return mreza.reshape(mapa.DIMENZIJA, mapa.DIMENZIJA)


def smanji(mreza):
    """Upola manja mreža: prosek 2x2 bloka, prazne ćelije se ne broje."""
    v = mreza.reshape(mreza.shape[0] // 2, 2, mreza.shape[1] // 2, 2)
    ima = ~np.isnan(v)
    zbir = np.where(ima, v, 0).sum(axis=(1, 3))
    koliko = ima.sum(axis=(1, 3))
    return np.where(koliko > 0, zbir / np.maximum(koliko, 1), np.nan).astype(np.float32)


def u_indekse(mreza, vmin, vmax):
    """Vrednosti -> indeksi palete (0 = prazno, 1..255 = skala vmin..vmax)."""
    idx = np.zeros(mreza.shape, dtype=np.uint8)
    ima = ~np.isnan(mreza)
    raspon = float(vmax - vmin) or 1.0
    udeo = np.clip((mreza[ima] - vmin) / raspon, 0.0, 1.0)
    idx[ima] = 1 + np.round(udeo * 254).astype(np.uint8)
    return idx


def upisi_zoom(mreza, vmin, vmax, koren, z, pal):
    """Peče sve pločice jednog zuma; vraća (broj_plocica, bajtova)."""
    n = mreza.shape[0] // mapa.VELICINA_PLOCICE
    idx = u_indekse(mreza, vmin, vmax)
    plocica, bajtova = 0, 0
    for x in range(n):
        folder = os.path.join(koren, str(z), str(x))
        os.makedirs(folder, exist_ok=True)
        for y in range(n):
            isecak = idx[y * mapa.VELICINA_PLOCICE:(y + 1) * mapa.VELICINA_PLOCICE,
                         x * mapa.VELICINA_PLOCICE:(x + 1) * mapa.VELICINA_PLOCICE]
            slika = Image.frombytes("P", isecak.shape[::-1],
                                    np.ascontiguousarray(isecak).tobytes())
            slika.putpalette(pal)
            putanja = os.path.join(folder, f"{y}.png")
            slika.save(putanja, optimize=True, transparency=0)
            plocica += 1
            bajtova += os.path.getsize(putanja)
    return plocica, bajtova


def generisi_sloj(naziv, izlaz):
    """Generiše sve pločice jednog sloja i njegov meta.json."""
    print(f"\nSloj „{naziv}“ — {mapa.OSOBINE[naziv]['opis']}")
    koren = os.path.join(izlaz, naziv)

    t0 = time.time()
    parametri, stanje = (parametri_ocene() if naziv == "ocena" else (None, {}))

    if os.path.isdir(koren):
        shutil.rmtree(koren)
    os.makedirs(koren, exist_ok=True)

    mreza = mreza_osobine(naziv, parametri)
    opseg = mapa.OSOBINE[naziv]["opseg"]
    if opseg is None:                       # `ocena`: opseg se meri, ne zna se unapred
        vmin = float(np.nanmin(mreza))
        vmax = float(np.nanmax(mreza))
        print(f"  izmeren opseg ocene: {vmin:.2f} .. {vmax:.2f}")
    else:
        vmin, vmax = opseg
    pal = paleta()

    ukupno_plocica, ukupno_bajtova = 0, 0
    for z in range(mapa.MAX_ZOOM, -1, -1):
        p, b = upisi_zoom(mreza, vmin, vmax, koren, z, pal)
        print(f"  zum {z}: {p} pločica, {b / 1024 / 1024:.1f} MB")
        ukupno_plocica += p
        ukupno_bajtova += b
        if z > 0:
            mreza = smanji(mreza)

    meta = {
        "sloj": naziv,
        "opis": mapa.OSOBINE[naziv]["opis"],
        "tip": mapa.OSOBINE[naziv]["tip"],
        "min": vmin,
        "max": vmax,
        "red_krive": mapa.RED_KRIVE,
        "dimenzija": mapa.DIMENZIJA,
        "velicina_plocice": mapa.VELICINA_PLOCICE,
        "max_zoom": mapa.MAX_ZOOM,
        "broj_kombinacija": mapa.UKUPNO_KOMBINACIJA,
        "broj_plocica": ukupno_plocica,
        "bajtova": ukupno_bajtova,
        "generisano": datetime.now().isoformat(timespec="seconds"),
        **stanje,
    }
    with open(os.path.join(koren, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  UKUPNO: {ukupno_plocica} pločica, {ukupno_bajtova / 1024 / 1024:.1f} MB, "
          f"{time.time() - t0:.1f}s")
    return meta


def main():
    p = argparse.ArgumentParser(description="Generisanje pločica za Mapu kombinacija")
    p.add_argument("--sloj", default="zbir",
                   help="naziv sloja, više naziva odvojeno zarezom, ili „sve“")
    p.add_argument("--izlaz", default=PODRAZUMEVANI_IZLAZ, help="izlazni folder")
    a = p.parse_args()

    slojevi = list(mapa.OSOBINE) if a.sloj == "sve" else [s.strip() for s in a.sloj.split(",")]
    nepoznati = [s for s in slojevi if s not in mapa.OSOBINE]
    if nepoznati:
        p.error(f"nepoznat sloj: {', '.join(nepoznati)}; dostupno: {', '.join(mapa.OSOBINE)}")

    print(f"Prostor: {mapa.UKUPNO_KOMBINACIJA:,} kombinacija na mreži "
          f"{mapa.DIMENZIJA}x{mapa.DIMENZIJA}".replace(",", "."))
    t0 = time.time()
    ukupno = sum(generisi_sloj(s, a.izlaz)["bajtova"] for s in slojevi)
    print(f"\nGotovo za {time.time() - t0:.1f}s; ukupno {ukupno / 1024 / 1024:.1f} MB "
          f"u {a.izlaz}")


if __name__ == "__main__":
    main()
