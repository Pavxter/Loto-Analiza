"""FastAPI backend za Loto Analizator (web).

Servira statički frontend i izlaže REST API koji poziva core module.
Pokretanje:  python pokreni.py   (iz korena projekta)
"""

import io
import json
import os
import re

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp.core import (konfig, baza, analitika, rangiranje, generator, bektest,
                         prognoza, razlicitost, istorija, mapa,
                         razlicitost_teorija as teorija)

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")

app = FastAPI(title="Loto Analizator Web", version="1.0")

# Jednostavan keš analize po periodu (invalidira se pri promeni podataka)
_kes = {}


@app.on_event("startup")
def _startup():
    """Osigurava šemu (nove kolone) i doračunava preklapanje starih bektestova (§8)."""
    baza.postavi_bazu()
    conn = baza.konekcija()
    try:
        bektest.migriraj_preklapanje_bektesta(conn)
    finally:
        conn.close()


def _osvezi_df():
    conn = baza.konekcija()
    try:
        return analitika.ucitaj_df(conn)
    finally:
        conn.close()


def _analiza(period=0, granica=None):
    """Analiza za dati period; opciono samo nad kolima ≤ granica (Faza 5, vremeplov).

    granica=None → cela baza (podrazumevano, ponašanje nepromenjeno).
    """
    kljuc = ("analiza", period, granica, _kes.get("verzija", 0))
    if kljuc not in _kes:
        df = _osvezi_df()
        if granica is not None:
            df = df[df["kolo"] <= granica]
        _kes[kljuc] = analitika.Analiza(df, period_analize=period)
    return _kes[kljuc]


def _invalidiraj():
    _kes["verzija"] = _kes.get("verzija", 0) + 1
    for k in [k for k in _kes if isinstance(k, tuple)]:
        del _kes[k]


# ---------------------------------------------------------------------------
# Analiza / Dashboard / Statistika
# ---------------------------------------------------------------------------

@app.get("/api/dashboard")
def dashboard(period: int = 0):
    return _analiza(period).kao_dashboard()


@app.get("/api/statistika")
def statistika(period: int = 0):
    a = _analiza(period)
    podaci = a.kao_statistika()
    podaci["vremenska_serija"] = a.vremenska_serija()
    return podaci


@app.get("/api/hi-kvadrat")
def hi_kvadrat():
    return _analiza(0).hi_kvadrat_pozicija()


@app.get("/api/bazen")
def bazen(period: int = 0):
    """Liste vrućih/hladnih/svežih za kreator bazena (kao analiziraj_period_za_bazen)."""
    df = _osvezi_df()
    if period <= 0 or period > len(df):
        period = len(df)
    df_p = df.tail(period)
    if df_p.empty:
        return {"vruci": [], "hladni": [], "svezi": []}
    brojevi = df_p[konfig.KOLONE_ZA_BROJEVE].melt(value_name="broj")["broj"].dropna().astype(int)
    freq = brojevi.value_counts()
    vruci = [int(x) for x in freq.index.tolist()]
    svi = set(range(1, konfig.MAX_BROJ + 1))
    neizvuceni = sorted(svi - set(freq.index))
    najredji = freq.sort_values(ascending=True).index.tolist()
    hladni = list(dict.fromkeys([int(x) for x in (neizvuceni + najredji)]))
    posl_10 = df.tail(konfig.PERIOD_SVEZIH_KOLA)
    svezi_s = posl_10[konfig.KOLONE_ZA_BROJEVE].melt(value_name="broj")["broj"].dropna().astype(int)
    svezi = [int(x) for x in svezi_s.value_counts().index.tolist()]
    return {"vruci": vruci, "hladni": hladni, "svezi": svezi, "period": period}


# ---------------------------------------------------------------------------
# Rangiranje
# ---------------------------------------------------------------------------

@app.get("/api/rangiranje")
def api_rangiranje(metoda: str = "frekvencija"):
    df = _osvezi_df()
    return {"metoda": metoda, "rang": rangiranje.rangiraj(df, metoda)}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class GeneratorZahtev(BaseModel):
    period: int = 0
    bazen: list[int] | None = None
    filteri: dict = {}
    granica: int | None = None            # vremeplov: analiziraj samo do ovog kola (Faza 5)


@app.post("/api/generator")
def api_generator(z: GeneratorZahtev):
    a = _analiza(z.period, z.granica)
    izvor = z.bazen if z.bazen else None
    if izvor is not None and len(set(izvor)) < konfig.BROJEVA_U_KOMBINACIJI:
        raise HTTPException(400, f"Bazen mora imati bar {konfig.BROJEVA_U_KOMBINACIJI} brojeva.")
    rez = generator.generisi(a, bazen=izvor, filteri=z.filteri or {})
    # Ograniči prikaz da odgovor ne bude ogroman
    rez["kombinacije"] = rez["kombinacije"][:200]
    # Panel „Različitost seta" (§8) — mereno nad prikazanim setom
    rez["razlicitost"] = razlicitost.razlicitost_seta([k["brojevi"] for k in rez["kombinacije"]])
    rez["granica"] = z.granica
    rez["broj_kola"] = int(len(a.loto_df))
    return rez


# ---------------------------------------------------------------------------
# Tiketi
# ---------------------------------------------------------------------------

class TiketZahtev(BaseModel):
    kombinacija: str


@app.get("/api/tiketi")
def api_tiketi():
    conn = baza.konekcija()
    try:
        return baza.svi_tiketi(conn)
    finally:
        conn.close()


@app.post("/api/tiketi")
def api_dodaj_tiket(z: TiketZahtev):
    conn = baza.konekcija()
    try:
        nov = baza.dodaj_tiket(conn, z.kombinacija.strip())
        return {"dodato": nov}
    finally:
        conn.close()


@app.delete("/api/tiketi/{tiket_id}")
def api_obrisi_tiket(tiket_id: int):
    conn = baza.konekcija()
    try:
        baza.obrisi_tiket(conn, tiket_id)
        return {"ok": True}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Bektest
# ---------------------------------------------------------------------------

class BektestZahtev(BaseModel):
    kolo: int
    bazen: list[int] = []
    opis: str = ""
    tip: str = "lista"                     # 'lista' ili 'ceo_bazen'
    kombinacije: list[list[int]] | None = None


@app.get("/api/bektest")
def api_bektest():
    conn = baza.konekcija()
    try:
        return baza.svi_bektestovi(conn)
    finally:
        conn.close()


@app.post("/api/bektest")
def api_sacuvaj_bektest(z: BektestZahtev):
    conn = baza.konekcija()
    try:
        nov = baza.sacuvaj_bektest(conn, z.kolo, z.bazen, z.opis, z.tip, z.kombinacije)
        return {"sacuvano": nov}
    finally:
        conn.close()


@app.delete("/api/bektest/{bektest_id}")
def api_obrisi_bektest(bektest_id: int):
    conn = baza.konekcija()
    try:
        baza.obrisi_bektest(conn, bektest_id)
        return {"ok": True}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Prognoza (predviđanje jednog broja — statistički eksperiment)
# ---------------------------------------------------------------------------

@app.get("/api/prognoza/predlozi")
def api_prognoza_predlozi(period: int = 0):
    """Uživo predlozi za sledeće kolo; računa i zaključava ih ako ne postoje."""
    conn = baza.konekcija()
    try:
        return prognoza.generisi_uzivo(conn, period)
    finally:
        conn.close()


@app.post("/api/prognoza/preracunaj")
def api_prognoza_preracunaj(period: int = 0):
    """Eksplicitno preračunava neocenjene uživo predloge (npr. posle promene perioda)."""
    conn = baza.konekcija()
    try:
        return prognoza.preracunaj_uzivo(conn, period)
    finally:
        conn.close()


@app.get("/api/prognoza/rezultati")
def api_prognoza_rezultati(izvor: str = "uzivo"):
    """Statistika po metodu + serije za kumulativni grafikon (jedan izvor: uzivo|retro)."""
    if izvor not in ("uzivo", "retro"):
        raise HTTPException(400, "izvor mora biti 'uzivo' ili 'retro'")
    conn = baza.konekcija()
    try:
        return {"statistika": prognoza.statistika(conn, izvor),
                "grafikon": prognoza.serije(conn, izvor), "izvor": izvor}
    finally:
        conn.close()


@app.get("/api/prognoza/istorija")
def api_prognoza_istorija(izvor: str = "", metod: str = "", limit: int = 50):
    """Poslednje prognoze (podrazumevano 50), sa izvučenim brojevima ciljnog kola."""
    conn = baza.konekcija()
    try:
        redovi = baza.prognoze_lista(conn, izvor or None, metod or None, limit, samo_ocenjene=False)
        kola = {r["kolo"] for r in redovi}
        izvuceni = {}
        for k in kola:
            red = conn.execute(
                "SELECT b1,b2,b3,b4,b5,b6,b7 FROM istorijski_rezultati WHERE kolo=?", (k,)).fetchone()
            if red:
                izvuceni[k] = list(red)
        for r in redovi:
            r["izvuceni"] = izvuceni.get(r["kolo"])
        return redovi
    finally:
        conn.close()


@app.post("/api/prognoza/retro")
def api_prognoza_retro():
    """Pokreće retroaktivni bektest (jednobrojni + kombinacijski; briše stare retro redove)."""
    conn = baza.konekcija()
    try:
        return prognoza.retro_bektest(conn)
    finally:
        conn.close()


# --- Kombinacijske prognoze (PLAN_PROGNOZA_KOMBINACIJE §7) ---

@app.get("/api/prognoza/komb/predlozi")
def api_prognoza_komb_predlozi(period: int = 0):
    conn = baza.konekcija()
    try:
        return prognoza.generisi_uzivo_komb(conn, period)
    finally:
        conn.close()


@app.post("/api/prognoza/komb/preracunaj")
def api_prognoza_komb_preracunaj(period: int = 0):
    conn = baza.konekcija()
    try:
        return prognoza.preracunaj_uzivo_komb(conn, period)
    finally:
        conn.close()


@app.get("/api/prognoza/komb/rezultati")
def api_prognoza_komb_rezultati(izvor: str = "uzivo"):
    if izvor not in ("uzivo", "retro"):
        raise HTTPException(400, "izvor mora biti 'uzivo' ili 'retro'")
    conn = baza.konekcija()
    try:
        return {"statistika": prognoza.statistika_komb(conn, izvor),
                "grafikon": prognoza.serije_komb(conn, izvor), "izvor": izvor}
    finally:
        conn.close()


@app.get("/api/prognoza/komb/histogram")
def api_prognoza_komb_histogram(izvor: str = "uzivo", metod: str = ""):
    conn = baza.konekcija()
    try:
        return prognoza.histogram_komb(conn, izvor, metod or None)
    finally:
        conn.close()


@app.get("/api/prognoza/komb/istorija")
def api_prognoza_komb_istorija(izvor: str = "", metod: str = "", limit: int = 50):
    conn = baza.konekcija()
    try:
        redovi = baza.prognoze_lista(conn, izvor or None, metod or None, limit,
                                     samo_ocenjene=False, vrsta="komb")
        kola = {r["kolo"] for r in redovi}
        izvuceni = {}
        for k in kola:
            red = conn.execute(
                "SELECT b1,b2,b3,b4,b5,b6,b7 FROM istorijski_rezultati WHERE kolo=?", (k,)).fetchone()
            if red:
                izvuceni[k] = list(red)
        for r in redovi:
            r["izvuceni"] = izvuceni.get(r["kolo"])
            r["komb_lista"] = [int(x) for x in (r["kombinacija"] or "").split(",") if x.strip()]
        return redovi
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Različitost (analiza preklapanja kombinacija)
# ---------------------------------------------------------------------------

@app.get("/api/razlicitost")
def api_razlicitost(period: int = 0):
    """Sve analize različitosti (keširano po periodu; rekordi uvek cela istorija)."""
    kljuc = ("razlicitost", period, _kes.get("verzija", 0))
    if kljuc not in _kes:
        conn = baza.konekcija()
        try:
            _kes[kljuc] = razlicitost.sve_analize(conn, period)
        finally:
            conn.close()
    return _kes[kljuc]


@app.get("/api/razlicitost/profil")
def api_razlicitost_profil(period: int = 0, tip: str = "sredina"):
    """Samo Analiza 4 za dati profil (brza promena dropdown-a bez preračuna ostatka)."""
    conn = baza.konekcija()
    try:
        return razlicitost.analiza_profil(razlicitost.istorija_iz_conn(conn), period, tip)
    finally:
        conn.close()


@app.get("/api/razlicitost/par")
def api_razlicitost_par(a: int, b: int, period: int = 0):
    """Detalj ćelije toplotne mape ko-okurencije (par a,b)."""
    if not (1 <= a <= konfig.MAX_BROJ and 1 <= b <= konfig.MAX_BROJ) or a == b:
        raise HTTPException(400, "Neispravan par brojeva.")
    conn = baza.konekcija()
    try:
        return razlicitost.ko_okurencija_par(razlicitost.istorija_iz_conn(conn), a, b, period)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Istraži istoriju (vremeplov) — UPUTSTVO_PROGRAMER_ISTORIJA.md, Faza 1
# granica = poslednje poznato kolo (uključivo); cilj = prvo kolo posle nje.
# Svi endpointi traže eksplicitnu `granica` (bez fallbacka na celu bazu → 422).
# ---------------------------------------------------------------------------

@app.get("/api/istorija/granice")
def api_istorija_granice():
    """Meta bez analize: najstarije/najnovije kolo i ukupno (bootstrap taba)."""
    conn = baza.konekcija()
    try:
        return istorija.granice(conn)
    finally:
        conn.close()


@app.get("/api/istorija/kontekst")
def api_istorija_kontekst(granica: int, prozor: int = 0):
    """Sve za početni ekran u jednom pozivu: prozor kola, cilj, granica, navigacija."""
    conn = baza.konekcija()
    try:
        return istorija.kontekst(conn, granica, prozor)
    finally:
        conn.close()


@app.get("/api/istorija/kola")
def api_istorija_kola(granica: int, prozor: int = 0):
    """Samo lista kola ≤ granica (poslednjih prozor) — za osvežavanje tabele."""
    conn = baza.konekcija()
    try:
        return istorija.kola_do(conn, granica, prozor)
    finally:
        conn.close()


@app.get("/api/istorija/kolo/{kolo}")
def api_istorija_kolo(kolo: int):
    """Detalj jednog kola + prethodno/sledeće (za klik u tabeli)."""
    conn = baza.konekcija()
    try:
        d = istorija.detalj_kola(conn, kolo)
        if d is None:
            raise HTTPException(404, f"Kolo {kolo} ne postoji u bazi.")
        return d
    finally:
        conn.close()


@app.get("/api/istorija/broj/{broj}")
def api_istorija_broj(broj: int, granica: int, prozor: int = 0):
    """Istorija jednog broja na granici (pojavljivanja, razmaci, pozicije, timeline)."""
    if not (1 <= broj <= konfig.MAX_BROJ):
        raise HTTPException(400, f"Broj {broj} nije u opsegu 1-{konfig.MAX_BROJ}.")
    conn = baza.konekcija()
    try:
        return istorija.detalj_broja(conn, broj, granica, prozor)
    finally:
        conn.close()


@app.get("/api/istorija/razlicitost")
def api_istorija_razlicitost(cilj: int, prozor: int = 0):
    """Preklapanje izvučenog kola `cilj` sa istorijom pre njega (Faza 3)."""
    conn = baza.konekcija()
    try:
        d = istorija.razlicitost_cilja(conn, cilj, prozor)
        if d is None:
            raise HTTPException(404, f"Kolo {cilj} ne postoji u bazi.")
        return d
    finally:
        conn.close()


@app.get("/api/istorija/rangiranje")
def api_istorija_rangiranje(granica: int, prozor: int = 0):
    """Rangiranje brojeva (frekvencija/Bajes/hibrid) kakvo bi bilo na granici (Faza 3)."""
    conn = baza.konekcija()
    try:
        return istorija.rangiranje(conn, granica, prozor)
    finally:
        conn.close()


@app.get("/api/istorija/prognoza")
def api_istorija_prognoza(granica: int):
    """Vremeplov: šta bi sistem predvideo na granici, za cilj (Faza 4)."""
    conn = baza.konekcija()
    try:
        p = istorija.prognoza_u_tacki(conn, granica)
        if p is None:
            raise HTTPException(404, f"Nema kola ≤ {granica} u bazi.")
        return p
    finally:
        conn.close()


@app.get("/api/istorija/prognoza/ishod")
def api_istorija_prognoza_ishod(granica: int):
    """Vremeplov: prognoza na granici + stvarni ishod cilja + evaluacija (Faza 4)."""
    conn = baza.konekcija()
    try:
        p = istorija.prognoza_ishod(conn, granica)
        if p is None:
            raise HTTPException(404, f"Nema kola ≤ {granica} u bazi.")
        return p
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Istorija / unos kola / uvoz
# ---------------------------------------------------------------------------

class KoloZahtev(BaseModel):
    kolo: int
    datum: str
    brojevi: list[int]


def _validiraj_brojeve(brojevi):
    if len(brojevi) != konfig.BROJEVA_U_KOMBINACIJI:
        raise HTTPException(400, f"Potrebno je tačno {konfig.BROJEVA_U_KOMBINACIJI} brojeva.")
    if len(set(brojevi)) != len(brojevi):
        raise HTTPException(400, "Svi brojevi moraju biti jedinstveni.")
    for b in brojevi:
        if not (1 <= b <= konfig.MAX_BROJ):
            raise HTTPException(400, f"Broj {b} nije u opsegu 1-{konfig.MAX_BROJ}.")


@app.get("/api/istorija")
def api_istorija(limit: int = 50):
    conn = baza.konekcija()
    try:
        redovi = conn.execute(
            "SELECT * FROM istorijski_rezultati ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in redovi]
    finally:
        conn.close()


@app.post("/api/istorija")
def api_dodaj_kolo(z: KoloZahtev):
    _validiraj_brojeve(z.brojevi)
    conn = baza.konekcija()
    try:
        rezime = bektest.dodaj_kolo_i_proveri(conn, z.kolo, z.datum, z.brojevi)
    finally:
        conn.close()
    _invalidiraj()
    return rezime


@app.put("/api/istorija/{unos_id}")
def api_izmeni_kolo(unos_id: int, z: KoloZahtev):
    _validiraj_brojeve(z.brojevi)
    conn = baza.konekcija()
    try:
        baza.izmeni_kolo(conn, unos_id, z.kolo, z.datum, z.brojevi)
    finally:
        conn.close()
    _invalidiraj()
    return {"ok": True}


@app.delete("/api/istorija/{unos_id}")
def api_obrisi_kolo(unos_id: int):
    conn = baza.konekcija()
    try:
        baza.obrisi_kolo(conn, unos_id)
    finally:
        conn.close()
    _invalidiraj()
    return {"ok": True}


@app.post("/api/uvoz")
async def api_uvoz(fajl: UploadFile = File(...), zameni: bool = False):
    """Uvoz CSV/Excel: kolone kolo, datum, b1..b7.

    zameni=False -> dodaje kola postojećoj istoriji (duplikati se preskaču).
    zameni=True  -> prvo pravi backup, obriše SVU istoriju, pa uveze iz fajla (čist uvoz).
    """
    sadrzaj = await fajl.read()
    try:
        if fajl.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(sadrzaj))
        else:
            df = pd.read_csv(io.BytesIO(sadrzaj))
    except Exception as e:
        raise HTTPException(400, f"Ne mogu da pročitam fajl: {e}")

    obavezne = ["kolo", "datum"] + konfig.KOLONE_ZA_BROJEVE
    if not all(c in df.columns for c in obavezne):
        raise HTTPException(400, f"Fajl mora imati kolone: {', '.join(obavezne)}")

    # Validacija sadržaja PRE brisanja (da ne obrišemo istoriju zbog lošeg fajla)
    try:
        redovi = []
        for i, (_, red) in enumerate(df.iterrows(), start=2):  # 2 = prvi red posle zaglavlja
            brojevi = [int(red[c]) for c in konfig.KOLONE_ZA_BROJEVE]
            for b in brojevi:
                if not (1 <= b <= konfig.MAX_BROJ):
                    raise ValueError(f"red {i}: broj {b} van opsega 1-{konfig.MAX_BROJ}")
            if len(set(brojevi)) != konfig.BROJEVA_U_KOMBINACIJI:
                raise ValueError(f"red {i}: ponovljeni brojevi {brojevi}")
            redovi.append((int(red["kolo"]), str(red["datum"]), brojevi))
    except Exception as e:
        raise HTTPException(400, f"Neispravan sadržaj fajla: {e}")

    backup_putanja = None
    conn = baza.konekcija()
    obrisano = 0
    uvezeno = 0
    try:
        if zameni:
            backup_putanja = baza.napravi_backup()
            obrisano = baza.obrisi_svu_istoriju(conn)
            # retro prognoze postaju nevažeće nad novom istorijom; uživo se zadržavaju
            baza.obrisi_retro_prognoze(conn)
        for kolo, datum, brojevi in redovi:
            if baza.dodaj_kolo(conn, kolo, datum, brojevi):
                uvezeno += 1
    finally:
        conn.close()
    _invalidiraj()
    return {
        "uvezeno": uvezeno,
        "ukupno_u_fajlu": len(df),
        "zamenjeno": zameni,
        "obrisano": obrisano,
        "backup": os.path.basename(backup_putanja) if backup_putanja else None,
    }


# ---------------------------------------------------------------------------
# Mapa kombinacija (plan_mapa_kombinacija.md, Faza 2)
# Pločice su statika (/mapa/{sloj}/{z}/{x}/{y}.png), ovde je samo ono što se
# računa: šta je na ćeliji i gde je uneta kombinacija.
# ---------------------------------------------------------------------------

MAPA_DIR = os.path.join(STATIC_DIR, "mapa")


def _mapa_slojevi():
    """Slojevi za koje su pločice zaista generisane i slažu se sa rasporedom."""
    slojevi = []
    for naziv, osobina in mapa.OSOBINE.items():
        put = os.path.join(MAPA_DIR, naziv, "meta.json")
        if not os.path.isfile(put):
            continue
        try:
            with open(put, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if (meta.get("red_krive") != mapa.RED_KRIVE
                or meta.get("dimenzija") != mapa.DIMENZIJA):
            continue    # stare pločice za drugi raspored — ne nudi ih
        slojevi.append({"sloj": naziv, "opis": osobina["opis"], "tip": osobina["tip"],
                        "min": meta["min"], "max": meta["max"],
                        "generisano": meta.get("generisano")})
    return slojevi


def _mapa_detalj(conn, brojevi):
    """Detalj jedne kombinacije: mesto na mapi, osobine i odnos prema izvučenim."""
    d = mapa.detalj_kombinacije(brojevi)
    sve_izvuceno = razlicitost.istorija_iz_conn(conn)
    trazena = tuple(d["brojevi"])
    d["izvucena"] = [kolo for kolo, br in sve_izvuceno if tuple(sorted(br)) == trazena]
    d["preklapanje"] = (razlicitost.preklapanje_sa_istorijom(sve_izvuceno, d["brojevi"])
                        if sve_izvuceno else None)
    return d


@app.get("/api/mapa/info")
def api_mapa_info():
    """Raspored, dostupni slojevi i skala boja — bootstrap taba."""
    conn = baza.konekcija()
    try:
        broj_kola = conn.execute("SELECT COUNT(*) FROM istorijski_rezultati").fetchone()[0]
    finally:
        conn.close()
    return {
        "dimenzija": mapa.DIMENZIJA,
        "velicina_plocice": mapa.VELICINA_PLOCICE,
        "max_zoom": mapa.MAX_ZOOM,
        "ukupno_kombinacija": mapa.UKUPNO_KOMBINACIJA,
        "praznih_celija": mapa.DIMENZIJA * mapa.DIMENZIJA - mapa.UKUPNO_KOMBINACIJA,
        "broj_kola": int(broj_kola),
        "slojevi": _mapa_slojevi(),
        "skala_boja": ["#%02x%02x%02x" % boja for boja in mapa.SKALA_BOJA],
    }


def _tacke(rangovi, oznake):
    """Rangovi -> tačke sa ćelijom i preklapanjem sa prethodnom tačkom u nizu.

    Preklapanje se računa istom funkcijom kao svuda u projektu (teorija.preklapanje_brojeva),
    da stvarne i kontrolne tačke ne bi merile istu stvar na dva načina.
    """
    x, y = mapa.hilbert_xy(rangovi) if rangovi else ([], [])
    tacke, prethodni = [], None
    for i, r in enumerate(rangovi):
        brojevi = mapa.unrang(r)
        tacke.append({
            "kolo": oznake[i],
            "rang": int(r),
            "x": int(x[i]),
            "y": int(y[i]),
            "preklapanje_sa_prethodnim": (None if prethodni is None
                                          else teorija.preklapanje_brojeva(prethodni, brojevi)),
        })
        prethodni = brojevi
    return tacke


@app.get("/api/mapa/tacke")
def api_mapa_tacke(granica: int | None = None):
    """Izvučene kombinacije kao tačke na mapi, hronološki (granica=None → sve)."""
    conn = baza.konekcija()
    try:
        izvucena = razlicitost.istorija_iz_conn(conn)
    finally:
        conn.close()
    if granica is not None:
        izvucena = [(kolo, br) for kolo, br in izvucena if kolo <= granica]
    rangovi = [mapa.rang(br) for _kolo, br in izvucena]
    return {"granica": granica, "broj": len(rangovi),
            "tacke": _tacke(rangovi, [kolo for kolo, _br in izvucena])}


@app.get("/api/mapa/slucajno")
def api_mapa_slucajno(n: int | None = None, seed: int = mapa.SEED_KONTROLE):
    """Kontrolni sloj: isto toliko slučajnih kombinacija, isti oblik odgovora.

    Bez `n` uzima onoliko tačaka koliko ima izvučenih kola, da bi dve slike bile
    uporedive po broju tačaka, a ne samo po rasporedu.
    """
    if n is None:
        conn = baza.konekcija()
        try:
            n = conn.execute("SELECT COUNT(*) FROM istorijski_rezultati").fetchone()[0]
        finally:
            conn.close()
    if not (0 <= n <= 20000):
        raise HTTPException(400, "Broj tačaka mora biti između 0 i 20000.")
    rangovi = [int(r) for r in mapa.slucajni_rangovi(n, seed)]
    return {"seed": int(seed), "broj": len(rangovi),
            "tacke": _tacke(rangovi, [None] * len(rangovi))}


@app.get("/api/mapa/komb")
def api_mapa_komb(x: int, y: int):
    """Šta je na ćeliji (x, y): kombinacija, osobine i preklapanje sa izvučenim."""
    try:
        d = mapa.detalj_celije(x, y)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if d["brojevi"] is None:
        return d            # prazan deo krive: nema kombinacije na toj ćeliji
    conn = baza.konekcija()
    try:
        return _mapa_detalj(conn, d["brojevi"])
    finally:
        conn.close()


@app.get("/api/mapa/rang")
def api_mapa_rang(brojevi: str):
    """„Gde je moj tiket": 7 brojeva -> rang, ćelija na mapi i isti detalj."""
    delovi = [d for d in re.split(r"[^0-9]+", brojevi.strip()) if d]
    try:
        b = [int(d) for d in delovi]
    except ValueError:
        raise HTTPException(400, "Brojevi moraju biti celi brojevi.")
    conn = baza.konekcija()
    try:
        return _mapa_detalj(conn, b)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Statički frontend (mora biti poslednje da ne preuzme /api rute)
# ---------------------------------------------------------------------------

class _NoCacheStatic(StaticFiles):
    """StaticFiles sa 'Cache-Control: no-cache' — browser sme da kešira ali MORA
    da revalidira (uz ETag/Last-Modified: 304 ako nije menjano, inače nova verzija).
    Rešava „stara verzija posle update-a" bez ručnog hard refresh-a."""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        put = str(args[0]) if args else ""
        if put.endswith(".png") and os.path.join("static", "mapa") in put:
            # Pločice mape su nepromenljive dok se ne pokrene generisi_mapu.py;
            # bez ovoga bi svako pomeranje mape slalo stotine revalidacija.
            resp.headers["Cache-Control"] = "public, max-age=604800, immutable"
        else:
            resp.headers["Cache-Control"] = "no-cache"
        return resp


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"),
                        headers={"Cache-Control": "no-cache"})


app.mount("/", _NoCacheStatic(directory=STATIC_DIR), name="static")
