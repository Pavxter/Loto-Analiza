"""Statistička analiza istorijskih rezultata.

Verno portovano iz analiza.py -> ucitaj_i_analiziraj_podatke (linije 302-348) i
pratećih metoda, ali kao čiste funkcije nad pandas DataFrame-om.
"""

import pandas as pd

from . import konfig

KOLONE = konfig.KOLONE_ZA_BROJEVE
MAX_BROJ = konfig.MAX_BROJ
BROJEVA_U_KOMBINACIJI = konfig.BROJEVA_U_KOMBINACIJI
BROJ_KATEGORIJA_FREKV = konfig.BROJ_KATEGORIJA_FREKV
PERIOD_SVEZIH_KOLA = konfig.PERIOD_SVEZIH_KOLA


def ucitaj_df(conn):
    """Učitava sve istorijske rezultate kao DataFrame (id ASC)."""
    return pd.read_sql_query("SELECT * FROM istorijski_rezultati ORDER BY id ASC", conn)


class Analiza:
    """Nosi rezultate statističke analize za dati period.

    Atributi odgovaraju istoimenim poljima u originalnoj klasi LotoAnalizator,
    tako da su formule 1:1 iste.
    """

    def __init__(self, loto_df, period_analize=0):
        self.loto_df = loto_df.reset_index(drop=True)
        self.period_analize = period_analize
        self._izracunaj()

    def _izracunaj(self):
        df = self.loto_df
        sve_komb = df[KOLONE].dropna().astype(int)
        self.set_istorijskih_kombinacija = {tuple(sorted(row)) for row in sve_komb.values}

        n = len(df)
        if self.period_analize > 0 and self.period_analize <= n:
            analizirani = df.tail(self.period_analize)
            self.naslov_sufiks = f"(Poslednjih {self.period_analize} kola)"
        else:
            analizirani = df
            self.naslov_sufiks = f"(Sva Kola - {n})"

        self.brojevi_po_kolima = analizirani[KOLONE].dropna().astype(int)
        self.srednje_vrednosti = self.brojevi_po_kolima.mean(axis=1)
        self.globalni_prosek = float(self.srednje_vrednosti.mean()) if not self.srednje_vrednosti.empty else 0.0
        self.globalna_std_dev = float(self.srednje_vrednosti.std()) if len(self.srednje_vrednosti) > 1 else 0.0

        # Par/nepar odnos
        if self.brojevi_po_kolima.empty:
            self.najcesci_par_nepar = None
        else:
            pn = self.brojevi_po_kolima.apply(
                lambda row: tuple(sorted([sum(1 for x in row if x % 2 == 0),
                                          sum(1 for x in row if x % 2 != 0)])), axis=1).value_counts()
            self.najcesci_par_nepar = tuple(pn.index[0]) if not pn.empty else None

        # Frekvencija i kategorije
        svi_izvuceni = pd.concat([self.brojevi_po_kolima[c] for c in self.brojevi_po_kolima]) \
            if not self.brojevi_po_kolima.empty else pd.Series(dtype=int)
        self.frekvencija = svi_izvuceni.value_counts()
        sortirani = self.frekvencija.sort_values(ascending=False)
        self.vruci_brojevi = set(sortirani.head(BROJ_KATEGORIJA_FREKV).index)
        self.hladni_brojevi = set(sortirani.tail(BROJ_KATEGORIJA_FREKV).index)
        self.neutralni_brojevi = set(range(1, MAX_BROJ + 1)) - self.vruci_brojevi - self.hladni_brojevi

        poslednjih_10 = analizirani.tail(PERIOD_SVEZIH_KOLA)
        self.svezi_brojevi = set(pd.concat([poslednjih_10[c] for c in KOLONE]).dropna().astype(int).unique())

        # Ritam ponavljanja (prosečan razmak između pojavljivanja broja)
        ritam = {}
        for broj in range(1, MAX_BROJ + 1):
            kola_sa_brojem = analizirani[analizirani[KOLONE].eq(broj).any(axis=1)]["id"]
            ritam[broj] = kola_sa_brojem.diff().dropna().mean() if len(kola_sa_brojem) > 1 else 0
        self.analiza_ponavljanja = pd.Series(ritam)

        # Uzastopni brojevi
        self.analiza_uzastopnih = pd.Series([
            sum(1 for i in range(len(k) - 1) if k[i + 1] == k[i] + 1)
            for k in [sorted(list(red)) for _, red in self.brojevi_po_kolima.iterrows()]
        ]).value_counts().sort_index()

        # Dekade
        self.analiza_dekada = pd.DataFrame([{
            "1-9": sum(1 for b in red if 1 <= b <= 9),
            "10-19": sum(1 for b in red if 10 <= b <= 19),
            "20-29": sum(1 for b in red if 20 <= b <= 29),
            "30-39": sum(1 for b in red if 30 <= b <= 39),
        } for _, red in self.brojevi_po_kolima.iterrows()]).mean() if not self.brojevi_po_kolima.empty \
            else pd.Series({"1-9": 0, "10-19": 0, "20-29": 0, "30-39": 0})

        # Poziciona frekvencija
        self.poziciona_frekvencija = pd.DataFrame(
            0, index=range(1, MAX_BROJ + 1),
            columns=[f"poz_{i}" for i in range(1, BROJEVA_U_KOMBINACIJI + 1)])
        for i, col in enumerate(KOLONE, 1):
            counts = analizirani[col].dropna().astype(int).value_counts()
            if not counts.empty:
                self.poziciona_frekvencija.loc[counts.index, f"poz_{i}"] = counts
        self.pozicioni_prosek = analizirani[KOLONE].mean()

        # Model pristrasnosti: stvarna / očekivana frekvencija po (broj, pozicija)
        self.model_pristrasnosti = {}
        broj_kola = len(analizirani)
        if broj_kola > 0:
            ocekivana = broj_kola / MAX_BROJ
            if ocekivana > 0:
                for poz_idx, poz_col in enumerate(self.poziciona_frekvencija.columns, 1):
                    for broj, stvarna in self.poziciona_frekvencija[poz_col].items():
                        self.model_pristrasnosti[(broj, poz_idx)] = stvarna / ocekivana

    # --- JSON izlazi za API ---

    def kao_dashboard(self):
        vruci = sorted(int(x) for x in self.vruci_brojevi)
        hladni = sorted(int(x) for x in self.hladni_brojevi)
        return {
            "naslov_sufiks": self.naslov_sufiks,
            "broj_kola": int(len(self.loto_df)),
            "globalni_prosek": round(self.globalni_prosek, 2),
            "globalna_std_dev": round(self.globalna_std_dev, 2),
            "par_nepar": (f"{self.najcesci_par_nepar[0]} parna / {self.najcesci_par_nepar[1]} neparna"
                          if self.najcesci_par_nepar else "N/A"),
            "vruci": vruci,
            "hladni": hladni,
            "svezi": sorted(int(x) for x in self.svezi_brojevi),
            "predlog_bazena": self.predlog_bazena(),
        }

    def predlog_bazena(self):
        """Fuzija top-12 vrućih i top-12 svežih brojeva (kao dashboard u originalu)."""
        sortirani_vruci = self.frekvencija.sort_values(ascending=False).index.tolist()
        top_vruci = sortirani_vruci[:12]
        poslednjih_10 = self.brojevi_po_kolima.tail(PERIOD_SVEZIH_KOLA)
        if poslednjih_10.empty:
            top_svezi = []
        else:
            svezi = poslednjih_10.melt(value_name="broj")["broj"].dropna().astype(int)
            top_svezi = svezi.value_counts().index.tolist()[:12]
        return sorted(int(x) for x in set(top_vruci + top_svezi))

    def kao_frekvencija(self):
        """Lista brojeva sa frekvencijom i kategorijom, sortirano po broju."""
        rezultat = []
        for broj in range(1, MAX_BROJ + 1):
            if broj in self.vruci_brojevi:
                kat = "vruc"
            elif broj in self.hladni_brojevi:
                kat = "hladan"
            else:
                kat = "neutralan"
            rezultat.append({
                "broj": broj,
                "frekvencija": int(self.frekvencija.get(broj, 0)),
                "kategorija": kat,
                "svez": broj in self.svezi_brojevi,
            })
        return rezultat

    def kao_statistika(self):
        return {
            "naslov_sufiks": self.naslov_sufiks,
            "frekvencija": self.kao_frekvencija(),
            "srednje_vrednosti": [round(float(x), 3) for x in self.srednje_vrednosti.tolist()],
            "globalni_prosek": round(self.globalni_prosek, 3),
            "globalna_std_dev": round(self.globalna_std_dev, 3),
            "ritam": {int(b): (round(float(v), 2) if pd.notna(v) else 0)
                      for b, v in self.analiza_ponavljanja.items()},
            "uzastopni": {int(k): int(v) for k, v in self.analiza_uzastopnih.items()},
            "dekade": {k: round(float(v), 3) for k, v in self.analiza_dekada.items()},
            "poziciona": self._poziciona_json(),
            "pozicioni_prosek": {c: round(float(v), 2) for c, v in self.pozicioni_prosek.items()},
        }

    def _poziciona_json(self):
        # Heatmap: red = broj (1..39), kolona = pozicija (1..7)
        pf = self.poziciona_frekvencija
        return {
            "brojevi": list(range(1, MAX_BROJ + 1)),
            "pozicije": list(range(1, BROJEVA_U_KOMBINACIJI + 1)),
            "vrednosti": [[int(pf.loc[b, f"poz_{p}"]) for p in range(1, BROJEVA_U_KOMBINACIJI + 1)]
                          for b in range(1, MAX_BROJ + 1)],
        }

    def hi_kvadrat_pozicija(self):
        """Hi-kvadrat test pristrasnosti pozicija izvlačenja (nad celom istorijom)."""
        from scipy.stats import chisquare
        observed = self.poziciona_frekvencija.values.flatten()
        broj_kola = len(self.loto_df)
        if self.loto_df.empty or observed.sum() == 0:
            return {"ok": False, "poruka": "Nema dovoljno podataka za test."}
        ukupno_izvlacenja = broj_kola * BROJEVA_U_KOMBINACIJI
        expected_value = ukupno_izvlacenja / (MAX_BROJ * BROJEVA_U_KOMBINACIJI)
        valid = [(o, expected_value) for o in observed if o > 0]
        if not valid:
            return {"ok": False, "poruka": "Nema validnih podataka."}
        obs = [v[0] for v in valid]
        exp = [v[1] for v in valid]
        chi2, p = chisquare(f_obs=obs, f_exp=exp)
        prag = 0.05
        pristrasno = bool(p < prag)
        return {
            "ok": True,
            "broj_kola": broj_kola,
            "chi2": round(float(chi2), 2),
            "p": round(float(p), 4),
            "prag": prag,
            "pristrasno": pristrasno,
            "zakljucak": ("Postoji statistički značajan dokaz pristrasnosti pozicija."
                          if pristrasno else
                          "Nema dokaza pristrasnosti — odstupanja su u granicama nasumičnog procesa."),
        }

    def vremenska_serija(self):
        """Srednja vrednost po kolu kroz celu istoriju (za trend liniju)."""
        if self.loto_df.empty:
            return {"kola": [], "vrednosti": []}
        sr = self.loto_df[KOLONE].mean(axis=1)
        kola = self.loto_df["kolo"].tolist()
        return {"kola": [int(k) for k in kola], "vrednosti": [round(float(x), 3) for x in sr.tolist()]}
