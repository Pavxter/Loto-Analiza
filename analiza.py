# analiza.py (v10.1)

import sys
import io

# Postavljanje UTF-8 kodiranja za stdout i stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import itertools
import matplotlib.pyplot as plt
import sqlite3 
import os.path
from datetime import datetime
from statistics import mean 
import os
from dotenv import load_dotenv
import seaborn as sns 
import json
import time
import math
import re

import google.generativeai as genai
import ml_generator
from data_manager import DatabaseManager
from scipy.stats import chisquare

# --- Konstante za Loto igru ---
MAX_BROJ = 39
BROJEVA_U_KOMBINACIJI = 7
BROJ_KATEGORIJA_FREKV = 13 # Broj brojeva u "vrućim" i "hladnim"
PERIOD_SVEZIH_KOLA = 10

# -----------------------------

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                               QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
                               QSpinBox, QPushButton, QTextEdit, QListWidget, QGridLayout,
                               QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
                               QAbstractItemView, QLineEdit, QMessageBox, QMenu, QDateEdit,
                               QDialog, QDialogButtonBox, QDoubleSpinBox, QCheckBox, QComboBox)
from PySide6.QtCore import Qt, QDate, QThread, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pandas.plotting import autocorrelation_plot

class EditTicketDialog(QDialog):
    def __init__(self, trenutna_kombinacija="", parent=None):
        super().__init__(parent); self.setWindowTitle("Izmeni/Dodaj Tiket"); layout = QVBoxLayout(self); self.unos_linija = QLineEdit(self)
        if trenutna_kombinacija: self.unos_linija.setText(trenutna_kombinacija.strip("()"))
        else: self.unos_linija.setPlaceholderText("Unesite 7 brojeva odvojenih zarezom...")
        self.dugmici = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); self.dugmici.accepted.connect(self.accept); self.dugmici.rejected.connect(self.reject)
        layout.addWidget(QLabel("Unesite kombinaciju (brojevi odvojeni zarezom):")); layout.addWidget(self.unos_linija); layout.addWidget(self.dugmici)
    def get_kombinacija(self):
        kombinacija_tekst = self.unos_linija.text()
        try:
            brojevi_str = kombinacija_tekst.split(',');
            if len(brojevi_str) != BROJEVA_U_KOMBINACIJI: raise ValueError(f"Potrebno je tačno {BROJEVA_U_KOMBINACIJI} brojeva.")
            brojevi = set()
            for b_str in brojevi_str:
                broj = int(b_str.strip());
                if not (1 <= broj <= MAX_BROJ): raise ValueError(f"Svi brojevi moraju biti između 1 i {MAX_BROJ}.")
                brojevi.add(broj)
            if len(brojevi) != BROJEVA_U_KOMBINACIJI: raise ValueError("Svi brojevi moraju biti jedinstveni.")
            return str(tuple(sorted(list(brojevi))))
        except Exception as e: QMessageBox.critical(self, "Greška u Unosu", str(e)); return None

class ConfirmAIDialog(QDialog):
    def __init__(self, prompt_text, parent=None):
        super().__init__(parent); self.setWindowTitle("Potvrda Slanja Upita AI Modelu"); self.setMinimumSize(600, 500); layout = QVBoxLayout(self)
        info_label = QLabel("Sledeći upit će biti poslat Google AI modelu na obradu. Pregledajte i potvrdite:"); info_label.setWordWrap(True)
        self.prompt_display = QTextEdit(); self.prompt_display.setPlainText(prompt_text); self.prompt_display.setReadOnly(True)
        self.dugmici = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.dugmici.button(QDialogButtonBox.StandardButton.Ok).setText("Pošalji Upit"); self.dugmici.accepted.connect(self.accept); self.dugmici.rejected.connect(self.reject)
        layout.addWidget(info_label); layout.addWidget(self.prompt_display); layout.addWidget(self.dugmici)

class EditHistoryDialog(QDialog):
    def __init__(self, red_podataka, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Izmeni Istorijski Unos"); self.kolo, self.datum, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7 = red_podataka
        layout = QFormLayout(self)
        self.kolo_input = QSpinBox(); self.kolo_input.setRange(1, 10000); self.kolo_input.setValue(self.kolo)
        self.datum_input = QDateEdit(); self.datum_input.setDate(QDate.fromString(self.datum, "yyyy-MM-dd")); self.datum_input.setCalendarPopup(True)
        komb_str = ", ".join(map(str, [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7]))
        self.kombinacija_input = QLineEdit(komb_str)
        layout.addRow("Broj kola:", self.kolo_input); layout.addRow("Datum:", self.datum_input); layout.addRow("Kombinacija:", self.kombinacija_input)
        self.dugmici = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.dugmici.accepted.connect(self.accept); self.dugmici.rejected.connect(self.reject); layout.addRow(self.dugmici)
    def get_podaci(self):
        try:
            kolo = self.kolo_input.value(); datum = self.datum_input.date().toString("yyyy-MM-dd")
            brojevi_str = self.kombinacija_input.text().split(',');
            if len(brojevi_str) != BROJEVA_U_KOMBINACIJI: raise ValueError(f"Potrebno je tačno {BROJEVA_U_KOMBINACIJI} brojeva.")
            brojevi = [int(b.strip()) for b in brojevi_str]
            if len(set(brojevi)) != BROJEVA_U_KOMBINACIJI: raise ValueError("Svi brojevi moraju biti jedinstveni.")
            for b in brojevi:
                if not (1 <= b <= MAX_BROJ): raise ValueError(f"Svi brojevi moraju biti između 1 i {MAX_BROJ}.")
            return [kolo, datum, *brojevi]
        except Exception as e: QMessageBox.critical(self, "Greška u Unosu", str(e)); return None

class MLWorker(QThread):
    finished = Signal(str)
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def run(self):
        result = self.fn()
        self.finished.emit(result)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, subplot_spec=(1, 1)):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#2E2E2E')
        if subplot_spec == (2, 1):
            self.axes1 = self.fig.add_subplot(2, 1, 1)
            self.axes2 = self.fig.add_subplot(2, 1, 2)
            self.axes = self.axes1 
            self.axes_list = [self.axes1, self.axes2]
        else:
            self.axes = self.fig.add_subplot(1, 1, 1)
            self.axes_list = [self.axes]

        for ax in self.axes_list:
            ax.set_facecolor('#3C3C3C') # Nešto svetlija pozadina za sam grafik
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white') 
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')

        super(MplCanvas, self).__init__(self.fig)

    def nacrtaj_grafikon_frekvencije(self, frekvencija_podaci, vruci, hladni, svezi, naslov_sufiks=""):
        self.axes.clear()
        boje = []
        frekvencije_za_prikaz = []
        for i in range(1, MAX_BROJ + 1):
            if i in vruci:
                boje.append('#FF6347') # Tomato red
            elif i in hladni:
                boje.append('#66B2FF') # Svetlo plava
            else:
                boje.append('#888888') # Svetlo siva
            frekvencije_za_prikaz.append(frekvencija_podaci.get(i, 0))
        
        bars = self.axes.bar(range(1, MAX_BROJ + 1), frekvencije_za_prikaz, color=boje)
        
        for i, bar in enumerate(bars):
            broj = i + 1
            if broj in svezi:
                height = bar.get_height()
                self.axes.text(bar.get_x() + bar.get_width() / 2.0, height, '*', ha='center', va='bottom', color='yellow', fontsize=14, weight='bold')

        self.axes.set_title(f'Frekvencija Izvučenih Brojeva {naslov_sufiks}', color='white')
        self.axes.set_xlabel('Loto Broj', color='white')
        self.axes.set_ylabel('Broj Ponavljanja', color='white')
        self.axes.set_xticks(range(1, MAX_BROJ + 1))
        self.axes.grid(axis='y', linestyle='--', alpha=0.4, color='gray')
        self.fig.tight_layout()

    def nacrtaj_grafikon_sr_vrednosti(self, sr_vrednosti_podaci, prosek, std_dev, naslov_sufiks=""):
        self.axes.clear()
        self.axes.hist(sr_vrednosti_podaci, bins=25, color='#00FFFF', edgecolor='black', alpha=0.7)
        self.axes.axvline(prosek, color='#FF4747', linestyle='dashed', linewidth=2, label=f'Prosek: {prosek:.2f}')
        self.axes.axvline(prosek - std_dev, color='orange', linestyle='dotted', linewidth=2, label=f'Opseg 1 Std Dev ({prosek-std_dev:.2f} - {prosek+std_dev:.2f})')
        self.axes.axvline(prosek + std_dev, color='orange', linestyle='dotted', linewidth=2)
        self.axes.set_title(f'Distribucija Srednjih Vrednosti {naslov_sufiks}', color='white')
        self.axes.set_xlabel('Srednja Vrednost 7 Brojeva u Kombinaciji', color='white')
        self.axes.set_ylabel('Broj Kola (Učestalost)', color='white')
        legend = self.axes.legend()
        legend.get_frame().set_facecolor('#4D4D4D')
        legend.get_frame().set_edgecolor('gray')
        for text in legend.get_texts():
            text.set_color("white")
        self.axes.grid(axis='y', linestyle='--', alpha=0.4, color='gray')
        self.fig.tight_layout()

    def nacrtaj_grafikon_ponavljanja(self, ponavljanja_podaci, naslov_sufiks=""):
        self.axes.clear()
        podaci_za_prikaz = ponavljanja_podaci[ponavljanja_podaci > 0]
        if not podaci_za_prikaz.empty:
            self.axes.bar(podaci_za_prikaz.index, podaci_za_prikaz.values, color='#9370DB') # MediumPurple
        self.axes.set_title(f'Prosečan Razmak Ponavljanja {naslov_sufiks}', color='white')
        self.axes.set_xlabel('Loto Broj', color='white')
        self.axes.set_ylabel('Prosečan Broj Kola (Razmak)', color='white')
        self.axes.set_xticks(range(1, MAX_BROJ + 1))
        self.axes.grid(axis='y', linestyle='--', alpha=0.4, color='gray')
        self.fig.tight_layout()

    def nacrtaj_grafikon_uzastopnih(self, uzastopni_podaci, naslov_sufiks=""):
        self.axes.clear()
        self.axes.bar(uzastopni_podaci.index, uzastopni_podaci.values, color='#FFD700', edgecolor='black') # Gold
        self.axes.set_title(f'Učestalost Uzastopnih Parova {naslov_sufiks}', color='white')
        self.axes.set_xlabel('Broj Uzastopnih Parova', color='white')
        self.axes.set_ylabel('Broj Kola', color='white')
        self.axes.grid(axis='y', linestyle='--', alpha=0.4, color='gray')
        self.fig.tight_layout()

    def nacrtaj_grafikon_dekada(self, dekada_podaci, naslov_sufiks=""):
        self.axes.clear()
        self.axes.bar(dekada_podaci.index, dekada_podaci.values, color='#FF6347', edgecolor='black') # Tomato
        self.axes.set_title(f'Prosečan Broj Brojeva po Dekadi {naslov_sufiks}', color='white')
        self.axes.set_xlabel('Dekada', color='white')
        self.axes.set_ylabel('Prosečan Broj Brojeva po Kolu', color='white')
        self.axes.grid(axis='y', linestyle='--', alpha=0.4, color='gray')
        self.fig.tight_layout()

    def nacrtaj_vremensku_seriju(self, sr_vrednosti_podaci):
        if hasattr(self, 'axes2'):
            self.axes1.clear()
            self.axes2.clear()
            
            self.axes1.set_title('Kretanje Srednje Vrednosti Kombinacija Kroz Vreme', color='white')
            self.axes2.set_title('Test Autokorelacije (Da li postoji "memorija"?)', color='white')

            if sr_vrednosti_podaci.empty:
                self.axes1.text(0.5, 0.5, 'Nema podataka za prikaz', horizontalalignment='center', verticalalignment='center')
                self.axes2.text(0.5, 0.5, 'Nema podataka za test', horizontalalignment='center', verticalalignment='center')
                self.fig.tight_layout()
                return

            x_osa = range(len(sr_vrednosti_podaci))
            self.axes1.plot(x_osa, sr_vrednosti_podaci.values, label='Sr. vrednost po kolu', alpha=0.6, marker='.', linestyle='-')
            pokretni_prosek = sr_vrednosti_podaci.rolling(window=10).mean()
            self.axes1.plot(x_osa, pokretni_prosek.values, color='#FF4747', linewidth=2, label='Pokretni prosek (10 kola)')
            self.axes1.set_xlabel('Broj Izvlačenja (Vreme)', color='white')
            self.axes1.set_ylabel('Srednja Vrednost', color='white')
            legend1 = self.axes1.legend()
            legend1.get_frame().set_facecolor('#4D4D4D')
            legend1.get_frame().set_edgecolor('gray')
            for text in legend1.get_texts(): text.set_color("white")
            self.axes1.grid(True, linestyle='--', alpha=0.4, color='gray')
            
            autocorrelation_plot(sr_vrednosti_podaci.dropna(), ax=self.axes2, color='white', markerfacecolor='white')
            self.axes2.lines[0].set_color('white') # Glavna linija
            for line in self.axes2.lines[1:]: line.set_color('gray') # Linije poverenja
            
            self.fig.tight_layout()

    def nacrtaj_pozicionu_analizu(self, poziciona_frekvencija, pozicioni_prosek, naslov_sufiks=""):
        if hasattr(self, 'axes2'):
            self.axes1.clear()
            self.axes2.clear()
            
            sns.heatmap(poziciona_frekvencija, ax=self.axes1, cmap="viridis", annot=True, fmt="d", linewidths=.5)
            self.axes1.set_title(f'Mapa Frekvencije Brojeva po Poziciji Izvlačenja {naslov_sufiks}', color='white')
            self.axes1.set_xlabel('Pozicija Izvlačenja (1. do 7.)', color='white')
            self.axes1.set_ylabel('Loto Broj', color='white')
            
            self.axes2.bar(pozicioni_prosek.index, pozicioni_prosek.values, color='#20B2AA') # LightSeaGreen
            self.axes2.axhline(y=20, color='#FF4747', linestyle='--', label='Globalni prosek (20.0)')
            self.axes2.set_title(f'Prosečna Vrednost Izvučenog Broja po Poziciji {naslov_sufiks}', color='white')
            self.axes2.set_xlabel('Pozicija Izvlačenja', color='white')
            self.axes2.set_ylabel('Prosečna Vrednost', color='white')
            legend2 = self.axes2.legend()
            legend2.get_frame().set_facecolor('#4D4D4D')
            legend2.get_frame().set_edgecolor('gray')
            for text in legend2.get_texts(): text.set_color("white")
            self.axes2.grid(axis='y', linestyle='--', alpha=0.4, color='gray')
            
            self.fig.tight_layout()

class LotoAnalizator(QMainWindow):
    def __init__(self):
        super().__init__()
        load_dotenv(); api_key = os.getenv("GEMINI_API_KEY")
        try:
            if not api_key: raise ValueError("GEMINI_API_KEY nije pronađen u .env fajlu ili .env fajl ne postoji.")
            genai.configure(api_key=api_key)
            self.ai_model = genai.GenerativeModel('gemini-1.5-flash'); print("Gemini AI uspešno konfigurisan.")
        except Exception as e:
            print(f"GREŠKA pri konfigurisanju AI: {e}"); self.ai_model = None
            QMessageBox.critical(self, "Greška AI Konfiguracije", f"Nije moguće konfigurisati Google AI.\n\nGreška: {e}")
        
        self.db_manager = DatabaseManager() # Inicijalizacija menadžera baze
        self.ocisti_format_datuma_u_bazi()
        self.ucitaj_i_analiziraj_podatke() # Prvo učitavanje podataka
        
        self.initUI()
        
        self.osvezi_sve_analize() # Zatim osvežavanje prikaza
        self.osvezi_tabelu_tiketa()
        self.osvezi_tabelu_istorije()
        self.osvezi_tabelu_bektesta()
        self.osvezi_dashboard_prikaz() # Zadatak 2.4: Inicijalno popunjavanje dashboard-a
        
    def ocisti_format_datuma_u_bazi(self):
        print("--- Proveravam format datuma u bazi... ---"); cursor = self.db_manager.db_conn.cursor(); cursor_update = self.db_manager.db_conn.cursor()
        cursor.execute("SELECT id, datum FROM istorijski_rezultati"); svi_datumi = cursor.fetchall(); ispravljeno_redova = 0
        for red_id, datum_str in svi_datumi:
            if datum_str and '.' in datum_str:
                try:
                    ispravan_datum = pd.to_datetime(datum_str, dayfirst=True).strftime('%Y-%m-%d')
                    if ispravan_datum != datum_str:
                        cursor_update.execute("UPDATE istorijski_rezultati SET datum = ? WHERE id = ?", (ispravan_datum, red_id)); ispravljeno_redova += 1
                except Exception as e: print(f"Nije moguće konvertovati datum '{datum_str}' za ID {red_id}: {e}")
        if ispravljeno_redova > 0: self.db_manager.db_conn.commit(); print(f"Završeno čišćenje. Ispravljeno {ispravljeno_redova} datuma.")
        else: print("Svi datumi su već u ispravnom formatu.")

    def ucitaj_i_analiziraj_podatke(self, period_analize=0):
        self.loto_df = self.db_manager.get_all_historical_results_as_df() # Koristi se nova metoda
        sve_kombinacije_df = self.loto_df[['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']].dropna().astype(int)
        self.set_istorijskih_kombinacija = {tuple(sorted(row)) for row in sve_kombinacije_df.values}
        if period_analize > 0 and period_analize <= len(self.loto_df):
            analizirani_df = self.loto_df.tail(period_analize); self.naslov_sufiks = f"(Poslednjih {period_analize} kola)"
        else:
            analizirani_df = self.loto_df; self.naslov_sufiks = f"(Sva Kola - {len(self.loto_df)})"
        print(f"--- Analiza se vrši na: {self.naslov_sufiks} ---")
        self.kolone_za_brojeve = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']; self.brojevi_po_kolima = analizirani_df[self.kolone_za_brojeve].dropna().astype(int); self.srednje_vrednosti = self.brojevi_po_kolima.mean(axis=1); self.globalni_prosek = self.srednje_vrednosti.mean(); self.globalna_std_dev = self.srednje_vrednosti.std()
        
        # Centralizovana analiza par/nepar odnosa
        if self.brojevi_po_kolima.empty:
            self.najcesci_par_nepar = "N/A"
        else:
            par_nepar_counts = self.brojevi_po_kolima.apply(lambda row: tuple(sorted([sum(1 for x in row if x % 2 == 0), sum(1 for x in row if x % 2 != 0)])), axis=1).value_counts()
            self.najcesci_par_nepar = par_nepar_counts.index[0] if not par_nepar_counts.empty else "N/A"

        svi_izvuceni_brojevi = pd.concat([self.brojevi_po_kolima[col] for col in self.brojevi_po_kolima]); self.frekvencija = svi_izvuceni_brojevi.value_counts(); sortirani_po_frekvenciji = self.frekvencija.sort_values(ascending=False)
        self.vruci_brojevi = set(sortirani_po_frekvenciji.head(BROJ_KATEGORIJA_FREKV).index); self.hladni_brojevi = set(sortirani_po_frekvenciji.tail(BROJ_KATEGORIJA_FREKV).index); self.neutralni_brojevi = set(range(1, MAX_BROJ + 1)) - self.vruci_brojevi - self.hladni_brojevi
        poslednjih_10_kola = analizirani_df.tail(PERIOD_SVEZIH_KOLA); self.svezi_brojevi = set(pd.concat([poslednjih_10_kola[col] for col in self.kolone_za_brojeve]).unique())
        self.analiza_ponavljanja = {broj: kola_sa_brojem.diff().dropna().mean() if len(kola_sa_brojem := analizirani_df[analizirani_df[self.kolone_za_brojeve].eq(broj).any(axis=1)]['id']) > 1 else 0 for broj in range(1, MAX_BROJ + 1)}
        self.analiza_ponavljanja = pd.Series(self.analiza_ponavljanja)
        self.analiza_uzastopnih = pd.Series([sum(1 for i in range(len(k)-1) if k[i+1] == k[i] + 1) for k in [sorted(list(red)) for i, red in self.brojevi_po_kolima.iterrows()]]).value_counts().sort_index()
        self.analiza_dekada = pd.DataFrame([{'1-9': sum(1 for b in red if 1<=b<=9), '10-19': sum(1 for b in red if 10<=b<=19), '20-29': sum(1 for b in red if 20<=b<=29), '30-39': sum(1 for b in red if 30<=b<=39)} for i, red in self.brojevi_po_kolima.iterrows()]).mean()
        self.poziciona_frekvencija = pd.DataFrame(0, index=range(1, MAX_BROJ + 1), columns=[f'poz_{i}' for i in range(1, BROJEVA_U_KOMBINACIJI + 1)])
        for i, col_name in enumerate(self.kolone_za_brojeve, 1):
            counts = analizirani_df[col_name].value_counts()
            if not counts.empty: self.poziciona_frekvencija.loc[counts.index, f'poz_{i}'] = counts
        self.pozicioni_prosek = analizirani_df[self.kolone_za_brojeve].mean()

        # Zadatak 1: Kreiranje "Modela Pristrasnosti"
        self.model_pristrasnosti = {}
        broj_kola_u_analizi = len(analizirani_df)
        if broj_kola_u_analizi > 0:
            # Očekivana frekvencija je broj kola podeljen sa brojem mogućih brojeva.
            # Svaki broj ima istu šansu da bude izvučen na bilo kojoj poziciji u idealnom scenariju.
            ocekivana_frekvencija = broj_kola_u_analizi / MAX_BROJ
            if ocekivana_frekvencija > 0:
                for poz_idx, poz_col in enumerate(self.poziciona_frekvencija.columns, 1):
                    for broj_idx, stvarna_frekvencija in self.poziciona_frekvencija[poz_col].items():
                        broj = broj_idx
                        # Skor = Stvarno / Očekivano. > 1 je "vruće", < 1 je "hladno".
                        skor = stvarna_frekvencija / ocekivana_frekvencija
                        self.model_pristrasnosti[(broj, poz_idx)] = skor
        
        print("--- Sve analize uspešno završene (uključujući model pristrasnosti)! ---")

    def osvezi_sve_analize(self):
        period = self.analiza_period_input.value() if hasattr(self, 'analiza_period_input') else 0
        self.ucitaj_i_analiziraj_podatke(period_analize=period)
        self.grafikon_frekvencije.nacrtaj_grafikon_frekvencije(self.frekvencija, self.vruci_brojevi, self.hladni_brojevi, self.svezi_brojevi, self.naslov_sufiks)
        self.grafikon_sr_vrednosti.nacrtaj_grafikon_sr_vrednosti(self.srednje_vrednosti, self.globalni_prosek, self.globalna_std_dev, self.naslov_sufiks)
        self.grafikon_ponavljanja.nacrtaj_grafikon_ponavljanja(self.analiza_ponavljanja, self.naslov_sufiks)
        self.grafikon_uzastopni.nacrtaj_grafikon_uzastopnih(self.analiza_uzastopnih, self.naslov_sufiks)
        self.grafikon_dekade.nacrtaj_grafikon_dekada(self.analiza_dekada, self.naslov_sufiks)
        if hasattr(self, 'grafikon_vremenske_serije'):
            srednje_vrednosti_ukupno = self.loto_df[self.kolone_za_brojeve].mean(axis=1)
            self.grafikon_vremenske_serije.nacrtaj_vremensku_seriju(srednje_vrednosti_ukupno)
        if hasattr(self, 'grafikon_redosleda'):
            self.grafikon_redosleda.nacrtaj_pozicionu_analizu(self.poziciona_frekvencija, self.pozicioni_prosek, self.naslov_sufiks)
        for graf in ['grafikon_frekvencije', 'grafikon_sr_vrednosti', 'grafikon_ponavljanja', 'grafikon_uzastopni', 'grafikon_dekade', 'grafikon_vremenske_serije', 'grafikon_redosleda']:
            if hasattr(self, graf): getattr(self, graf).draw()
        print("Svi analitički prikazi su osveženi i ponovo iscrtani.")
        if hasattr(self, 'unos_kola') and not self.loto_df.empty:
            self.unos_kola.setValue(int(self.loto_df['kolo'].max()) + 1)

    def osvezi_dashboard_prikaz(self):
        """
        Glavna funkcija za osvežavanje prikaza na dashboard-u.
        Prikuplja podatke, vrši analizu i popunjava UI elemente.
        (Zadatak 2.1)
        """
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            period = self.dashboard_period_input.value()
            
            # 1. Učitavanje i analiza podataka (koristi postojeću, moćnu funkciju)
            # Ovo će popuniti self.vruci_brojevi, self.hladni_brojevi, self.svezi_brojevi, itd.
            self.ucitaj_i_analiziraj_podatke(period_analize=period)

            # 2. Popunjavanje numeričkih labela
            self.db_prosek_label.setText(f"{self.globalni_prosek:.2f}")
            self.db_stddev_label.setText(f"{self.globalna_std_dev:.2f}")
            
            # Prikaz centralizovanog rezultata za par/nepar odnos
            if self.najcesci_par_nepar != "N/A":
                self.db_par_nepar_label.setText(f"{self.najcesci_par_nepar[0]} parna / {self.najcesci_par_nepar[1]} neparna")
            else:
                self.db_par_nepar_label.setText("N/A")

            vruci_str = ", ".join(map(str, sorted(list(self.vruci_brojevi))))
            hladni_str = ", ".join(map(str, sorted(list(self.hladni_brojevi))))
            self.db_vruci_lista_label.setText(vruci_str)
            self.db_hladni_lista_label.setText(hladni_str)

            # 3. Iscrtavanje mini-grafikona
            self.db_grafikon_frekvencije.nacrtaj_grafikon_frekvencije(self.frekvencija, self.vruci_brojevi, self.hladni_brojevi, self.svezi_brojevi, self.naslov_sufiks)
            self.db_grafikon_sr_vrednosti.nacrtaj_grafikon_sr_vrednosti(self.srednje_vrednosti, self.globalni_prosek, self.globalna_std_dev, self.naslov_sufiks)
            self.db_grafikon_frekvencije.draw()
            self.db_grafikon_sr_vrednosti.draw()

            # 4. Kreiranje i prikaz predloga bazena (fuzija vrućih i svežih)
            # Stara logika: predlog_bazena = sorted(list(self.vruci_brojevi.union(self.svezi_brojevi)))
            
            # NOVA LOGIKA za kontrolisanu veličinu bazena
            # Uzimamo prvih 12 najčešćih ("vrućih") brojeva iz analiziranog perioda
            sortirani_vruci = self.frekvencija.sort_values(ascending=False).index.tolist()
            top_vruci = sortirani_vruci[:12]

            # Uzimamo prvih 12 "najsvežijih" brojeva (najčešći u poslednjih 10 kola unutar analiziranog perioda)
            poslednjih_10_kola_df = self.brojevi_po_kolima.tail(PERIOD_SVEZIH_KOLA)
            svezi_series = poslednjih_10_kola_df.melt(value_name='broj')['broj'].dropna().astype(int)
            frekvencija_svezih = svezi_series.value_counts()
            top_svezi = frekvencija_svezih.index.tolist()[:12]

            # Fuzija i kreiranje jedinstvenog, sortiranog bazena
            predlog_bazena = sorted(list(set(top_vruci + top_svezi)))
            self.db_bazen_output.setText(", ".join(map(str, predlog_bazena)))
            print(f"Dashboard osvežen. Predlog bazena sadrži {len(predlog_bazena)} brojeva.")
        finally:
            QApplication.restoreOverrideCursor()

    def dashboard_prebaci_u_generator(self):
        """
        Preuzima predloženi bazen sa dashboard-a, popunjava polja
        na Generator tabu i prebacuje fokus na taj tab.
        (Zadatak 2.2)
        """
        predlozeni_bazen = self.db_bazen_output.text()
        if not predlozeni_bazen: return
        self.bazen_brojeva_input.setText(predlozeni_bazen)
        self.koristi_bazen_checkbox.setChecked(True)
        self.tabs.setCurrentWidget(self.tab_generator)

    def analiziraj_period_za_bazen(self):
        """Izvršava analizu na klik dugmeta i popunjava liste brojeva."""
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            period = self.bazen_period_input.value()
            if period <= 0:
                period = len(self.loto_df)
            
            if period > len(self.loto_df):
                period = len(self.loto_df)
                self.bazen_period_input.setValue(period)

            df_period = self.loto_df.tail(period)
            
            if df_period.empty:
                self.bazen_vruci_prikaz.clear()
                self.bazen_hladni_prikaz.clear()
                self.bazen_svezi_prikaz.clear()
                QMessageBox.warning(self, "Nema Podataka", "Nema podataka za izabrani period.")
                return

            # Vrući brojevi: sortirani po frekvenciji opadajuće
            brojevi_series_period = df_period[self.kolone_za_brojeve].melt(value_name='broj')['broj'].dropna().astype(int)
            frekvencija_period = brojevi_series_period.value_counts()
            self.prikaz_vruci = frekvencija_period.index.tolist()
            self.bazen_vruci_prikaz.setText(", ".join(map(str, self.prikaz_vruci)))

            # Hladni brojevi: prvo neizvučeni (numerički), pa najređi (po frekvenciji rastuće)
            all_lotto_numbers = set(range(1, 40))
            drawn_in_period = set(frekvencija_period.index)
            undrawn_in_period = sorted(list(all_lotto_numbers - drawn_in_period))
            least_frequent_drawn = frekvencija_period.sort_values(ascending=True).index.tolist()
            full_cold_list = undrawn_in_period + least_frequent_drawn
            self.prikaz_hladni = list(dict.fromkeys(full_cold_list)) # Uklanja duplikate ako postoje
            self.bazen_hladni_prikaz.setText(", ".join(map(str, self.prikaz_hladni)))

            # Sveži brojevi: jedinstveni iz poslednjih 10 kola, sortirani po UČESTALOSTI
            poslednjih_10_kola_df = self.loto_df.tail(10)
            svezi_series = poslednjih_10_kola_df[self.kolone_za_brojeve].melt(value_name='broj')['broj'].dropna().astype(int)
            frekvencija_svezih = svezi_series.value_counts()
            self.prikaz_svezi = frekvencija_svezih.index.tolist()
            self.bazen_svezi_prikaz.setText(", ".join(map(str, self.prikaz_svezi)))
            
            QMessageBox.information(self, "Analiza Završena", f"Prikazani su rezultati analize za poslednjih {period} kola.")

        except Exception as e:
            print(f"Greška pri analizi perioda za bazen: {e}")
            QMessageBox.critical(self, "Greška", f"Došlo je do greške pri analizi: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def kreiraj_bazen_fuzijom(self):
        try:
            if not hasattr(self, 'prikaz_vruci') or not hasattr(self, 'prikaz_hladni') or not hasattr(self, 'prikaz_svezi'):
                QMessageBox.warning(self, "Greška", "Morate prvo kliknuti na 'Analiziraj Period' da biste dobili liste brojeva.")
                return

            broj_vrucih = self.bazen_uzmi_vrucih_input.value()
            broj_hladnih = self.bazen_uzmi_hladnih_input.value()
            broj_svezih = self.bazen_uzmi_svezih_input.value()

            finalni_set = set()
            finalni_set.update(self.prikaz_vruci[:broj_vrucih])
            finalni_set.update(self.prikaz_hladni[:broj_hladnih])
            finalni_set.update(self.prikaz_svezi[:broj_svezih])

            finalni_bazen = sorted(list(finalni_set))
            rezultat_str = ", ".join(map(str, finalni_bazen))
            self.bazen_rezultat_output.setText(rezultat_str)
            QMessageBox.information(self, "Uspeh", f"Uspešno kreiran bazen od {len(finalni_bazen)} brojeva.")

        except Exception as e:
            QMessageBox.critical(self, "Greška", f"Došlo je do greške pri kreiranju bazena: {e}")
            print(f"Greška kod kreiranja bazena: {e}")

    def prebaci_bazen_u_generator(self):
        bazen_text = self.bazen_rezultat_output.text()
        if not bazen_text:
            QMessageBox.warning(self, "Greška", "Prvo morate kreirati bazen.")
            return

        # Pronalazi tab "Generator" po imenu
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Generator":
                self.tabs.setCurrentIndex(i)
                break
        
        self.bazen_brojeva_input.setText(bazen_text)
        self.koristi_bazen_checkbox.setChecked(True)
        print(f"Bazen '{bazen_text}' je prebačen u generator.")

    def initUI(self):
        self.setWindowTitle('Loto Analizator v10.1 - Napredne Analize')
        self.setGeometry(100, 100, 1200, 800)
        
        self.tabs = QTabWidget()

        # --- Kreiranje novog Dashboard Taba (Zadatak 1.1) ---
        self.tab_dashboard = QWidget()
        glavni_layout_db = QVBoxLayout(self.tab_dashboard)

        # Panel sa globalnim kontrolama (Zadatak 1.2)
        kontrole_panel = QWidget()
        kontrole_layout = QHBoxLayout(kontrole_panel)
        kontrole_layout.addWidget(QLabel("<b>Period Analize:</b>"))
        self.dashboard_period_input = QSpinBox()
        self.dashboard_period_input.setRange(10, 10000)
        self.dashboard_period_input.setValue(300)
        self.dashboard_period_input.setSuffix(" kola")
        self.dashboard_period_input.setToolTip("Broj poslednjih kola za stratešku analizu na ovom dashboard-u.")
        kontrole_layout.addWidget(self.dashboard_period_input)
        self.dashboard_osvezi_dugme = QPushButton("Osveži Prikaz")
        kontrole_layout.addWidget(self.dashboard_osvezi_dugme)
        kontrole_layout.addStretch(1)
        glavni_layout_db.addWidget(kontrole_panel)

        # Mreža za prikaz podataka (Zadatak 1.3)
        grid_layout = QGridLayout()

        # Panel: Ključni Numerički Podaci
        kljucni_pokazatelji_box = QGroupBox("Ključni Pokazatelji")
        form_layout_kp = QFormLayout(kljucni_pokazatelji_box)
        self.db_prosek_label = QLabel("N/A"); self.db_prosek_label.setStyleSheet("font-size: 11pt; font-weight: bold; color: #66FFCC;")
        self.db_stddev_label = QLabel("N/A"); self.db_stddev_label.setStyleSheet("font-size: 11pt; font-weight: bold; color: #E0E0E0;")
        self.db_par_nepar_label = QLabel("N/A"); self.db_par_nepar_label.setStyleSheet("font-size: 11pt; font-weight: bold; color: #E0E0E0;")
        form_layout_kp.addRow("Prosek srednjih vrednosti:", self.db_prosek_label)
        form_layout_kp.addRow("Standardna devijacija:", self.db_stddev_label)
        form_layout_kp.addRow("Najčešći Par/Nepar odnos:", self.db_par_nepar_label)
        grid_layout.addWidget(kljucni_pokazatelji_box, 0, 0)

        # Panel: Vrući & Hladni Brojevi
        vruci_hladni_box = QGroupBox("Vrući i Hladni Brojevi")
        form_layout_vh = QFormLayout(vruci_hladni_box)
        self.db_vruci_lista_label = QLabel("N/A"); self.db_vruci_lista_label.setWordWrap(True); self.db_vruci_lista_label.setStyleSheet("font-size: 10pt; color: #FF6347; font-weight: bold;")
        self.db_hladni_lista_label = QLabel("N/A"); self.db_hladni_lista_label.setWordWrap(True); self.db_hladni_lista_label.setStyleSheet("font-size: 10pt; color: #66B2FF; font-weight: bold;")
        form_layout_vh.addRow("Vrući brojevi:", self.db_vruci_lista_label)
        form_layout_vh.addRow("Hladni brojevi:", self.db_hladni_lista_label)
        grid_layout.addWidget(vruci_hladni_box, 0, 1)

        # Panel: Mini-Grafikon Frekvencija
        self.db_grafikon_frekvencije = MplCanvas(self, width=5, height=3.5)
        grid_layout.addWidget(self.db_grafikon_frekvencije, 1, 0)

        # Panel: Mini-Grafikon Srednjih Vrednosti
        self.db_grafikon_sr_vrednosti = MplCanvas(self, width=5, height=3.5)
        grid_layout.addWidget(self.db_grafikon_sr_vrednosti, 1, 1)

        glavni_layout_db.addLayout(grid_layout)

        # Panel za Akciju (Zadatak 1.4)
        akcija_box = QGroupBox("Predlog Bazena i Akcija")
        akcija_layout = QHBoxLayout(akcija_box)
        self.db_bazen_output = QLineEdit(); self.db_bazen_output.setReadOnly(True); self.db_bazen_output.setPlaceholderText("Ovde će biti prikazan predlog bazena brojeva...")
        self.db_bazen_output.setStyleSheet("font-size: 12pt; font-weight: bold; color: #FFFF99; background-color: #2E2E2E;")
        self.db_prebaci_u_generator_dugme = QPushButton("🚀 Iskoristi Ovu Strategiju u Generatoru")
        akcija_layout.addWidget(QLabel("<b>Predlog Bazena (Vrući + Sveži):</b>")); akcija_layout.addWidget(self.db_bazen_output); akcija_layout.addWidget(self.db_prebaci_u_generator_dugme)
        glavni_layout_db.addWidget(akcija_box)
        
        # --- Kraj Dashboard Taba ---

        self.grafikon_frekvencije = MplCanvas(self); self.grafikon_sr_vrednosti = MplCanvas(self); self.grafikon_ponavljanja = MplCanvas(self); self.grafikon_uzastopni = MplCanvas(self); self.grafikon_dekade = MplCanvas(self); self.grafikon_vremenske_serije = MplCanvas(self, subplot_spec=(2, 1)); self.grafikon_redosleda = MplCanvas(self, subplot_spec=(2, 1))
        
        tab_frekvencija = QWidget(); layout_frekvencija = QVBoxLayout(tab_frekvencija); layout_frekvencija.addWidget(self.grafikon_frekvencije)
        tab_sr_vrednosti = QWidget(); layout_sr_vrednosti = QVBoxLayout(tab_sr_vrednosti); layout_sr_vrednosti.addWidget(self.grafikon_sr_vrednosti)
        tab_ponavljanja = QWidget(); layout_ponavljanja = QVBoxLayout(tab_ponavljanja); layout_ponavljanja.addWidget(self.grafikon_ponavljanja)
        tab_uzastopni = QWidget(); layout_uzastopni = QVBoxLayout(tab_uzastopni); layout_uzastopni.addWidget(self.grafikon_uzastopni)
        tab_dekade = QWidget(); layout_dekade = QVBoxLayout(tab_dekade); layout_dekade.addWidget(self.grafikon_dekade)
        tab_vremenska_serija = QWidget(); layout_vremenska_serija = QVBoxLayout(tab_vremenska_serija); layout_vremenska_serija.addWidget(self.grafikon_vremenske_serije)
        tab_redosled = QWidget(); layout_redosled = QVBoxLayout(tab_redosled); layout_redosled.addWidget(self.grafikon_redosleda)
        
        # ML Generator Tab
        self.tab_ml_generator = QWidget(); ml_layout = QVBoxLayout(self.tab_ml_generator)
        ml_info = QLabel("Ovaj panel koristi Veštačku Inteligenciju (VAE neuronsku mrežu) da nauči 'suštinu' dobitnih kombinacija i generiše potpuno nove, statistički slične predloge.")
        ml_info.setWordWrap(True); ml_layout.addWidget(ml_info)
        self.treniraj_dugme = QPushButton("Istreniraj ML Model (sporo, radi se jednom)"); self.treniraj_dugme.clicked.connect(self.pokreni_trening)
        ml_layout.addWidget(self.treniraj_dugme)
        generisi_layout = QHBoxLayout(); self.ml_broj_predloga = QSpinBox(); self.ml_broj_predloga.setRange(1, 50); self.ml_broj_predloga.setValue(10)
        self.generisi_ml_dugme = QPushButton("Generiši ML Predloge"); self.generisi_ml_dugme.clicked.connect(self.generisi_ml_predloge)
        generisi_layout.addWidget(QLabel("Broj predloga:")); generisi_layout.addWidget(self.ml_broj_predloga); generisi_layout.addWidget(self.generisi_ml_dugme)
        ml_layout.addLayout(generisi_layout)
        self.ml_rezultati_output = QListWidget()
        self.ml_rezultati_output.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.ml_rezultati_output.customContextMenuRequested.connect(self.prikazi_kontekstni_meni_ml)
        ml_layout.addWidget(self.ml_rezultati_output)
        self.sacuvaj_ml_set_dugme = QPushButton("Sačuvaj Ovaj ML Set za Bektest"); self.sacuvaj_ml_set_dugme.clicked.connect(self.sacuvaj_ml_set_za_bektest)
        ml_layout.addWidget(self.sacuvaj_ml_set_dugme)
        self.ml_status_label = QLabel("Status: Model nije istreniran."); self.ml_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter); ml_layout.addWidget(self.ml_status_label)

        # Generator Tab
        self.tab_generator = QWidget(); glavni_layout_gen = QVBoxLayout(self.tab_generator); forma_layout = QFormLayout();
        
        # NOVE KONTROLE ZA BAZEN BROJEVA
        forma_layout.addRow(QLabel("<b>=== Bazen Brojeva za Generisanje (Opciono) ===</b>"))
        self.bazen_brojeva_input = QLineEdit(); self.bazen_brojeva_input.setPlaceholderText("Unesite 10-20 brojeva odvojenih zarezom, npr: 1,5,8,12...")
        forma_layout.addRow(QLabel("Bazen brojeva:"), self.bazen_brojeva_input)
        self.koristi_bazen_checkbox = QCheckBox("Aktiviraj generisanje samo iz gornjeg bazena"); forma_layout.addRow(self.koristi_bazen_checkbox)

        self.analiza_period_input = QSpinBox(); self.analiza_period_input.setRange(0, 10000); self.analiza_period_input.setValue(0); self.analiza_period_input.setToolTip("Unesite broj poslednjih kola za analizu (0 = cela istorija)"); self.unikat_filter_checkbox = QCheckBox("Izbaci već izvučene kombinacije"); self.unikat_filter_checkbox.setChecked(True); forma_layout.addRow(QLabel("<b>=== Glavna Podešavanja Analize ===</b>")); forma_layout.addRow(QLabel("Analiziraj samo poslednjih [N] kola:"), self.analiza_period_input); forma_layout.addRow(self.unikat_filter_checkbox)
        self.strategija_svezine_input = QComboBox(); self.strategija_svezine_input.addItems(["Favorizuj 'sveže' brojeve (Prati Trend)", "Kažnjavaj 'sveže' brojeve (Izbegavaj Ponavljanje)", "Ignoriši 'svežinu' (Neutralno)"]); forma_layout.addRow(QLabel("<b>--- Strategija Bodovanja 'Svežine' ---</b>")); forma_layout.addRow(QLabel("Brojevi iz poslednjih 10 kola:"), self.strategija_svezine_input)
        self.diverzitet_checkbox = QCheckBox("Optimizuj za raznovrsnost (ukloni slične)"); self.diverzitet_checkbox.setChecked(True)
        self.slicnost_input = QSpinBox(); self.slicnost_input.setRange(1, 5); self.slicnost_input.setValue(3); self.slicnost_input.setToolTip("Kombinacija se smatra 'sličnom' ako deli više od ovog broja brojeva.")
        forma_layout.addRow(QLabel("<b>--- Fuzija i Optimizacija ---</b>")); forma_layout.addRow(self.diverzitet_checkbox); forma_layout.addRow(QLabel("Max dozvoljena sličnost:"), self.slicnost_input)
        self.min_sv_input = QDoubleSpinBox(); self.min_sv_input.setRange(4, 36); self.min_sv_input.setDecimals(2); self.min_sv_input.setValue(17.14); self.max_sv_input = QDoubleSpinBox(); self.max_sv_input.setRange(4, 36); self.max_sv_input.setDecimals(2); self.max_sv_input.setValue(22.86); self.parni_input = QSpinBox(); self.parni_input.setRange(0, 7); self.parni_input.setValue(3); self.neparni_input = QSpinBox(); self.neparni_input.setRange(0, 7); self.neparni_input.setValue(4); self.vruci_input = QSpinBox(); self.vruci_input.setRange(0, 7); self.vruci_input.setValue(2); self.neutralni_input = QSpinBox(); self.neutralni_input.setRange(0, 7); self.neutralni_input.setValue(3); self.hladni_input = QSpinBox(); self.hladni_input.setRange(0, 7); self.hladni_input.setValue(2); self.uzastopni_input = QSpinBox(); self.uzastopni_input.setRange(0, 6); self.uzastopni_input.setValue(1); self.dekada_max_input = QSpinBox(); self.dekada_max_input.setRange(1, 7); self.dekada_max_input.setValue(3);
        forma_layout.addRow(QLabel("<b>--- Filteri Srednje Vrednosti ---</b>")); forma_layout.addRow(QLabel("Minimalna sr. vrednost:"), self.min_sv_input); forma_layout.addRow(QLabel("Maksimalna sr. vrednost:"), self.max_sv_input); forma_layout.addRow(QLabel("<b>--- Filteri Tipa Broja ---</b>")); forma_layout.addRow(QLabel("Broj parnih:"), self.parni_input); forma_layout.addRow(QLabel("Broj neparnih:"), self.neparni_input); forma_layout.addRow(QLabel("<b>--- Filteri Frekvencije (Istorijski) ---</b>")); forma_layout.addRow(QLabel("Broj 'vrućih' brojeva:"), self.vruci_input); forma_layout.addRow(QLabel("Broj 'neutralnih' brojeva:"), self.neutralni_input); forma_layout.addRow(QLabel("Broj 'hladnih' brojeva:"), self.hladni_input); forma_layout.addRow(QLabel("<b>--- Strukturni Filteri ---</b>")); forma_layout.addRow(QLabel("Broj uzastopnih parova (tačno):"), self.uzastopni_input); forma_layout.addRow(QLabel("Max brojeva iz jedne dekade:"), self.dekada_max_input)
        
        # Zadatak 3: Dodavanje Nove Kontrole na UI
        forma_layout.addRow(QLabel("<b>--- Bodovanje Pristrasnosti Mašine (Hi-Kvadrat) ---</b>"))
        self.primeni_pristrasnost_checkbox = QCheckBox("Primeni bodovanje po pristrasnosti mašine")
        self.primeni_pristrasnost_checkbox.setToolTip("Ako je uključeno, favorizuje kombinacije koje odgovaraju 'vrućim' pozicijama za brojeve, naučenim iz istorije.")
        forma_layout.addRow(self.primeni_pristrasnost_checkbox)

        self.analiza_period_input.valueChanged.connect(self.osvezi_sve_analize)
        dugmici_layout = QHBoxLayout(); self.generisi_dugme = QPushButton("Generiši Kombinacije"); self.generisi_dugme.clicked.connect(self.pokreni_generisanje); self.ai_dugme = QPushButton("🤖 Pitaj AI za Preporuku"); self.ai_dugme.clicked.connect(self.pokreni_ai_preporuku); self.ai_dugme.setEnabled(False); dugmici_layout.addWidget(self.generisi_dugme); dugmici_layout.addWidget(self.ai_dugme);
        self.broj_kombinacija_label = QLabel("Broj pronađenih kombinacija: 0"); self.broj_kombinacija_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.rangiraj_checkbox = QCheckBox("Prikaži samo rangiranih Top 50"); self.rangiraj_checkbox.setChecked(True)
        self.rezultati_output = QListWidget(); self.dodaj_tiket_dugme = QPushButton("Dodaj Izabranu Kombinaciju u Praćenje"); self.dodaj_tiket_dugme.clicked.connect(self.dodaj_tiket_u_pracenje)
        self.sacuvaj_set_dugme = QPushButton("Sačuvaj Prikazani Set za Bektest"); self.sacuvaj_set_dugme.clicked.connect(self.sacuvaj_set_za_bektest)
        glavni_layout_gen.addLayout(forma_layout); glavni_layout_gen.addLayout(dugmici_layout); glavni_layout_gen.addWidget(self.rangiraj_checkbox); glavni_layout_gen.addWidget(self.broj_kombinacija_label); glavni_layout_gen.addWidget(QLabel("Generisane kombinacije (kliknite da izaberete):")); glavni_layout_gen.addWidget(self.rezultati_output); glavni_layout_gen.addWidget(self.dodaj_tiket_dugme); glavni_layout_gen.addWidget(self.sacuvaj_set_dugme)
        
        # Moji Tiketi Tab
        self.tab_moji_tiketi = QWidget(); layout_moji_tiketi = QVBoxLayout(self.tab_moji_tiketi); gornji_panel_layout = QHBoxLayout(); unos_layout = QFormLayout(); self.unos_kola = QSpinBox(); self.unos_kola.setRange(1, 10000); self.unos_datuma = QDateEdit(QDate.currentDate()); self.unos_datuma.setCalendarPopup(True); self.unos_dobitne_kombinacije = QLineEdit(); self.unos_dobitne_kombinacije.setPlaceholderText("npr. 5,12,18,23,29,31,35"); unos_layout.addRow("Broj kola:", self.unos_kola); unos_layout.addRow("Datum izvlačenja:", self.unos_datuma); unos_layout.addRow("Dobitna kombinacija (zarez):", self.unos_dobitne_kombinacije); gornji_panel_layout.addLayout(unos_layout)
        desni_dugmici_layout = QVBoxLayout(); self.proveri_dugme = QPushButton("Dodaj Kolo i Proveri Tikete"); self.proveri_dugme.clicked.connect(self.proveri_i_dodaj_kolo); self.samo_aktivni_checkbox = QCheckBox("Analiziraj samo aktivne tikete"); desni_dugmici_layout.addWidget(self.samo_aktivni_checkbox); self.ai_analiza_tiketa_dugme = QPushButton("🤖 AI Analiza Mojih Tiketa"); self.ai_analiza_tiketa_dugme.clicked.connect(self.pokreni_ai_analizu_tiketa); self.manuelno_dodaj_dugme = QPushButton("Dodaj Tiket Manuelno"); self.manuelno_dodaj_dugme.clicked.connect(self.manuelno_dodaj_tiket); desni_dugmici_layout.addWidget(self.proveri_dugme); desni_dugmici_layout.addWidget(self.ai_analiza_tiketa_dugme); desni_dugmici_layout.addWidget(self.manuelno_dodaj_dugme); gornji_panel_layout.addLayout(desni_dugmici_layout); layout_moji_tiketi.addLayout(gornji_panel_layout)
        self.tabela_tiketa = QTableWidget(); layout_moji_tiketi.addWidget(self.tabela_tiketa); self.tabela_tiketa.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu); self.tabela_tiketa.customContextMenuRequested.connect(self.prikazi_kontekstni_meni)
        
        # Istorija Tab
        self.tab_istorija = QWidget(); layout_istorija = QVBoxLayout(self.tab_istorija)
        self.izvezi_istoriju_dugme = QPushButton("Izvezi Istoriju u CSV"); self.izvezi_istoriju_dugme.clicked.connect(self.izvezi_istoriju_u_csv)
        layout_istorija.addWidget(self.izvezi_istoriju_dugme)
        layout_istorija.addWidget(QLabel("Pregled svih unetih kola (desni klik na red za izmenu ili brisanje)")); self.tabela_istorije = QTableWidget(); layout_istorija.addWidget(self.tabela_istorije); self.tabela_istorije.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu); self.tabela_istorije.customContextMenuRequested.connect(self.prikazi_kontekstni_meni_istorije)

        # Bektesting Tab
        self.tab_bektest = QWidget()
        layout_bektest = QVBoxLayout(self.tab_bektest)
        
        gornji_layout_bektest = QHBoxLayout()
        gornji_layout_bektest.addWidget(QLabel("Pregled rezultata sačuvanih setova (virtualnih igara) - Desni klik za brisanje"))
        gornji_layout_bektest.addStretch() 
        
        self.ai_analiza_bektesta_dugme = QPushButton("🤖 AI Analiza Bektestova")
        self.ai_analiza_bektesta_dugme.clicked.connect(self.pokreni_ai_analizu_bektestova)
        gornji_layout_bektest.addWidget(self.ai_analiza_bektesta_dugme)
        
        layout_bektest.addLayout(gornji_layout_bektest)
        
        self.tabela_bektesta = QTableWidget()
        layout_bektest.addWidget(self.tabela_bektesta)
        self.tabela_bektesta.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tabela_bektesta.customContextMenuRequested.connect(self.prikazi_kontekstni_meni_bektesta)
        
        # Kreiranje menija na vrhu
        self.kreiraj_meni()
        
        # Kreator Bazena Tab
        self.tab_kreator_bazena = QWidget()
        kreator_layout = QVBoxLayout(self.tab_kreator_bazena)
        
        # Sekcija 1: Podesite Period i Pokrenite Analizu
        kreator_layout.addWidget(QLabel("<b>Sekcija 1: Podesite Period i Pokrenite Analizu</b>"))
        sekcija1_layout = QHBoxLayout()
        form_layout1 = QFormLayout()
        self.bazen_period_input = QSpinBox()
        self.bazen_period_input.setRange(0, 10000)
        self.bazen_period_input.setValue(300)
        self.bazen_period_input.setToolTip("Unesite broj poslednjih kola za analizu (0 = cela istorija)")
        form_layout1.addRow("Analiziraj samo poslednjih [N] kola:", self.bazen_period_input)
        self.analiziraj_period_dugme = QPushButton("Analiziraj Period")
        sekcija1_layout.addLayout(form_layout1)
        sekcija1_layout.addWidget(self.analiziraj_period_dugme)
        sekcija1_layout.addStretch()
        kreator_layout.addLayout(sekcija1_layout)
        
        # Sekcija 2: Analitički Prikaz
        kreator_layout.addWidget(QLabel("<b>Sekcija 2: Analitički Prikaz (rezultati analize)</b>"))
        sekcija2_layout = QFormLayout()
        self.bazen_vruci_prikaz = QLineEdit(); self.bazen_vruci_prikaz.setReadOnly(True); self.bazen_vruci_prikaz.setPlaceholderText("Kliknite na 'Analiziraj Period' za prikaz...")
        self.bazen_hladni_prikaz = QLineEdit(); self.bazen_hladni_prikaz.setReadOnly(True); self.bazen_hladni_prikaz.setPlaceholderText("Kliknite na 'Analiziraj Period' za prikaz...")
        self.bazen_svezi_prikaz = QLineEdit(); self.bazen_svezi_prikaz.setReadOnly(True); self.bazen_svezi_prikaz.setPlaceholderText("Kliknite na 'Analiziraj Period' za prikaz...")
        sekcija2_layout.addRow("Sortirani Vrući brojevi:", self.bazen_vruci_prikaz)
        sekcija2_layout.addRow("Sortirani Hladni brojevi:", self.bazen_hladni_prikaz)
        sekcija2_layout.addRow("Sortirani Sveži brojevi (posl. 10 kola):", self.bazen_svezi_prikaz)
        kreator_layout.addLayout(sekcija2_layout)

        # Sekcija 3: Kreirajte Vaš Bazen
        kreator_layout.addWidget(QLabel("<b>Sekcija 3: Kreirajte Vaš Bazen</b>"))
        sekcija3_layout = QFormLayout()
        self.bazen_uzmi_vrucih_input = QSpinBox(); self.bazen_uzmi_vrucih_input.setRange(0, 39); self.bazen_uzmi_vrucih_input.setValue(7)
        self.bazen_uzmi_hladnih_input = QSpinBox(); self.bazen_uzmi_hladnih_input.setRange(0, 39); self.bazen_uzmi_hladnih_input.setValue(7)
        self.bazen_uzmi_svezih_input = QSpinBox(); self.bazen_uzmi_svezih_input.setRange(0, 39); self.bazen_uzmi_svezih_input.setValue(5)
        sekcija3_layout.addRow("Uzmi prvih [N] vrućih brojeva:", self.bazen_uzmi_vrucih_input)
        sekcija3_layout.addRow("Uzmi prvih [N] hladnih brojeva:", self.bazen_uzmi_hladnih_input)
        sekcija3_layout.addRow("Uzmi prvih [N] svežih brojeva:", self.bazen_uzmi_svezih_input)
        kreator_layout.addLayout(sekcija3_layout)
        self.kreiraj_bazen_dugme = QPushButton("Kreiraj Bazen Fuzijom Trendova")
        kreator_layout.addWidget(self.kreiraj_bazen_dugme)

        # Sekcija 4: Rezultat i Akcija
        kreator_layout.addWidget(QLabel("<b>Sekcija 4: Rezultat i Akcija</b>"))
        sekcija4_layout = QFormLayout()
        self.bazen_rezultat_output = QLineEdit(); self.bazen_rezultat_output.setReadOnly(True)
        sekcija4_layout.addRow("Kreirani Bazen:", self.bazen_rezultat_output)
        kreator_layout.addLayout(sekcija4_layout)
        self.prebaci_bazen_dugme = QPushButton("Prebaci Bazen u Generator")
        kreator_layout.addWidget(self.prebaci_bazen_dugme)
        kreator_layout.addStretch()

        # Povezivanje signala
        self.analiziraj_period_dugme.clicked.connect(self.analiziraj_period_za_bazen)
        self.kreiraj_bazen_dugme.clicked.connect(self.kreiraj_bazen_fuzijom)
        self.prebaci_bazen_dugme.clicked.connect(self.prebaci_bazen_u_generator)
        
        # Povezivanje signala za Dashboard (Zadatak 2.3)
        self.dashboard_osvezi_dugme.clicked.connect(self.osvezi_dashboard_prikaz)
        self.db_prebaci_u_generator_dugme.clicked.connect(self.dashboard_prebaci_u_generator)

        self.tabs.insertTab(0, self.tab_dashboard, "📈 Strateški Dashboard"); self.tabs.addTab(self.tab_generator, "Generator"); self.tabs.addTab(self.tab_kreator_bazena, "Kreator Bazena"); self.tabs.addTab(self.tab_ml_generator, "ML Generator"); self.tabs.addTab(self.tab_moji_tiketi, "Moji Tiketi"); self.tabs.addTab(self.tab_istorija, "Istorija Izvlačenja"); self.tabs.addTab(self.tab_bektest, "Bektest Strategija")
        
        # --- Faza 1: Kreiranje UI za Napredne Analize ---
        self.tab_napredne_analize = QWidget()
        layout_napredne_analize = QVBoxLayout(self.tab_napredne_analize)

        # Zadatak 1.2: Dizajniranje Kontrola
        gornji_panel_na = QWidget()
        gornji_layout_na = QHBoxLayout(gornji_panel_na)
        gornji_layout_na.addWidget(QLabel("Izaberite test koji želite da sprovedete:"))
        
        self.test_selector_combo = QComboBox()
        self.test_selector_combo.addItems([
            "Hi-Kvadrat Test: Pristrasnost Pozicija Izvlačenja"
        ])
        gornji_layout_na.addWidget(self.test_selector_combo)
        
        self.pokreni_test_dugme = QPushButton("Pokreni Analizu")
        gornji_layout_na.addWidget(self.pokreni_test_dugme)
        gornji_layout_na.addStretch(1)
        
        layout_napredne_analize.addWidget(gornji_panel_na)
        
        self.rezultat_testa_output = QTextEdit()
        self.rezultat_testa_output.setReadOnly(True)
        self.rezultat_testa_output.setStyleSheet("font-family: 'Courier New', monospace; background-color: #2E2E2E; color: #E0E0E0;")
        layout_napredne_analize.addWidget(self.rezultat_testa_output)
        
        # Povezivanje signala (Zadatak 2.2)
        self.pokreni_test_dugme.clicked.connect(self.pokreni_naprednu_analizu)
        
        self.tabs.addTab(self.tab_napredne_analize, "Napredne Analize")
        # --- Kraj Faze 1 ---

        self.tabs.addTab(tab_frekvencija, "Frekvencija"); self.tabs.addTab(tab_sr_vrednosti, "Sr. Vrednosti"); self.tabs.addTab(tab_ponavljanja, "Ponavljanja"); self.tabs.addTab(tab_uzastopni, "Uzastopni"); self.tabs.addTab(tab_dekade, "Dekade"); self.tabs.addTab(tab_vremenska_serija, "Vremenska Serija"); self.tabs.addTab(tab_redosled, "Analiza Redosleda")
        self.setCentralWidget(self.tabs); self.show()

    def kreiraj_meni(self):
        """Kreira glavni meni aplikacije."""
        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("&Pomoć")
        about_action = help_menu.addAction("&O Programu")
        about_action.triggered.connect(self.prikazi_about_prozor)

    def prikazi_about_prozor(self):
        """Prikazuje 'About' prozor sa informacijama."""
        tekst = """<b>Loto Analizator v10.1</b><br><br>
                   Aplikacija za statističku analizu, generisanje i bektestiranje Loto 7/39 strategija.<br>
                   Razvijena u saradnji sa Google AI.<br><br>
                   Sva prava zadržana."""
        QMessageBox.about(self, "O Programu", tekst)

    def osvezi_tabelu_tiketa(self):
        print("Osvežavam prikaz tiketa...")
        try:
            cursor = self.db_manager.db_conn.cursor()
            cursor.execute("SELECT id, kombinacija, status, poslednji_rezultat, datum_provere, dodatne_metrike FROM odigrani_tiketi ORDER BY id")
            svi_tiketi = cursor.fetchall()
            self.tabela_tiketa.setRowCount(len(svi_tiketi))
            kolone = ["ID", "Kombinacija", "Status", "Poslednji Pogodak", "Datum Provere", "Sr. Vrednost"]
            self.tabela_tiketa.setColumnCount(len(kolone)); self.tabela_tiketa.setHorizontalHeaderLabels(kolone)
            for i, red_podataka in enumerate(svi_tiketi):
                id_tiketa, kombinacija, status, pogodak, datum_provere, dodatne_metrike_json = red_podataka
                self.tabela_tiketa.setItem(i, 0, QTableWidgetItem(str(id_tiketa)))
                self.tabela_tiketa.setItem(i, 1, QTableWidgetItem(kombinacija))
                self.tabela_tiketa.setItem(i, 2, QTableWidgetItem(status))

                pogodak_str = str(red_podataka[3]) if red_podataka[3] is not None else ""; self.tabela_tiketa.setItem(i, 3, QTableWidgetItem(pogodak_str))
                datum_str = str(red_podataka[4]) if red_podataka[4] is not None else ""; self.tabela_tiketa.setItem(i, 4, QTableWidgetItem(datum_str))
                
                # Korišćenje regularnog izraza za uklanjanje bilo kog prefiksa u zagradama (npr. (ML), (GEN), (POOL))
                komb_str_sa_prefiskom = red_podataka[1]
                komb_str_bez_prefiksa = re.sub(r'^\(\w+\)', '', komb_str_sa_prefiskom)

                brojevi_int = [int(b) for b in komb_str_bez_prefiksa.strip("()").split(",")]; srednja_vrednost = mean(brojevi_int)
                item_sr_vrednost = QTableWidgetItem(f"{srednja_vrednost:.2f}"); self.tabela_tiketa.setItem(i, 5, item_sr_vrednost)

                # NOVO: Postavljanje tooltip-a sa dodatnim metrikama
                if dodatne_metrike_json:
                    try:
                        metrike = json.loads(dodatne_metrike_json)
                        tooltip_text = f"<b>Detalji Provere:</b><br>"
                        tooltip_text += f"Indeks Promašaja: <b>{metrike.get('promasaj', 'N/A')}</b><br>"
                        tooltip_text += f"Indeks Iznenađenja: <b>{metrike.get('iznenadjenje', 'N/A'):.2f}</b>"
                        self.tabela_tiketa.item(i, 3).setToolTip(tooltip_text)
                    except (json.JSONDecodeError, TypeError): pass # Ignorišemo greške ako JSON nije validan
                elif pogodak is None:
                    self.tabela_tiketa.item(i, 3).setToolTip("Metrike će biti dostupne nakon unosa rezultata kola.")
            self.tabela_tiketa.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); self.tabela_tiketa.resizeColumnsToContents()
        except Exception as e: print(f"Greška prilikom osvežavanja tabele tiketa: {e}")
    
    def osvezi_tabelu_istorije(self):
        print("Osvežavam prikaz istorije izvlačenja...")
        try:
            cursor = self.db_manager.db_conn.cursor()
            cursor.execute("SELECT id, kolo, datum, b1, b2, b3, b4, b5, b6, b7 FROM istorijski_rezultati ORDER BY id DESC")
            svi_rezultati = cursor.fetchall()
            self.tabela_istorije.setRowCount(len(svi_rezultati))
            kolone = ["ID", "Kolo", "Datum", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
            self.tabela_istorije.setColumnCount(len(kolone)); self.tabela_istorije.setHorizontalHeaderLabels(kolone)
            for i, red_podataka in enumerate(svi_rezultati):
                for j, podatak in enumerate(red_podataka):
                    self.tabela_istorije.setItem(i, j, QTableWidgetItem(str(podatak)))
            self.tabela_istorije.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); self.tabela_istorije.resizeColumnsToContents()
        except Exception as e: print(f"Greška prilikom osvežavanja tabele istorije: {e}")

    def osvezi_tabelu_bektesta(self):
        print("Osvežavam prikaz bektestova...")
        try:
            cursor = self.db_manager.db_conn.cursor()
            cursor.execute("SELECT id, kolo, datum_kreiranja, filter_podesavanja, broj_kombinacija, rezultat, indeks_promasaja, indeks_iznenadjenja FROM virtualne_igre ORDER BY id DESC")
            svi_bektestovi = cursor.fetchall()
            self.tabela_bektesta.setRowCount(len(svi_bektestovi))
            kolone = ["ID", "Kolo za Igru", "Datum Kreiranja", "Podešavanja Filtera", "Br. Komb.", "Rezultat", "Min. Promašaj", "Min. Iznenađenje"]
            self.tabela_bektesta.setColumnCount(len(kolone)); self.tabela_bektesta.setHorizontalHeaderLabels(kolone)
            for i, red_podataka in enumerate(svi_bektestovi):
                for j, podatak in enumerate(red_podataka):
                    item_text = ""
                    if podatak is not None:
                        # j == 7 odgovara koloni "Min. Iznenađenje"
                        if j == 7 and isinstance(podatak, float):
                            item_text = f"{podatak:.2f}"
                        else:
                            item_text = str(podatak)
                    
                    self.tabela_bektesta.setItem(i, j, QTableWidgetItem(item_text))
            self.tabela_bektesta.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); self.tabela_bektesta.resizeColumnsToContents()
            self.tabela_bektesta.setColumnHidden(0, True)
        except Exception as e: print(f"Greška prilikom osvežavanja tabele bektesta: {e}")

    def izvezi_istoriju_u_csv(self):
        try:
            putanja_fajla = "istorija_izvlacenja.csv"
            df = self.db_manager.get_all_historical_results_as_df()
            df.to_csv(putanja_fajla, index=False, encoding='utf-8-sig')
            QMessageBox.information(self, "Uspeh", f"Istorija je uspešno izvezena u fajl '{putanja_fajla}' unutar Vašeg projektnog foldera.")
        except Exception as e:
            QMessageBox.critical(self, "Greška pri Izvozu", f"Došlo je do greške: {e}")

    def prikazi_kontekstni_meni(self, position):
        indeksi = self.tabela_tiketa.selectedIndexes()
        if not indeksi:
            return
        
        red = indeksi[0].row()
        item = self.tabela_tiketa.item(red, 0)
        if not item or not item.text():
            return

        id_tiketa = int(item.text())
        
        kontekstni_meni = QMenu(self)
        izmeni_akcija = kontekstni_meni.addAction("Izmeni Tiket")
        kontekstni_meni.addSeparator()
        promeni_status_akcija = kontekstni_meni.addAction("Promeni Status (Aktivan/Neaktivan)")
        obrisi_akcija = kontekstni_meni.addAction("Obriši Tiket")
        
        izabrana_akcija = kontekstni_meni.exec(self.tabela_tiketa.viewport().mapToGlobal(position))
        
        if izabrana_akcija == izmeni_akcija:
            self.izmeni_tiket(id_tiketa)
        elif izabrana_akcija == promeni_status_akcija:
            self.promeni_status_tiketa(id_tiketa)
        elif izabrana_akcija == obrisi_akcija:
            self.obrisi_tiket(id_tiketa)
    
    def prikazi_kontekstni_meni_istorije(self, position):
        indeksi = self.tabela_istorije.selectedIndexes()
        if not indeksi: return
        red = indeksi[0].row(); id_unosa = int(self.tabela_istorije.item(red, 0).text())
        kontekstni_meni = QMenu(self); izmeni_akcija = kontekstni_meni.addAction("Izmeni Unos"); obrisi_akcija = kontekstni_meni.addAction("Obriši Unos")
        izabrana_akcija = kontekstni_meni.exec(self.tabela_istorije.viewport().mapToGlobal(position))
        if izabrana_akcija == izmeni_akcija: self.izmeni_unos_istorije(id_unosa)
        elif izabrana_akcija == obrisi_akcija: self.obrisi_unos_istorije(id_unosa)
    
    def prikazi_kontekstni_meni_bektesta(self, position):
        indeksi = self.tabela_bektesta.selectedIndexes()
        if not indeksi: return
        red = indeksi[0].row(); id_unosa = int(self.tabela_bektesta.item(red, 0).text())
        kontekstni_meni = QMenu(self); obrisi_akcija = kontekstni_meni.addAction("Obriši Ovaj Bektest")
        izabrana_akcija = kontekstni_meni.exec(self.tabela_bektesta.viewport().mapToGlobal(position))
        if izabrana_akcija == obrisi_akcija: self.obrisi_bektest(id_unosa)

    def prikazi_kontekstni_meni_ml(self, position):
        izabrani_item = self.ml_rezultati_output.itemAt(position)
        if not izabrani_item or not izabrani_item.text() or "---" in izabrani_item.text() or "Generišem" in izabrani_item.text():
            return

        kontekstni_meni = QMenu(self)
        dodaj_u_pracenje_akcija = kontekstni_meni.addAction("Dodaj u Praćenje Tiketa")
        
        izabrana_akcija = kontekstni_meni.exec(self.ml_rezultati_output.viewport().mapToGlobal(position))
        
        if izabrana_akcija == dodaj_u_pracenje_akcija:
            self.ml_rezultati_output.setCurrentItem(izabrani_item)
            self.dodaj_ml_tiket_u_pracenje()

    def izmeni_unos_istorije(self, unos_id):
        cursor = self.db_manager.db_conn.cursor(); cursor.execute("SELECT kolo, datum, b1, b2, b3, b4, b5, b6, b7 FROM istorijski_rezultati WHERE id=?", (unos_id,)); podaci = cursor.fetchone()
        if not podaci: return
        dialog = EditHistoryDialog(red_podataka=podaci, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            novi_podaci = dialog.get_podaci()
            if novi_podaci:
                cursor.execute("UPDATE istorijski_rezultati SET kolo=?, datum=?, b1=?, b2=?, b3=?, b4=?, b5=?, b6=?, b7=? WHERE id=?", (*novi_podaci, unos_id))
                self.db_manager.db_conn.commit(); self.osvezi_tabelu_istorije(); self.osvezi_sve_analize()
                QMessageBox.information(self, "Uspeh", f"Unos ID {unos_id} je uspešno izmenjen.")

    def obrisi_unos_istorije(self, unos_id):
        potvrda = QMessageBox.question(self, "Potvrda Brisanja", f"Da li ste SIGURNI da želite trajno da obrišete unos ID {unos_id} iz istorije? Ovo će uticati na sve analize.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if potvrda == QMessageBox.StandardButton.Yes:
            cursor = self.db_manager.db_conn.cursor(); cursor.execute("DELETE FROM istorijski_rezultati WHERE id = ?", (unos_id,)); self.db_manager.db_conn.commit()
            self.osvezi_tabelu_istorije(); self.osvezi_sve_analize()
            print(f"Unos ID {unos_id} je obrisan iz istorije.")

    def obrisi_bektest(self, unos_id):
        potvrda = QMessageBox.question(self, "Potvrda Brisanja", 
                                       f"Da li ste sigurni da želite da obrišete sačuvani bektest ID {unos_id}?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if potvrda == QMessageBox.StandardButton.Yes:
            cursor = self.db_manager.db_conn.cursor(); cursor.execute("DELETE FROM virtualne_igre WHERE id = ?", (unos_id,)); self.db_manager.db_conn.commit(); self.osvezi_tabelu_bektesta(); print(f"Bektest ID {unos_id} je obrisan.")

    def promeni_status_tiketa(self, tiket_id):
        cursor = self.db_manager.db_conn.cursor(); cursor.execute("SELECT status FROM odigrani_tiketi WHERE id = ?", (tiket_id,)); trenutni_status = cursor.fetchone()[0]
        novi_status = "neaktivan" if trenutni_status == "aktivan" else "aktivan"
        cursor.execute("UPDATE odigrani_tiketi SET status = ? WHERE id = ?", (novi_status, tiket_id)); self.db_manager.db_conn.commit(); self.osvezi_tabelu_tiketa(); print(f"Status tiketa ID {tiket_id} promenjen u '{novi_status}'.")
    
    def obrisi_tiket(self, tiket_id):
        potvrda = QMessageBox.question(self, "Potvrda Brisanja", f"Da li ste sigurni da želite trajno da obrišete tiket ID {tiket_id}?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if potvrda == QMessageBox.StandardButton.Yes: cursor = self.db_manager.db_conn.cursor(); cursor.execute("DELETE FROM odigrani_tiketi WHERE id = ?", (tiket_id,)); self.db_manager.db_conn.commit(); self.osvezi_tabelu_tiketa(); print(f"Tiket ID {tiket_id} je obrisan.")
            
    def izmeni_tiket(self, tiket_id):
        cursor = self.db_manager.db_conn.cursor(); cursor.execute("SELECT kombinacija FROM odigrani_tiketi WHERE id = ?", (tiket_id,)); trenutna_kombinacija = cursor.fetchone()[0]
        dialog = EditTicketDialog(trenutna_kombinacija=trenutna_kombinacija, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            nova_kombinacija = dialog.get_kombinacija()
            if nova_kombinacija: cursor.execute("UPDATE odigrani_tiketi SET kombinacija = ? WHERE id = ?", (nova_kombinacija, tiket_id)); self.db_manager.db_conn.commit(); self.osvezi_tabelu_tiketa()
    
    def manuelno_dodaj_tiket(self):
        dialog = EditTicketDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            nova_kombinacija = dialog.get_kombinacija()
            if nova_kombinacija: cursor = self.db_manager.db_conn.cursor(); cursor.execute("INSERT OR IGNORE INTO odigrani_tiketi (kombinacija) VALUES (?)", (nova_kombinacija,)); self.db_manager.db_conn.commit(); self.osvezi_tabelu_tiketa()

    def sacuvaj_set_za_bektest(self):
        broj_stavki = self.rezultati_output.count()
        if broj_stavki == 0 or "---" in self.rezultati_output.item(0).text():
            QMessageBox.warning(self, "Greška", "Nema generisanih kombinacija za čuvanje.")
            return
        cursor = self.db_manager.db_conn.cursor(); cursor.execute("SELECT max(kolo) FROM istorijski_rezultati"); poslednje_poznato_kolo = cursor.fetchone()[0]
        kolo_za_igru = (poslednje_poznato_kolo or 0) + 1
        sve_kombinacije = [self.rezultati_output.item(i).text().split("| Kombinacija: ")[-1] for i in range(broj_stavki)]
        lista_kombinacija_str = ";".join(sve_kombinacije)
        period_text = f"Posl. {self.analiza_period_input.value()}" if self.analiza_period_input.value() > 0 else "Sva kola"
        unikat_text = "Da" if self.unikat_filter_checkbox.isChecked() else "Ne"
        svezina_text = self.strategija_svezine_input.currentText().split(" ")[0]

        # Dodavanje prefiksa za tip strategije radi lakšeg prepoznavanja
        if self.koristi_bazen_checkbox.isChecked() and self.bazen_brojeva_input.text():
            tip_strategije = "Tip: Bazen"
        else:
            tip_strategije = "Tip: Generator"

        filter_podesavanja = (f"{tip_strategije} | Period: {period_text}, Unikat: {unikat_text}, Svežina: {svezina_text}, "
                              f"SrV: {self.min_sv_input.value()}-{self.max_sv_input.value()}, "
                              f"P/N: {self.parni_input.value()}/{self.neparni_input.value()}, V/H/N: {self.vruci_input.value()}/{self.hladni_input.value()}/{self.neutralni_input.value()}")
        bazen_brojeva_str = self.bazen_brojeva_input.text() if self.koristi_bazen_checkbox.isChecked() else ""
        try:
            cursor = self.db_manager.db_conn.cursor()
            cursor.execute("INSERT INTO virtualne_igre (kolo, datum_kreiranja, filter_podesavanja, lista_kombinacija, broj_kombinacija, bazen_brojeva) VALUES (?, ?, ?, ?, ?, ?)",
                           (kolo_za_igru, datetime.now().strftime("%Y-%m-%d %H:%M"), filter_podesavanja, lista_kombinacija_str, len(sve_kombinacije), bazen_brojeva_str))
            self.db_manager.db_conn.commit(); self.osvezi_tabelu_bektesta()
            QMessageBox.information(self, "Uspeh", f"Set od {len(sve_kombinacije)} kombinacija je sačuvan za bektest kola {kolo_za_igru}.")
        except sqlite3.IntegrityError: QMessageBox.warning(self, "Greška", f"Set za kolo {kolo_za_igru} sa istim filter podešavanjima je već sačuvan.")
        except Exception as e: QMessageBox.critical(self, "Greška Baze", f"Došlo je do greške pri čuvanju seta: {e}")

    def sacuvaj_ml_set_za_bektest(self):
        broj_stavki = self.ml_rezultati_output.count()
        if broj_stavki == 0 or "---" in self.ml_rezultati_output.item(0).text():
            QMessageBox.warning(self, "Greška", "Nema generisanih ML predloga za čuvanje.")
            return
        cursor = self.db_manager.db_conn.cursor(); cursor.execute("SELECT max(kolo) FROM istorijski_rezultati"); poslednje_poznato_kolo = cursor.fetchone()[0]
        kolo_za_igru = (poslednje_poznato_kolo or 0) + 1
        sve_kombinacije = [self.ml_rezultati_output.item(i).text() for i in range(broj_stavki)]
        lista_kombinacija_str = ";".join(sve_kombinacije)
        filter_podesavanja = f"ML Model v1 (VAE, LatentDim={ml_generator.LATENT_DIM})"
        try:
            cursor.execute("INSERT OR IGNORE INTO virtualne_igre (kolo, datum_kreiranja, filter_podesavanja, lista_kombinacija, broj_kombinacija, bazen_brojeva) VALUES (?, ?, ?, ?, ?, ?)",
                           (kolo_za_igru, datetime.now().strftime("%Y-%m-%d %H:%M"), filter_podesavanja, lista_kombinacija_str, len(sve_kombinacije), ""))
            self.db_manager.db_conn.commit()
            self.osvezi_tabelu_bektesta()
            QMessageBox.information(self, "Uspeh", f"ML set od {len(sve_kombinacije)} kombinacija je sačuvan za bektest kola {kolo_za_igru}.")
        except sqlite3.IntegrityError: QMessageBox.warning(self, "Greška", f"Set za kolo {kolo_za_igru} sa istim ML podešavanjima je već sačuvan.")
        except Exception as e: QMessageBox.critical(self, "Greška Baze", f"Došlo je do greške pri čuvanju seta: {e}")

    def dodaj_tiket_u_pracenje(self):
        izabrani_tiket = self.rezultati_output.currentItem()
        if not izabrani_tiket or "---" in izabrani_tiket.text():
            QMessageBox.warning(self, "Greška", "Nije izabrana validna kombinacija.")
            return
        
        ceo_tekst = izabrani_tiket.text()
        kombinacija_za_upis = ""
        if "| Kombinacija:" in ceo_tekst:
            kombinacija_za_upis = ceo_tekst.split("| Kombinacija:")[1].strip()
        elif ceo_tekst.startswith("(") and ceo_tekst.endswith(")"):
            kombinacija_za_upis = ceo_tekst
        else:
            QMessageBox.warning(self, "Greška", "Nije izabrana validna kombinacija.")
            return

        # Dodavanje prefiksa za lakšu identifikaciju porekla tiketa
        if self.koristi_bazen_checkbox.isChecked() and self.bazen_brojeva_input.text():
            prefiks = "(POOL)"
        else:
            prefiks = "(GEN)"
        kombinacija_za_upis = prefiks + kombinacija_za_upis

        print(f"Dodajem tiket u praćenje: {kombinacija_za_upis}")
        try:
            cursor = self.db_manager.db_conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO odigrani_tiketi (kombinacija) VALUES (?)", (kombinacija_za_upis,))
            self.db_manager.db_conn.commit()
            self.osvezi_tabelu_tiketa()
            QMessageBox.information(self, "Uspeh", f"Kombinacija {kombinacija_za_upis} je uspešno dodata u praćenje.")
        except Exception as e:
            QMessageBox.critical(self, "Greška Baze", f"Došlo je do greške pri upisu u bazu: {e}")

    def pokreni_generisanje(self):
        self.rezultati_output.clear()
        self.ai_dugme.setEnabled(False)
        self.broj_kombinacija_label.setText("Broj pronađenih kombinacija: 0")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        # --- NOVA LOGIKA ZA ODREĐIVANJE IZVORA BROJEVA ---
        izvor_brojeva = range(1, 40)
        if self.koristi_bazen_checkbox.isChecked():
            bazen_text = self.bazen_brojeva_input.text()
            if bazen_text:
                try:
                    brojevi_u_bazenu = [int(b.strip()) for b in bazen_text.split(',')]
                    if len(brojevi_u_bazenu) < 7: raise ValueError("Bazen mora sadržati bar 7 brojeva.")
                    for b in brojevi_u_bazenu:
                        if not (1 <= b <= 39): raise ValueError(f"Broj {b} nije u opsegu 1-39.")
                    izvor_brojeva = sorted(list(set(brojevi_u_bazenu)))
                    print(f"--- Generisanje iz prilagođenog bazena od {len(izvor_brojeva)} brojeva ---")
                except Exception as e:
                    QMessageBox.critical(self, "Greška u Bazenu Brojeva", f"Unos nije ispravan: {e}")
                    QApplication.restoreOverrideCursor(); return

        min_sv = self.min_sv_input.value()
        max_sv = self.max_sv_input.value()
        parni_filter = self.parni_input.value()
        neparni_filter = self.neparni_input.value()
        vruci_filter = self.vruci_input.value()
        neutralni_filter = self.neutralni_input.value()
        hladni_filter = self.hladni_input.value()
        uzastopni_filter = self.uzastopni_input.value()
        dekada_max_filter = self.dekada_max_input.value()
        filtriraj_unikate = self.unikat_filter_checkbox.isChecked()

        if parni_filter + neparni_filter != 7:
            self.rezultati_output.addItem("GREŠKA: Zbir parnih i neparnih brojeva mora biti 7.")
            QApplication.restoreOverrideCursor()
            return
        if vruci_filter + neutralni_filter + hladni_filter != 7:
            self.rezultati_output.addItem("GREŠKA: Zbir vrućih, neutralnih i hladnih brojeva mora biti 7.")
            QApplication.restoreOverrideCursor()
            return

        validne_kombinacije = []
        
        for kombinacija in itertools.combinations(izvor_brojeva, 7):
            if filtriraj_unikate and kombinacija in self.set_istorijskih_kombinacija:
                continue
            
            srednja_vrednost_kombinacije = sum(kombinacija) / 7.0
            if not (min_sv <= srednja_vrednost_kombinacije <= max_sv):
                continue
            
            if sum(1 for broj in kombinacija if broj % 2 == 0) != parni_filter:
                continue
            
            if sum(1 for broj in kombinacija if broj in self.vruci_brojevi) != vruci_filter:
                continue
            
            if sum(1 for broj in kombinacija if broj in self.hladni_brojevi) != hladni_filter:
                continue
            
            if sum(1 for i in range(len(kombinacija)-1) if kombinacija[i+1] == kombinacija[i] + 1) != uzastopni_filter:
                continue
            
            dekade = {'1-9': 0, '10-19': 0, '20-29': 0, '30-39': 0}
            for broj in kombinacija:
                if 1 <= broj <= 9: dekade['1-9'] += 1
                elif 10 <= broj <= 19: dekade['10-19'] += 1
                elif 20 <= broj <= 29: dekade['20-29'] += 1
                else: dekade['30-39'] += 1
            if max(dekade.values()) > dekada_max_filter:
                continue
            
            validne_kombinacije.append(kombinacija)

        if validne_kombinacije:
            self.broj_kombinacija_label.setText(f"Pronađeno {len(validne_kombinacije)} kombinacija. Bodujem i rangiram...")
            QApplication.processEvents()
            
            kombinacije_sa_skorom = [(self.izracunaj_skor(k), k) for k in validne_kombinacije]
            kombinacije_sa_skorom.sort(key=lambda x: x[0], reverse=True)
            
            if self.diverzitet_checkbox.isChecked():
                max_slicnost = self.slicnost_input.value()
                kombinacije_sa_skorom = self.primeni_filter_diverziteta(kombinacije_sa_skorom, max_slicnost, 1)
            
            self.rezultati_output.clear()
            broj_za_prikaz = 50 if self.rangiraj_checkbox.isChecked() else len(kombinacije_sa_skorom)
            for skor, k in kombinacije_sa_skorom[:broj_za_prikaz]:
                self.rezultati_output.addItem(f"Skor: {skor:<8.2f} | Kombinacija: {k}")
            
            self.ai_dugme.setEnabled(True)
            self.broj_kombinacija_label.setText(f"Pronađeno {len(validne_kombinacije)} (filtrirano za raznovrsnost: {len(kombinacije_sa_skorom)}). Prikazano Top {broj_za_prikaz}.")
        else:
            self.rezultati_output.addItem("Nijedna kombinacija ne zadovoljava zate uslove.")
            self.broj_kombinacija_label.setText("Broj pronađenih kombinacija: 0")
        
        QApplication.restoreOverrideCursor()

    def pokreni_trening(self):
        self.treniraj_dugme.setEnabled(False)
        self.generisi_ml_dugme.setEnabled(False)
        self.ml_status_label.setText("Status: Trening u toku... Molim sačekajte, ovo može potrajati.")
        
        self.worker = MLWorker(fn=ml_generator.treniraj_i_sacuvaj_model)
        self.worker.finished.connect(self.trening_zavrsen)
        self.worker.start()

    def trening_zavrsen(self, rezultat):
        QMessageBox.information(self, "Trening Modela", rezultat)
        self.ml_status_label.setText(f"Status: {rezultat}")
        self.treniraj_dugme.setEnabled(True)
        self.generisi_ml_dugme.setEnabled(True)
        
    def generisi_ml_predloge(self):
        broj = self.ml_broj_predloga.value()
        self.ml_rezultati_output.clear()
        self.ml_rezultati_output.addItem("Generišem predloge...")
        QApplication.processEvents()
        
        predlozi, greska = ml_generator.generisi_kombinacije(broj)
        
        self.ml_rezultati_output.clear()
        if greska:
            self.ml_status_label.setText(f"Status: Greška!")
        else:
            self.ml_status_label.setText(f"Status: Uspešno generisano {len(predlozi)} predloga.")

        for p in predlozi:
            self.ml_rezultati_output.addItem(str(p))    

    def izracunaj_skor(self, kombinacija):
        skor = 0
        
        sr_vrednost_kombinacije = sum(kombinacija) / 7.0
        udaljenost_od_proseka = abs(sr_vrednost_kombinacije - self.globalni_prosek)
        skor_sv = max(0, 100 * (1 - udaljenost_od_proseka / (2 * self.globalna_std_dev)))
        skor += skor_sv
        
        strategija_index = self.strategija_svezine_input.currentIndex()
        broj_svezih = sum(1 for broj in kombinacija if broj in self.svezi_brojevi)
        if strategija_index == 0: # Favorizuj
            skor += broj_svezih * 10
        elif strategija_index == 1: # Kaznjavaj
            skor -= broj_svezih * 10
            
        if self.analiza_ponavljanja.count() > 0 and self.analiza_ponavljanja[self.analiza_ponavljanja > 0].any():
            prosek_svih_ritmova = self.analiza_ponavljanja[self.analiza_ponavljanja > 0].mean()
            skor_ritma = 0
            for broj in kombinacija:
                ritam_broja = self.analiza_ponavljanja.get(broj, prosek_svih_ritmova)
                if abs(ritam_broja - prosek_svih_ritmova) < 3:
                    skor_ritma += 5
            skor += skor_ritma

        # Zadatak 2: Unapređenje Sistema Bodovanja (Bonus/Kazna za Pristrasnost)
        if self.primeni_pristrasnost_checkbox.isChecked():
            skor_pristrasnosti = 0
            # Kombinacija je sortirana, tako da je pozicija i-tog elementa (i+1)
            for i, broj in enumerate(kombinacija):
                pozicija = i + 1
                # Uzimamo skor iz modela, ako ne postoji, podrazumevana vrednost je 1 (neutralno)
                bonus_skor = self.model_pristrasnosti.get((broj, pozicija), 1.0)
                # Dodajemo bonus/kaznu. Množimo sa 10 da bi imalo veći uticaj.
                # (skor - 1) pretvara skorove u opseg oko nule (npr. 1.2 -> 0.2, 0.8 -> -0.2)
                skor_pristrasnosti += (bonus_skor - 1) * 10 
            skor += skor_pristrasnosti
            
        return round(skor, 2)

    def primeni_filter_diverziteta(self, kandidati, max_slicnost, broj_kola_za_izbegavanje):
        print(f"Primenjujem filter diverziteta sa max sličnošću od {max_slicnost} broja...")
        if not kandidati:
            return []
        
        zona_izbegavanja = []
        if broj_kola_za_izbegavanje > 0:
            poslednja_kola_df = self.loto_df[self.kolone_za_brojeve].tail(broj_kola_za_izbegavanje)
            zona_izbegavanja = [set(row) for row in poslednja_kola_df.values]

        finalne_preporuke = []
        skorovi_finalnih = []
        for skor, kandidat_tuple in kandidati:
            kandidat_set = set(kandidat_tuple)
            previse_slican = any(len(kandidat_set.intersection(postojeca_komb)) > max_slicnost for postojeca_komb in finalne_preporuke + zona_izbegavanja)
            if not previse_slican:
                finalne_preporuke.append(kandidat_set)
                skorovi_finalnih.append((skor, kandidat_tuple))
        return skorovi_finalnih

    def izracunaj_promasaj_za_kombinaciju(self, kombinacija_lista, dobitna_kombinacija_set):
        """
        Izračunava "indeks promašaja" za JEDNU kombinaciju.
        """
        if not kombinacija_lista or not dobitna_kombinacija_set:
            return None
        try:
            trenutni_promasaj = 0
            for dobitni_broj in dobitna_kombinacija_set:
                najmanja_razlika = min(abs(dobitni_broj - broj_u_komb) for broj_u_komb in kombinacija_lista)
                trenutni_promasaj += najmanja_razlika
            return trenutni_promasaj
        except Exception: return None

    def izracunaj_indeks_promasaja(self, set_kombinacija, dobitna_kombinacija_set):
        """
        Izračunava "indeks promašaja" za dati set kombinacija.
        Indeks je najmanja ukupna "udaljenost" jedne od kombinacija u setu od dobitne kombinacije.
        """
        minimalni_ukupni_promasaj = float('inf')

        for komb_str in set_kombinacija:
            try:
                # Kombinacije su sačuvane kao stringovi "(1, 2, 3, 4, 5, 6, 7)"
                kombinacija_lista = list(eval(komb_str))
                
                trenutni_promasaj = 0
                for dobitni_broj in dobitna_kombinacija_set:
                    # Za svaki dobitni broj, nalazimo njemu najbliži broj u našoj kombinaciji
                    najmanja_razlika = min(abs(dobitni_broj - broj_u_komb) for broj_u_komb in kombinacija_lista)
                    trenutni_promasaj += najmanja_razlika
                
                minimalni_ukupni_promasaj = min(minimalni_ukupni_promasaj, trenutni_promasaj)
            except Exception:
                continue # Ako parsiranje kombinacije ne uspe, preskačemo je
        
        return minimalni_ukupni_promasaj if minimalni_ukupni_promasaj != float('inf') else None

    def izracunaj_model_verovatnoce(self, df_istorija, period_analize):
        """
        Kreira model verovatnoće na osnovu frekvencije brojeva u datom periodu.
        Koristi Laplace (Add-one) smoothing da izbegne verovatnoće jednake nuli.
        """
        if period_analize > 0 and period_analize <= len(df_istorija):
            analizirani_df = df_istorija.tail(period_analize)
        else:
            analizirani_df = df_istorija

        if analizirani_df.empty:
            # Ako nema podataka, vraćamo uniformnu distribuciju
            verovatnoca = 1 / MAX_BROJ
            return {broj: verovatnoca for broj in range(1, MAX_BROJ + 1)}

        kolone_za_brojeve = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
        svi_izvuceni_brojevi = pd.concat([analizirani_df[col] for col in kolone_za_brojeve]).dropna().astype(int)
        
        # Laplace (Add-one) Smoothing
        frekvencije = {broj: 1 for broj in range(1, MAX_BROJ + 1)} # Počinjemo sa 1 za svaki broj
        for broj in svi_izvuceni_brojevi:
            if broj in frekvencije:
                frekvencije[broj] += 1
        
        ukupan_broj_pojavljivanja = sum(frekvencije.values())
        
        model_verovatnoce = {broj: freq / ukupan_broj_pojavljivanja for broj, freq in frekvencije.items()}
        
        return model_verovatnoce

    def izracunaj_indeks_iznenadjenja(self, kombinacija, model_verovatnoce):
        """
        Računa "indeks iznenađenja" za jednu kombinaciju na osnovu datog modela verovatnoće.
        Veći indeks znači statistički "ređu" ili "iznenađujuću" kombinaciju.
        """
        try:
            # Sabiranje logaritama je ekvivalent množenju verovatnoća
            # Koristimo model_verovatnoce.get() da izbegnemo KeyError ako broj ne postoji, mada ne bi trebalo sa smoothingom
            log_verovatnoca = sum(math.log(model_verovatnoce.get(broj, 1e-9)) for broj in kombinacija) # 1e-9 za svaki slučaj
            # Množimo sa -1 da veće iznenađenje (manja verovatnoća) bude veći pozitivan broj
            indeks = -1 * log_verovatnoca
            return indeks
        except (ValueError, TypeError):
            return None # U slučaju matematičke greške

    def proveri_i_dodaj_kolo(self):
        try:
            kolo = self.unos_kola.value()
            datum = self.unos_datuma.date().toString("yyyy-MM-dd")
            kombinacija_tekst = self.unos_dobitne_kombinacije.text()
            brojevi_str = kombinacija_tekst.split(',')
            if len(brojevi_str) != 7:
                raise ValueError("Potrebno je tačno 7 brojeva.")
            
            dobitni_brojevi_lista = []
            dobitni_brojevi_set = set()
            for b_str in brojevi_str:
                broj = int(b_str.strip())
                if not (1 <= broj <= 39):
                    raise ValueError("Svi brojevi moraju biti između 1 i 39.")
                dobitni_brojevi_lista.append(broj)
                dobitni_brojevi_set.add(broj)
            
            if len(dobitni_brojevi_set) != 7:
                raise ValueError("Svi brojevi moraju biti jedinstveni.")
        except ValueError as e:
            QMessageBox.critical(self, "Greška u Unosu", f"Unos nije ispravan: {e}")
            return

        uspeh_dodavanja = self.db_manager.add_historical_result(kolo, datum, dobitni_brojevi_lista)
        
        cursor = self.db_manager.db_conn.cursor()
        cursor.execute("SELECT id, kombinacija FROM odigrani_tiketi") # Proveravamo sve, ne samo aktivne
        tiketi_za_proveru = cursor.fetchall()

        # Priprema modela verovatnoće za Indeks Iznenađenja (radimo jednom za sve tikete)
        istorija_pre_kola = self.loto_df[self.loto_df['kolo'] < kolo]
        model_verovatnoce = self.izracunaj_model_verovatnoce(istorija_pre_kola, 0) # Analiza cele istorije pre ovog kola

        for tiket_id, tiket_kombinacija_str in tiketi_za_proveru:
            try:
                # NOVO: Robusno rukovanje sa bilo kojim prefiksom (ML), (GEN), (POOL)
                komb_str_bez_prefiksa = re.sub(r'^\(\w+\)', '', tiket_kombinacija_str)
                tiket_brojevi_lista = [int(b) for b in komb_str_bez_prefiksa.strip("()").split(",")]
                tiket_brojevi = set(tiket_brojevi_lista)

                # Računanje dodatnih metrika
                promasaj = self.izracunaj_promasaj_za_kombinaciju(tiket_brojevi_lista, dobitni_brojevi_set)
                iznenadjenje = self.izracunaj_indeks_iznenadjenja(tiket_brojevi_lista, model_verovatnoce)
                metrike_json = json.dumps({"promasaj": promasaj, "iznenadjenje": iznenadjenje}) if promasaj is not None and iznenadjenje is not None else None

                broj_pogodaka = len(dobitni_brojevi_set.intersection(tiket_brojevi))
                datum_sada = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute("UPDATE odigrani_tiketi SET poslednji_rezultat = ?, datum_provere = ?, dodatne_metrike = ? WHERE id = ?", 
                               (broj_pogodaka, datum_sada, metrike_json, tiket_id))
            except (ValueError, IndexError):
                print(f"Greška: Nije moguće parsirati tiket ID: {tiket_id} sa kombinacijom: '{tiket_kombinacija_str}'. Preskačem proveru za ovaj tiket.")
                continue
        
        self.db_manager.db_conn.commit()

        # Provera bektestova za uneto kolo
        cursor.execute("SELECT id, lista_kombinacija, bazen_brojeva, filter_podesavanja FROM virtualne_igre WHERE kolo = ?", (kolo,))
        svi_bektestovi_za_kolo = cursor.fetchall()

        if svi_bektestovi_za_kolo:
            print(f"Pronađeno {len(svi_bektestovi_za_kolo)} bektestova za kolo {kolo}. Ažuriram rezultate...")
            for bektest_id, lista_kombinacija_str, bazen_brojeva_str, filter_podesavanja in svi_bektestovi_za_kolo:
                # 1. Provera uspešnosti bazena
                bazen_rezultat_str = ""
                if bazen_brojeva_str:
                    try:
                        bazen_set = set(int(b.strip()) for b in bazen_brojeva_str.split(','))
                        pogodaka_u_bazenu = len(dobitni_brojevi_set.intersection(bazen_set))
                        bazen_rezultat_str = f"Bazen: {pogodaka_u_bazenu}/{len(bazen_set)} | "
                    except:
                        bazen_rezultat_str = "Bazen: Greška | "

                # 2. Provera uspešnosti kombinacija
                pogoci = {7:0, 6:0, 5:0, 4:0, 3:0}
                sve_kombinacije_u_setu = lista_kombinacija_str.split(';')
                for komb_str in sve_kombinacije_u_setu:
                    try:
                        tiket_brojevi = set(eval(komb_str))
                        pogodak = len(dobitni_brojevi_set.intersection(tiket_brojevi))
                        if pogodak in pogoci: pogoci[pogodak] += 1
                    except: continue
                
                komb_rezultat_str = f"Komb: 7:{pogoci[7]}, 6:{pogoci[6]}, 5:{pogoci[5]}, 4:{pogoci[4]}"
                finalni_rezultat = bazen_rezultat_str + komb_rezultat_str

                # 3. Izračunavanje Indeksa Promašaja (NOVO)
                indeks_promasaja = self.izracunaj_indeks_promasaja(sve_kombinacije_u_setu, dobitni_brojevi_set)

                # 4. Izračunavanje Indeksa Iznenađenja (NOVO)
                indeks_iznenadjenja = None
                try:
                    # A. Odredi period analize iz podešavanja
                    period_match = re.search(r"Period: Posl\. (\d+)", filter_podesavanja)
                    period_analize = int(period_match.group(1)) if period_match else 0
                    
                    # B. Uzmi istoriju PRE ovog kola
                    istorija_pre_kola = self.loto_df[self.loto_df['kolo'] < kolo]
                    
                    # C. Kreiraj model verovatnoće
                    model_verovatnoce = self.izracunaj_model_verovatnoce(istorija_pre_kola, period_analize)
                    
                    # D. Nađi najmanji indeks iznenađenja u setu
                    minimalni_indeks_iznenadjenja = float('inf')
                    for komb_str in sve_kombinacije_u_setu:
                        try:
                            kombinacija_lista = list(eval(komb_str))
                            trenutni_indeks = self.izracunaj_indeks_iznenadjenja(kombinacija_lista, model_verovatnoce)
                            if trenutni_indeks is not None:
                                minimalni_indeks_iznenadjenja = min(minimalni_indeks_iznenadjenja, trenutni_indeks)
                        except: continue
                    
                    if minimalni_indeks_iznenadjenja != float('inf'):
                        indeks_iznenadjenja = minimalni_indeks_iznenadjenja
                except Exception as e: print(f"Greška pri računanju indeksa iznenađenja za bektest ID {bektest_id}: {e}")

                # 5. Ažuriranje baze sa svim rezultatima (IZMENJENO)
                cursor.execute("UPDATE virtualne_igre SET rezultat = ?, indeks_promasaja = ?, indeks_iznenadjenja = ? WHERE id = ?", (finalni_rezultat, indeks_promasaja, indeks_iznenadjenja, bektest_id))
            self.db_manager.db_conn.commit()

        self.osvezi_tabelu_tiketa()
        self.osvezi_tabelu_istorije()
        self.osvezi_tabelu_bektesta()
        
        if uspeh_dodavanja:
            self.osvezi_sve_analize() # Osvežavamo sve analize sa novim kolom
            QMessageBox.information(self, "Uspeh", f"Kolo {kolo} je uspešno dodato, tiketi i bektest su provereni!")
        else:
            QMessageBox.warning(self, "Provera Završena", f"Tiketi su provereni. Kolo {kolo} već postoji u bazi i nije ponovo dodato.")

    def dodaj_ml_tiket_u_pracenje(self):
        izabrani_tiket = self.ml_rezultati_output.currentItem()
        if not izabrani_tiket:
            QMessageBox.warning(self, "Greška", "Nije izabrana nijedna ML kombinacija.")
            return
        
        kombinacija_za_upis = "(ML)" + izabrani_tiket.text()
            
        print(f"Dodajem ML tiket u praćenje: {kombinacija_za_upis}")
        try:
            cursor = self.db_manager.db_conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO odigrani_tiketi (kombinacija) VALUES (?)", (kombinacija_za_upis,))
            self.db_manager.db_conn.commit()
            self.osvezi_tabelu_tiketa()
            QMessageBox.information(self, "Uspeh", f"ML Kombinacija {kombinacija_za_upis} je uspešno dodata u praćenje.")
        except Exception as e:
            QMessageBox.critical(self, "Greška Baze", f"Došlo je do greške pri upisu ML tiketa u bazu: {e}")

    def closeEvent(self, event):
        self.db_manager.close()
        event.accept()
        
    def pokreni_ai_preporuku(self):
        if not self.ai_model:
            QMessageBox.critical(self, "Greška", "AI model nije uspešno konfigurisan. Proverite API ključ u .env fajlu.")
            return
            
        broj_stavki = self.rezultati_output.count()
        if broj_stavki == 0:
            QMessageBox.warning(self, "Nema Kombinacija", "Prvo morate generisati listu kombinacija.")
            return
            
        kombinacije_za_slanje = [self.rezultati_output.item(i).text().split("| Kombinacija: ")[-1] for i in range(broj_stavki)]
        
        prompt_tekst = f"""Ti si stručni analitičar za Loto 7/39. Ja sam koristio program da filtriram i dobijem sledeću listu od {broj_stavki} potencijalno dobrih kombinacija:\n{', '.join(kombinacije_za_slanje)}\n\nTvoj zadatak je da iz ove liste izabereš i preporučiš mi TAČNO 8 kombinacija. Ne sme biti ni manje ni više od 8. Kriterijumi za tvoj izbor treba da budu raznovrsnost (pokrivanje što više različitih brojeva) i balans. Odgovor mi formatiraj isključivo kao listu od 8 kombinacija, svaku u novom redu, bez ikakvog dodatnog teksta ili objašnjenja."""
        
        dialog = ConfirmAIDialog(prompt_tekst, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.rezultati_output.clear()
            self.rezultati_output.addItem("🤖 Komuniciram sa Google AI... Molim sačekajte...")
            QApplication.processEvents()
            
            try:
                response = self.ai_model.generate_content(prompt_tekst)
                ai_odgovor = response.text
                self.rezultati_output.clear()
                self.rezultati_output.addItem("--- PREPORUKA OD GOOGLE AI ---")
                for line in ai_odgovor.strip().split('\n'):
                    cista_linija = line.strip().replace("*", "").replace("-", "").strip()
                    if cista_linija:
                        self.rezultati_output.addItem(cista_linija)
                self.db_manager.save_ai_log("Preporuka iz Generatora", prompt_tekst, ai_odgovor)
            except Exception as e:
                self.rezultati_output.clear()
                self.rezultati_output.addItem(f"Došlo je do greške pri komunikaciji sa AI: {e}")
            
            QApplication.restoreOverrideCursor()

    def pokreni_ai_analizu_tiketa(self):
        if not self.ai_model:
            QMessageBox.critical(self, "Greška", "AI model nije uspešno konfigurisan.")
            return
            
        sql_upit = "SELECT kombinacija, status, poslednji_rezultat FROM odigrani_tiketi"
        if self.samo_aktivni_checkbox.isChecked():
            sql_upit += " WHERE status = 'aktivan'"
            
        lista_tiketa_za_analizu = []
        cursor = self.db_manager.db_conn.cursor()
        cursor.execute(sql_upit)
        for red in cursor.fetchall():
            komb, stat, rez = red
            rez_str = str(rez) if rez is not None else "N/A"
            lista_tiketa_za_analizu.append(f"- Kombinacija: {komb}, Status: {stat}, Poslednji broj pogodaka: {rez_str}")
            
        if not lista_tiketa_za_analizu:
            QMessageBox.warning(self, "Nema Tiketa", "Nema izabranih tiketa za analizu.")
            return
            
        tiketi_string = "\n".join(lista_tiketa_za_analizu)
        
        prompt_tekst = f"""Ti si iskusan Loto analitičar i strateški savetnik. Tvoj zadatak je da se ponašaš kao ohrabrujući partner i da analiziraš moj stil igranja na osnovu liste tiketa koje pratim.\n\nKontekst opšte statistike:\n- Vrući brojevi (najčešći): {sorted(list(self.vruci_brojevi))}\n- Hladni brojevi (najređi): {sorted(list(self.hladni_brojevi))}\n- Prosečna SREDNJA VREDNOST dobitnih kombinacija je oko {self.globalni_prosek:.2f}.\n\nOvo je lista mojih tiketa za praćenje:\n{tiketi_string}\n\nMolim te, napiši mi analizu u partnerskom i konstruktivnom tonu. Struktura tvog odgovora treba da bude:\n1. **Pozitivna zapažanja:** Započni analizu tako što ćeš prvo istaći šta je dobro u mom izboru tiketa.\n2. **Konstruktivan predlog:** Nakon toga, daj mi jedan konkretan i prijateljski savet za budućnost.\n3. **Zaključak:** Završi sa ohrabrujućom porukom."""
        
        dialog = ConfirmAIDialog(prompt_tekst, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            print("Šaljem podatke o tiketima na AI analizu...")
            try:
                response = self.ai_model.generate_content(prompt_tekst)
                ai_odgovor = response.text
                QMessageBox.information(self, "AI Analiza Vaše Strategije", ai_odgovor)
                self.db_manager.save_ai_log("Analiza Stila Igranja", prompt_tekst, ai_odgovor)
            except Exception as e:
                QMessageBox.critical(self, "Greška", f"Došlo je do greške pri komunikaciji sa AI: {e}")
            QApplication.restoreOverrideCursor()

    def pokreni_ai_analizu_bektestova(self):
        if not self.ai_model:
            QMessageBox.critical(self, "Greška", "AI model nije uspešno konfigurisan. Proverite API ključ u .env fajlu.")
            return

        try:
            cursor = self.db_manager.db_conn.cursor()
            cursor.execute("SELECT kolo, filter_podesavanja, rezultat FROM virtualne_igre WHERE rezultat IS NOT NULL AND rezultat != '' ORDER BY kolo ASC")
            svi_bektestovi = cursor.fetchall()

            if len(svi_bektestovi) < 5:
                QMessageBox.warning(self, "Nedovoljno Podataka", "Nema dovoljno završenih bektestova (minimum 5) za smislenu AI analizu.")
                return

            formatirani_podaci = "\n".join([f"- Kolo {kolo} | Strategija: {podesavanja} | Rezultat: {rezultat}" for kolo, podesavanja, rezultat in svi_bektestovi])

            prompt_tekst = f"""**PERZONA:**
Ponašaj se kao vrhunski Data Scientist specijalizovan za analizu statističkih igara i prepoznavanje trendova. Budi temeljan, objektivan i traži zakonitosti u podacima koje ti dostavljam.

**KONTEKST:**
Analiziraš uspešnost različitih strategija za generisanje Loto 7/39 kombinacija. Svaka strategija je definisana periodom analize, načinom tretiranja "svežih" brojeva, i drugim filterima. Rezultat strategije se meri po tome koliko je pogodaka bilo u generisanom setu.

**PODACI:**
Evo podataka iz mojih poslednjih bektestova. Podaci su u formatu: Kolo | Strategija | Rezultat
{formatirani_podaci}

**TVOJ ZADATAK:**
Analiziraj ove podatke i odgovori mi na sledeća pitanja:
1.  **Identifikuj Pobedničku Strategiju:** Koja kombinacija filtera (koji tip redova) konzistentno daje najbolje rezultate (najviše pogodaka sa 4, 5 ili više)?
2.  **Pronađi Gubitničku Strategiju:** Postoji li neka strategija koja konstantno podbacuje i koju bi trebalo da napustim?
3.  **Analiziraj Parametre:** Da li primećuješ neku vezu? Na primer:
    - Da li kraći periodi analize (npr. 50-150 kola) daju bolje rezultate od dužih (preko 500)?
    - Da li strategija "Kažnjavaj sveže" daje bolje rezultate od "Favorizuj sveže"?
    - Da li određeni opseg srednje vrednosti u filterima češće dovodi do uspeha?
4.  **Daj Preporuku:** Na osnovu svega, koju strategiju ili kombinaciju filtera bi mi preporučio da nastavim da koristim i testiram u narednim kolima?
"""

            dialog = ConfirmAIDialog(prompt_tekst, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                print("Šaljem podatke o bektestovima na AI analizu...")
                try:
                    response = self.ai_model.generate_content(prompt_tekst)
                    ai_odgovor = response.text
                    self.db_manager.save_ai_log("Analiza Bektestova", prompt_tekst, ai_odgovor)
                    
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("AI Analiza Vaših Bektest Strategija")
                    msg_box.setText(ai_odgovor)
                    msg_box.setIcon(QMessageBox.Icon.Information)
                    msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
                    msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                    msg_box.exec()

                except Exception as e:
                    QMessageBox.critical(self, "Greška", f"Došlo je do greške pri komunikaciji sa AI: {e}")
                finally:
                    QApplication.restoreOverrideCursor()

        except Exception as e:
            QMessageBox.critical(self, "Greška Baze Podataka", f"Došlo je do greške prilikom čitanja bektestova: {e}")
            print(f"Greška kod AI analize bektesta: {e}")

    # --- Faza 2: Implementacija Logike Hi-Kvadrat Testa ---

    def pokreni_naprednu_analizu(self):
        """
        Glavna funkcija koja se poziva na klik dugmeta "Pokreni Analizu".
        Proverava koji je test izabran i poziva odgovarajuću funkciju.
        (Zadatak 2.3)
        """
        izabrani_test = self.test_selector_combo.currentText()
        
        if "Hi-Kvadrat Test" in izabrani_test:
            self.sprovedi_hi_kvadrat_test_pozicija()
        else:
            self.rezultat_testa_output.setText("Izaberite validan test.")

    def sprovedi_hi_kvadrat_test_pozicija(self):
        """
        Sprovodi Hi-Kvadrat test za analizu pristrasnosti pozicija izvlačenja.
        (Zadatak 2.4)
        """
        self.rezultat_testa_output.clear()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # 1. Prikupljanje Posmatranih Vrednosti (Observed)
            # Koristimo podatke iz cele istorije za najtačniji test
            self.ucitaj_i_analiziraj_podatke(period_analize=0) 
            observed_freq = self.poziciona_frekvencija.values.flatten()

            # Ako nema podataka, prekini
            if self.loto_df.empty or observed_freq.sum() == 0:
                self.rezultat_testa_output.setText("Nema dovoljno podataka za sprovođenje testa.")
                return

            # 2. Računanje Očekivanih Vrednosti (Expected)
            ukupan_broj_kola = len(self.loto_df)
            ukupan_broj_izvlacenja = ukupan_broj_kola * BROJEVA_U_KOMBINACIJI
            
            # Očekivana frekvencija za svako od 39*7 polja
            expected_value = ukupan_broj_izvlacenja / (MAX_BROJ * BROJEVA_U_KOMBINACIJI)
            
            # Kreiramo listu očekivanih vrednosti koja odgovara obliku posmatranih
            expected_freq = [expected_value] * len(observed_freq)

            # 3. Sprovođenje Testa
            # Uklanjamo polja gde je i posmatrana i očekivana vrednost 0 da ne utiču na test
            valid_indices = [i for i, obs in enumerate(observed_freq) if obs > 0]
            observed_valid = [observed_freq[i] for i in valid_indices]
            expected_valid = [expected_freq[i] for i in valid_indices]

            if not observed_valid:
                 self.rezultat_testa_output.setText("Nema validnih podataka za testiranje (sve frekvencije su 0).")
                 return

            chi2_stat, p_value = chisquare(f_obs=observed_valid, f_exp=expected_valid)

            # 4. Formatiranje Izlaza i Tumačenje
            prag_znacajnosti = 0.05
            
            izvestaj = []
            izvestaj.append("--- Hi-Kvadrat Test: Pristrasnost Pozicija Izvlačenja ---")
            izvestaj.append("")
            izvestaj.append(f"Analizirani period: Sva Kola ({ukupan_broj_kola})")
            izvestaj.append("-" * 55)
            izvestaj.append("Rezultati testa:")
            izvestaj.append(f"- Hi-kvadrat statistika: {chi2_stat:.2f}")
            izvestaj.append(f"- P-vrednost (p-value):   {p_value:.4f}")
            izvestaj.append(f"- Prag značajnosti (α): {prag_znacajnosti}")
            izvestaj.append("-" * 55)
            izvestaj.append("Zaključak:")

            if p_value < prag_znacajnosti:
                izvestaj.append(f"P-vrednost ({p_value:.4f}) je MANJA od praga značajnosti ({prag_znacajnosti}).")
                izvestaj.append("Na osnovu dostupnih podataka, POSTOJI statistički značajan dokaz")
                izvestaj.append("koji ukazuje na to da proces izvlačenja brojeva po pozicijama NIJE")
                izvestaj.append("potpuno nasumičan, tj. da postoji određena pristrasnost.")
            else:
                izvestaj.append(f"P-vrednost ({p_value:.4f}) je VEĆA od praga značajnosti ({prag_znacajnosti}).")
                izvestaj.append("Na osnovu dostupnih podataka, NE POSTOJI statistički značajan dokaz")
                izvestaj.append("koji bi ukazao na to da je proces izvlačenja brojeva po pozicijama pristrasan.")
                izvestaj.append("Uočena odstupanja su u granicama očekivanog za nasumičan proces.")

            self.rezultat_testa_output.setText("\n".join(izvestaj))

        except Exception as e:
            self.rezultat_testa_output.setText(f"Došlo je do greške prilikom sprovođenja testa:\n{e}")
            print(f"Greška u Hi-Kvadrat testu: {e}")
        finally:
            QApplication.restoreOverrideCursor()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    glavni_prozor = LotoAnalizator()
    sys.exit(app.exec())