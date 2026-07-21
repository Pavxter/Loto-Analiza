"""Konfiguracija pravila igre i putanja.

Konstante su izvučene iz analiza.py i parametrizovane tako da aplikacija može da
podrži i druge formate igre (npr. 6/45) bez menjanja logike.
"""

import os

# --- Pravila igre (podrazumevano Loto 7/39) ---
MAX_BROJ = 39                    # Najveći mogući broj u igri
BROJEVA_U_KOMBINACIJI = 7        # Koliko se brojeva izvlači
BROJ_KATEGORIJA_FREKV = 13       # Koliko brojeva ulazi u "vruće" i "hladne"
PERIOD_SVEZIH_KOLA = 10          # Prozor za "sveže" brojeve

# Nazivi kolona sa brojevima u tabeli istorijskih rezultata
KOLONE_ZA_BROJEVE = [f"b{i}" for i in range(1, BROJEVA_U_KOMBINACIJI + 1)]

# --- Putanje ---
# Koren projekta je roditelj foldera 'webapp'
KOREN_PROJEKTA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PUTANJA_BAZE = os.path.join(KOREN_PROJEKTA, "loto_baza.db")


def sve_brojeve():
    """Vraća listu svih mogućih brojeva [1..MAX_BROJ]."""
    return list(range(1, MAX_BROJ + 1))
