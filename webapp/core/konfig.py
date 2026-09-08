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

# --- Sekvencijalni prediktor (PLAN_SEKVENCIJALNI_PREDIKTOR §2.3, §6) ---
# Parametri se biraju na sintetici i FIKSIRAJU. Podešavanje na istorijskim podacima
# „dok K ne padne ispod 1" je preučavanje i plan ga izričito zabranjuje (§8).
ETA_HEDGE = 0.05             # korak eksponencijalnih težina (Hedge)
TEMPERATURA_SOFTMAX = 1.0    # oštrina pretvaranja ocena prediktora u raspodelu
ALFA_DELJENJA = 0.01         # fixed-share: deo težine koji se posle svakog kola
                             # ravnomerno preraspodeli, da nijedan ekspert ne umre
LAMBDA_OSTRINE = 0.3         # svaki ekspert = (1−λ)·uniformni + λ·njegova raspodela;
                             # isti λ za sve, pa težine mere sadržaj a ne ton
SEKV_PERIOD = 100            # prozor koji eksperti vide; jednak prognoza.RETRO_PERIOD
SEKV_MIN_START = 50          # preskoči prva kola; jednak prognoza.MIN_START
SNAGA_PRIORA_PRELAZA = 50    # pseudo-posmatranja koja drže eksperte prelaza na teoriji
                             # dok stvarni brojači ne skupe dovoljno podataka (§2.2)

# --- Korak izbora sedmorke (PLAN_KORAK_IZBORA §2.4) ---
# Prag iznad kog raspon verovatnoća prestaje da bude šum. NIJE izabran po osećaju:
# model je pušten na pet uniformnih sintetičkih istorija od po 1.500 kola (čista
# slučajnost, ništa za naučiti) i uzet je 95. percentil izmerenog `raspon_udeo`.
# Ispod praga raspon je ono što šum sam po sebi proizvodi, pa model nema
# preferenciju. Postupak je zapisan u test_raspon_p_mix_na_sintetici; vrednost se
# menja samo ponovnim merenjem, sa novim datumom.
SEKV_BAZEN = 15              # koliko najverovatnijih brojeva ulazi u bazen iz kog
                             # Generator bira tiket. Predlog modela NE prolazi kroz
                             # Generator (§2.1): filteri rade na razlikama reda 10⁻²
                             # dok je razlika kandidata 4·10⁻⁴, pa bi Generator u
                             # potpunosti preuzeo izbor i predlog bi prestao da
                             # svedoči o modelu. Zato dva odvojena izlaza.
PRAG_RASPONA = 0.0814        # 95. percentil na 7.250 koraka čistog šuma
                             # (semena 17/23/31/47/59), izmereno 2026-09-08.
                             # Za poređenje: medijana šuma je 0,0597, a raspon
                             # izmeren na kolu 2026072 je 0,057 — ispod medijane.

# --- Putanje ---
# Koren projekta je roditelj foldera 'webapp'
KOREN_PROJEKTA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PUTANJA_BAZE = os.path.join(KOREN_PROJEKTA, "loto_baza.db")


def sve_brojeve():
    """Vraća listu svih mogućih brojeva [1..MAX_BROJ]."""
    return list(range(1, MAX_BROJ + 1))
