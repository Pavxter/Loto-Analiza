# data_manager.py

import sqlite3
import pandas as pd
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='loto_baza.db'):
        """Inicijalizuje konekciju i postavlja bazu ako je potrebno."""
        self.db_conn = sqlite3.connect(db_path)
        self.setup_database()

    def setup_database(self):
        """Kreira tabele ako ne postoje i vrši migracije."""
        cursor = self.db_conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS istorijski_rezultati (id INTEGER PRIMARY KEY AUTOINCREMENT, kolo INTEGER, datum TEXT, b1 INTEGER, b2 INTEGER, b3 INTEGER, b4 INTEGER, b5 INTEGER, b6 INTEGER, b7 INTEGER, UNIQUE(kolo, datum))'''); cursor.execute('''CREATE TABLE IF NOT EXISTS odigrani_tiketi (id INTEGER PRIMARY KEY AUTOINCREMENT, kombinacija TEXT UNIQUE, status TEXT DEFAULT 'aktivan', poslednji_rezultat INTEGER, datum_provere TEXT, dodatne_metrike TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS ai_log (id INTEGER PRIMARY KEY AUTOINCREMENT, datum_vreme TEXT, tip_zahteva TEXT, prompt TEXT, odgovor TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS virtualne_igre (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            kolo INTEGER, 
            datum_kreiranja TEXT, 
            filter_podesavanja TEXT, 
            lista_kombinacija TEXT, 
            broj_kombinacija INTEGER, 
            rezultat TEXT, 
            bazen_brojeva TEXT, 
            indeks_promasaja INTEGER,
            indeks_iznenadjenja REAL,
            UNIQUE(kolo, filter_podesavanja)
        )''')
        
        # Migracija: Dodavanje kolone 'bazen_brojeva' ako ne postoji
        try:
            cursor.execute("PRAGMA table_info(virtualne_igre)")
            kolone = [info[1] for info in cursor.fetchall()]
            if 'bazen_brojeva' not in kolone:
                print("--- MIGRACIJA BAZE: Dodajem kolonu 'bazen_brojeva' u tabelu 'virtualne_igre'... ---")
                cursor.execute("ALTER TABLE virtualne_igre ADD COLUMN bazen_brojeva TEXT")
                self.db_conn.commit()
                print("--- Migracija uspešna! ---")
        except Exception as e:
            print(f"Greška prilikom migracije baze (dodavanje kolone 'bazen_brojeva'): {e}")

        # Migracija: Dodavanje kolone 'indeks_promasaja' ako ne postoji
        try:
            cursor.execute("PRAGMA table_info(virtualne_igre)")
            kolone = [info[1] for info in cursor.fetchall()]
            if 'indeks_promasaja' not in kolone:
                print("--- MIGRACIJA BAZE: Dodajem kolonu 'indeks_promasaja' u tabelu 'virtualne_igre'... ---")
                cursor.execute("ALTER TABLE virtualne_igre ADD COLUMN indeks_promasaja INTEGER")
                self.db_conn.commit()
                print("--- Migracija uspešna! ---")
        except Exception as e:
            print(f"Greška prilikom migracije baze (dodavanje kolone 'indeks_promasaja'): {e}")

        # Migracija: Dodavanje kolone 'indeks_iznenadjenja' ako ne postoji
        try:
            cursor.execute("PRAGMA table_info(virtualne_igre)")
            kolone = [info[1] for info in cursor.fetchall()]
            if 'indeks_iznenadjenja' not in kolone:
                print("--- MIGRACIJA BAZE: Dodajem kolonu 'indeks_iznenadjenja' u tabelu 'virtualne_igre'... ---")
                cursor.execute("ALTER TABLE virtualne_igre ADD COLUMN indeks_iznenadjenja REAL")
                self.db_conn.commit()
                print("--- Migracija uspešna! ---")
        except Exception as e:
            print(f"Greška prilikom migracije baze (dodavanje kolone 'indeks_iznenadjenja'): {e}")

        # Migracija: Dodavanje kolone 'dodatne_metrike' u tabelu 'odigrani_tiketi'
        try:
            cursor.execute("PRAGMA table_info(odigrani_tiketi)")
            kolone = [info[1] for info in cursor.fetchall()]
            if 'dodatne_metrike' not in kolone:
                print("--- MIGRACIJA BAZE: Dodajem kolonu 'dodatne_metrike' u tabelu 'odigrani_tiketi'... ---")
                cursor.execute("ALTER TABLE odigrani_tiketi ADD COLUMN dodatne_metrike TEXT")
                self.db_conn.commit(); print("--- Migracija uspešna! ---")
        except Exception as e:
            print(f"Greška prilikom migracije baze (dodavanje kolone 'dodatne_metrike'): {e}")

        self.db_conn.commit()
        self.import_from_csv_if_needed('loto_podaci.csv')

    def import_from_csv_if_needed(self, csv_path='loto_podaci.csv'):
        """Proverava da li je baza prazna i ako jeste, uvozi podatke iz CSV fajla."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM istorijski_rezultati")
            if cursor.fetchone()[0] > 0:
                # Baza već ima podatke, ne radimo ništa.
                return

            if not os.path.exists(csv_path):
                print(f"Info: Baza je prazna, ali fajl '{csv_path}' nije pronađen. Preskačem automatski uvoz.")
                return

            print(f"--- Info: Baza je prazna. Pokušavam da uvezem podatke iz '{csv_path}'... ---")
            df = pd.read_csv(csv_path)
            # Očekivane kolone: 'kolo', 'datum', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'
            
            uspesno_uvezeno = 0
            for index, row in df.iterrows():
                brojevi = [row['b1'], row['b2'], row['b3'], row['b4'], row['b5'], row['b6'], row['b7']]
                if self.add_historical_result(row['kolo'], row['datum'], brojevi):
                    uspesno_uvezeno += 1
            
            print(f"--- Automatski uvoz završen. Uvezeno {uspesno_uvezeno} kola. ---")

        except Exception as e:
            print(f"GREŠKA prilikom automatskog uvoza iz CSV fajla: {e}")

    def get_all_historical_results_as_df(self):
        """Vraća sve istorijske rezultate kao pandas DataFrame."""
        return pd.read_sql_query("SELECT * FROM istorijski_rezultati ORDER BY id ASC", self.db_conn)

    def add_historical_result(self, kolo, datum, brojevi):
        """Dodaje novi istorijski rezultat i vraća True ako je uspešno, inače False."""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("INSERT OR IGNORE INTO istorijski_rezultati (kolo, datum, b1, b2, b3, b4, b5, b6, b7) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                           (kolo, datum, *brojevi))
            self.db_conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            print(f"Greška pri dodavanju istorijskog rezultata: {e}")
            return False

    def save_ai_log(self, tip_zahteva, prompt, odgovor):
        """Čuva zapis o komunikaciji sa AI modelom."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("INSERT INTO ai_log (datum_vreme, tip_zahteva, prompt, odgovor) VALUES (?, ?, ?, ?)", 
                           (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tip_zahteva, prompt, odgovor))
            self.db_conn.commit()
        except sqlite3.Error as e:
            print(f"Greška pri čuvanju AI loga: {e}")

    def close(self):
        """Zatvara konekciju sa bazom."""
        if self.db_conn:
            self.db_conn.close()
            print("Konekcija sa bazom podataka je zatvorena.")