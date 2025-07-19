# data_manager.py

import sqlite3
import pandas as pd
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='loto_baza.db'):
        """Inicijalizuje konekciju i postavlja bazu ako je potrebno."""
        self.db_conn = sqlite3.connect(db_path)
        self.setup_database()

    def setup_database(self):
        """Kreira tabele ako ne postoje i vrši migracije."""
        cursor = self.db_conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS istorijski_rezultati (id INTEGER PRIMARY KEY AUTOINCREMENT, kolo INTEGER, datum TEXT, b1 INTEGER, b2 INTEGER, b3 INTEGER, b4 INTEGER, b5 INTEGER, b6 INTEGER, b7 INTEGER, UNIQUE(kolo, datum))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS odigrani_tiketi (id INTEGER PRIMARY KEY AUTOINCREMENT, kombinacija TEXT UNIQUE, status TEXT DEFAULT 'aktivan', poslednji_rezultat INTEGER, datum_provere TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS ai_log (id INTEGER PRIMARY KEY AUTOINCREMENT, datum_vreme TEXT, tip_zahteva TEXT, prompt TEXT, odgovor TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS virtualne_igre (id INTEGER PRIMARY KEY AUTOINCREMENT, kolo INTEGER, datum_kreiranja TEXT, filter_podesavanja TEXT, lista_kombinacija TEXT, broj_kombinacija INTEGER, rezultat TEXT, bazen_brojeva TEXT, UNIQUE(kolo, filter_podesavanja))''')
        
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

        self.db_conn.commit()

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