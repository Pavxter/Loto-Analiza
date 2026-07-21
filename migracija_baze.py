"""Jednokratna migracija: smanjenje baze podataka.

Stari format bektesta je čuvao SVE kombinacije kao ogroman tekstualni blob
(kolona virtualne_igre.lista_kombinacija), zbog čega je baza narasla na ~59 MB.
Ovi stari bektestovi su za već izvučena kola i imaju izračunat rezultat, pa nam
sirove kombinacije više nisu potrebne.

Skripta:
  1. Pravi rezervnu kopiju (loto_baza_backup.db).
  2. Prazni lista_kombinacija za stare (semicolon) zapise koji su već ocenjeni.
  3. Pokreće VACUUM da fizički oslobodi prostor.

Pokretanje:  python migracija_baze.py
Zaustavi web server pre pokretanja (da nema aktivnih konekcija za VACUUM).
"""

import os
import shutil
import sqlite3
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from webapp.core import konfig

PUTANJA = konfig.PUTANJA_BAZE
BACKUP = os.path.join(konfig.KOREN_PROJEKTA, "loto_baza_backup.db")


def velicina_mb(putanja):
    return os.path.getsize(putanja) / (1024 * 1024)


def main():
    if not os.path.exists(PUTANJA):
        print(f"Baza nije pronađena: {PUTANJA}")
        return

    pre = velicina_mb(PUTANJA)
    print(f"Veličina pre: {pre:.1f} MB")

    print(f"Pravim rezervnu kopiju -> {BACKUP}")
    shutil.copy2(PUTANJA, BACKUP)

    conn = sqlite3.connect(PUTANJA)
    try:
        cur = conn.cursor()
        # Broj i veličina blobova pre
        red = cur.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(lista_kombinacija)),0) FROM virtualne_igre "
            "WHERE lista_kombinacija IS NOT NULL AND lista_kombinacija != '' "
            "AND lista_kombinacija NOT LIKE '[%'").fetchone()
        print(f"Starih (blob) bektesta: {red[0]}, ukupno {red[1]/1024/1024:.1f} MB teksta")

        # Prazni samo stari format (ne dira nove JSON zapise koji počinju sa '[')
        cur.execute(
            "UPDATE virtualne_igre SET lista_kombinacija='' "
            "WHERE lista_kombinacija IS NOT NULL AND lista_kombinacija != '' "
            "AND lista_kombinacija NOT LIKE '[%'")
        obrisano = cur.rowcount
        conn.commit()
        print(f"Ispražnjeno zapisa: {obrisano}")

        print("Pokrećem VACUUM...")
        cur.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()

    posle = velicina_mb(PUTANJA)
    print(f"Veličina posle: {posle:.1f} MB  (ušteda {pre - posle:.1f} MB)")
    print("Gotovo. Ako nešto nije u redu, vrati kopiju loto_baza_backup.db.")


if __name__ == "__main__":
    main()
