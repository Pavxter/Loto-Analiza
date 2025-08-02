# Loto Analizator

Napredna desktop aplikacija za statističku analizu, generisanje i praćenje Loto 7/39 kombinacija, napisana u Python-u uz pomoć PySide6 i Google Gemini AI.

## O Projektu

Loto Analizator je sveobuhvatan alat dizajniran da pomogne igračima Lota 7/39 da donose informisane odluke. Aplikacija pruža detaljne statističke analize istorijskih podataka, moćan generator kombinacija sa finim podešavanjima, kao i integraciju sa veštačkom inteligencijom za dublje uvide i preporuke.

Ceo projekat, uključujući kod i korisnički interfejs, je napisan na **srpskom jeziku**.

## Ključne Funkcionalnosti

Aplikacija je organizovana u tabove, od kojih svaki nudi specifičnu funkcionalnost:

*   **Strateški Dashboard:**
    *   Početni ekran aplikacije koji pruža brz strateški pregled.
    *   Prikazuje ključne pokazatelje (prosek, standardna devijacija, par/nepar odnos) za izabrani period.
    *   Lista "vrućih" i "hladnih" brojeva.
    *   Mini-grafikoni za vizuelni uvid u frekvenciju i distribuciju srednjih vrednosti.
    *   Automatski generiše **predlog bazena brojeva** kombinovanjem vrućih i svežih trendova.
    *   Omogućava direktno prebacivanje predložene strategije u **Generator Kombinacija** jednim klikom.

*   **Generator Kombinacija:**
    *   Generisanje kombinacija na osnovu niza detaljnih filtera:
        *   Opseg srednje vrednosti kombinacije.
        *   Broj parnih/neparnih brojeva.
        *   Broj "vrućih", "hladnih" i "neutralnih" brojeva na osnovu istorijske frekvencije.
        *   Broj uzastopnih parova.
        *   Maksimalan broj brojeva iz iste dekade.
    *   Mogućnost generisanja iz celokupnog opsega (1-39) ili iz prilagođenog **bazena brojeva**.
    *   Bodovanje i rangiranje generisanih kombinacija.
    *   **Bodovanje po Pristrasnosti Mašine:** Opciono primenjuje dodatni nivo bodovanja koji favorizuje kombinacije koje se poklapaju sa statistički "vrućim" pozicijama za određene brojeve, na osnovu Hi-Kvadrat analize.
    *   Filter za **diverzitet** koji eliminiše previše slične kombinacije.

*   **Kreator Bazena:**
    *   Interaktivni alat za kreiranje prilagođenih bazena brojeva.
    *   Analizira željeni period (npr. poslednjih 100 kola) i izdvaja vruće, hladne i "sveže" brojeve.
    *   Omogućava fuziju ovih grupa kako bi se kreirao optimizovan bazen za generator.

*   **ML Generator (Veštačka Inteligencija):**
    *   Koristi **Varijacioni Autoenkoder (VAE)**, neuronsku mrežu koja uči "suštinu" dobitnih kombinacija.
    *   Mogućnost treniranja modela na postojećim istorijskim podacima.
    *   Generisanje potpuno novih, statistički jedinstvenih kombinacija koje liče na dobitne.

*   **Integracija sa Google Gemini AI:**
    *   **Preporuka Kombinacija:** Pošaljite listu generisanih kombinacija AI modelu i zatražite preporuku za 8 najboljih.
    *   **Analiza Stila Igranja:** AI analizira vaše sačuvane tikete i daje konstruktivne savete i zapažanja o vašoj strategiji.
    *   **Analiza Bektestova:** AI analizira uspešnost vaših sačuvanih strategija i identifikuje koje kombinacije filtera daju najbolje rezultate.

*   **Statističke Analize i Vizualizacije:**
    *   Grafikoni frekvencije, srednjih vrednosti, prosečnog razmaka ponavljanja, uzastopnih parova, broja brojeva po dekadi i drugi.
    *   Analiza kretanja srednje vrednosti kroz vreme (vremenska serija).
    *   Mapa učestalosti brojeva po poziciji izvlačenja.
    *   Mogućnost analize celokupne istorije ili samo određenog broja poslednjih kola.

*   **Praćenje Tiketa i Istorije:**
    *   Dodajte, menjajte i brišite vaše odigrane tikete.
    *   Automatska provera tiketa nakon unosa novog kola.
    *   Pregled, izmena i brisanje istorijskih izvlačenja.
    *   Izvoz istorije u CSV format.

*   **Bektest Strategija:**
    *   Sačuvajte setove generisanih kombinacija (iz glavnog ili ML generatora) kao "virtualnu igru".
    *   Aplikacija automatski proverava uspešnost sačuvanih setova kada se unese rezultat za to kolo, pružajući uvid u efikasnost različitih strategija.
    *   **Indeks Promašaja:** Pored direktnih pogodaka, sistem meri i "blizinu" promašaja. Za svaki bektest, izračunava se najmanja ukupna numerička udaljenost od dobitne kombinacije, dajući uvid u to koliko je strategija bila "blizu".
    *   **Indeks Iznenađenja:** Kvantifikuje koliko je neka strategija statistički "neočekivana" ili "retka" u odnosu na istorijske podatke. Veći indeks znači da je strategija generisala kombinacije koje su ređe viđene u prošlosti.

*   **Napredne Analize:**
    *   Sprovodi rigorozne statističke testove za proveru hipoteza.
    *   **Hi-Kvadrat Test:** Analizira da li postoji statistički značajna pristrasnost u pozicijama na kojima se brojevi izvlače.

## Tehnologije

*   **GUI:** PySide6
*   **Analiza Podataka:** pandas
*   **Statistički Testovi:** scipy
*   **Baza Podataka:** sqlite3
*   **Grafikoni:** matplotlib, seaborn
*   **AI Model:** Google Gemini (preko `google-generativeai`)
*   **ML Model:** Custom VAE (pomoću `tensorflow` - naveden u `ml_generator.py`)

## Pokretanje Aplikacije

1.  **Klonirajte repozitorijum:**
    ```bash
    git clone https://github.com/Pavxter/Loto-Analiza.git
    cd Loto-Analiza
    ```

2.  **Instalirajte zavisnosti:**
    Preporučuje se kreiranje virtualnog okruženja.
    ```bash
    python -m venv .venv
    # Na Windows-u:
    .venv\Scripts\activate
    # Na macOS/Linux-u:
    # source .venv/bin/activate
    
    pip install -r requirements.txt
    ```

3.  **Podesite API Ključ:**
    *   Kreirajte fajl pod nazivom `.env` u glavnom direktorijumu projekta.
    *   U `.env` fajl dodajte svoj Google Gemini API ključ na sledeći način:
        ```
        GEMINI_API_KEY="VAS_API_KLJUC_OVDE"
        ```

4.  **Pokrenite program:**
    ```bash
    python analiza.py
    ```
    Prilikom prvog pokretanja, ako je baza podataka `loto_baza.db` prazna i ako postoji `loto_podaci.csv` fajl u direktorijumu, baza će biti automatski popunjena istorijskim podacima.
