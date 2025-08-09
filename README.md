# Loto Analizator v10.4

Aplikacija za naprednu statističku analizu Loto 7/39 (i sličnih) igara, razvijena u Python-u korišćenjem PySide6 i Pandas biblioteka.

## Ključne Funkcionalnosti

*   **Strateški Dashboard:** Centralni pregled sa ključnim pokazateljima, mini-grafikonima i predlogom bazena brojeva.
*   **Detaljne Analize:**
    *   Frekvencija brojeva (vrući, hladni, sveži).
    *   Distribucija srednjih vrednosti kombinacija.
    *   Prosečan razmak između ponavljanja brojeva.
    *   Učestalost uzastopnih brojeva i brojeva po dekadama.
    *   Analiza redosleda izvlačenja (poziciona analiza).
*   **Generator Kombinacija:** Moćan alat za generisanje kombinacija na osnovu preko 10 različitih filtera i strategija.
*   **Kreator Bazena:** Interfejs za kreiranje prilagođenih bazena brojeva fuzijom različitih trendova (vrući, hladni, sveži).
*   **Bajesovska Analiza:** Iterativni model koji rangira brojeve po "stepenu verovanja" na osnovu učenja iz celokupne istorije.
*   **Bajesovski Hibridni Model:** Napredni model koji kombinuje Bajesovsko učenje sa analizom povezanosti brojeva za kreiranje optimizovanog bazena. Model je stabilizovan da uvek daje iste rezultate za isti set podataka.
*   **ML Generator (VAE):** Generator koji koristi Variational Autoencoder (VAE) neuronsku mrežu za učenje "suštine" dobitnih kombinacija i generisanje novih.
*   **Praćenje Tiketa i Bektest:** Mogućnost unosa i praćenja sopstvenih tiketa, kao i čuvanje i analiza uspešnosti generisanih setova.
*   **AI Integracija:** Korišćenje Google AI (Gemini) za analizu strategija, bektestova i preporuke kombinacija.

## Poslednje Izmene (v10.4)

*   **Stabilizacija Hibridnog Modela:** Logika Bajesovskog hibridnog modela je ispravljena kako bi se osigurali dosledni i deterministički rezultati pri svakom pokretanju. Model sada kreće od neutralne početne tačke i obrađuje podatke hronološki.
*   **Dodata Funkcionalnost:** U tab "Bajesovski Hibrid" dodato je dugme "Koristi kao Bazen u Generatoru" za lakši prenos strategije.
*   **Ispravka Logike:** Rešena je greška u prethodnoj implementaciji hibridnog modela koja je mogla dovesti do nedoslednosti.

## Pokretanje

1.  Instalirati sve potrebne biblioteke:
    `pip install -r requirements.txt`
2.  Kreirati `.env` fajl u glavnom direktorijumu i uneti API ključ za Google AI:
    `GEMINI_API_KEY='VAS_API_KLJUC'`
3.  Pokrenuti aplikaciju:
    `python analiza.py`