# Loto Analizator v10.6

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
*   **Kreator Bazena:** Interfejs za kreiranje prilagođenih bazena brojeva fuzijom različitih trendova (vrući, hladni, sveži). Od sada podržava i direktno bektestiranje kompletnog generisanog bazena.
*   **Bajesovska Analiza:** Iterativni model koji rangira brojeve po "stepenu verovanja" na osnovu učenja iz celokupne istorije.
*   **Bajesovski Hibridni Model:** Napredni model koji kombinuje Bajesovsko učenje sa analizom povezanosti brojeva za kreiranje optimizovanog bazena. Model je stabilizovan da uvek daje iste rezultate za isti set podataka.
*   **ML Generator (VAE):** Generator koji koristi Variational Autoencoder (VAE) neuronsku mrežul. Sada podržava dve strategije:
    *   **Generisanje Bazena Brojeva:** Analizira veliki broj generisanih kombinacija i rangira sve brojeve po ML skoru (frekvenciji pojavljivanja), omogućavajući kreiranje strategija zasnovanih na najverovatnijim brojevima.
    *   **Generisanje Gotovih Kombinacija:** Klasičan pristup generisanja kompletnih, gotovih kombinacija.
*   **Praćenje Tiketa i Bektest:** Mogućnost unosa i praćenja sopstvenih tiketa, kao i čuvanje i analiza uspešnosti generisanih setova.
*   **AI Integracija:** Korišćenje Google AI (Gemini) za analizu strategija, bektestova i preporuke kombinacija.

## Poslednje Izmene (v10.6)

*   **Direktno Bektestiranje Bazena:** U "Kreator Bazena" dodata je opcija "Sačuvaj CEO Bazen za Bektest". Ova funkcija omogućava korisniku da generiše i sačuva SVE moguće kombinacije iz kreiranog bazena, pružajući način za testiranje punog potencijala strategije bez dodatnih filtera.

## Poslednje Izmene (v10.5)

*   **Unapređen ML Generator:** Tab "ML Generator" je redizajniran. Umesto jedne akcije, sada nudi dve odvojene: "Generiši Bazen Brojeva" i "Generiši Gotove Kombinacije".
*   **Prikaz ML Bazena:** Dodata je tabela za jasan prikaz rangiranih brojeva i njihovog ML skora.
*   **Čuvanje ML Strategije:** Implementirana je logika za čuvanje strategije zasnovane na "Top N" brojeva iz generisanog ML bazena, omogućavajući budući bektesting.

## Pokretanje

1.  Instalirati sve potrebne biblioteke:
    `pip install -r requirements.txt`
2.  Kreirati `.env` fajl u glavnom direktorijumu i uneti API ključ za Google AI:
    `GEMINI_API_KEY='VAS_API_KLJUC'`
3.  Pokrenuti aplikaciju:
    `python analiza.py`