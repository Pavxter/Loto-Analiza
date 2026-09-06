# Loto Analizator — Web (v1)

Interaktivna web verzija Loto Analizatora. Ponovo koristi svu dokazanu analitiku iz
originalne desktop aplikacije (`analiza.py`), ali kroz moderan browser interfejs sa
interaktivnim grafikonima i čistijom arhitekturom.

> **Napomena:** Loto izvlačenja su nezavisni slučajni događaji. Ova aplikacija služi za
> istraživanje istorije i generisanje kombinacija po pravilima — **ne predviđa** buduće brojeve.

## Pokretanje

```bash
pip install -r requirements-web.txt
python pokreni.py
```

Otvara se `http://127.0.0.1:8000`. Opcije: `python pokreni.py --port 9000 --bez-otvaranja`.

Za grafikone i mapu je potreban internet (ECharts, Alpine.js i Leaflet se učitavaju
preko CDN-a).

Statički fajlovi se serviraju sa `Cache-Control: no-cache`, pa posle izmena običan
refresh (F5) uvek pokaže najnoviju verziju (nepromenjeni fajlovi → HTTP 304 preko ETag-a).
Izuzetak su pločice mape: one se ne menjaju dok se ne pokrene `generisi_mapu.py`, pa se
keširaju dugoročno da pomeranje mape ne bi slalo stotine revalidacija.

### Pločice mape (jednokratno)

Strana „Mapa kombinacija" se ne crta iz sirovih podataka nego iz unapred ispečenih
slika, koje **ne idu u git** (`webapp/static/mapa/`). Pre prvog korišćenja:

```bash
python -X utf8 generisi_mapu.py --sloj sve
```

Traje oko minut i po i zauzima oko 46 MB. Dok pločice ne postoje, tab pokazuje uputstvo
umesto mape; sve ostale strane rade normalno. Sloj „Ocena Generatora" čita bazu, pa se
peče za stanje u tom trenutku — ponovi skriptu (`--sloj ocena`) kad želiš da ga osvežiš.

## Strane

Redosledom kako se pojavljuju u aplikaciji (detaljan opis svake je u `FUNKCIJE.md`):

1. **Dashboard** — ključni pokazatelji, frekvencija, predlog bazena.
2. **Istraži istoriju** — vremeplov kroz kola: bira se bilo koje odigrano kolo i vidi
   tačno ono što je sistem znao „tada" (bez curenja budućnosti) — detalj broja, sažetak
   perioda, istorijska različitost/rangiranje i „predikcija tada" naspram stvarnog ishoda.
3. **Mapa kombinacija** — ceo prostor od 15.380.937 kombinacija kao zumabilna mapa:
   izvučena kola kao tačke, njihov hronološki redosled kao putanja obojena preklapanjem,
   uz obavezni kontrolni (slučajni) sloj, vremenski slajder i „gde je moj tiket".
4. **Statistika** — frekvencija, srednje vrednosti, trend, ritam, uzastopni, dekade,
   poziciona heatmapa i hi-kvadrat test.
5. **Različitost** — koliko se izvučene kombinacije preklapaju vs. teorijska slučajnost
   (rekordi, uzastopna/svi parovi, profil vs. sadržaj, ko-okurencija).
6. **Rangiranje** — brojevi rangirani metodom Frekvencija / Bajes / Hibrid.
7. **Prognoza** — statistički eksperiment: predviđanje jednog broja i cele kombinacije
   za sledeće kolo (7 metoda + kontrola, uživo i retro-bektest, testovi značajnosti).
8. **Generator** — kombinacije po filterima (par/nepar, vrući/hladni, sredina, uzastopni,
   dekade, diverzitet) uz bodovanje; opcioni „vremeplov" (analiza do zadatog kola).
9. **Bektest** — uspešnost sačuvanih strategija (rezultat, indeks promašaja/iznenađenja).
10. **Moji tiketi** — evidencija odigranih tiketa i njihovih pogodaka.
11. **Podaci** — unos novog kola (auto-provera tiketa, bektesta i prognoza) + uvoz CSV/Excel.

## Arhitektura

```
webapp/
  core/                  # čista analitika (bez UI-ja) — deljena logika
    konfig.py              # pravila igre (MAX_BROJ, itd.)
    baza.py                # SQLite sloj
    analitika.py           # frekvencija, srednje vrednosti, poziciona, model pristrasnosti, detalj broja
    rangiranje.py          # frekvencija / Bajes / hibrid + matrica povezanosti
    generator.py           # filteri + bodovanje + diverzitet
    bektest.py             # indeksi + "dodaj kolo i proveri sve"
    razlicitost_teorija.py # hipergeometrijska teorija preklapanja + bitmaske i testovi
    razlicitost.py         # analize preklapanja izvučenih kombinacija
    prediktori.py          # 7 jednobrojnih prediktora (čiste funkcije, bez curenja)
    prediktori_komb.py     # kombinacijski prediktori (7 brojeva po metodu)
    prognoza.py            # uživo/retro prognoza, evaluacija, prognoza_u_tacki (vremeplov)
    istorija.py            # „Istraži istoriju": sečenje po granica/prozor + prosleđivanje
    mapa.py                # „Mapa kombinacija": rang/unrang, Hilbert, osobine, skokovi
  api/app.py             # FastAPI endpointi (JSON)
  static/                # frontend (index.html, app.js, styles.css)
    mapa/                  # generisane pločice mape (van git-a; pravi ih generisi_mapu.py)
  tests/                 # smoke + invarijant testovi (core, razlicitost, prognoza, istorija, mapa)
pokreni.py               # pokretač servera
generisi_mapu.py         # jednokratno pečenje pločica mape
migracija_baze.py        # jednokratno smanjenje baze (57 MB -> ~0.2 MB)
```

Baza (`loto_baza.db`) je ista kao u desktop verziji — obe aplikacije mogu da je koriste.

## Uvoz podataka

CSV/Excel mora imati kolone: `kolo, datum, b1, b2, b3, b4, b5, b6, b7`.

## Testovi

Pet modula (svi se pokreću sa `-X utf8` radi ćiriličnih/latiničnih ispisa):

```bash
python -X utf8 -m webapp.tests.test_core         # analitika, rangiranje, generator, bektest
python -X utf8 -m webapp.tests.test_razlicitost  # teorija i analize preklapanja
python -X utf8 -m webapp.tests.test_prognoza     # bez curenja, determinizam, ekvivalencija, brzina
python -X utf8 -m webapp.tests.test_istorija     # granica/prozor, vremeplov == retro, anti-curenje
python -X utf8 -m webapp.tests.test_mapa         # rang↔unrang, Hilbert, ocena == Generator, pločice
```

`test_mapa` traje oko pola minuta; ako pločice nisu generisane, provera pločica se
preskoči, a ostalo se izvrši.

## Šta je izmenjeno u odnosu na desktop v10.6

- **Spojene metode rangiranja** (Frekvencija/Bajes/Hibrid) u jednu stranu — sve daju
  praktično isti redosled jer reflektuju istu frekvenciju.
- **ML/VAE i Gemini AI** privremeno izostavljeni iz v1 (mogu se vratiti kasnije).
- **Baza smanjena** ~300× jer bektest više ne čuva sve kombinacije kao tekst.
- `eval()` zamenjen sigurnim `json` parsiranjem.

## Novo u web verziji (posle v1)

- **Prognoza** i **Različitost** — dve nove analitičke strane sa strogim testovima
  značajnosti (bez curenja budućnosti, kontrolne grupe, Bonferroni korekcija).
- **Istraži istoriju (vremeplov)** — interaktivno putovanje kroz kola sa garancijom da
  se nikad ne koriste podaci posle izabrane granice; „predikcija tada" je bit-identična
  retro-bektestu (pokriveno testom). Generator dobija opcioni parametar granice.
- **Mapa kombinacija** — ceo prostor igre kao jedna zumabilna slika (Hilbertova kriva
  reda 12 nad leksikografskim rangom), sa izvučenim kolima kao tačkama i putanjom kroz
  vreme. Svaki sloj ima kontrolni slučajni parnjak, jer je poruka strane upravo to da se
  dve slike ne razlikuju.
