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

Za grafikone je potreban internet (ECharts + Alpine.js se učitavaju preko CDN-a).

## Strane

1. **Dashboard** — ključni pokazatelji, frekvencija, predlog bazena.
2. **Statistika** — frekvencija, srednje vrednosti, trend, ritam, uzastopni, dekade,
   poziciona heatmapa i hi-kvadrat test.
3. **Rangiranje** — brojevi rangirani metodom Frekvencija / Bajes / Hibrid.
4. **Generator** — kombinacije po filterima (par/nepar, vrući/hladni, sredina, uzastopni,
   dekade, diverzitet) uz bodovanje.
5. **Bektest** — uspešnost sačuvanih strategija (rezultat, indeks promašaja/iznenađenja).
6. **Moji tiketi** — evidencija odigranih tiketa i njihovih pogodaka.
7. **Podaci** — unos novog kola (auto-provera tiketa i bektesta) + uvoz CSV/Excel.

## Arhitektura

```
webapp/
  core/          # čista analitika (bez UI-ja) — deljena logika
    konfig.py      # pravila igre (MAX_BROJ, itd.)
    baza.py        # SQLite sloj
    analitika.py   # frekvencija, srednje vrednosti, poziciona, model pristrasnosti
    rangiranje.py  # frekvencija / Bajes / hibrid + matrica povezanosti
    generator.py   # filteri + bodovanje + diverzitet
    bektest.py     # indeksi + "dodaj kolo i proveri sve"
  api/app.py     # FastAPI endpointi (JSON)
  static/        # frontend (index.html, app.js, styles.css)
  tests/         # smoke + invarijant testovi
pokreni.py       # pokretač servera
migracija_baze.py# jednokratno smanjenje baze (57 MB -> ~0.2 MB)
```

Baza (`loto_baza.db`) je ista kao u desktop verziji — obe aplikacije mogu da je koriste.

## Uvoz podataka

CSV/Excel mora imati kolone: `kolo, datum, b1, b2, b3, b4, b5, b6, b7`.

## Testovi

```bash
python -m webapp.tests.test_core
```

## Šta je izmenjeno u odnosu na desktop v10.6

- **Spojene metode rangiranja** (Frekvencija/Bajes/Hibrid) u jednu stranu — sve daju
  praktično isti redosled jer reflektuju istu frekvenciju.
- **ML/VAE i Gemini AI** privremeno izostavljeni iz v1 (mogu se vratiti kasnije).
- **Baza smanjena** ~300× jer bektest više ne čuva sve kombinacije kao tekst.
- `eval()` zamenjen sigurnim `json` parsiranjem.
