# Plan razvoja — „Mapa kombinacija“ (prostor svih kombinacija + putanja kroz vreme)

> Verzija: 1.0
> Datum: 2026-09-05
> Projekat: Loto Analizator — web
> Osnova: `README-web.md` (stanje posle „Istraži istoriju“), postojeći `istorija.py`, `razlicitost.py`, `generator.py`

---

## 1. Svrha

Aplikacija ima analitiku, vremeplov i testove, ali korisniku i dalje nedostaje **osećaj razmere**: koliko je zapravo velik prostor od 15.380.937 mogućih kombinacija, i koliko je 1.422 izvučenih u odnosu na njega.

Nova strana **„Mapa kombinacija“** prikazuje ceo prostor kao zumabilnu 2D mapu, sa izvučenim kombinacijama kao tačkama i njihovim hronološkim redosledom kao putanjom. To je **vizuelizacija poruke koju cela aplikacija nosi**, ne novi analitički metod.

Ključni princip, kao i dosad: nijedan prikaz ne sme sugerisati obrazac koji teorija ne podržava. Zato svaki sloj ima **kontrolni (slučajni) parnjak** koji se može uključiti jednim klikom.

---

## 2. Fiksne odluke

### 2.1. Indeksiranje kombinacija

Svaka kombinacija `{a<b<c<d<e<f<g}` iz 1..39 dobija **leksikografski rang** `0 … 15.380.936` preko kombinatornog brojevnog sistema:

```
rang = C(a-1,1) + C(b-1,2) + C(c-1,3) + C(d-1,4) + C(e-1,5) + C(f-1,6) + C(g-1,7)
```

(uz konvenciju indeksa 0-bazirano; tačan oblik izvesti i pokriti testom rang→unrang→rang).

Funkcije `rang(komb) -> int` i `unrang(r) -> tuple` su O(7). Obe idu u `core/mapa.py`.

### 2.2. Raspored na 2D

Rang se preslikava na koordinate `(x, y)` u kvadratu **4096×4096** (16.777.216 ćelija, popunjeno 15.380.937; ostatak prazan) preko **Hilbertove krive reda 12**. Razlog: susedni rangovi ostaju prostorno bliski, pa mapa ima „teksturu“ umesto šuma.

Raspored je **deterministički i trajan**. Ne menjati red krive ni dimenziju bez regeneracije svih pločica.

### 2.3. Boja pozadine = osobina, ne identitet

Svaki piksel na najvišem zumu je jedna kombinacija; boja kodira **izabranu osobinu**:

| Sloj | Osobina | Skala |
|---|---|---|
| `zbir` | zbir 7 brojeva (28–252) | sekvencijalna |
| `raspon` | max − min | sekvencijalna |
| `parni` | broj parnih (0–7) | diskretna |
| `dekade` | broj dekada koje kombinacija dodiruje (1–4) | diskretna |
| `ocena` | ocena postojećeg Generatora (`generator.py`) | sekvencijalna |

Na nižim zumovima jedan piksel pokriva više kombinacija → prikazuje se **prosek** osobine (izračunat pri generisanju pločica).

Sloj `ocena` je najvredniji: vizuelno pokazuje koliki deo prostora filteri Generatora odbacuju.

### 2.4. Pločice (tiles)

Mapa se **ne renderuje u browseru iz sirovih podataka**. Za svaki sloj se unapred generišu PNG pločice 256×256 za zoom nivoe 0–4 (2×2 → 32×32 pločica), standardnim `z/x/y` rasporedom. Ukupno ~1.400 pločica po sloju; generisanje u numpy-ju jednom, čuvanje u `webapp/static/mapa/{sloj}/{z}/{x}/{y}.png`.

Generisanje je jednokratna skripta `generisi_mapu.py` (kao `migracija_baze.py`), ne deo servera.

Ograničenje zuma: na zoomu 4 piksel = 1 kombinacija (4096/256 = 16 = 2⁴). Prikaz preko toga je samo CSS uvećanje.

### 2.5. Sloj izvučenih kombinacija

1.422 tačke se **ne peku u pločice** — dinamički su sloj (canvas preko pločica), da bi se menjale sa vremenskim slajderom i novim kolima. Endpoint vraća `[{kolo, rang, x, y}]`.

### 2.6. Sloj putanje

Segmenti od kola *i* do *i+1* hronološki. **Boja segmenta = preklapanje sa prethodnim kolom (0–7)**, iz `razlicitost.py`. Prikazuje se „rep“ od poslednjih N koraka (podrazumevano 50, opcije 10/50/200/sve).

Linije se **nikad ne prikazuju bez boje preklapanja** — gola linija sugeriše pravac koji ne postoji.

### 2.7. Kontrolni slojevi (obavezni)

- **Slučajne tačke**: isti broj kombinacija kao izvučenih, generisan sa fiksnim seed-om, isti stil.
- **Slučajna putanja**: kroz te slučajne tačke, ista boja po preklapanju.

Prekidač „Stvarno / Slučajno / Oba“ je vidljiv na ekranu stalno. Cilj: korisnik vidi da su dve slike nerazlučive.

### 2.8. Veza sa istorijom

Vremenski slajder koristi isti pojam `granica` iz `istorija.py`: prikazuju se samo kola ≤ granica. Klik na tačku otvara „Istraži istoriju“ na tom kolu.

---

## 3. Arhitektura

```
webapp/
  core/
    mapa.py              # NOVO: rang/unrang, hilbert(rang)->(x,y), inverz, osobine, slucajni set
  api/app.py             # + /api/mapa/*
  static/
    mapa/                # NOVO: generisane pločice (gitignore ili LFS — vidi §7)
    index.html           # + tab „Mapa kombinacija“
    app.js               # + stanje mape, slojevi, slajder
    styles.css
  tests/
    test_mapa.py         # NOVO
generisi_mapu.py         # NOVO: jednokratno generisanje pločica
```

Biblioteka za pločice: **Leaflet** (CDN, `CRS.Simple`), jer je lagana, radi bez geografskih koordinata i ima gotov canvas overlay. Ne uvoditi ništa teže.

---

## 4. API

Sve GET, JSON:

| Endpoint | Vraća |
|---|---|
| `/api/mapa/info` | dimenzija, broj kombinacija, dostupni slojevi, max zoom |
| `/api/mapa/tacke?granica=` | izvučene kombinacije ≤ granica: `[{kolo, rang, x, y, preklapanje_sa_prethodnim}]` |
| `/api/mapa/slucajno?n=&seed=` | kontrolni set, isti oblik |
| `/api/mapa/komb?x=&y=` | kombinacija na pikselu, da li je izvučena (kad), najbliža izvučena po preklapanju, osobine |
| `/api/mapa/rang?brojevi=1,2,3,4,5,6,7` | rang + koordinate za unet tiket („gde je moj tiket“) |
| `/api/mapa/skokovi?granica=` | raspodela dužina koraka po mapi za stvarne vs. slučajne tačke (za sekciju „Test“) |

Pločice se serviraju kao statika, ne kroz API.

---

## 5. UI

### 5.1. Raspored

```
┌──────────────────────────────────────────────────────────────┐
│ MAPA KOMBINACIJA        Sloj: [ocena ▼]   Prikaz: [Stvarno|Slučajno|Oba] │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                    (Leaflet mapa, zum + pan)                 │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│ Vreme: [◀] ──────●───────── [▶]  kolo 2026-184   rep: [50 ▼]  │
│ Gde je moj tiket: [ _ _ _ _ _ _ _ ] [Pronađi]                 │
├──────────────────────────────────────────────────────────────┤
│ Izabrano: 3 7 12 19 24 31 38 — izvučeno 2026-184             │
│ Najbliže izvučeno: 1 7 11 18 25 29 35 (2026-183), preklapanje 2 │
└──────────────────────────────────────────────────────────────┘
```

### 5.2. Obavezni tekst na ekranu

Ispod mape, stalno vidljivo:

> „Prikazano je 1.422 izvučenih od 15.380.937 mogućih kombinacija (0,009%). Tačke na mapi su raspoređene onako kako teorija predviđa za slučajno izvlačenje; uključi „Slučajno“ da uporediš.“

Brojevi se izvode iz baze i `konfig.py`, ne hardkoduju.

### 5.3. Collapsible „Test“ sekcija

Histogram dužina koraka po mapi: stvarno vs. slučajno, uz jednu rečenicu zaključka. Ovo je isti test različitosti u drugom obliku — reći to eksplicitno u UI.

---

## 6. Faze

### Faza 1 — Jezgro i pločice
1. `core/mapa.py`: rang, unrang, Hilbert i inverz, osobine.
2. `generisi_mapu.py`: sloj `zbir` za sve zoome. Izmeriti vreme i veličinu.
3. `test_mapa.py`: rang↔unrang bijekcija na uzorku + ivicama (prva/poslednja kombinacija), Hilbert bijekcija, svi rangovi unutar 4096².

**Gotovo kad:** pločice sloja `zbir` postoje i test prolazi.

### Faza 2 — Mapa u browseru
1. Tab, Leaflet sa `CRS.Simple`, pločice sloja `zbir`.
2. Klik na piksel → `/api/mapa/komb` → panel sa kombinacijom.
3. „Gde je moj tiket“.

**Gotovo kad:** može da se zumira do pojedinačne kombinacije i klikne na nju.

### Faza 3 — Tačke i kontrola
1. Sloj izvučenih (canvas overlay), `/api/mapa/tacke`.
2. Slučajni set, prekidač Stvarno/Slučajno/Oba.
3. Obavezni tekst iz §5.2.

**Gotovo kad:** dve slike se vide jedna pored druge.

### Faza 4 — Vreme i putanja
1. Slajder vezan na `granica`, tačke se pojavljuju kolo po kolo.
2. Putanja sa bojom preklapanja, rep N.
3. Slučajna putanja.
4. Klik na tačku → „Istraži istoriju“.

**Gotovo kad:** animacija kolo po kolo radi za stvarno i slučajno, i boja segmenata odgovara `razlicitost.py`.

### Faza 5 — Ostali slojevi i test
1. Slojevi `raspon`, `parni`, `dekade`, `ocena`.
2. Sekcija „Test“ sa histogramom skokova.

**Gotovo kad:** svi slojevi generisani, histogram poklapa stvarno i slučajno.

---

## 7. Tehnička upozorenja

- **Veličina pločica**: 5 slojeva × ~1.400 PNG-a. Proceniti ukupno (verovatno 20–80 MB). Ako je preko ~30 MB, pločice **ne idu u git** — `.gitignore` + `generisi_mapu.py` se pokreće pri instalaciji (dodati u README).
- **Generisanje**: 15,4M rangova × osobina u numpy-ju je sekunde; Hilbert nad 15,4M je najskuplji korak — vektorizovati, ne petlja u Pythonu. Cilj: ceo sloj < 2 min.
- **Sloj `ocena`**: zavisi od stanja baze (frekvencije). Generisati ga za fiksnu granicu (npr. poslednje kolo u trenutku generisanja) i to jasno napisati u UI. Ne regenerisati automatski pri svakom unosu kola.
- **Browser**: canvas overlay sa 1.422 tačaka i 1.421 segmenata je trivijalan. Ne pokušavati WebGL.
- **Leaflet** sa CDN-a — isto ograničenje kao ECharts/Alpine (internet obavezan).

---

## 8. Šta NE raditi

- Ne renderovati 15M piksela u browseru.
- Ne dodeljivati jedinstvenu boju svakoj kombinaciji.
- Ne crtati putanju bez boje preklapanja.
- Ne prikazivati stvarne tačke bez dostupnog slučajnog parnjaka.
- Ne izvoditi zaključke iz „grozdova“ ili „pravaca“ na mapi — mapa ima teksturu rasporeda, ne strukturu podataka.
- Ne uvoditi novu statistiku; histogram skokova je jedini test i on je preformulacija postojećeg.

---

## 9. Commit redosled

```
1. feat(mapa): rang/unrang, hilbert i generisanje plocica
2. feat(mapa): tab sa zumabilnom mapom i klikom na kombinaciju
3. feat(mapa): sloj izvucenih i kontrolni slucajni sloj
4. feat(mapa): vremenski slajder i putanja sa preklapanjem
5. feat(mapa): dodatni slojevi osobina i test skokova
6. docs: README i FUNKCIJE sekcija „Mapa kombinacija“
```

---

## 10. Kriterijum uspeha

Korisnik koji otvori stranu treba za manje od minuta da:

1. vidi koliko je prostor ogroman, a izvučenih malo;
2. zumira do jedne kombinacije i pronađe svoj tiket;
3. pusti vreme i gleda kako tačke „skaču“ bez reda;
4. uključi „Slučajno“ i ne vidi razliku;
5. iz toga sam izvuče zaključak koji aplikacija nikad ne mora da mu kaže rečima.
