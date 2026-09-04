# Stanje projekta — Loto Analizator (web)

> Pregled napravljen 2026-09-04 da bi se lako nastavilo sa razvojem.
> Ukratko: aplikacija radi, sve tri velike nove funkcije su **završene i testirane**,
> ali **ceo taj rad još nije komitovan u git**. To je jedino „nedovršeno".

---

## 1. Šta je aplikacija

Interaktivna **web verzija** Loto Analizatora (loto 7/39). FastAPI backend + čist
HTML/JS frontend (Alpine.js + ECharts preko CDN-a). Ponovo koristi analitiku iz stare
desktop aplikacije (`analiza.py`), ali kroz browser i sa čistom modularnom arhitekturom.

Radi nad bazom `loto_baza.db` (**1.422 kola**). Numeracija kola: `godina*1000 + redni_broj`;
brojevi se unose **redosledom izvlačenja** (bitno za `fresh`/poziciju).

**Poruka koju cela app nosi:** ovo nije proricanje — izvlačenja su nezavisna. App je
istraživački i statistički alat koji meri da li se ijedan metod ponaša značajno drugačije
od slučajnosti (i dosad — ne ponaša se, što je i očekivano).

### Pokretanje
```bash
python pokreni.py            # otvara http://127.0.0.1:8000
python pokreni.py --port 9000 --bez-otvaranja
```

### Testovi (Windows — obavezan `-X utf8` zbog μ/σ u ispisu)
```bash
.venv311/Scripts/python.exe -X utf8 -m webapp.tests.test_prognoza
.venv311/Scripts/python.exe -X utf8 -m webapp.tests.test_razlicitost
.venv311/Scripts/python.exe -X utf8 -m webapp.tests.test_core
```
Status na dan pregleda: **sva tri prolaze u celini.** (Bez `-X utf8` padne samo `print`
grčkih slova na cp1252 konzoli — kozmetika, ne logika.)

---

## 2. Arhitektura

```
webapp/
  core/                     # čista analitika (bez UI-ja)
    konfig.py                 pravila igre (MAX_BROJ=39, BROJEVA_U_KOMBINACIJI=7)
    baza.py                   SQLite sloj  (+ tabela prognoze, hook evaluacije)
    analitika.py              frekvencija, srednje vrednosti, poziciona, pristrasnost
    rangiranje.py             frekvencija / Bajes / hibrid + matrica povezanosti
    generator.py              filteri + bodovanje + diverzitet
    bektest.py                indeksi + „dodaj kolo i proveri sve" (+ mere različitosti)
    prediktori.py            ★ jednobrojni prediktori (hot/cold/bayes/hybrid/rhythm/fresh/random)
    prediktori_komb.py       ★ kombinacijski prediktori (k_hot7 … k_cooc, k_random)
    prognoza.py              ★ uživo tok + evaluacija + retro-bektest + statistika
    razlicitost_teorija.py   ★ hipergeom. raspodela, bitmaske, σ, z-test, hi-kvadrat
    razlicitost.py           ★ 5 analiza različitosti + mere za Generator/Bektest
  api/app.py                 FastAPI endpointi (JSON)
  static/                    frontend: index.html, app.js, styles.css
  tests/                     test_core, test_prognoza ★, test_razlicitost ★
pokreni.py                   pokretač servera

★ = novo, nekomitovano (vidi §4)
```

Baza je ista kao desktop verzija — obe app mogu da je koriste. Zavisnosti:
`requirements-web.txt` (fastapi, uvicorn, pandas, **scipy**, openpyxl). *scipy je nov —
koristi ga `prognoza.py` za `binomtest`.*

---

## 3. Strane aplikacije (9 tabova)

| # | Strana | Šta radi |
|---|--------|----------|
| 1 | **Dashboard** | ključni pokazatelji, frekvencija, predlog bazena |
| 2 | **Statistika** | frekvencija, srednje vrednosti, trend, ritam, uzastopni, dekade, poziciona heatmapa, hi-kvadrat |
| 3 | **Različitost** ★ | koliko se kombinacije razlikuju vs. čista slučajnost (5 analiza) |
| 4 | **Rangiranje** | brojevi rangirani: Frekvencija / Bajes / Hibrid |
| 5 | **Prognoza** ★ | predviđanje 1 broja **i** kombinacije od 7 — statistički eksperiment |
| 6 | **Generator** | kombinacije po filterima + bodovanje + panel „Različitost seta" ★ |
| 7 | **Bektest** | uspešnost strategija + kolone prosečnog/maks preklapanja ★ |
| 8 | **Moji tiketi** | evidencija odigranih tiketa i pogodaka |
| 9 | **Podaci** | unos kola (auto-provera tiketa/bektesta/**prognoza** ★) + uvoz CSV/Excel |

★ = dodato u poslednjem, nekomitovanom talasu rada.

---

## 4. Gde smo stali (najvažnije)

Postoje **tri plana** u rootu — svaki je isporučen u celini, ali **rad nije u gitu**:

| Plan | Strana | Status | Testovi |
|------|--------|--------|---------|
| `PLAN_PROGNOZA.md` | Prognoza → „Jedan broj" | **Završeno** (sve 4 faze) | `test_prognoza.py` ✔ |
| `PLAN_PROGNOZA_KOMBINACIJE.md` | Prognoza → „Kombinacija" | **Završeno** (sve 4 faze) | `test_prognoza.py` ✔ |
| `PLAN_RAZLICITOST.md` | Različitost (+ Generator/Bektest) | **Završeno** (sve 4 faze) | `test_razlicitost.py` ✔ |

Dokazano radi nad tvojom bazom:
- **Retro-bektest** (9.450 + 9.450 redova) determinističan, **~7 s** — unutar ciljanih granica.
- **Prognoza (jedan broj):** svih 7 metoda 16,6–18,5% ≈ baseline 17,95% → nerazlučivo od
  slučajnosti (kontrolna grupa čak nadmašila Bajesa). Tačno kako teorija predviđa.
- **Prognoza (kombinacija):** svi metodi prosek preklapanja 1,25–1,32 ≈ μ=1,256 → nerazlučivo.
- **Različitost:** uzastopna/svi parovi ≈ slučajnost (p ≫ 0,05); najveće preklapanje ikad
  6/7 (nikad identično); „vrući parovi" van intervala ≈ očekivanih 37 → šum, ne signal.
- **Zaštita od curenja budućnosti** pokrivena unit testovima (retro prvo predviđa pa dodaje kolo).

### Nekomitovane izmene (git status)
```
 M  FUNKCIJE.md               (+ sekcije 8 Prognoza, 9 Različitost)
 M  webapp/api/app.py         (+ ~15 endpointa: /api/prognoza/*, /api/razlicitost/*)
 M  webapp/core/baza.py       (+ tabela prognoze, hook oceni_prognoze, migracija)
 M  webapp/core/bektest.py    (+ mere preklapanja sa dobitnom)
 M  webapp/static/app.js      (+ ucitajPrognozu/Komb, ucitajRazlicitost, grafikoni)
 M  webapp/static/index.html  (+ tabovi Prognoza i Različitost)
 ?? PLAN_PROGNOZA_KOMBINACIJE.md, PLAN_RAZLICITOST.md
 ?? webapp/core/prediktori.py, prediktori_komb.py, prognoza.py,
       razlicitost.py, razlicitost_teorija.py
 ?? webapp/tests/test_prognoza.py, test_razlicitost.py
```

**Prvi korak za nastavak:** komituj ovaj rad (predlog: tri commita — Prognoza,
Prognoza-kombinacije, Različitost — ili jedan zbirni). Bez toga se lako izgubi pregled.

---

## 5. Mogući sledeći koraci

Iz „Van opsega v1" sekcija u planovima i README-web.md — kandidati za dalje:

- **Vraćanje ML/VAE i Gemini AI** (svesno izostavljeni iz web v1; postoje u desktop
  `analiza.py`, `ml_generator.py`, `loto_decoder_model.keras`).
- **Prognoza:** ensemble prediktori (težinsko kombinovanje) — tek ako nešto ikad izađe
  van pojasa; dropdown perioda 50/100/200 kao odvojeni skupovi rezultata.
- **Različitost:** trojke/četvorke u ko-okurenciji (v2, uz strožu kontrolu lažnih alarma);
  klasterovanje istorije.
- **Generator:** optimizacija „pokrivenosti" preko više metoda zajedno.
- **Tehnički:** dodati `pytest` u okruženje (sad se testovi zovu kao moduli); CI; možda
  offline verzija ECharts/Alpine (sad zavise od interneta preko CDN-a).

---

## 6. Reference u kodu

- Endpointi: `webapp/api/app.py` (grep `@app.get`/`@app.post`).
- Registri prediktora se šire jednom linijom: `PREDIKTORI` u `prediktori.py`,
  `PREDIKTORI_KOMB` u `prediktori_komb.py`.
- Teorija (ne hardkodovati vrednosti — sve izvedeno): `razlicitost_teorija.py`.
- Puna korisnička dokumentacija svih strana: `FUNKCIJE.md` (sekcije 8 i 9 su nove).
