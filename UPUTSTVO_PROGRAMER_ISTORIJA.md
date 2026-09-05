# Uputstvo za programera — „Istraži istoriju“

> Datum: 2026-09-04
> Projekat: Loto Analizator — web
> Osnova: `Stanje_projekta.md` (stanje) + `plan_ineraktivna_istorija.md` (plan v1.0)
> Ovaj dokument je **operativno uputstvo**: šta tačno raditi, kojim redom, i kako se proverava da je gotovo.

---

## 0. Pre bilo čega — komit postojećeg rada

Tri završene funkcije (Prognoza, Prognoza-kombinacije, Različitost) **nisu u gitu**. Ništa novo ne počinje dok ovo nije urađeno.

```bash
git add PLAN_PROGNOZA.md PLAN_PROGNOZA_KOMBINACIJE.md PLAN_RAZLICITOST.md FUNKCIJE.md
git add webapp/core/prediktori.py webapp/core/prediktori_komb.py webapp/core/prognoza.py
git add webapp/core/baza.py webapp/api/app.py webapp/static/ webapp/tests/test_prognoza.py
git commit -m "feat: prognoza jednog broja i kombinacije (retro-bektest, evaluacija, UI)"

git add webapp/core/razlicitost.py webapp/core/razlicitost_teorija.py
git add webapp/core/bektest.py webapp/tests/test_razlicitost.py
git commit -m "feat: analiza razlicitosti + mere preklapanja u Generatoru/Bektestu"
```

(Ako se `app.py`/`app.js`/`index.html` ne daju čisto podeliti, jedan zbirni commit je prihvatljiv — bitno je da `git status` bude čist.)

Zatim pokreni sva tri testa i potvrdi da prolaze:

```bash
.venv311/Scripts/python.exe -X utf8 -m webapp.tests.test_core
.venv311/Scripts/python.exe -X utf8 -m webapp.tests.test_prognoza
.venv311/Scripts/python.exe -X utf8 -m webapp.tests.test_razlicitost
```

Tek onda: `git checkout -b feat/istorija`.

---

## 1. Ključne odluke (obavezujuće)

Ove stvari su u planu ostale nedorečene. Ovde su fiksirane — ne menjaj ih bez dogovora.

### 1.1. Jedna konvencija za istorijsku tačku

Svuda (core, API, UI, testovi) koriste se **dva pojma**, nikad jedan pomešan:

| Pojam | Značenje | Primer |
|---|---|---|
| `granica` | **poslednje kolo koje sistem zna** (uključivo) | `2025119` |
| `cilj` | **naredno stvarno kolo** = prvo kolo posle granice u bazi | `2025120` |

Pravilo: **dostupni podaci = sva kola ≤ granica**. Kolo `cilj` i sve posle njega su nedostupni sve do evaluacije.

U UI se korisniku prikazuje oboje („Sistem zna zaključno sa 2025-119 · Naredno kolo: 2025-120“). Kad korisnik klikne na kolo u tabeli, to kolo postaje **cilj**, a granica je automatski kolo pre njega. Tako je „predikcija tada“ uvek predikcija *za* kliknuto kolo.

### 1.2. Navigacija ide po redosledu u bazi, ne po broju kola

Numeracija `godina*1000 + redni` nije kontinualna (`2025052 → 2026001`). Prethodno/sledeće kolo se traži kao `ORDER BY kolo` + offset, nikad kao `kolo ± 1`. `istorija.py` mora da ima:

```python
def prethodno_kolo(kolo) -> int | None
def sledece_kolo(kolo) -> int | None
def kola_do(granica, prozor=None) -> list[Kolo]   # ≤ granica, poslednjih `prozor` (None = sva)
```

### 1.3. Nema nove statistike — samo prosleđivanje podskupa

`istorija.py` **ne računa** frekvencije, Bajesa, različitost ni prognozu. On samo:
1. napravi podskup podataka `kola_do(granica, prozor)`,
2. prosledi ga postojećim funkcijama iz `analitika.py`, `rangiranje.py`, `razlicitost.py`, `prognoza.py`,
3. spakuje rezultat za UI.

Ako neka postojeća funkcija **čita bazu globalno** umesto da prima podatke kao argument — refaktoriši je da prima podatke (dodaj opcioni parametar, podrazumevano = cela baza). To je jedini dozvoljeni zahvat u postojećoj analitici i **ne sme promeniti rezultate** (proveriti postojećim testovima).

### 1.4. Jedan korak vremeplova = isti kod kao retro-bektest

`prognoza.py` već ima logiku „predvidi za kolo K nad podacima < K, pa evaluiraj“. Izdvoji je u jednu javnu funkciju:

```python
def prognoza_u_tacki(granica, metode=None) -> dict
    # vraća: {granica, cilj, jedan_broj: {metod: broj}, kombinacija: {metod: [7]}}
def oceni_u_tacki(prognoza, stvarno_kolo) -> dict
    # koristi POSTOJEĆU evaluaciju (pogodak / preklapanje)
```

Retro-bektest treba da **poziva istu funkciju u petlji**. Dva odvojena puta = garantovano razilaženje rezultata (test 30.5 iz plana proverava baš to).

### 1.5. API konvencije

- Prefiks `/api/istorija/...`, **samo ASCII** u putanjama (plan ima ćirilično „а“ u `kolа` — to je greška).
- Svaki endpoint prima **eksplicitno** `granica` (int) i, gde ima smisla, `prozor` (int ili izostavljen = sva).
- Nijedan endpoint ne sme imati implicitni fallback na „celu bazu“ kad `granica` nedostaje — vrati 422.
- Jedan zbirni endpoint `/api/istorija/kontekst` vraća sve što početni ekran treba u jednom pozivu (kola u prozoru, cilj, granica, sažetak). Ostali endpointi su za detalje na klik.

---

## 2. Faze, redom, sa kriterijumima završetka

Svaka faza = jedan ili više commita + testovi prolaze + ručna provera u browseru. **Ne prelaziti na sledeću dok prethodna nije zatvorena.**

### Faza 1 — Osnova (core + API + tab + tabela)

**Uraditi:**
1. `webapp/core/istorija.py` sa funkcijama iz §1.2 + `kontekst(granica, prozor)`.
2. Endpointi:
   - `GET /api/istorija/kontekst?granica=&prozor=`
   - `GET /api/istorija/kolo/{kolo}` (detalj jednog kola + prethodno/sledeće)
   - `GET /api/istorija/kola?granica=&prozor=` (samo lista, za tabelu)
3. Novi tab **„Istraži istoriju“** kao 2. stavka (odmah posle Dashboarda).
4. Toolbar: `[<<] [<] Kolo XXXX [>] [>>]`, dugmad perioda `20 / 50 / 100 / 200 / sva`, „Idi na najnovije“. `<<`/`>>` = skok za `prozor` kola.
5. Tabela prethodnih kola: kolo, datum (ako kolona postoji u bazi — proveri `baza.py`; ako ne postoji, ne izmišljati), 7 brojeva **redosledom izvlačenja**.
6. Klik na red = to kolo postaje cilj.
7. Alpine stanje:
   ```js
   istorija: { granica: null, cilj: null, prozor: 100, broj: null, loading: false, kontekst: null, detalj: null }
   ```

**Gotovo kad:** bilo koje kolo od 1.422 može da se izabere, navigacija radi preko granice godine, tabela prikazuje tačan broj prethodnih kola za svaki prozor.

**Commit:** `feat(istorija): core, API i tab sa navigacijom i tabelom kola`

### Faza 2 — Interaktivni brojevi

**Uraditi:**
1. Svaki broj u tabeli i u detalju kola je klikabilan → otvara panel `BrojDetalj` (ne napušta stranu).
2. `GET /api/istorija/broj/{broj}?granica=&prozor=` vraća: ukupno pojavljivanja, poslednje, prethodno, trenutni razmak, prosečan/min/max razmak, pojavljivanja u prozoru, raspodela po poziciji izvlačenja, timeline (lista kola u prozoru sa flagom pojavio/nije).
3. Sve iz **postojeće** `analitika.py` (frekvencija, ritam, poziciona). Ako nešto nedostaje (npr. razmaci kao lista), dodati u `analitika.py`, ne u `istorija.py`.
4. Timeline kao jednostavan ECharts scatter/bar ili čist HTML — ne komplikovati.
5. Jedna rečenica objašnjenja iznad brojki (§25 plana): *„Broj 17 se pojavio 19 puta u poslednjih 100 kola; očekivanje ≈ 17,95; razlika nije značajna.“* Očekivanje se **izvodi** (`prozor * 7/39`), ne hardkoduje.

**Gotovo kad:** iz tabele se klikom istraži bilo koji broj; promena prozora menja rezultate; rezultati zavise samo od kola ≤ granica.

**Commit:** `feat(istorija): detalj broja, razmaci, pozicije i timeline`

### Faza 3 — Kontekst kola

**Uraditi:**
1. Sekcija „Šta se dešavalo pre ovog kola“ — tabela broj / pojavljivanja / poslednji put / razmak za ceo prozor (§10 plana). Ulazi u `/api/istorija/kontekst`.
2. `GET /api/istorija/razlicitost?cilj=&prozor=` → preklapanje cilja sa prethodnim kolom, sa poslednjih N, ponovljeni parovi, maks. istorijsko preklapanje. **Poziva** `razlicitost.py`; koristi `razlicitost_teorija.py` za μ/σ.
3. `GET /api/istorija/rangiranje?granica=&prozor=` → tabela broj / frekvencija / Bajes / hibrid **kakva bi bila na granici**. Poziva `rangiranje.py` nad podskupom.
4. Sve napredno u collapsible sekcijama (Osnovno ▾ / Detaljna statistika ▸ / Test ▸ / Teorija ▸). Početni ekran ostaje miran.

**Gotovo kad:** za izabrano kolo se vidi sažetak prozora, različitost i rangiranje „tada“, i to bez ikakvog uticaja kola ≥ cilj.

**Commit:** `feat(istorija): sazetak perioda, istorijska razlicitost i rangiranje`

### Faza 4 — Vremeplov (najvažnija faza)

**Uraditi:**
1. Refaktor u `prognoza.py` iz §1.4 (`prognoza_u_tacki`, `oceni_u_tacki`), retro-bektest prebačen da ih koristi. **`test_prognoza.py` mora i dalje proći identično** — retro-bektest je determinističan, uporedi izlaz pre/posle.
2. `GET /api/istorija/prognoza?granica=` → svi prediktori (jedan broj + kombinacija) za cilj. Sporo je samo koliko i jedan korak retro-bektesta (~ms), pa nema keša u v1.
3. `GET /api/istorija/prognoza/ishod?granica=` → prognoza + stvarno kolo cilj + evaluacija (pogodak / preklapanje k/7).
4. UI panel „Predikcija tada“ sa dugmetom **[Izračunaj šta bi sistem tada predvideo]** → ispod toga **[Prikaži stvarni ishod]**. Ishod se ne prikazuje dok korisnik ne klikne (namerno — to je poenta eksperimenta).
5. Ispod rezultata obavezna rečenica konteksta: *„Očekivano preklapanje slučajne kombinacije: μ = 1,256.“* — izvedeno iz teorije, ne hardkodovano.
6. „Korak po korak“: dugmad `[◀ prethodno] [sledeće ▶]` zadržavaju otvoren panel prognoze i automatski preračunavaju. Bez animacije.

**Gotovo kad:** iz proizvoljne tačke se ponovi eksperiment, rezultat je identičan onom iz retro-bektesta za isto kolo, i testovi curenja (§3) prolaze.

**Commit:** `feat(istorija): vremeplov — istorijska prognoza i poredjenje sa ishodom`

### Faza 5 — Integracija (opciono za v1, ali poželjno)

1. Linkovi iz `BrojDetalj` → Statistika / Rangiranje / Prognoza sa prenetim brojem.
2. Linkovi iz detalja kola → Različitost / Generator / Bektest.
3. Generator: opcioni parametar `granica` (podrazumevano = cela baza, ponašanje nepromenjeno).

**Commit:** `refactor: povezivanje postojecih tabova sa istorijskim kontekstom`

---

## 3. Testovi — `webapp/tests/test_istorija.py`

Poziv kao i ostali: `.venv311/Scripts/python.exe -X utf8 -m webapp.tests.test_istorija`.

Obavezni testovi (nazivi orijentacioni):

| Test | Šta proverava |
|---|---|
| `test_granica_iskljucuje_buducnost` | `kola_do(granica)` ne sadrži ni jedno kolo > granica |
| `test_prethodno_sledece_preko_godine` | navigacija `2025-052 ↔ 2026-001` (ili stvarni prelaz u bazi) |
| `test_prozor_velicina` | za 20/50/100/200 vraća tačno toliko kola kad ih ima; manje samo na početku baze |
| `test_broj_zavisi_samo_od_prozora` | detalj broja za (granica, 50) ≠ za (granica, 100) kad se stvarno razlikuju; nikad ne uključuje kola > granica |
| `test_prognoza_jednaka_retro_bektestu` | `prognoza_u_tacki(g)` == red retro-bektesta za isto kolo, za sve metode |
| `test_leakage` | **najvažniji**: kopiraj bazu u temp, izmeni brojeve u svim kolima > granica na proizvoljne vrednosti, ponovo izračunaj kontekst/broj/rangiranje/prognozu za granicu → rezultati **bit-identični** originalu |
| `test_api_bez_granice_422` | endpointi bez `granica` vraćaju 422, ne rezultat nad celom bazom |

Za `test_leakage` koristi tempfile kopiju `loto_baza.db` — nikad ne diraj pravu bazu.

Postojeći testovi (`test_core`, `test_prognoza`, `test_razlicitost`) moraju proći **nepromenjeni** posle svake faze.

---

## 4. Performance — pravila

- Interaktivni ekran radi nad **prozorom**, ne nad celom bazom, osim kad korisnik izabere „sva“.
- Nijedna UI akcija ne pokreće batch retro-bektest. Batch ostaje isključivo na tabu Bektest.
- Ciljno vreme odgovora svakog `/api/istorija/*` endpointa: < 200 ms za prozor ≤ 200; „sva“ (1.422 kola) < 1 s.
- Keš tek ako se izmeri sporost; ako se doda, ključ `(granica, prozor, ...)`, i **mora se invalidirati** u `baza.py` pri unosu novog kola (isti hook gde se već pozivaju evaluacije tiketa/prognoza).

---

## 5. Šta se NE radi (ponovljeno iz plana, jer je bitno)

- Nema ML/VAE/Gemini.
- Nema novih prediktora.
- Nema novih statističkih testova.
- Nema promene šeme baze.
- Nema novog frontend frameworka — ostaje Alpine + ECharts.
- Nema paralelne implementacije frekvencije/rangiranja/različitosti/prognoze u `istorija.py`.

Ako ti se u toku rada učini da nešto od ovoga „mora“ — stani i pitaj.

---

## 6. Definicija „gotovo“ za celu funkciju

Korisnik bez znanja o implementaciji može da:

1. izabere bilo koje kolo,
2. vidi šta se dešavalo pre njega u prozoru po izboru,
3. klikne na broj i vidi njegovu istoriju,
4. vidi rangiranje i različitost „tada“,
5. pokrene prognozu i tek onda otkrije stvarni ishod,
6. pročita jednu rečenicu koja kaže da li je rezultat drugačiji od slučajnosti (i skoro uvek — nije).

Plus: `git status` čist, sva četiri test modula prolaze, `FUNKCIJE.md` dobio sekciju 10 „Istraži istoriju“, `Stanje_projekta.md` ažuriran.
