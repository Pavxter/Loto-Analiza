# Loto Analizator — Opis svih funkcija

Ovaj dokument objašnjava sve funkcije web aplikacije: šta svaka radi, šta pojedini
pojmovi znače i kako se koristi.

> **Važno pre svega:** Loto izvlačenja su nezavisni slučajni događaji. Ni jedna
> statistika ovde (frekvencija, Bajes, hibrid) ne može da predvidi buduće brojeve.
> Aplikacija služi za **istraživanje istorije i sastavljanje kombinacija po pravilima**
> — kao alat za analizu i zabavu, ne kao proricanje.

Igra: **Loto 7/39** (izvlači se 7 brojeva iz opsega 1–39). U bazi je 1.422 odigranih kola.

---

## Zajednički pojmovi

| Pojam | Značenje |
|---|---|
| **Vrući brojevi** | 13 brojeva koji su se najčešće izvlačili u posmatranom periodu. |
| **Hladni brojevi** | 13 brojeva koji su se najređe izvlačili (prvo neizvučeni, pa najređi). |
| **Neutralni brojevi** | Svi ostali (niti vrući, niti hladni). |
| **Sveži brojevi** | Brojevi izvučeni u poslednjih 10 kola. |
| **Srednja vrednost kombinacije** | Prosek 7 brojeva u jednoj kombinaciji (zbir ÷ 7). |
| **Bazen brojeva** | Suženi skup brojeva (npr. 15–20) iz kojih generator pravi kombinacije. |
| **Period** | Koliko poslednjih kola se analizira (Sva / 50 / 100 / 200 / 500). Menja se gore desno. |

---

## 1. Dashboard

Centralni pregled stanja na osnovu izabranog perioda.

- **Prosečna sredina** — prosek srednjih vrednosti svih kola u periodu (obično ~20).
- **Std. devijacija** — koliko srednje vrednosti variraju oko proseka.
- **Najčešći par/nepar** — najčešći odnos parnih i neparnih brojeva (npr. 3 parna / 4 neparna).
- **Analizirano** — koliko je kola ušlo u analizu.
- **Vrući / Hladni brojevi** — obojene kuglice (crveno = vruć, plavo = hladan).
- **Frekvencija brojeva** — stubičasti grafikon učestalosti svakog broja, obojen po kategoriji.
- **Predlog bazena** — automatski predlog bazena kao fuzija top-12 vrućih i top-12 svežih brojeva.
- **Prebaci u Generator** — šalje predloženi bazen direktno na stranu Generator.

**Kako koristiti:** Promeni period gore desno da vidiš kako se vrući/hladni menjaju.
Klik na „Prebaci u Generator" ako želiš odmah da praviš kombinacije iz predloga.

---

## 2. Statistika

Detaljne raspodele i napredna analiza.

- **Frekvencija po broju** — veliki grafikon učestalosti (isto kao na Dashboardu, obojeno).
- **Raspodela srednjih vrednosti** — histogram: koliko kola ima koju srednju vrednost
  (obično zvonolik oblik oko 20).
- **Trend srednje vrednosti** — linija kroz celu istoriju; može se zumirati (dole klizač).
- **Ritam ponavljanja** — prosečan razmak (u broju kola) između dva pojavljivanja istog broja.
- **Uzastopni brojevi** — koliko kola ima 0, 1, 2… para susednih brojeva (npr. 12 i 13).
- **Raspodela po dekadama** — prosečan broj izvučenih brojeva iz grupa 1–9, 10–19, 20–29, 30–39.

### Napredno — poziciona analiza
- **Frekvencija broja po poziciji izvlačenja** — toplotna mapa: red = broj (1–39),
  kolona = redni broj izvlačenja (1–7). Pokazuje da li se neki brojevi češće pojavljuju
  kao „prvi izvučeni", „drugi" itd.
- **Hi-kvadrat test pristrasnosti pozicija** — statistički test koji proverava da li je
  raspored po pozicijama nasumičan. Prikazuje Hi-kvadrat vrednost, P-vrednost i zaključak
  („Pristrasno" / „Nasumično").

**Kako koristiti:** Prelazi mišem preko grafikona za tačne vrednosti (tooltip).
Klik na „Pokreni test" za hi-kvadrat rezultat.

---

## 3. Rangiranje brojeva

Rangira svih 39 brojeva trima metodama (biraš gore desno segmentiranim dugmetom):

- **Frekvencija** — rang po ukupnom broju pojavljivanja.
- **Bajesovski** — iterativni model „verovanja": kroz celu istoriju svako kolo blago
  podiže verovatnoću izvučenih, a spušta ostalih (learning rate 0.005), uz normalizaciju.
- **Hibridni** — 80% Bajesovski skor + 20% bonus „povezanosti" (koliko se broj često
  izvlači zajedno sa top-20 Bajesovih brojeva).

Prikaz: grafikon skora po broju + tabela sa rangom, skorom i (za hibrid) povezanošću.

> **Napomena:** Sve tri metode u praksi daju skoro isti redosled, jer sve reflektuju
> istu istorijsku frekvenciju. Zato su spojene u jednu stranu radi poređenja.

---

## 4. Generator

Srce aplikacije — pravi kombinacije po zadatim pravilima i boduje ih.

### Izvor brojeva
- **Koristi prilagođeni bazen** — ako je uključeno, generiše samo iz brojeva koje uneseš;
  inače koristi svih 1–39.

### Filteri (svaki se može uključiti/isključiti)
- **Min./Max. sredina** — zadržava samo kombinacije čija je srednja vrednost u opsegu.
- **Tačno parnih** — koliko parnih brojeva mora biti u kombinaciji.
- **Uzastopnih** — koliko parova susednih brojeva.
- **Vrućih / Hladnih** — koliko brojeva iz tih kategorija.
- **Max po dekadi** — najviše koliko brojeva iz jedne dekade (npr. da ne budu svi 30-ih).
- **Izbaci već izvučene** — preskače kombinacije koje su se tačno tako već pojavile u istoriji.

### Bodovanje i rangiranje
- **Strategija svežine** — favorizuj / kažnjavaj / ignoriši sveže brojeve u skoru.
- **Model pozicione pristrasnosti** — dodaje bonus/kaznu na osnovu pozicione analize.
- **Filter diverziteta** — izbacuje međusobno preslične kombinacije (podešava se „max sličnost").

**Skor** kombinuje blizinu prosečnoj srednjoj vrednosti, svežinu, ritam i (opciono) pristrasnost.
Rezultat je tabela kombinacija sa skorom; klik „+ tiket" dodaje kombinaciju u „Moje tikete".

**Kako koristiti:** Podesi filtere → „Generiši". Ako je rezultat prazan, filteri su prestrogi
(npr. zbir uslova nemoguć) — olabavi ih.

---

## 5. Bektest

Prati koliko su uspešne sačuvane strategije za buduća kola.

- Svaki red je jedna sačuvana strategija: kolo, opis, bazen, broj kombinacija.
- Kad uneseš to kolo (na strani „Podaci"), aplikacija **automatski** izračuna:
  - **Rezultat** — koliko brojeva iz bazena je pogođeno i najbolji rezultat kombinacija
    (7:x, 6:x, 5:x, 4:x pogodaka).
  - **Indeks promašaja** — zbir „udaljenosti" najbliže kombinacije od dobitne (manje = bliže).
  - **Indeks iznenađenja** — koliko je kombinacija statistički „retka" (veće = ređa).
- Dugme ✕ briše bektest.

**Kako koristiti:** Bektestovi se prave iz drugih delova (npr. čuvanjem bazena/strategije),
a rezultat se popuni sam kad uneseš odgovarajuće kolo.

---

## 6. Moji tiketi

Evidencija stvarno odigranih (ili planiranih) kombinacija.

- **Dodaj tiket** — unese 7 brojeva zarezom razdvojenih.
- Tabela prikazuje sve tikete, poslednji rezultat (broj pogodaka) i datum provere.
- Kad uneseš novo kolo, svi tiketi se automatski provere i ažuriraju.
- Dugme ✕ briše tiket.

---

## 7. Podaci

Unos i uvoz istorijskih rezultata.

- **Unesi novo kolo** — broj kola, datum, 7 brojeva. Po čuvanju automatski proverava
  sve tikete i bektestove za to kolo. Validacija: tačno 7 jedinstvenih brojeva u opsegu 1–39.

> **Numeracija kola:** `kolo = godina * 1000 + broj kola u toj godini`.
> Npr. 56. kolo 2026. godine → **2026056**; sledeće kolo unosiš kao **2026057**.
> Ovako je broj jedinstven i hronološki raste (aplikacija ga koristi da poveže
> bektestove i da odredi „istoriju pre ovog kola"), a čuva zvanični broj kola.

> **Redosled brojeva:** unosi ih **kako su izvučeni**, ne sortirano. Za većinu analiza
> je svejedno, ali poziciona analiza, model pristrasnosti i hi-kvadrat test imaju smisla
> samo sa pravim redosledom izvlačenja.
- **Uvoz iz fajla** — CSV ili Excel sa kolonama `kolo, datum, b1, b2, b3, b4, b5, b6, b7`.
  Dva režima:
  - **Uvezi (dodaj)** — dodaje kola postojećoj istoriji; duplikati se preskaču.
  - **Zameni istoriju uvozom** (čekboks) — obriše SVU postojeću istoriju pa uveze iz fajla.
    Za slučaj kada je baza neispravna ili treba potpuna zamena. Sigurnosne mere:
    (1) pre brisanja se automatski pravi rezervna kopija `loto_baza_backup_<datum_vreme>.db`,
    (2) fajl se prvo proveri — ako je neispravan, ništa se ne briše,
    (3) traži se potvrda pre izvršenja.
- **Poslednja kola** — tabela poslednjih unetih kola sa obojenim kuglicama.

> **Vraćanje kopije:** ako nešto pođe naopako, ugasi aplikaciju i preimenuj poslednji
> `loto_baza_backup_….db` nazad u `loto_baza.db`.

### Priprema sirovog CSV-a (`pripremi_csv.py`)

Sirovi fajlovi sa rezultatima obično dolaze u drugom formatu (`BrKola, datum, br1..br7`,
datum kao `DD.MM.YYYY`, broj kola se resetuje svake godine). Skripta ih konvertuje u
format koji aplikacija očekuje i usput ispravlja greške:

```bash
python pripremi_csv.py "C:\Users\pavks\Desktop\Lotto.csv"
```

Radi sledeće: mapira kolone → `kolo, datum, b1..b7`, prevodi datum u `YYYY-MM-DD`,
popravlja godine kojima fali cifra (`226` → `2026`), primenjuje poznate ručne ispravke,
računa `kolo = godina*1000 + BrKola`, sortira hronološki i **validira sve** (opseg 1–39,
7 jedinstvenih brojeva, jedinstven broj kola). Ako nešto nije u redu — prijavi red i
ne napravi izlazni fajl. Rezultat je `<ime>_za_uvoz.csv` spreman za uvoz u aplikaciju.

**Kako koristiti (redovna upotreba):** posle svakog izvlačenja otvoriš „Podaci", uneseš
kolo i brojeve, klik „Sačuvaj kolo" — sve analize i provere se osveže automatski.

---

## Podešavanja i tehnički detalji

- **Period analize** (gore desno) utiče na Dashboard, Statistiku i bodovanje generatora.
- Baza podataka je `loto_baza.db` (ista kao u desktop verziji).
- Grafikoni zahtevaju internet (učitavaju se preko CDN-a).
- Pokretanje: dupli klik na `Pokreni Loto.bat` ili `python pokreni.py`.

## Šta je izostavljeno iz v1 (moguće dodati kasnije)
- **ML/VAE generator** (neuronska mreža) — izbačen jer za loto uči istu frekvenciju uz
  veliku složenost i tešku zavisnost (TensorFlow).
- **Gemini AI preporuke** — tekstualne AI analize; mogu se vratiti kao opcioni modul.
