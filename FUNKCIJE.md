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

**Panel „Različitost seta"** (ispod skora) posle svakog generisanja pokazuje: **prosečno i
maksimalno preklapanje** po svim parovima generisanih kombinacija (uz referencu — nasumičan
set bi imao ~1,26) i **pokrivenost** (koliko od 39 brojeva set ukupno sadrži, vizuelno kao
39 kuglica). Time filter diverziteta dobija merljiv, vidljiv efekat.

**Kako koristiti:** Podesi filtere → „Generiši". Ako je rezultat prazan, filteri su prestrogi
(npr. zbir uslova nemoguć) — olabavi ih.

---

## 5. Bektest

Prati koliko su uspešne sačuvane strategije za buduća kola.

- Svaki red je jedna sačuvana strategija: kolo, opis, bazen, broj kombinacija.
- Kad uneseš to kolo (na strani „Podaci"), aplikacija **automatski** izračuna:
  - **Rezultat** — koliko brojeva iz bazena je pogođeno i najbolji rezultat kombinacija
    (7:x, 6:x, 5:x, 4:x pogodaka).
  - **Prosek / Maks preklapanja** — prosečan i najbolji broj zajedničkih brojeva svih
    kombinacija sa dobitnom (slučajno očekivanje po kombinaciji: **1,26**).
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

## 8. Prognoza

Statistički eksperiment: predviđanje **jednog broja** (1–39) za sledeće kolo, sa
sedam paralelnih metoda i kontrolnom grupom.

> **Svrha:** ovo NIJE proricanje. Meri se da li ijedan metod pogađa statistički
> značajno više od nasumične osnovne linije **17,95%** (verovatnoća da nasumično
> izabran broj bude među 7 izvučenih od 39). Očekivani ishod poštene igre: nijedan.

### Metode (prediktori)
| Metod | Logika |
|---|---|
| **Najvrući** | Broj sa najviše pojavljivanja u periodu. |
| **Najhladniji** | Najmanje pojavljivanja; prioritet neizvučenim („due" hipoteza). |
| **Bajesovski** | Najviši skor Bajesovog modela (isti kao na strani Rangiranje). |
| **Hibridni** | Najviši hibridni skor (80% Bajes + 20% povezanost). |
| **Ritam koji kasni** | Najveći odnos D/R — broj koji najviše „kasni" za svojim ritmom. |
| **Najsvežiji** | Prvi izvučen broj iz poslednjeg kola. |
| **Nasumični (kontrola)** | Nasumičan broj, seedovan brojem kola. Referenca — ako i on „odskoči", greška je u sistemu. |

### Kako radi
- **Predlozi:** otvaranjem strane računaju se i **zaključavaju** predlozi za sledeće
  kolo. „Preračunaj predloge" ih menja samo dok kolo nije uneto — ocenjena prognoza
  je nepromenljiva (poenta eksperimenta).
- **Automatska ocena:** pri unosu kola na strani „Podaci" sve prognoze za to kolo
  dobijaju pogodak ✓/✗ bez ikakve akcije.
- **Retro-bektest:** dugme prolazi kroz celu istoriju (walk-forward, period 100 kola,
  preskače prvih 50) i ocenjuje sve metode retroaktivno. Determinističan — svako
  pokretanje daje isti rezultat. Retro i uživo rezultati se vode **odvojeno** (retro
  je metodološki slabiji).
- **Grafikon:** kumulativna uspešnost po metodu, isprekidana linija = baseline 17,95%,
  osenčana zona = 95% pojas pouzdanosti (sužava se sa brojem kola). Linija unutar
  zone = nerazlučivo od slučajnosti.
- **Tabela:** za svaki metod n, pogoci, uspešnost, p-vrednost (dvostrani binomni test)
  i zaključak. Prag je pooštren Bonferroni korekcijom (0,05 / 7 ≈ **0,007**) jer se
  7 metoda testira paralelno.
- Ako metod „odskače": najverovatniji uzroci su greška u podacima, curenje budućnosti
  ili slučajnost uprkos korekciji — **proveriti pre bilo kakvog zaključka**.

### Rezultat retro-bektesta nad tvojom bazom (1.358 ocenjenih kola)
Svih 7 metoda završilo je između 16,6% i 18,5% — statistički nerazlučivo od
slučajnosti (svi p ≫ 0,007). Kontrolna grupa (18,19%) je čak nadmašila Bajesa.
Upravo to teorija i predviđa za poštenu igru.

### Tab „Kombinacija" — predviđanje 7 brojeva

Pored jednog broja, svaki metod predlaže i **jednu kompletnu kombinaciju od 7 brojeva**.
Umesto pogotka ✓/✗ meri se **preklapanje sa dobitnom kombinacijom** — koliko od 7
predloženih brojeva se poklopi (0–7).

| Metod | Logika |
|---|---|
| **Top-7 vrućih / hladnih** | 7 najfrekventnijih, odnosno 7 najređih brojeva u periodu. |
| **Top-7 Bajes / hibrid** | Vrh Bajesove, odnosno hibridne rang-liste. |
| **Top-7 po ritmu** | 7 brojeva sa najvećim odnosom kašnjenja i ritma (D/R). |
| **Ko-okurencijski** | Pohlepno bira brojeve koji najčešće izlaze zajedno (jedini algoritamski nov). |
| **Nasumični (kontrola)** | 7 nasumičnih brojeva (seed = kolo·2). Referenca za poređenje. |

- **Osnovna linija:** ako su izvlačenja slučajna, svaka kombinacija ima isto očekivano
  preklapanje **μ = 1,256** sa sledećim kolom (7·7/39), bez obzira kako je izabrana.
- **Grafikon:** kumulativni prosek preklapanja po metodu; osenčena zona je pojas
  `μ ± 1,96·σ/√n` (σ ≈ 0,932) koji se sužava sa brojem kola.
- **Histogram:** raspodela preklapanja (0–7) izabranog metoda naspram teorijske
  (hipergeometrijske) krive, sa hi-kvadrat testom. Metod može imati prosek ≈ μ a ipak
  drugačiji **oblik** raspodele — i to je nalaz.
- **Tabela:** prosek preklapanja, najbolje postignuto, p-vrednost (z-test proseka;
  „—" za n < 30) i zaključak. Prag je Bonferroni preko **svih** metoda strane
  (jednobrojni + kombinacijski = isti eksperiment): 0,05 / 14 ≈ **0,0036**.
- **„+ tiket":** dugme na kartici predloga dodaje kombinaciju u „Moje tikete" da se
  može stvarno pratiti.

**Rezultat nad tvojom bazom (1.359 ocenjenih kola, retro):** svih 7 metoda ima prosek
u rasponu 1,25–1,32, svi statistički nerazlučivi od μ = 1,256 (svi p ≫ 0,0036).
Kontrolna grupa ima najviši prosek (1,32) — čist šum, i Bonferroni ga ispravno ne
proglašava odskačućim. Poštena igra, kako teorija i predviđa.

---

## 9. Različitost

Meri koliko se **izvučene kombinacije međusobno razlikuju** po sadržaju (broj
zajedničkih brojeva) i poredi istoriju sa egzaktnom teorijskom raspodelom slučajnosti.

> **Svrha:** kombinacije sličnih statističkih profila (sredina, par/nepar, dekade) po
> pravilu dele vrlo malo zajedničkih brojeva. Ova strana to kvantifikuje i testira: da
> li se istorija ponaša kao čista slučajnost, ili ne.

### Teorijska osnova
Za dve nezavisne slučajne kombinacije 7/39, broj zajedničkih brojeva k prati
**hipergeometrijsku raspodelu** P(k) = C(7,k)·C(32,7−k)/C(39,7). Očekivano preklapanje
je **1,256**, a najverovatnije je da dele tačno **1** broj (≈41%). Deliti 5+ brojeva je
izuzetno retko (< 0,08%).

### Analize
- **Rekordi (cela istorija):** da li je ikad ponovljena identična kombinacija (7/7),
  najveće preklapanje ikad (koja dva kola, koji brojevi) i broj parova sa 5+ zajedničkih
  naspram slučajnog očekivanja. Uz objašnjenje **paradoksa rođendana** — ogroman broj
  parova čini „neverovatna" poklapanja očekivanim.
- **Uzastopna kola:** histogram preklapanja svakog para (N, N+1) preko teorijske krive,
  sa hi-kvadrat testom.
- **Svi parovi (sklopivo):** isto nad svim parovima kola (~milion). Test je ovde
  orijentacioni jer parovi nisu potpuno nezavisni.
- **Profil ne predviđa sadržaj:** za uzorak parova crta razliku profila (sredina /
  broj parnih / raspored po dekadama) naspram broja zajedničkih brojeva, uz **Pearson-ov
  koeficijent korelacije**. Slaba veza = profil ne nosi informaciju o sadržaju.
- **Ko-okurencija parova:** toplotna mapa z-skorova (koliko češće/ređe od slučajnosti
  se svaki par 39×39 pojavljuje zajedno), klik na ćeliju za detalje. Ispod je „kontrola
  lažnih alarma": koliko parova je van 95% intervala naspram očekivanih **≈ 37** — ako
  je posmatrano ≈ 37, „vrući parovi" su šum, ne signal.

### Rezultat nad tvojom bazom
Preklapanje uzastopnih kola i svih parova ne razlikuje se od slučajnosti (p ≫ 0,05);
najveće preklapanje ikad je **6/7** (nikad identično), broj „vrućih parova" van intervala
je ≈ očekivani. Sve u skladu sa čistom slučajnošću.

---

## 10. Istraži istoriju (vremeplov)

Putovanje kroz kola: bira se bilo koje odigrano kolo i vidi **tačno ono što je sistem
znao u tom trenutku** — nijedna brojka ne „viri" iz budućnosti. Služi za učenje i
proveru: šta bi svaka analiza pokazala „tada" i da li se to razlikuje od slučajnosti.

### Dva ključna pojma
| Pojam | Značenje | Primer |
|---|---|---|
| **Granica** | poslednje kolo koje sistem „zna" (uključivo) | 2026-060 |
| **Cilj** | prvo stvarno kolo posle granice — ono što pokušavamo da „pogodimo" | 2026-061 |

Pravilo kroz celu stranu: **dostupni podaci = sva kola ≤ granica**. Cilj i sve posle njega
su nedostupni sve do trenutka kada svesno zatražiš ishod. Klik na kolo u tabeli bira to
kolo **kao cilj** (granica postaje kolo pre njega), pa je „predikcija tada" uvek predikcija
*za* kliknuto kolo.

### Navigacija i prozor
- Toolbar: `«` / `‹` (skok za ceo prozor / jedno kolo unazad), tekuće kolo, `›` / `»`
  (napred), „Idi na najnovije". Kretanje ide **po redosledu u bazi**, ne aritmetikom
  broja kola (prelaz godine 2025-052 → 2026-001 radi ispravno).
- **Prozor** (20 / 50 / 100 / 200 / sva) — koliko poslednjih kola ≤ granice ulazi u
  analize. Cela strana radi nad prozorom (brzo), osim kad izabereš „sva".
- **Info linija:** sistem zna zaključno sa …, naredno kolo (cilj) …, kola u prozoru,
  raspon datuma. Tu je i dugme **„Generiši sa znanjem do ovog kola →"** (vidi dole).
- **Tabela prethodnih kola:** kolo, datum, 7 brojeva **redosledom izvlačenja**. Klik na
  red bira kolo kao cilj; klik na broj otvara detalj broja.

### Detalj broja (klik na bilo koji broj)
Panel bez napuštanja strane: pojavljivanja u prozoru i ukupno (≤ granice), trenutni
razmak, prosečan/min/maks razmak, poslednje/prethodno pojavljivanje, raspodela po
poziciji izvlačenja i **timeline** (kada je broj izlazio u prozoru). Iznad je rečenica
tumačenja („Broj 22 se pojavio 22 puta u poslednjih 100 kola; očekivanje ≈ 17,95;
razlika nije značajna."). Linkovi **„Vidi broj u: Statistici / Rangiranju →"** skoče na
taj tab i istaknu broj u grafikonu.

### Kontekst kola (sklopive sekcije)
- **Šta se dešavalo pre ovog kola** — sažetak prozora za svih 39 brojeva: pojavljivanja,
  poslednji put, trenutni razmak (sortirano po učestalosti).
- **Različitost cilja** — koliko naredno kolo liči na prošlost: preklapanje sa neposredno
  prethodnim kolom, prosek u prozoru naspram μ, najveće istorijsko preklapanje i
  ponovljeni parovi iz cilja. Link **„Cela analiza različitosti →"** vodi na stranu 9.
- **Rangiranje „tada"** — kako bi tri metode (Frekvencija / Bajes / Hibrid) rangirale
  brojeve nad podacima ≤ granice.

### Predikcija „tada" (vremeplov prognoze)
Najvažniji deo. Dugme **„Izračunaj šta bi sistem tada predvideo"** pokaže predloge svih
sedam jednobrojnih i svih kombinacijskih metoda za cilj — **isključivo iz podataka ≤
granica**. Tek dugme **„Prikaži stvarni ishod"** otkriva stvarno kolo i ocenu (pogodak
✓/✗ za jedan broj, preklapanje 0–7 za kombinaciju) — ishod je namerno skriven do klika,
to je poenta eksperimenta. Dugmad **◀ prethodno / sledeće ▶** pomeraju tačku uz
automatski preračun (ishod se ponovo sakrije). Ispod je rečenica konteksta (μ = 1,256).

> **Garancija tačnosti:** ovaj vremeplov koristi isti kod kao retro-bektest sa strane
> Prognoza — rezultat za bilo koje kolo je bit-identičan odgovarajućem redu retro-bektesta
> (pokriveno testom). Zato je i ovde ishod skoro uvek „nerazlučivo od slučajnosti".

### Veza sa Generatorom (vremeplov generisanja)
Dugme **„Generiši sa znanjem do ovog kola →"** otvara Generator koji analizira **samo
kola ≤ granica** (polje „Analiziraj do kola" u Generatoru, prazno = cela baza). Tako se
generisanje kombinacija može reprodukovati „iz prošlosti", bez uticaja kasnijih kola.

**Kako koristiti:** izaberi kolo (ili se kreći strelicama), podesi prozor, otvori sekcije
koje te zanimaju, pa u „Predikcija tada" prvo izračunaj predikciju a onda otkrij ishod.

---

## 11. Mapa kombinacija

Ceo prostor igre kao jedna zumabilna slika: svih **15.380.937** kombinacija 7/39, svaka
kao jedna ćelija, a 1.422 izvučena kola kao tačke na njoj.

> **Svrha:** osećaj razmere. Ostale strane mere prošlost brojevima; ova pokazuje koliko
> je prostor velik i koliko je izvučenih malo (0,009%). Nije nova analiza — jedini test
> na strani je preformulacija testa različitosti.

### Kako je prostor pretvoren u mapu

| Pojam | Značenje |
|---|---|
| **Rang** | redni broj kombinacije, 0 … 15.380.936, po leksikografskom redosledu (1-2-3-4-5-6-7 je 0, 33-34-35-36-37-38-39 je poslednja). |
| **Ćelija** | mesto na mreži 4096 × 4096 koje Hilbertova kriva reda 12 dodeljuje rangu. |
| **Prazan deo** | mreža ima 16.777.216 ćelija, a kombinacija je 15.380.937; preostalih ~8% nema kombinaciju i providno je (stepenasta oblast gore desno). |

Hilbertova kriva je izabrana zato što **susedni rangovi ostaju prostorno blizu**, pa mapa
ima teksturu umesto šuma. Raspored je trajan: ista kombinacija uvek pada na isto
mesto. Mapa **nije slika verovatnoće** — svaka ćelija ima potpuno istu šansu.

### Slojevi (boja pozadine)

Boja kodira osobinu, a ne identitet kombinacije:

| Sloj | Šta boji | Opseg |
|---|---|---|
| **Zbir sedam brojeva** | zbir kombinacije | 28 – 252 |
| **Razlika najvećeg i najmanjeg** | raspon | 6 – 38 |
| **Koliko je parnih brojeva** | 0 – 7 | diskretno |
| **Koliko dekada dodiruje** | 1 – 4 | diskretno |
| **Ocena Generatora** | skor kojim Generator rangira kombinacije | izmereno 55 – 204 |

**Ocena** je jedini sloj koji zavisi od baze: računa se iz svežih brojeva i ritma, pa se
peče za stanje baze u trenutku pokretanja `generisi_mapu.py` (aplikacija ispod mape piše
za koje kolo važi) i **ne menja se pri unosu novog kola**. Komponenta pristrasnosti je
izostavljena jer zavisi od pozicije u izvlačenju, a kombinacije na mapi su sortirane.
Ocena je pravilo po kom Generator bira, ne verovatnoća; sloj je koristan da se vidi
**koliki deo prostora bodovanje uopšte razlikuje**.

Legenda pored birača sloja pokazuje skalu (tamno = malo, svetlo = mnogo).

### Zum i pločice

Mapa se ne crta u browseru nego iz unapred ispečenih PNG pločica (256 × 256, zumovi 0–4).
Na zumu 4 **jedan piksel je jedna kombinacija**; ispod toga je piksel prosek celog bloka
(traka iznad mape stalno piše koliko). Dalje uvećanje je samo razvlačenje slike.
Točak miša skroluje stranu, **`Ctrl` + točak zumira**; dugme „Cela mapa" vraća pogled.

Pločice ne idu u git. Ako nisu generisane, tab pokazuje komandu:
`python -X utf8 generisi_mapu.py --sloj sve` (oko minut i po, oko 46 MB).

### Tačke: izvučeno i kontrola

- **Stvarno** — 1.422 izvučene kombinacije kao tačke.
- **Slučajno** — kontrolni set iste veličine, izvučen ravnomerno iz celog prostora sa
  fiksnim seed-om (uvek ista slika). Dugme „Drugi slučajan uzorak" menja seed.
- **Oba** — oba seta odjednom; tek tada su različite boje (izvučeno narandžasto, kontrola
  bela). Kada se gleda jedan set, oba izgledaju **identično** — u tome je i poenta.

Poluprečnik tačke raste sa zumom: krupne tačke na malom zumu bi lagale da je prostor pun.

### Vreme i putanja

- **Slajder** ide kroz kola hronološki; „vreme" je isti pojam **granice** kao na strani
  „Istraži istoriju" — prikazuju se samo kola ≤ izabranog. Uz njega su korak nazad/napred,
  **Pusti / Pauza** (cela istorija prođe za oko pola minuta) i „Do kraja".
- **Putanja** spaja kola redom izvlačenja. **Boja segmenta = koliko brojeva to kolo deli
  sa prethodnim (0–7)**, po istom pojmu preklapanja kao strana „Različitost"; legenda je
  ispod slajdera. Gola linija se nikad ne crta, da ne bi sugerisala pravac koji ne postoji.
- **Rep** bira koliko poslednjih segmenata se vidi (10 / 50 / 200 / sve).
- Kontrolni set ima **svoju putanju po istim pravilima**; u prikazu „Oba" je isprekidana
  samo da bi se znalo koji je koji.

> Linija pokazuje **redosled u vremenu, ne kretanje kroz prostor**. Dve susedne ćelije na
> mapi mogu biti izvučene godinama jedna od druge.

### Klik i „Gde je moj tiket"

Klik na ćeliju (ili na tačku) otvara panel: 7 brojeva, redni broj u prostoru, ćelija,
da li je i kada izvučena, osobine, i rečenica o preklapanju sa istorijom (najveće
poklapanje, poređenje sa μ = 1,256). Odatle vode **„Otvori kolo … u Istraži istoriju →"**
i **„Pomeri vreme na ovo kolo"**. Prelaz mišem preko tačke pokazuje kolo i preklapanje.

Polje **„Gde je moj tiket"** prima 7 brojeva i skoči na njihovu ćeliju uz pun zum; dugme
**„Slučajna"** bira nasumičnu kombinaciju iz celog prostora. Na malom zumu jedan piksel
pokriva više kombinacija, pa klik bira jednu iz tog bloka — zumiraj do kraja za tačnu.

### Sekcija „Test" (sklopiva)

Histogram **dužina skokova**: koliko je daleko na mapi svako kolo od prethodnog, za
stvarna kola i za kontrolni set, po istim kantama. Dužina skoka je druga mera razlike dva
uzastopna kola, pa je ovo **isti test različitosti u drugom obliku**, ne nova statistika.

Nad tvojom bazom: prosečan skok **2.074** ćelije za stvarna kola i **1.982** za kontrolu
(medijane 2.007 i 1.938), na 1.421 koraku — raspodele se poklapaju.

### Šta mapa ne pokazuje

- **Mrlje i prelazi u boji** dolaze od redosleda rangiranja (susedne ćelije se razlikuju
  samo u poslednjim brojevima), ne od izvlačenja.
- **Grozdovi tačaka** su očekivani i za čist slučaj; kontrolni sloj postoji baš zato da se
  to vidi, umesto da se veruje utisku.
- **Nijedan pravac** na mapi nije trend. Putanja je redosled u vremenu.
- **Ocena** nije verovatnoća da će kombinacija izaći.

**Kako koristiti:** izaberi sloj, klikni „Slučajno" pa „Stvarno" nekoliko puta i uporedi
slike; pusti vreme da vidiš kako tačke skaču bez reda; nađi svoj tiket; na kraju otvori
„Test" da vidiš isto to kao brojku.

---

## Podešavanja i tehnički detalji

- **Period analize** (gore desno) utiče na Dashboard, Statistiku i bodovanje generatora.
  Strane „Istraži istoriju" i „Mapa kombinacija" ne koriste period nego granicu (vreme).
- Baza podataka je `loto_baza.db` (ista kao u desktop verziji).
- Grafikoni i mapa zahtevaju internet (ECharts, Alpine.js i Leaflet se učitavaju preko CDN-a).
- **Pločice mape** se prave jednom, skriptom `python -X utf8 generisi_mapu.py --sloj sve`
  (oko minut i po, oko 46 MB u `webapp/static/mapa/`). Ne idu u git. Sloj „Ocena
  Generatora" čita bazu, pa se osvežava ponovnim pokretanjem (`--sloj ocena`).
- Pokretanje: dupli klik na `Pokreni Loto.bat` ili `python pokreni.py`.
- **Osvežavanje posle izmena:** statički fajlovi se serviraju sa `Cache-Control: no-cache`,
  pa običan refresh (F5) uvek povuče najnoviju verziju — nema potrebe za `Ctrl`+`F5`.
  Nepromenjeni fajlovi se i dalje serviraju brzo (HTTP 304, revalidacija preko ETag-a).
  Pločice mape su izuzetak — keširaju se dugoročno jer se menjaju samo kad se ponovo
  pokrene `generisi_mapu.py`.

## Šta je izostavljeno iz v1 (moguće dodati kasnije)
- **ML/VAE generator** (neuronska mreža) — izbačen jer za loto uči istu frekvenciju uz
  veliku složenost i tešku zavisnost (TensorFlow).
- **Gemini AI preporuke** — tekstualne AI analize; mogu se vratiti kao opcioni modul.
