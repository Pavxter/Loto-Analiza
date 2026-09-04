# Plan implementacije: strana „Različitost" (analiza preklapanja kombinacija)

Dokument za programera. Nova strana aplikacije **Loto Analizator** koja meri koliko se
izvučene kombinacije međusobno razlikuju po sadržaju (zajednički brojevi), poredi
istoriju sa egzaktnom teorijskom raspodelom slučajnosti i dodaje mere različitosti
u Generator i Bektest.

> **Svrha (preneti u UI):** kombinacije sličnih statističkih profila (sredina,
> par/nepar, dekade) po pravilu dele vrlo malo zajedničkih brojeva. Ova strana to
> kvantifikuje i testira: da li se istorija ponaša kao čista slučajnost, ili ne.

---

## 1. Teorijska osnova (referenca za sve grafikone)

**Hipergeometrijska raspodela preklapanja.** Za dve nezavisne nasumične kombinacije
7/39, verovatnoća da dele tačno k brojeva:

```
P(k) = C(7,k) · C(32, 7−k) / C(39,7),   k = 0..7
```

Vrednosti (izračunati u kodu, ne hardkodovati — ovo je za proveru):

| k | P(k) |
|---|---|
| 0 | ≈ 21,9 % |
| 1 | ≈ 41,2 % |
| 2 | ≈ 27,5 % |
| 3 | ≈ 8,2 % |
| 4 | ≈ 1,13 % |
| 5 | ≈ 0,068 % |
| 6 | ≈ 0,0015 % |
| 7 | ≈ 1 / 15.380.937 |

Očekivano preklapanje: **E[k] = 7·7/39 ≈ 1,256**.

**Ko-okurencija para brojeva:** verovatnoća da se konkretan par (a, b) nađe zajedno
u jednom kolu: `C(37,5)/C(39,7) = 7·6/(39·38) ≈ 2,834 %` → očekivano ≈ 40,3 puta u
1.422 kola. Ukupno parova: C(39,2) = 741.

Implementirati kao modul `razlicitost_teorija.py` sa čistim funkcijama:
`hipergeom_pmf(k)`, `ocekivano_preklapanje()`, `p_par_u_kolu()`, plus binomni interval
pouzdanosti (reuse iz strane Prognoza ako je već implementiran).

---

## 2. Infrastruktura: bitmaske

Sve analize se svode na preklapanje skupova → predstaviti svako kolo kao 39-bitni
integer (bit i−1 = broj i izvučen):

```python
def maska(brojevi: list[int]) -> int:
    m = 0
    for b in brojevi:
        m |= 1 << (b - 1)
    return m

def preklapanje(m1: int, m2: int) -> int:
    return (m1 & m2).bit_count()   # Python 3.8: bin(x).count('1')
```

- Maske za sva kola izračunati **jednom** pri učitavanju strane (lista uparena sa
  brojem kola, hronološki) i proslediti svim analizama.
- Svih parova je C(1422,2) ≈ 1,01 miliona popcount operacija — ispod sekunde u čistom
  Pythonu; ako zatreba brže, `numpy` nad matricom nije potreban za v1.
- Isti modul koristiti i za Generator/Bektest mere (§8) — ne duplirati.

---

## 3. Analiza 1: preklapanje uzastopnih kola

- Za svaki par (kolo N, N+1) hronološki: `preklapanje` → 1.421 vrednosti.
- **Grafikon:** histogram udela po k (0..7), preko njega teorijska kriva P(k)
  (linija ili tačke sa markerima).
- **Hi-kvadrat test:** posmatrano vs. očekivano `n·P(k)`; kategorije k ≥ 4 spojiti u
  jednu ćeliju (pravilo očekivane frekvencije ≥ 5). Prikazati χ², df, p-vrednost i
  zaključak istim stilom kao postojeći test pozicione pristrasnosti.
- Poštovati globalni period aplikacije (analizira se poslednjih X kola).

## 4. Analiza 2: preklapanje svih parova kola

- Dupla petlja preko svih parova maski (ili preko parova unutar perioda).
- Isti prikaz kao Analiza 1: histogram + teorijska kriva + hi-kvadrat.
- **Napomena za test:** parovi nisu potpuno nezavisni (svako kolo učestvuje u 1.421
  paru), pa je formalni hi-kvadrat ovde aproksimativan — u UI tooltip: „test je
  orijentacioni; merodavna je vizuelna podudarnost sa krivom i Analiza 1".

## 5. Analiza 3: najbliži sused i rekordi

Za svako kolo naći maksimalno preklapanje sa **bilo kojim ranijim** kolom (hronološki,
da se svaki „rekord" pripiše trenutku kada se desio).

**Kartice na vrhu strane:**
- „Ponovljena identična kombinacija (7/7)?" — da/ne, i ako da, koja kola.
- „Najveće preklapanje ikad" — k, i par kola sa brojevima (zajednički obojeni).
- „Parova sa 5+ zajedničkih" — posmatrano vs. teorijski očekivano
  (`broj_parova · Σ P(k≥5)`), sa kratkim objašnjenjem paradoksa rođendana:
  veliki broj parova čini „neverovatna" poklapanja očekivanim.

**Grafikon:** linija „maksimalno preklapanje sa istorijom" kroz vreme (stepenasta,
raste retko) — vizuelno pokazuje kako se rekordi gomilaju rano pa proređuju.

## 6. Analiza 4: profil ne predviđa sadržaj (srž strane)

- **Scatter:** za uzorak parova kola (svi parovi = milion tačaka je previše za chart;
  uzeti nasumičan uzorak od ~20.000 parova, seedovan radi reproducibilnosti):
  - X = |razlika srednjih vrednosti dva kola|
  - Y = broj zajedničkih brojeva (uz mali vertikalni jitter da se tačke ne preklope)
- Preko scatter-a: linija prosečnog preklapanja po binovima X (bin širine 1) i
  horizontalna referenca na 1,256.
- **Pearson-ov koeficijent korelacije** r između X i Y, prikazan uz grafikon sa
  interpretacijom: |r| < 0,05 → „profil ne nosi informaciju o sadržaju".
- Isti grafikon ponuditi i za razliku u broju parnih (dropdown izbora profila:
  sredina / broj parnih / raspored po dekadama kao L1 razlika).

## 7. Analiza 5: ko-okurencija parova brojeva

- Matrica 39×39 (simetrična, dijagonala prazna): broj kola u kojima su se a i b
  pojavili zajedno.
- **Toplotna mapa** obojena po odstupanju od očekivanja (z-skor:
  `(posmatrano − n·p) / sqrt(n·p·(1−p))`, p ≈ 2,834 %), divergentna paleta
  (plavo = ređe, crveno = češće od očekivanog).
- Ispod mape obavezan blok „kontrola lažnih alarma":
  - broj parova van 95 % intervala: posmatrano vs. očekivano ≈ 0,05 · 741 ≈ 37;
  - tekst: „ako je posmatrano ≈ 37, 'vrući parovi' su šum, ne signal".
- Klik na ćeliju → tooltip: par brojeva, posmatrano, očekivano, z-skor, spisak
  poslednjih ~5 kola gde su izašli zajedno.
- Toplotnu mapu graditi istom tehnikom kao postojeću pozicionu mapu (reuse komponente).

## 8. Mere za Generator i Bektest (izmene postojećih strana)

### Generator — panel „Različitost seta" ispod tabele rezultata:
- **Unutrašnja različitost:** prosečno i maksimalno preklapanje po svim parovima
  generisanih kombinacija; uz referencu 1,256 („nasumičan set bi imao ~1,26").
- **Pokrivenost:** koliko različitih brojeva od 39 set ukupno sadrži (i vizuelno:
  39 kuglica, pokrivene obojene).
- Vrednosti se osvežavaju posle svakog generisanja i posle primene filtera diverziteta
  (čime dosadašnji filter dobija merljiv, vidljiv efekat).

### Bektest — dve nove kolone po redu:
- **Prosečno preklapanje sa dobitnom** — prosek `preklapanje(kombinacija, dobitna)`
  po svim kombinacijama strategije.
- **Maks. preklapanje** — najbolja kombinacija (ovo je srodno postojećem „najboljem
  rezultatu", ali izraženo u istoj metrici kao ostatak strane).
- U zaglavlju tabele tooltip: „slučajno očekivanje po kombinaciji: 1,26".
- Kolone se popunjavaju u istom hooku koji već računa rezultat bektesta pri unosu kola;
  za postojeće, već ocenjene bektestove — jednokratna migracija koja ih doračuna.

---

## 9. UI strana „Različitost" — raspored

1. Kartice rekorda (Analiza 3).
2. Histogram uzastopnih kola + test (Analiza 1).
3. Histogram svih parova + test (Analiza 2) — sklopivo (collapsible), jer je sadržajno
   sličan prethodnom.
4. Scatter „profil vs. sadržaj" sa izborom profila (Analiza 4).
5. Toplotna mapa ko-okurencije + kontrola lažnih alarma (Analiza 5).

- Svi grafikoni poštuju globalni period; kartice rekorda uvek rade nad celom istorijom
  (uz oznaku „cela istorija").
- Boje i stil kuglica/grafikona preuzeti iz postojećih strana.

---

## 10. Redosled implementacije (faze)

1. **Faza 1:** modul teorije + bitmaske; Analiza 1 (uzastopna kola) sa histogramom
   i hi-kvadrat testom.
2. **Faza 2:** Analize 2 i 3 (svi parovi, rekordi, kartice).
3. **Faza 3:** Analiza 4 (scatter + korelacija) i Analiza 5 (toplotna mapa).
4. **Faza 4:** integracija u Generator i Bektest (§8), migracija starih bektestova,
   dokumentacija u `FUNKCIJE.md` (nova sekcija „9. Različitost").

---

## 11. Kriterijumi prihvatanja

- [ ] Jedinični testovi teorijskog modula: Σ P(k) = 1; E[k] ≈ 1,2564 (tolerancija 1e−4);
      P(7) = 1/C(39,7).
- [ ] `preklapanje` testiran na ručnim primerima (disjunktne, identične, delimične).
- [ ] Analiza svih parova (~1 milion) izvršava se < 2 s; strana se renderuje < 3 s.
- [ ] Sanity test na sintetičkim podacima: 1.000 stvarno nasumičnih kola (seedovano) →
      hi-kvadrat testovi Analiza 1 i 2 daju p > 0,05 (sistem ne prijavljuje lažni signal).
- [ ] Scatter uzorak je seedovan → identičan grafikon pri svakom učitavanju.
- [ ] Broj parova van 95 % intervala u Analizi 5 se prikazuje uporedo sa očekivanih ~37.
- [ ] Generator prikazuje unutrašnju različitost i pokrivenost posle svakog generisanja.
- [ ] Stari bektestovi posle migracije imaju popunjene nove kolone.

---

## 12. Van opsega v1

- Klasterovanje kombinacija (grupisanje istorije po sličnosti) — zanimljivo, ali bez
  jasnog pitanja na koje odgovara.
- Analiza preklapanja sa kombinacijama drugih igrača (nema podataka).
- Trojke i četvorke brojeva u ko-okurenciji — kombinatorna eksplozija
  (C(39,3) = 9.139 trojki) uz još manje očekivane frekvencije; eventualno v2 sa
  strožom kontrolom lažnih alarma.
