# Plan implementacije: „Prognoza kombinacije" (predviđanje 7 brojeva)

Dokument za programera. Proširenje postojeće strane **Prognoza**: pored predviđanja
jednog broja, svaki metod predlaže i **jednu kompletnu kombinaciju od 7 brojeva** po
kolu. Meri se **preklapanje sa dobitnom kombinacijom** (0–7 zajedničkih brojeva),
prati kroz vreme i poredi sa teorijskom osnovnom linijom.

**Zavisnosti:** ovaj plan pretpostavlja da su implementirani:
- `PLAN_PROGNOZA.md` — tabela prognoza, registar prediktora, hook evaluacije,
  retro-bektest infrastruktura, Bonferroni prag, grafikon sa pojasom pouzdanosti.
- `PLAN_RAZLICITOST.md` — modul teorije (`hipergeom_pmf`, bitmaske, `preklapanje()`).

> **Svrha (preneti u UI):** ako su izvlačenja slučajna, svaka kombinacija ima
> identično očekivano preklapanje **1,256** sa sledećim izvlačenjem, bez obzira kako
> je izabrana. Funkcija testira da li ijedan metod dugoročno drži prosek značajno
> iznad te linije.

---

## 1. Teorijska osnova (metrika i baseline)

- **Metrika po kolu:** `k = preklapanje(predložena_kombinacija, dobitna)` ∈ {0..7}.
- **Raspodela pod nultom hipotezom:** hipergeometrijska P(k) iz modula Različitost.
- **Očekivanje:** μ = 7·7/39 ≈ **1,2564**.
- **Standardna devijacija:** izračunati iz P(k):
  `σ = sqrt(Σ k²·P(k) − μ²)` ≈ **0,9317** (ne hardkodovati; izvesti u modulu teorije
  i pokriti testom). Napomena: ovo je hipergeometrijska σ sa korekcijom za konačnu
  populaciju — `Var = n·p·(1−p)·(N−n)/(N−1)`, p = 7/39.
- **Pojas pouzdanosti za kumulativni prosek posle n kola:**
  `μ ± 1,96 · σ / sqrt(n)`.
- **Test značajnosti po metodu:** z-test proseka:
  `z = (prosek − μ) / (σ / sqrt(n))`, dvostrana p-vrednost. Za male n (< 30) prikazati
  „premalo podataka" umesto p-vrednosti.
- **Prag:** Bonferroni preko ukupnog broja metoda na strani Prognoza
  (jednobrojni + kombinacijski zajedno — svi su deo istog eksperimenta).

Dodati u modul teorije: `sigma_preklapanja()`, `z_test_proseka(prosek, n)`.

---

## 2. Prediktori kombinacija (v1 skup)

Interfejs analogan jednobrojnim, ali vraća sortiranu 7-torku:

```python
def prediktor_komb(istorija: list[Kolo], period: int | None) -> tuple[int, ...]:
    """Vraća 7 jedinstvenih brojeva 1–39, sortirano rastuće (kanonski oblik)."""
```

| ID | Naziv | Logika |
|---|---|---|
| `k_hot7` | Top-7 vrućih | 7 najfrekventnijih u periodu. Tie-break: skoriji poslednji nastup, pa manji broj. |
| `k_cold7` | Top-7 hladnih | 7 najređih (prvo neizvučeni u periodu). Tie-break: najduže neizvučen, pa manji broj. |
| `k_bayes7` | Top-7 Bajes | Vrh Bajesove rang-liste — **reuse** refaktorisane funkcije iz plana Prognoza. |
| `k_hybrid7` | Top-7 hibrid | Vrh hibridne rang-liste — reuse. |
| `k_rhythm7` | Top-7 po ritmu | 7 najvećih D/R (metrika iz plana Prognoza, §2). Ako manje od 7 brojeva ima definisan ritam, dopuniti po dužini kašnjenja D. |
| `k_cooc` | Ko-okurencijski pohlepni | Videti §3 — jedini algoritamski novi. |
| `k_random` | **Kontrolna grupa** | 7 nasumičnih jedinstvenih brojeva, seedovan RNG po kolu (`seed = kolo * 2` — različit od seeda jednobrojnog random prediktora da ne dele slučajnost). |

Pravila:
- Sva tie-break pravila deterministička; kanonski oblik (sortirano) obavezan radi
  poređenja i UNIQUE ograničenja.
- Registar proširiti: `PREDIKTORI_KOMB = {id: (naziv, fn)}` — odvojen od jednobrojnog
  registra, jer su tipovi rezultata različiti.
- Zabrana curenja budućnosti identična planu Prognoza (§6 tamo); unit test proširiti
  da pokrije i kombinacijske prediktore.

---

## 3. Ko-okurencijski pohlepni algoritam (`k_cooc`)

Jedini prediktor koji koristi strukturu parova (Analiza 5 strane Različitost):

```
C = matrica ko-okurencije 39×39 nad periodom (reuse iz strane Različitost,
    ali računata SAMO nad istorijom pre ciljnog kola)
1. start: par (a, b) sa najvećim C[a][b]; tie-break: manji a, pa manji b
2. dok |S| < 7:
       kandidat x ∉ S sa najvećim Σ_{s∈S} C[x][s]
       tie-break: veća ukupna frekvencija x, pa manji x
3. vrati sorted(S)
```

- Složenost zanemarljiva (39 kandidata × ≤ 7 koraka).
- U retro-bektestu matricu C održavati inkrementalno (dodavanje kola = +1 na 21
  ćeliju para; klizni prozor za period = i −1 na izlazećem kolu), po istom principu
  kao inkrementalni brojači iz plana Prognoza §5.

---

## 4. Model podataka

Proširenje tabele `prognoze` (bez nove tabele — isti životni ciklus):

```sql
ALTER TABLE prognoze ADD COLUMN vrsta TEXT NOT NULL DEFAULT 'broj';
    -- 'broj' = jednobrojna prognoza, 'komb' = kombinacija
ALTER TABLE prognoze ADD COLUMN kombinacija TEXT;
    -- CSV 7 brojeva sortirano ("3,8,12,19,22,31,37"); NULL za vrstu 'broj'
ALTER TABLE prognoze ADD COLUMN preklapanje INTEGER;
    -- 0..7 posle evaluacije; NULL pre. Za vrstu 'broj' ostaje NULL.
```

- Postojeća kolona `broj` je NULL za `vrsta='komb'`; kolona `pogodak` se za
  kombinacije ne koristi (ostaje NULL) — merodavno je `preklapanje`.
- UNIQUE proširiti na `(kolo, metod, izvor, vrsta)`.
- Migracija: postojeći redovi dobijaju `vrsta='broj'` (DEFAULT to već rešava).
- Pravilo nepromenljivosti ocenjenih prognoza važi identično (preklapanje ≠ NULL →
  red zaključan).

---

## 5. Tok uživo i evaluacija

Identičan planu Prognoza §4, sa dopunama:

- Pri generisanju predloga za sledeće kolo upisuju se i svi kombinacijski prediktori
  (`vrsta='komb'`, `kombinacija` popunjena, `preklapanje=NULL`).
- Hook pri unosu kola (strana „Podaci") dobija drugi korak:

  ```
  za svaku prognozu WHERE kolo = uneto AND vrsta='komb' AND preklapanje IS NULL:
      preklapanje = preklapanje_bitmask(maska(kombinacija), maska(dobitna))
  ```

- **Obavezno reuse `preklapanje()` iz bitmask modula** — u aplikaciji već postoje
  dva mesta koja porede skupove brojeva (rezultat tiketa, rezultat bektesta); ne
  uvoditi treću nezavisnu implementaciju. Preporuka: u okviru ove faze prevesti i
  proveru tiketa/bektesta na isti modul (mala, izolovana izmena, isti rezultati).

---

## 6. Retro-bektest

Proširenje postojeće walk-forward petlje (plan Prognoza §5) — **ista petlja**, ne
nova: u svakom koraku N, pored jednobrojnih, izračunati i upisati kombinacijske
prognoze. Inkrementalna stanja koja petlja održava:

- brojači frekvencija (klizni prozor) — već postoji,
- Bajesovo stanje — već postoji,
- ritam (poslednje pojavljivanje + prosečni razmak po broju) — već postoji,
- **novo:** matrica ko-okurencije (klizni prozor, §3).

Ciljno vreme za ceo bektest (jednobrojni + kombinacijski, 1.422 kola): **< 15 s**.
Determinizam obavezan (uključujući `k_random` seed).

---

## 7. UI — proširenje strane „Prognoza"

Strana dobija dva taba (segmentirano dugme gore): **„Jedan broj"** (postojeći sadržaj)
i **„Kombinacija"** (novo). Zajednički su: izbor izvora (uživo/retro), dugme
retro-bektesta (pokreće oba), oznaka ciljnog kola.

### Tab „Kombinacija"

**7.1 Predlozi za sledeće kolo** — kartica po metodu: naziv, 7 kuglica (obojene po
postojećoj šemi vruć/hladan/neutralan), tooltip sa logikom. Dugme „+ tiket" na svakoj
kartici — dodaje kombinaciju u „Moje tikete" (reuse postojeće funkcije), čime se
predlog može stvarno odigrati i pratiti i kroz tikete.

**7.2 Grafikon kumulativnog proseka preklapanja**
- X: redni broj ocenjenog kola; Y: kumulativni prosek preklapanja po metodu.
- Horizontalna linija na **1,256** + pojas `μ ± 1,96·σ/√n` (sužava se sa n).
- Legenda sa uključivanjem linija; prekidač uživo/retro kao kod jednobrojnog.

**7.3 Histogram raspodele preklapanja po metodu**
- Dropdown izbora metode → histogram udela k=0..7 naspram teorijske krive P(k).
- Svrha: metod može imati prosek ≈ 1,256 a drugačiji oblik raspodele — i to je nalaz.
- Hi-kvadrat test (reuse iz Različitosti, spajanje ćelija k ≥ 4).

**7.4 Tabela rezultata po metodu**

| Kolona | Sadržaj |
|---|---|
| Metod | naziv |
| Ocenjenih kola | n |
| Prosek preklapanja | na 3 decimale |
| Očekivano | 1,256 |
| Maks. postignuto | najbolje k ikad (i u kom kolu) |
| p-vrednost | z-test proseka (§1); „—" za n < 30 |
| Zaključak | isti stil i formulacija opreza kao jednobrojna tabela |

**7.5 Istorija** — poslednjih ~50 kombinacijskih prognoza: kolo, metod, 7 kuglica
(pogođeni brojevi vizuelno istaknuti), preklapanje, izvor.

---

## 8. Redosled implementacije (faze)

1. **Faza 1:** migracija tabele, registar kombinacijskih prediktora,
   `k_hot7`/`k_cold7`/`k_random`, hook evaluacije, tab sa predlozima i tabelom.
2. **Faza 2:** retro-bektest proširenje + unit test curenja za kombinacijske
   prediktore; grafikon kumulativnog proseka sa baseline-om.
3. **Faza 3:** `k_bayes7`/`k_hybrid7`/`k_rhythm7`, `k_cooc` sa inkrementalnom
   matricom, histogram raspodele, z-test i zaključci.
4. **Faza 4:** „+ tiket" integracija, prelazak tiketa/bektesta na zajednički
   `preklapanje()` modul, dokumentacija u `FUNKCIJE.md` (dopuna sekcije Prognoza).

---

## 9. Kriterijumi prihvatanja

- [ ] Testovi teorije: σ ≈ 0,9317 iz P(k) (tolerancija 1e−3); pojas pouzdanosti
      monotono se sužava sa n.
- [ ] Svi kombinacijski prediktori vraćaju tačno 7 jedinstvenih brojeva 1–39,
      sortirano, deterministički (osim seedovanog `k_random`).
- [ ] Unit test curenja budućnosti prolazi za sve kombinacijske prediktore,
      uključujući `k_cooc` (matrica ne sme da sadrži ciljno kolo).
- [ ] Retro-bektest (svi prediktori, obe vrste) < 15 s, identičan pri svakom
      pokretanju.
- [ ] `k_random` završava unutar pojasa pouzdanosti oko 1,256 (sanity check).
- [ ] Evaluacija kombinacija koristi isti `preklapanje()` kao strana Različitost;
      provera tiketa daje identične rezultate kao pre prelaska na zajednički modul.
- [ ] „+ tiket" sa kartice predloga kreira tiket identičan ručnom unosu.
- [ ] Jednobrojne prognoze rade nepromenjeno posle migracije (regresioni test).

---

## 10. Van opsega v1

- Više kombinacija po metodu (set od N kombinacija) — to je domen Generatora;
  ovde strogo jedna po metodu radi čistoće eksperimenta.
- Optimizacija „pokrivenosti" preko više metoda zajedno.
- Prediktori sa težinskim kombinovanjem metoda („ensemble") — tek ako v1 pokaže
  bilo šta van pojasa pouzdanosti, što je malo verovatno.
