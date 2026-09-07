# Plan razvoja — „Sinteza“ (zajednički sud za sve metode) + „Rang kombinacije“

> Verzija: 1.0
> Datum: 2026-09-07
> Projekat: Loto Analizator — web
> Osnova: `README-web.md`, `prognoza.py` (retro-bektest, evaluacija, Bonferroni), `prediktori.py` / `prediktori_komb.py` (registri), `razlicitost_teorija.py`, `mapa.py` (`rang`)

---

## 1. Svrha

Aplikacija ima mnogo metoda i svaki je zasebno testiran. Nedostaje **jedno mesto gde se svi metodi mere istim alatom, protiv iste kontrole, sa istim pravilima**, i gde korisnik dobija jedan odgovor umesto deset tabela.

Strana **„Sinteza“** je taj zajednički sud. Ona:
- ne uvodi nov način predviđanja,
- ne uvodi nov način ocenjivanja,
- registruje **ensemble kao još jedan prediktor** i pušta ga kroz isti sud kao sve ostale,
- dodaje **jedan** nov test na nivou cele kombinacije (rang), jer ga aplikacija još nema.

Princip ostaje: *teorija izvodi očekivanje → podaci se porede → jedna rečenica zaključka*.

---

## 2. Fiksne odluke

### 2.1. Jedan objekat „Eksperiment“

Svaki red u Sintezi je instanca iste strukture (dataclass u `core/sinteza.py`):

```
Eksperiment
  metod            # ime iz registra (hot, cold, bayes, …, k_hot7, …, ensemble)
  tip              # "jedan_broj" | "kombinacija" | "test"
  granica_od, granica_do   # opseg retro-bektesta
  n                # broj evaluiranih kola
  rezultat         # pogodaka / prosečno preklapanje / statistika testa
  ocekivano        # iz teorije (7/39, μ=1,256, …) — nikad hardkodovano
  z, p             # iz postojeće evaluacije u prognoza.py
  p_korig          # Bonferroni preko SVIH redova u tabeli
  zakljucak        # jedna rečenica, generisana (vidi §5.3)
```

**Ne praviti drugu evaluaciju.** `rezultat`, `z`, `p` dolaze iz `prognoza.py` (isti kod koji koristi tab Prognoza i vremeplov). `sinteza.py` samo sakuplja i pakuje.

### 2.2. Ensemble = jedna linija u registru

`prediktori_komb.py` dobija `k_ensemble` (i po potrebi `prediktori.py` dobija `ensemble`), registrovan kao svaki drugi. Pravila:

- **Walk-forward**: težine se uče samo na kolima ≤ trenutna granica; pri retro-bektestu se ponovo uče za svaku tačku (ili na fiksnom prvom delu istorije — izbor dokumentovati). Nikad na kolima koja se evaluiraju.
- Težinski model: prosta linearna kombinacija ocena postojećih metoda (frekvencija, Bajes, hibrid, ritam, pozicija, parovi). Bez ML biblioteka. Maks. ~6 težina.
- Ako je walk-forward prespor za retro-bektest (~9.450 redova), učenje težina se radi jednom na prvih 30% istorije, testira na ostalih 70%. To takođe ne curi.

Ensemble **nema nikakav poseban tretman** u UI — isti red, iste kolone, ista korekcija.

### 2.3. Test „Rang kombinacije“

Novi modul-deo u `razlicitost.py` (ili `razlicitost_teorija.py` za teoriju), koristi `mapa.rang`:

| Test | Teorijsko očekivanje | Statistika |
|---|---|---|
| histogram rangova (50 korpi) | uniformno, N/50 po korpi | hi-kvadrat, df=49 |
| rastojanje uzastopnih |rang_i − rang_{i+1}| | trougaona, prosek Nmax/3 | KS test ili hi-kvadrat na korpama |
| autokorelacija ranga, pomak 1–5 | 0 | z = r·√n |
| raspodela najmanjeg broja | P(min=k) = C(39−k,6)/C(39,7) | hi-kvadrat |

Poslednji red je eksplicitno objašnjenje zašto se rangovi gomilaju u malim vrednostima (65% kola ima min ≤ 5) — da korisnik ne pomisli da mašina „voli male brojeve“.

Ova četiri testa ulaze u Sintezu kao redovi tipa `test`, sa istom Bonferroni korekcijom.

### 2.4. Kontrola je red, ne fusnota

`random` / `k_random` su obavezni redovi u tabeli, vizuelno jednaki ostalima. Bonferroni korekcija se računa preko svih redova **uključujući** kontrolu i testove.

---

## 3. Arhitektura

```
webapp/
  core/
    sinteza.py           # NOVO: Eksperiment, sakupljanje svih redova, korekcija, zaključci
    prediktori.py        # + ensemble
    prediktori_komb.py   # + k_ensemble
    razlicitost.py       # + testovi ranga (koristi mapa.rang)
  api/app.py             # + /api/sinteza/*
  static/                # + tab „Sinteza“
  tests/
    test_sinteza.py      # NOVO
```

Tab „Sinteza“ ide **odmah posle Dashboarda** ili kao poslednji analitički tab — dogovoriti; preporuka: poslednji, kao „zaključak“.

---

## 4. API

| Endpoint | Vraća |
|---|---|
| `GET /api/sinteza?granica=&prozor=` | sve redove Eksperimenta + globalni zaključak + `ocekivano_laznih = N·0,05` |
| `GET /api/sinteza/metod/{metod}` | detalj jednog reda: krivulja pogodaka kroz vreme, raspodela, tekst testa |
| `GET /api/sinteza/rang` | četiri testa ranga sa histogramima |
| `POST /api/sinteza/osvezi` | ponovo pokreće retro-bektest (eksplicitno, jer traje ~7 s + ensemble) |

Rezultat retro-bektesta se **kešira** u tabeli `prognoze` koja već postoji (ili u novoj `sinteza_kes`), invalidira se pri unosu kola istim hook-om kao evaluacije tiketa.

---

## 5. UI

### 5.1. Raspored

```
┌────────────────────────────────────────────────────────────────┐
│ SINTEZA — svi metodi, isti sud                                  │
│                                                                │
│ Od 19 metoda i testova, 0 je značajno na 5% posle korekcije.   │
│ Očekivano lažno pozitivnih bez korekcije: 0,95.                │
│                                     [Osveži retro-bektest]     │
├────────────────────────────────────────────────────────────────┤
│ JEDAN BROJ (n = 1.350 kola)                                    │
│ Metod     Pogodaka   Očekivano   z      p      p_kor   Zaključak│
│ random    243        242,3       0,05   0,96   1,00    ≈ slučajnost │
│ bayes     238        242,3      −0,29   0,77   1,00    ≈ slučajnost │
│ ensemble  247        242,3       0,32   0,75   1,00    ≈ slučajnost │
│ …                                                              │
├────────────────────────────────────────────────────────────────┤
│ KOMBINACIJA (n = 1.350)                                        │
│ Metod       Prosek prekl.  μ      z     p     p_kor   Zaključak │
│ k_random    1,26          1,256  …                              │
│ k_ensemble  1,29          1,256  …                              │
├────────────────────────────────────────────────────────────────┤
│ TESTOVI SLUČAJNOSTI                                            │
│ Test                 Statistika   p     p_kor   Zaključak       │
│ frekvencija brojeva  χ²=…                                       │
│ rang – uniformnost   χ²=…                                       │
│ rang – rastojanja    KS=…                                       │
│ rang – autokorel.    z=…                                        │
│ najmanji broj        χ²=…                                       │
│ različitost parova   …                                          │
├────────────────────────────────────────────────────────────────┤
│ ▸ Kako čitati ovu tabelu                                       │
│ ▸ Zašto ensemble nije bolji od delova                          │
│ ▸ Metodologija (walk-forward, Bonferroni, kontrola)            │
└────────────────────────────────────────────────────────────────┘
```

### 5.2. Redovi su klikabilni

Klik → panel sa krivuljom pogodaka kroz vreme (kumulativno, sa pojasom ±2σ oko očekivanja) i dugmetom „otvori u Prognozi“ / „otvori u Različitosti“. Ne duplirati grafikone koji tamo već postoje — samo linkovati.

### 5.3. Generisani zaključak

Jedna funkcija u `sinteza.py`, tri ishoda, bez nijansi:

- `p_kor ≥ 0,05` → „≈ slučajnost“
- `p_kor < 0,05` i metod ≠ random → „**odstupa** — proveriti“ (crveno, i to je *jedini* slučaj kad se nešto ističe)
- kontrola sa `p_kor < 0,05` → „kontrola odstupa — lažno pozitivan, očekivano ~N·0,05“

Globalna rečenica na vrhu se izvodi iz brojanja ovih ishoda.

### 5.4. Sekcija „Zašto ensemble nije bolji“

Kratak tekst, collapsible:
> Kombinovanje metoda pomaže kad svaki nosi malo signala. Ovde svaki metod pojedinačno daje ≈ 17,95%, koliko i slučajan izbor. Ensemble je prošao isti test kao ostali — vidi red iznad.

Ako ensemble ikad prođe ispod 0,05 posle korekcije, ovaj tekst se automatski **ne prikazuje**, a prikazuje se upozorenje „proveriti curenje, pa ponoviti walk-forward“.

---

## 6. Faze

### Faza 1 — Sakupljač
1. `core/sinteza.py` sa `Eksperiment` i `sakupi(granica)` koja poziva postojeći retro-bektest i pakuje sve postojeće metode + kontrolu.
2. Bonferroni preko svih redova.
3. `/api/sinteza`.
4. Tab sa tabelama „Jedan broj“ i „Kombinacija“ i globalnom rečenicom.

**Gotovo kad:** tabela prikazuje sve postojeće metode; brojevi su identični onima na tabu Prognoza (test).

### Faza 2 — Ensemble
1. `ensemble` i `k_ensemble` u registrima, walk-forward.
2. Pojavljuju se u Sintezi bez ikakve izmene UI-ja (dokaz da je registar dovoljan).
3. Sekcija „Zašto ensemble nije bolji“.

**Gotovo kad:** ensemble ima red, p_kor ≈ 1, retro-bektest i dalje < 15 s.

### Faza 3 — Rang kombinacije
1. Četiri testa u `razlicitost.py`.
2. `/api/sinteza/rang` + sekcija „Testovi slučajnosti“ (uključiti i postojeće: hi-kvadrat frekvencije, parovi).
3. Histogrami u detalj-panelu.

**Gotovo kad:** svi testovi u tabeli, teorijska očekivanja izvedena, ne hardkodovana.

### Faza 4 — Detalji i keš
1. Klik na red → krivulja kroz vreme sa ±2σ.
2. Keš rezultata + invalidacija na unos kola.
3. Linkovi ka Prognozi / Različitosti / Istoriji.

**Gotovo kad:** otvaranje Sinteze posle prvog izračunavanja traje < 300 ms.

---

## 7. Testovi — `test_sinteza.py`

| Test | Proverava |
|---|---|
| `test_isti_brojevi_kao_prognoza` | red `bayes` u Sintezi == rezultat retro-bektesta iz `prognoza.py` za isti opseg |
| `test_bonferroni_preko_svih` | p_kor = min(1, p·N) gde je N ukupan broj redova uključujući kontrolu i testove |
| `test_ensemble_walk_forward` | težine za granicu g zavise samo od kola ≤ g (isti anti-leakage obrazac kao `test_istorija`) |
| `test_ensemble_na_slucajnim_podacima` | na sintetičkoj uniformnoj bazi ensemble daje p_kor ≈ 1 (ne sme „naći“ signal u šumu) |
| `test_rang_uniformnost_sintetika` | 10.000 slučajnih kombinacija: hi-kvadrat rangova ne odbacuje uniformnost |
| `test_rang_rastojanje_teorija` | prosek |Δrang| na sintetici ≈ Nmax/3 unutar 2σ |
| `test_min_broj_raspodela` | P(min=k) sabira na 1 i ≈ empirija na sintetici |
| `test_zakljucak_tri_ishoda` | funkcija zaključka vraća tačno jedan od tri teksta |

Postojeći testovi ostaju nepromenjeni i prolaze.

---

## 8. Šta NE raditi

- Ne praviti drugu evaluaciju, drugi retro-bektest, drugi Bonferroni.
- Ne davati ensembleu više od ~6 težina; ne uvoditi ML/VAE/Gemini.
- Ne učiti težine na kolima koja se evaluiraju.
- Ne skrivati kontrolu; ne prikazivati „najbolji metod“ bez korekcije.
- Ne dodavati nove testove bez teorijskog očekivanja izvedenog u kodu.
- Ne bojiti tabelu osim jednog crvenog slučaja iz §5.3.

---

## 9. Commit redosled

```
1. feat(sinteza): Eksperiment, sakupljanje postojecih metoda i Bonferroni
2. feat(sinteza): tab sa tabelama i globalnim zakljuckom
3. feat(prediktori): ensemble i k_ensemble (walk-forward)
4. feat(razlicitost): testovi ranga kombinacije i najmanjeg broja
5. feat(sinteza): detalj reda, kes, linkovi
6. test: sinteza, ensemble anti-leakage, rang na sintetici
7. docs: README i FUNKCIJE sekcija „Sinteza“
```

---

## 10. Kriterijum uspeha

Korisnik otvori Sintezu i u prvoj rečenici pročita koliko metoda (od svih, uključujući ensemble) odstupa od slučajnosti posle korekcije. Zatim vidi da kontrola stoji u istom redu sa Bajesom i ensembleom, i da ih ništa ne razdvaja. Ako ikad nešto odstupi, to je jedino crveno na strani — i vodi na proveru curenja, ne na tiket.
