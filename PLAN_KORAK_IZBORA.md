# Plan razvoja — Korak izbora: „Predlog modela“ vs. „Tiket“

> Verzija: 1.0
> Datum: 2026-09-08
> Projekat: Loto Analizator — web
> Osnova: `sekvencijalni.py` (implementiran, K ≈ 1), `prelazi.py`, `generator.py`, `sinteza.py`
> Povod: dijagnostika kola 2026071/2026072 — raspon `p_mix` 5,4–5,7%, razlika 7. i 8. kandidata 0,000428, predlozi netipični (5 uzastopnih, 6 parnih)

---

## 1. Šta se rešava

Model radi ispravno: eksperti su naučeni, K ≈ 1, raspodela je gotovo ravna. Problem je **isključivo u koraku izbora sedmorke** i u tome što UI ne saopštava koliko je raspodela ravna.

Tri konkretna nedostatka:

1. **Nema informacije o ravnoći.** Korisnik vidi 7 brojeva i ne zna da je razlika između 7. i 8. kandidata 0,24% od baseline. Predlog izgleda kao odluka, a jeste izbor iz šuma.
2. **Deterministički tie-break ka manjem broju** — sistematska pristrasnost ka niskim brojevima kad su verovatnoće praktično jednake (a jesu, često, na četvrtoj decimali).
3. **Netipični predlozi zbunjuju** (5 uzastopnih, 6 parnih), jer top-7 po marginalama ne poznaje osobine skupa.

**Odluka o Generatoru (revidirana u odnosu na `PLAN_SEKVENCIJALNI_PREDIKTOR.md` §2.4):** predlog modela **ne prolazi** kroz Generator. Filteri Generatora rade na razlikama reda 10⁻² dok je razlika kandidata 4·10⁻⁴ — Generator bi u potpunosti preuzeo izbor, pa bi „predlog modela“ prestao da svedoči o modelu. Umesto toga se prikazuju **dva odvojena izlaza sa jasnim vlasništvom**.

---

## 2. Fiksne odluke

### 2.1. Dva izlaza, jasno razdvojena

| Izlaz | Ko bira | Kako |
|---|---|---|
| **Predlog modela** | `sekvencijalni.py` | goli top-7 po `p_mix`, bez ikakvih filtera |
| **Tiket** | `generator.py` | bazen = top-15 po `p_mix`, pa postojeći filteri i bodovanje Generatora |

Nikad se ne prikazuje jedan bez drugog i nikad bez oznake ko je birao. „Tiket“ je označen kao izbor Generatora, ne modela.

Bazen od 15 je podrazumevano, konfigurabilno u `konfig.py` (`SEKV_BAZEN = 15`). Generator se poziva sa postojećim podrazumevanim filterima; korisnik ih može menjati na tabu Generator, i to se odražava ovde.

### 2.2. Slučajan tie-break sa seed-om

Trenutno: kod jednakih `p_mix` bira se manji broj → sistematska pristrasnost ka niskim brojevima kroz vreme.

Novo: tie-break je pseudoslučajan sa seed-om izvedenim iz broja kola (`seed = kolo`), tako da ostaje **reproducibilan** (retro-bektest i vremeplov moraju ostati deterministični). Praktično: dodaj svakom broju šum reda 10⁻¹² iz `random.Random(kolo)` pre sortiranja, ili sortiraj po `(-p, rnd.random())`.

Pokriveno testom: dva izvršavanja isti rezultat; kroz 1.422 koraka raspodela izabranih brojeva bez trenda ka niskim.

### 2.3. Mera ravnoće raspodele

U `sekvencijalni.py`, uz svaki korak se računa i pamti:

```
p_min, p_max
raspon        = p_max - p_min
raspon_udeo   = raspon / (7/39)
zazor_7_8     = p[7. kandidat] - p[8. kandidat]
zazor_udeo    = zazor_7_8 / (7/39)
```

Ove veličine idu u `sekv_stanje` i u sve endpointe koji vraćaju predlog.

### 2.4. Prag „nema preferencije“

Prag se **ne bira proizvoljno** — izvodi se iz sintetike: pusti model na uniformnoj sintetičkoj bazi iste veličine, izmeri raspodelu `raspon_udeo`, uzmi 95. percentil kao prag `PRAG_RASPONA`. Zapisati u `konfig.py` sa komentarom kako je dobijen i datumom.

- `raspon_udeo ≤ PRAG_RASPONA` → „model nema značajnu preferenciju“
- iznad praga → prikazati kao izuzetak i uputiti na Sintezu (ne kao dobru vest)

---

## 3. Izmene po fajlu

```
core/sekvencijalni.py
  + izracunaj_ravnocu(p_mix) -> dict          (§2.3)
  + izaberi_top7(p_mix, kolo)                 (§2.2, seed = kolo)
  ~ predlog() vraća {brojevi, ravnoca, p_mix}
core/generator.py
  + generisi_iz_bazena(bazen, filteri)        (tanak omotač nad postojećim)
core/konfig.py
  + SEKV_BAZEN = 15
  + PRAG_RASPONA = <iz sintetike>             (§2.4)
core/baza.py
  ~ sekv_stanje: + kolone p_min, p_max, raspon_udeo, zazor_udeo
api/app.py
  ~ /api/sekv/stanje: + ravnoca, + tiket
  ~ /api/sekv/korak:  + ravnoca, + tiket
static/
  ~ panel „Sekvencijalni prediktor“ (§4)
tests/test_sekv.py
  + testovi iz §5
```

---

## 4. UI

### 4.1. Panel

```
┌──────────────────────────────────────────────────────────────┐
│ SEKVENCIJALNI PREDIKTOR — kolo 2026-072                      │
│                                                              │
│ Raspon verovatnoća: 0,1754 – 0,1857  (5,7% od 7/39)          │
│ Razlika 7. i 8. kandidata: 0,00043  (0,24%)                  │
│ → Model nema značajnu preferenciju. Predlog je praktično     │
│   nasumičan izbor iz gotovo ravne raspodele.                 │
│                                                              │
│ Koeficijent nepredvidivosti: 1,003  (pojas 0,988 – 1,012)    │
├──────────────────────────────────────────────────────────────┤
│ PREDLOG MODELA (top 7 po verovatnoći, bez filtera)           │
│   5  10  14  16  22  28  30                                  │
│   Bira model. Može izgledati netipično (uzastopni, parnost) — │
│   marginalne verovatnoće ne poznaju osobine cele kombinacije. │
├──────────────────────────────────────────────────────────────┤
│ TIKET (Generator, iz bazena od 15 najverovatnijih)           │
│   3  10  16  23  28  31  37     par/nepar 3/4, zbir 148      │
│   Bira Generator po filterima tipičnosti — ne model.          │
│   Nema veću šansu od predloga iznad. [Podesi filtere →]      │
└──────────────────────────────────────────────────────────────┘
```

### 4.2. Obavezna pravila prikaza

- Rečenica o ravnoći stoji **iznad** oba predloga, uvek.
- Uz „Tiket“ obavezno: „Nema veću šansu od predloga iznad.“
- Nijedan od dva izlaza se ne prikazuje sam.
- Nikakvo isticanje bojom osim slučaja `raspon_udeo > PRAG_RASPONA`.

### 4.3. Collapsible „Zašto predlog izgleda neobično“

> Model računa verovatnoću za svaki broj posebno. Parnost, raspon i uzastopni brojevi su osobine cele kombinacije — top-7 po pojedinačnim verovatnoćama ih ne vidi. Zato predlog povremeno ima 6 parnih ili nekoliko uzastopnih. Takve kombinacije su ređe kao *klasa*, ali svaka pojedinačna ima istu šansu kao bilo koja druga (1 : 15.380.937).

Brojevi u tekstu se izvode, ne hardkoduju.

---

## 5. Testovi (dopuna `test_sekv.py`)

| Test | Proverava |
|---|---|
| `test_raspon_p_mix_na_sintetici` | **ključni**: na uniformnoj sintetici raspodela `raspon_udeo` obuhvata izmerenih 5,7% → potvrda da je raspon šum, ne signal. Izlaz ovog testa daje `PRAG_RASPONA` |
| `test_tiebreak_reproducibilan` | dva izvršavanja istog koraka → identičan predlog |
| `test_tiebreak_bez_pristrasnosti` | kroz celu istoriju, prosek izabranih brojeva ≈ 20 (bez trenda ka niskim); poređenje sa starim tie-breakom u istom testu |
| `test_predlog_bez_filtera` | predlog modela ne zavisi od podešavanja Generatora |
| `test_tiket_iz_bazena` | tiket je podskup bazena od `SEKV_BAZEN` i zadovoljava aktivne filtere |
| `test_K_nepromenjen` | K pre i posle svih izmena identičan (računa se iz raspodele, ne iz izbora) |
| `test_ravnoca_u_stanju` | `raspon_udeo` i `zazor_udeo` se upisuju u `sekv_stanje` za svako kolo |

`test_K_nepromenjen` je regresioni: sačuvati postojeće K_t kao referencu pre izmena.

---

## 6. Faze

**Faza 1 — Merenje i iskrenost (bez promene izbora)**
`izracunaj_ravnocu`, upis u bazu, rečenica u UI, `test_raspon_p_mix_na_sintetici` → dobija se `PRAG_RASPONA`.
*Gotovo kad:* korisnik vidi koliko je raspodela ravna, pre nego što išta drugo bude promenjeno.

**Faza 2 — Tie-break**
Seed iz broja kola, oba testa tie-breaka, provera da K nije pomeren.
*Gotovo kad:* nema trenda ka niskim brojevima, retro-bektest i dalje determinističan.

**Faza 3 — Tiket iz bazena**
`generisi_iz_bazena`, drugi izlaz u UI, oznake vlasništva.
*Gotovo kad:* oba izlaza prikazana, jasno razdvojena, sa obaveznim rečenicama.

**Faza 4 — Objašnjenje i istorija ravnoće**
Collapsible tekst; krivulja `raspon_udeo` kroz vreme u detalj-panelu (uz krivulju K i težina).
*Gotovo kad:* vidi se da je raspodela ravna kroz celu istoriju, ne samo sada.

---

## 7. Šta NE raditi

- Ne provlačiti predlog modela kroz filtere — time model prestaje da svedoči o sebi.
- Ne prikazivati tiket bez oznake da ga bira Generator i bez rečenice o jednakoj šansi.
- Ne birati `PRAG_RASPONA` „po osećaju“ — samo iz sintetike, sa zapisanim datumom i postupkom.
- Ne menjati način računanja K.
- Ne dodavati eksperte za parnost/zbir/raspon kao „popravku“ — osobine skupa ne pripadaju marginalnim ekspertima; njihovo mesto su filteri Generatora.
- Ne tumačiti `raspon_udeo > PRAG_RASPONA` kao nalaz pre provere curenja i ponovljenog merenja na sintetici.

---

## 8. Commit redosled

```
1. feat(sekv): mera ravnoce raspodele (raspon, zazor) + upis u stanje
2. test(sekv): raspon p_mix na sintetici, izvodjenje PRAG_RASPONA
3. fix(sekv): reproducibilan slucajan tie-break umesto manjeg broja
4. feat(sekv): tiket iz bazena preko generatora, odvojen od predloga modela
5. feat(ui): recenica o ravnoci, dva izlaza sa oznakom vlasnistva, objasnjenje
6. docs: FUNKCIJE — revizija odeljka 2.4 plana sekvencijalnog prediktora
```

---

## 9. Kriterijum uspeha

Korisnik koji dobije `5 10 14 16 22 28 30` sa šest parnih odmah iznad pročita da je raspon verovatnoća 5,7% i da model nema preferenciju, ispod vidi „uredan“ tiket sa oznakom da ga je birao Generator i da nema veću šansu, a u objašnjenju sazna zašto marginalne verovatnoće prave netipične kombinacije. Nijedan od tih ekrana ne sugeriše da je bilo koji predlog bolji od bilo koje druge kombinacije — a K ostaje nepromenjen, jer se izbor sedmorke i merenje znanja ne mešaju.
