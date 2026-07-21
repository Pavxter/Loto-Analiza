# Plan implementacije: strana „Prognoza" (predviđanje jednog broja)

Dokument za programera. Opisuje novu funkciju aplikacije **Loto Analizator**: predviđanje
**jednog broja** (od 39) za sledeće kolo, više paralelnih metoda-prediktora, automatsko
ocenjivanje pri unosu kola, praćenje uspešnosti kroz vreme i retroaktivni bektest nad
celom istorijom.

> **Svrha funkcije (obavezno preneti u UI):** ovo je statistički eksperiment, ne
> proricanje. Cilj je da se izmeri da li bilo koji metod pogađa **statistički značajno
> više** od nasumične osnovne linije. Osnovna linija: 7/39 ≈ **17,95 %** (verovatnoća
> da nasumično izabran broj bude među 7 izvučenih).

---

## 1. Pregled arhitekture

Funkcija se sastoji iz četiri celine:

1. **Modul prediktora** (`prediktori.py`) — čiste funkcije koje na osnovu istorije
   do kola N-1 vraćaju jedan broj (predlog za kolo N).
2. **Baza** — dve nove tabele: `prognoze` (predlozi po kolu i metodu) i evidencija
   rezultata (pogodak da/ne), popunjava se automatski.
3. **Automatska evaluacija** — kuka (hook) u postojećoj logici čuvanja kola na strani
   „Podaci": pri unosu novog kola ocene se sve otvorene prognoze (isto mesto gde se
   već proveravaju tiketi i bektestovi).
4. **UI strana „Prognoza"** — trenutni predlozi, tabela istorije, kumulativni grafikon
   uspešnosti sa baseline linijom i intervalom pouzdanosti, dugme za retroaktivni bektest.

---

## 2. Prediktori (v1 skup)

Svi prediktori imaju isti interfejs:

```python
def prediktor(istorija: list[Kolo], period: int | None) -> int:
    """istorija = sva kola STROGO PRE ciljnog kola, hronološki.
    period = broj poslednjih kola koja se gledaju (None = sva).
    Vraća jedan broj 1–39."""
```

| ID | Naziv | Logika |
|---|---|---|
| `hot` | Najvrući | Broj sa najviše pojavljivanja u periodu. Tie-break: skoriji poslednji nastup. |
| `cold` | Najhladniji („due") | Broj sa najmanje pojavljivanja; prioritet imaju neizvučeni u periodu. Tie-break: najduže neizvučen. |
| `bayes` | Bajesovski | Broj sa najvišim skorom postojećeg Bajesovog modela (learning rate 0.005) — **reuse postojeće funkcije sa strane Rangiranje**, ne duplirati kod. |
| `hybrid` | Hibridni | Najviši hibridni skor (80 % Bajes + 20 % povezanost) — takođe reuse. |
| `rhythm` | Ritam koji kasni | Za svaki broj: prosečan razmak ponavljanja R i broj kola od poslednjeg pojavljivanja D. Predlaže broj sa najvećim odnosom D/R (najviše „kasni" u odnosu na svoj ritam). Brojevi koji se nisu pojavili ni jednom u periodu se preskaču. |
| `fresh` | Najsvežiji | Broj izvučen najskorije (poslednje kolo, prva pozicija izvlačenja kao tie-break). Testira hipotezu „vrući ostaju vrući" u najsirovijem obliku. |
| `random` | **Kontrolna grupa** | Nasumičan broj 1–39, uniformno. **Obavezan** — bez njega grafikon nema referencu iz prakse. Koristi seedovan RNG po kolu (`seed = kolo`) da retro-bektest bude reproducibilan. |

Napomene za implementaciju:

- Tie-break pravila moraju biti **deterministička** (osim `random`) da bi retro-bektest
  uvek davao isti rezultat.
- Prediktori ne smeju da vide ciljno kolo ni bilo šta posle njega (nikakvo curenje
  budućnosti — ovo je najčešća greška u ovakvim bektestovima; videti §6).
- Period za prediktore = globalni period aplikacije (gore desno), a u retro-bektestu
  fiksiran parametar (podrazumevano 100; videti §5).
- Skup prediktora učiniti **proširivim**: registar `PREDIKTORI = {id: (naziv, fn)}` da
  se novi metod dodaje jednom linijom.

---

## 3. Model podataka

Nova tabela `prognoze`:

```sql
CREATE TABLE prognoze (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kolo        INTEGER NOT NULL,          -- ciljno kolo (format godina*1000+br)
    metod       TEXT    NOT NULL,          -- id prediktora ('hot', 'bayes', ...)
    broj        INTEGER NOT NULL,          -- predlog 1..39
    period      INTEGER,                   -- period nad kojim je računato (NULL = sva)
    izvor       TEXT    NOT NULL,          -- 'uzivo' ili 'retro'
    pogodak     INTEGER,                   -- NULL = kolo još nije uneto; 0/1 posle
    kreirano    TEXT    NOT NULL,          -- ISO datum-vreme
    UNIQUE (kolo, metod, izvor)
);
```

- `izvor` razdvaja prognoze napravljene uživo (pre nego što je kolo poznato) od
  retroaktivnih. **Na grafikonu i u statistici se prikazuju odvojeno ili sa jasnom
  oznakom** — retro rezultati su metodološki slabiji (rizik od suptilnog curenja) i ne
  smeju se mešati sa uživo rezultatima bez oznake.
- `UNIQUE (kolo, metod, izvor)` sprečava duple prognoze; ponovni upis = zamena
  (`INSERT OR REPLACE`) samo dok je `pogodak IS NULL`. **Ocenjenu prognozu je zabranjeno
  menjati** — to je poenta eksperimenta.

---

## 4. Tok „uživo" prognoze

1. **Generisanje predloga.** Kada korisnik otvori stranu „Prognoza", aplikacija odredi
   sledeće očekivano kolo (`poslednje uneto + 1`, uz preskok godine po postojećoj
   numeraciji) i, ako za njega nema upisanih prognoza, izračuna i upiše predlog svakog
   prediktora (`izvor='uzivo'`, `pogodak=NULL`).
2. **Zaključavanje.** Predlozi za dato kolo se računaju jednom i ne preračunavaju se
   (osim ako korisnik promeni globalni period pre nego što je kolo uneto — tada dugme
   „Preračunaj predloge" eksplicitno pregazi neocenjene).
3. **Evaluacija.** U postojećoj funkciji čuvanja kola (strana „Podaci") dodati korak:

   ```
   posle provere tiketa i bektestova:
       za svaku prognozu WHERE kolo = uneto_kolo AND pogodak IS NULL:
           pogodak = 1 ako je broj u 7 izvučenih, inače 0
   ```

4. **Edge case:** ako korisnik unese kolo za koje ne postoje prognoze (npr. uvoz iz
   fajla, preskočeno kolo) — ništa se ne dešava, prognoze za to kolo prosto ne postoje.
   Pri uvozu sa opcijom „Zameni istoriju" — obrisati i sve `retro` prognoze (postaju
   nevažeće), a `uzivo` zadržati.

---

## 5. Retroaktivni bektest

Dugme „Pokreni retro-bektest" na strani „Prognoza".

Algoritam (walk-forward):

```
za N od (min_start) do poslednjeg kola, hronološki:
    istorija = sva kola strogo pre N
    za svaki prediktor:
        broj = prediktor(istorija, period=RETRO_PERIOD)
        pogodak = broj in izvuceni(N)
        upiši (kolo=N, metod, broj, izvor='retro', pogodak)
```

- `min_start`: preskočiti prvih ~50 kola (premalo istorije za smislen period).
- `RETRO_PERIOD`: fiksirati na 100 (dokumentovati; opciono dropdown 50/100/200 —
  ali onda svaka vrednost perioda ide kao poseban skup rezultata, ne mešati).
- **Performanse:** Bajes model je iterativan kroz istoriju — naivno pozivanje za svako
  N daje O(N²). Rešenje: inkrementalna varijanta — održavati stanje Bajesovog modela
  i ažurirati ga jednim korakom po kolu tokom prolaska. Isto za brojače frekvencija
  (klizni prozor za period: dodaj kolo N-1, izbaci kolo N-1-period). Ciljno vreme:
  ceo bektest < par sekundi za 1.422 kola.
- Ponovno pokretanje: obriše sve `retro` redove pa računa iznova (uz potvrdu).
- Progres-bar ili bar indikator „u toku" u UI.

---

## 6. Zaštita od curenja budućnosti (kritično)

Ovo je jedini deo gde bag daje **lažno pozitivan rezultat** koji izgleda kao otkriće:

- Prediktor sme da primi isključivo kola sa `kolo < N`. Napisati **unit test** koji
  ubaci sintetičku istoriju gde je broj X izvučen tek u kolu N i potvrdi da nijedan
  prediktor za kolo N ne koristi tu informaciju (npr. frekvencijski brojači pre/posle).
- Bajes/hibrid funkcije sa strane „Rangiranje" verovatno primaju „svu istoriju" —
  refaktorisati da primaju eksplicitnu listu kola, a strana Rangiranje im prosleđuje
  celu. Nikakav globalni keš stanja između poziva.
- `random` prediktor: seed vezan za kolo, ne za vreme poziva.

---

## 7. UI strana „Prognoza"

### 7.1 Gornji blok — „Predlozi za sledeće kolo"
- Kartica po prediktoru: naziv metode, predložen broj (velika kuglica, obojena po
  postojećoj šemi vruć/hladan/neutralan), kratak opis logike (tooltip).
- Oznaka ciljnog kola (npr. „za kolo 2026057").
- Dugme „Preračunaj predloge" (aktivno samo dok kolo nije uneto).

### 7.2 Grafikon — kumulativna uspešnost kroz vreme
- X-osa: redni broj ocenjenog kola; Y-osa: kumulativni % pogodaka.
- Jedna linija po metodu (legenda sa uključivanjem/isključivanjem linija — postojeći
  chart framework to podržava).
- **Horizontalna referentna linija na 17,95 %.**
- **Pojas pouzdanosti oko baseline-a:** za n ocenjenih kola, 95 % interval za
  binomnu raspodelu p=7/39. Dovoljna je normalna aproksimacija:
  `granice = p ± 1,96 * sqrt(p*(1-p)/n)` — osenčena zona koja se sužava sleva nadesno.
  Interpretacija u UI: „linija metoda unutar zone = nerazlučivo od slučajnosti".
- Prekidač „Uživo / Retro / Sve (označeno)" — podrazumevano odvojeno.

### 7.3 Tabela rezultata po metodu
| Kolona | Sadržaj |
|---|---|
| Metod | naziv |
| Ocenjenih kola | n |
| Pogodaka | k |
| Uspešnost | k/n u % |
| Očekivano | 17,95 % |
| P-vrednost | binomni test, dvostrani: verovatnoća da čista slučajnost da k ili ekstremnije od očekivanog (koristiti `scipy.stats.binomtest` ako je scipy već zavisnost; ako nije — egzaktan binomni test je ~15 linija koda, ne uvoditi scipy samo zbog ovoga) |
| Zaključak | „Nerazlučivo od slučajnosti" (p ≥ prag) / „Odskače (proveriti!)" (p < prag) |

- **Prag značajnosti:** zbog 7 paralelnih metoda primeniti Bonferroni korekciju:
  prag = 0,05 / broj_metoda ≈ **0,007**. Ovo upisati i u tooltip („zašto ne 0,05").
- Ako neki metod „odskače": poruka ne sme da glasi „metod radi", nego
  „rezultat odskače — najverovatniji uzroci: greška u podacima, curenje budućnosti,
  ili slučajnost uprkos korekciji; proveriti pre bilo kakvog zaključka".

### 7.4 Tabela istorije prognoza
- Poslednjih ~50 redova: kolo, metod, predlog, izvučeni brojevi, pogodak (✓/✗), izvor.
- Filter po metodu.

---

## 8. Redosled implementacije (predlog faza)

1. **Faza 1 — jezgro:** tabela `prognoze`, registar prediktora, `hot`/`cold`/`random`,
   hook evaluacije pri unosu kola, minimalna strana sa predlozima i tabelom po metodu.
2. **Faza 2 — retro-bektest:** walk-forward petlja sa inkrementalnim brojačima,
   unit test protiv curenja, progres indikator.
3. **Faza 3 — puni skup i grafikon:** `bayes`/`hybrid` (refaktor reuse), `rhythm`,
   `fresh`, kumulativni grafikon sa baseline-om i pojasom pouzdanosti, binomni test
   i tabela zaključaka.
4. **Faza 4 — poliranje:** filteri, tooltips, edge case-ovi uvoza, dokumentacija u
   `FUNKCIJE.md` (nova sekcija „8. Prognoza").

---

## 9. Kriterijumi prihvatanja (acceptance)

- [ ] Pri unosu novog kola sve otvorene prognoze za to kolo dobijaju `pogodak` 0/1 bez
      ikakve akcije korisnika.
- [ ] Retro-bektest nad 1.422 kola završava za < 10 s i daje identičan rezultat pri
      svakom ponovnom pokretanju (determinizam, uključujući `random` sa seedom).
- [ ] Unit test curenja budućnosti prolazi za sve prediktore.
- [ ] Grafikon prikazuje baseline 17,95 % i pojas pouzdanosti koji se sužava sa n.
- [ ] `random` metod posle retro-bektesta završava unutar pojasa pouzdanosti
      (sanity check celog sistema — ako kontrolna grupa „odskače", bag je u evaluaciji).
- [ ] Ocenjene prognoze su nepromenljive (pokušaj izmene se odbija).
- [ ] Uživo i retro rezultati se nigde ne sabiraju bez vidljive oznake.

---

## 10. Van opsega v1 (svesno izostavljeno)

- Predviđanje više brojeva ili kombinacija (to već pokriva Generator).
- ML modeli — ista odluka kao za VAE u glavnoj aplikaciji.
- Automatsko „prebacivanje na najbolji metod" — eksplicitno zabranjeno u v1, jer bi
  naknadni izbor pobednika poništio statističku validnost (multiple comparisons).
