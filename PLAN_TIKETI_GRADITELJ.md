# Plan razvoja — Dnevnik tiketa i Graditelj kombinacije

> Verzija: 1.0
> Datum: 2026-10-08
> Projekat: Loto Analizator — web
> Osnova: `baza.py` (`odigrani_tiketi`), `bektest.py` (`dodaj_kolo_i_proveri`), `generator.py`,
> `prediktori.py`, `prediktori_komb.py`, `prognoza.py`
> Povod: kartica „Moji tiketi" je pravljena za stalne tikete koji se proveravaju svako kolo, a
> korisnik igra 1–2 nove kombinacije po kolu i želi istoriju svoje igre.

---

## 1. Šta se rešava

1. **Tiket nije vezan za kolo.** `odigrani_tiketi` čuva kombinaciju bez kola; svako novo
   izvlačenje prepisuje `poslednji_rezultat`, pa se istorija igre gubi.
2. **Mera blizine je pristrasna.** `promasaj_kombinacije` za svaki dobitni broj traži najbliži
   broj tiketa — jedan broj tiketa (10) može „pokriti" više dobitnih (9 i 11), pa mera
   precenjuje blizinu. Uz to nije normalizovana: ne zna se da li je 14 dobro.
3. **Nema provere sličnosti sa istorijom** pri unosu kombinacije.
4. **Analize nisu razdvojene** na istorijske (opisne) i one koje grade predlog.
5. **Nema mesta gde se predlog gradi vidljivo, korak po korak** — signali postoje
   (`k_cooc`, `k_rhythm7`, filteri Generatora), ali su razbacani.

---

## 2. Fiksne odluke

### 2.1. Redosled faza

Dnevnik tiketa (faze 1–3) pre Graditelja: beleženje igre počinje odmah, a Graditelj (faza 5)
ima gde da upiše kombinaciju. Reorganizacija menija (faza 4) pre Graditelja, da nova strana
odmah sleti u pravu grupu.

### 2.2. Stara tabela ostaje

`odigrani_tiketi` se ne briše — koristi je desktop `analiza.py`. Postojećih 19 tiketa se
kopira u novu tabelu sa `kolo = NULL` („bez kola"), izvor `uvoz`. Provera starih tiketa u
`dodaj_kolo_i_proveri` ostaje dok desktop postoji.

### 2.3. Blizina se računa na sortiranim kombinacijama

Kombinacija se gleda kao skup. Redosled izvlačenja (bitan za pozicionu analizu) ovde se ne
koristi.

### 2.4. Pravila o strukturi nisu prediktivni signal

Dekade, uzastopni, zbir i parnost ne menjaju verovatnoću — svaka kombinacija je jednako
verovatna. U UI se označavaju kao „preferencija oblika", odvojeno od signala (potencijal,
ritam, kookurencija), koji tvrde nešto o budućnosti i zato idu u bektest.

### 2.5. Težine Graditelja se zaključavaju pre bektesta

Podrazumevane težine (jednake) i bazen (15) su fiksni u `konfig.py` pre prvog retro-bektesta
`k_graditelj`. Korisnik ih može menjati na strani, ali bektest uvek koristi zaključane
vrednosti — inače su težine naštimovane prema istoriji i test gubi vrednost.

---

## 3. Faza 1 — Dnevnik tiketa

### 3.1. Tabela `odigrano`

| Kolona | Tip | Napomena |
|---|---|---|
| `id` | INTEGER PK | |
| `kolo` | INTEGER NULL | godina*1000+broj; NULL samo za uvezene |
| `kombinacija` | TEXT | JSON lista od 7 sortiranih brojeva |
| `izvor` | TEXT | `rucno`, `generator`, `sekv`, `prognoza:<metod>`, `graditelj`, `uvoz` |
| `napomena` | TEXT NULL | slobodan tekst |
| `uneto` | TEXT | vreme unosa |
| `pogoci`, `skoro`, `rastojanje`, `percentil` | NULL dok kolo nije izvučeno | §4 |

`UNIQUE(kolo, kombinacija)` — ista kombinacija u istom kolu dva puta nema smisla; u različitim
kolima je dozvoljena.

### 3.2. Unos

- Podrazumevano kolo = poslednje u bazi + 1. Prelaz godine: ako je poslednje kolo
  `G*1000+n` i korisnik promeni godinu, predlaže se `(G+1)*1000+1`. Polje se može menjati.
- Validacija: 7 različitih brojeva iz 1–39; kolo ne sme biti već izvučeno (tiket se upisuje
  pre izvlačenja; naknadni unos za izvučeno kolo dozvoljen uz potvrdu, odmah se ocenjuje).
- Status u listi: „čeka izvlačenje" / „provereno".

### 3.3. Provera sličnosti sa istorijom (dok se kuca)

`GET /api/odigrano/slicnost?brojevi=...` vraća:
- najbliže istorijsko kolo (kolo, datum, broj poklapanja, ista kombinacija),
- broj istorijskih kola sa 7/6/5/4 poklapanja,
- očekivane vrednosti za nasumičnu kombinaciju (hipergeometrijski, nad trenutnim brojem kola).

Na 1431 kolo očekivano je:

| Poklapanja | Očekivano kola |
|---|---|
| 7 | 0,0001 |
| 6 | 0,02 |
| 5 | 0,97 |
| 4 | 16,2 |

Upozorenje (žuto) od **6+**. 5 i 4 se prikazuju informativno, uz očekivanje, da se ne
tumače kao loš izbor.

### 3.4. API

- `GET /api/odigrano` — lista, najnovije kolo prvo
- `POST /api/odigrano` — `{kolo, brojevi, izvor, napomena}`
- `DELETE /api/odigrano/{id}`
- `GET /api/odigrano/slicnost`

### 3.5. Dugmad „+ tiket"

Generator, Prognoza i Sekvencijalni upisuju direktno u `odigrano` za sledeće kolo, sa
popunjenim `izvor`. Stari endpoint `/api/tiketi` ostaje samo za kompatibilnost.

---

## 4. Faza 2 — Ocenjivanje pri unosu izvlačenja

`dodaj_kolo_i_proveri` za uneto kolo ocenjuje **samo** redove `odigrano` sa tim kolom.

| Mera | Definicija |
|---|---|
| `pogoci` | \|T ∩ D\| |
| `skoro` | broj dobitnih brojeva d ∉ T za koje postoji t ∈ T sa \|t − d\| = 1 |
| `rastojanje` | D(T, I) = Σ \|t₍ᵢ₎ − d₍ᵢ₎\| nad sortiranim kombinacijama — optimalno 1-na-1 uparivanje u 1D (Earth mover's); 0..224 |
| `percentil` | udeo svih C(39,7) kombinacija čije je rastojanje do izvučene **strogo veće**, + pola izjednačenih (srednji rang) |

**Percentil — tačan račun.** Raspodela D(X, I) po svim 7-podskupovima X za dato I računa se
dinamičkim programiranjem: stanje (pozicija i, poslednji izabrani broj, zbir rastojanja do i),
39 × 7 × 225 stanja. Bez simulacije, tačno, delić sekunde.

Testovi (`tests/test_odigrano.py`):
- D je simetrično, D(T,T)=0, D(1..7, 33..39)=224
- raspodela iz DP ima ukupnu masu C(39,7) i poklapa se sa grubom enumeracijom na manjem
  problemu (npr. 7/15)
- tiket se ocenjuje samo za svoje kolo; uvezeni (`kolo NULL`) se ne diraju
- „skoro" ne broji pogođene brojeve

`promasaj_kombinacije` ostaje za bektestove koji ga već koriste.

---

## 5. Faza 3 — Kumulativni pregled tiketa

- Tabela po kolima: kolo, kombinacija, izvlačenje, pogoci, skoro, percentil, izvor, napomena.
- Prosečan percentil. Pod H₀ (izbor bez veze sa ishodom) percentili su ~ravnomerni na [0,1]:
  test srednje vrednosti (normalna aproksimacija, var = 1/12n) i — od 20+ tiketa — KS test.
  Prikazuje se n i upozorenje kad je n malo.
- Pogoci prema očekivanju: očekivano 7·7/39 = 1,256 po tiketu.
- Poređenje po izvoru (isti test po grupi, Bonferroni po broju grupa).

---

## 6. Faza 4 — Reorganizacija menija

| Grupa | Strane |
|---|---|
| **Istorija i statistika** | Istorija, Statistika, Mapa kombinacija, Različitost |
| **Predviđanje** | Rangiranje, Prognoza, Sinteza, Generator, Bektest, *Graditelj (faza 5)* |
| **Moja igra** | Moji tiketi |
| — | Kontrolna tabla, Podaci |

Pre premeštanja proveriti sadržaj svake strane; ako strana meša opisno i prediktivno, deli se
ili se prediktivni deo seli. Rute (`strana===...`) se ne menjaju, menja se samo meni.

---

## 7. Faza 5 — Graditelj kombinacije (nova strana, grupa Predviđanje)

Modul `core/graditelj.py`; postojeće funkcije se uvoze, ne kopiraju.

### 7.1. Deo 1 — Potencijal brojeva

- **Prozor W.** Podrazumevano = očekivano vreme „obrta" (svih 39 brojeva izvučeno bar jednom),
  tačno: **22,2 kola** → W = 22. P(obrt za 24 kola) = 0,71, za 30 = 0,90, za 40 = 0,99.
  Klizač: 22 / 50 / 100 / 500 / sva kola.
- Za broj b: E = 7W/39, O_b = broj pojava u poslednjih W kola,
  **potencijal z_b = (E − O_b) / √(W·p·(1−p))**, p = 7/39. Pozitivno = kasni.
- Grafikon: x = 39 brojeva sortiranih po z (opadajuće), y = z, traka ±2σ; boja po znaku.
- Napomena u UI: opisna mera; da li ima prediktivnu vrednost, odlučuje bektest (§8).
- **Bazen** = top 15 po z (`GRADITELJ_BAZEN = 15` u `konfig.py`). Tie-break: reproducibilan
  slučajan, isto kao u sekvencijalnom.

### 7.2. Deo 2 — Signali za brojeve iz bazena

- **Ritam:** D/R iz postojećeg `k_rhythm7` primitiva (D = trenutno kašnjenje, R = prosečan
  razmak ponavljanja), na **celoj istoriji**.
- **Kookurencija:** lift para = posmatrano / očekivano, očekivano = n·42/(39·38).
  Na celoj istoriji ≈ 40,6 po paru; u 22 kola ≈ 0,6 → zato **uvek dug prozor** (cela
  istorija), nezavisno od W. Koristi `matrica_cooc`.
- Tabela: broj, z, D/R; mini toplotna mapa lifta 15×15.

### 7.3. Deo 3 — Sklapanje

- Svih C(15,7) = 6435 kombinacija iz bazena.
- Signali se pre sabiranja svode na [0,1] (min-max preko bazena), da težine budu uporedive.
- **Skor(K) = a·Σ z̃ + b·Σ ritam̃ + c·(Σ lift̃ parova)/3** (21 par → /3 da bude na skali 7
  brojeva). a = b = c = 1 podrazumevano, klizači.
- **Preferencije oblika** (isključuju, ne boduju) — iste funkcije kao Generator
  (`broj_uzastopnih`, `osobine_kombinacije`): dekada max 3, uzastopnih max 1; zbir i parnost
  opciono.
- Izlaz: najbolja kombinacija + 3 alternative, raščlanjen skor po komponenti i po broju,
  provera sličnosti sa istorijom (§3.3), dugme **„Odigraj"** → `odigrano`, izvor `graditelj`.
- Ako nijedna kombinacija ne prolazi pravila: poruka „popusti pravila", kao kod sekv tiketa.

### 7.4. API

- `GET /api/graditelj/potencijal?w=22`
- `POST /api/graditelj/sklopi` — `{w, bazen, a, b, c, pravila}`

---

## 8. Faza 6 — `k_graditelj` u Prognozi

- Dodaje se u `PREDIKTORI_KOMB` sa **zaključanim** parametrima (§2.5): W=22, bazen 15,
  a=b=c=1, dekada max 3, uzastopnih max 1.
- Ulazi u retro-bektest i kontrolnu grupu kao i ostale metode.
- `PRAG_KOMB` se automatski pooštrava (Bonferroni nad jednim prediktorom više) — navesti u
  commitu.
- Test: `k_graditelj` ne gleda u ciljno kolo (isti test curenja kao za ostale prediktore).

---

## 9. Van opsega

- Promena desktop `analiza.py`.
- Optimizacija težina Graditelja prema istoriji (namerno — §2.5).
- Brisanje tabele `odigrani_tiketi`.
