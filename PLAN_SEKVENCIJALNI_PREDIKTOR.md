# Plan razvoja — „Sekvencijalni prediktor“ i koeficijent nepredvidivosti

> Verzija: 1.0
> Datum: 2026-09-08
> Projekat: Loto Analizator — web
> Osnova: `prognoza.py` (retro-bektest, `prognoza_u_tacki`), registri `PREDIKTORI` / `PREDIKTORI_KOMB`, `razlicitost.py`, `istorija.py`, `sinteza.py`

---

## 1. Svrha

Algoritam koji prolazi kroz istoriju **kolo po kolo**, u svakom koraku:
1. gleda kako se novo kolo razlikovalo od prethodnog (otišli / došli / ostali, pomak zbira…),
2. uklapa to sa akumuliranom statistikom (frekvencija, srednja vrednost, parovi, ritam, pozicija),
3. **uči** iz sopstvene greške u prošlom koraku,
4. predlaže kombinaciju za sledeće kolo,
5. uz predlog daje **koeficijent nepredvidivosti** — meru koliko zapravo zna.

Koeficijent je centralni deo, ne dodatak. Prediktor bez njega bi bio još jedan red u Sintezi; prediktor sa njim je živ instrument koji pokazuje, iz kola u kolo, koliko se iz istorije *može* naučiti.

**Očekivani ishod** (zapisati unapred, u kod i u UI): kod nezavisnih izvlačenja koeficijent ostaje ≈ 1,00 kroz celu istoriju. Sistem je uspešan ako to **pokaže**, ne ako to „pobedi“.

---

## 2. Fiksne odluke

### 2.1. Izlaz svakog eksperta je raspodela, ne 7 brojeva

Svaki ekspert za kolo *t+1* daje vektor **p ∈ ℝ³⁹**, p ≥ 0, Σp = 7 (očekivani broj pogodaka; ekvivalentno verovatnoća da svaki broj bude izvučen). Uniformni ekspert daje p_i = 7/39 za sve *i*.

Postojeći prediktori (hot, cold, bayes, fresh, ritam, pozicija) se **omotavaju** funkcijom koja njihov rang/ocenu pretvara u raspodelu (softmax sa temperaturom, normalizovan na 7). Ne menjaju se.

### 2.2. Novi eksperti „prelaza“ (uče iz niza razlika)

U `core/prelazi.py`:

| Ekspert | Šta uči iz koraka *t−1 → t* |
|---|---|
| `povratak` | šansa da broj koji je upravo **otišao** dođe u sledećem kolu (vs. 7/39) |
| `zadrzavanje` | šansa da broj koji je **ostao** 2+ kola uzastopno ostane i dalje |
| `prelaz_prekl` | matrica prelaza preklapanja 8×8 → očekivano preklapanje sa poslednjim kolom → pojačava/slabi brojeve iz poslednjeg kola |
| `pomak_zbira` | raspodela Δzbir → pomera težinu ka nižim/višim brojevima |

Svi se ažuriraju online: brojači, ne ML. Svaki ima jasno teorijsko očekivanje (7/39, hipergeometrijska, …) i to očekivanje je njegova početna vrednost.

### 2.3. Mešanje: eksponencijalne težine (Hedge)

- Težine w_e, početno jednake.
- Predlog: p_mix = Σ w_e · p_e / Σ w_e.
- Posle stvarnog kola *S* (skup od 7): gubitak eksperta ℓ_e = −Σ_{i∈S} ln p_e(i) − Σ_{i∉S} ln(1 − p_e(i)) (Bernoulli log-loss po broju).
- Ažuriranje: w_e ← w_e · exp(−η · ℓ_e), zatim normalizacija. η fiksno (npr. 0,05), zapisano u `konfig.py`.
- **Uniformni ekspert je uvek u mešavini.** Ako ništa ne radi, težina mu raste ka 1 — to je ugrađena bezbednost protiv preučavanja.

### 2.4. Predložena kombinacija

7 brojeva sa najvećim p_mix. Opcija: proći kroz postojeći diverzitet Generatora. Prikazuje se **uvek** uz koeficijent, nikad sama.

> **REVIDIRANO 2026-09-08** (PLAN_KORAK_IZBORA §2.1). Opcija „proći kroz Generator" je odbačena. Filteri Generatora rade na razlikama reda 10⁻², dok je razlika 7. i 8. kandidata 4·10⁻⁴ — Generator bi u potpunosti preuzeo izbor, pa bi predlog prestao da svedoči o modelu. Umesto toga se prikazuju **dva odvojena izlaza sa jasnim vlasništvom**: predlog modela (goli top-7 po p_mix, bez ikakvih filtera) i tiket (Generator bira iz bazena od `SEKV_BAZEN` najverovatnijih brojeva). Nijedan se ne prikazuje sam, i uz tiket obavezno stoji da nema veću šansu od predloga.

### 2.5. Koeficijent nepredvidivosti (definisan pre implementacije)

```
K_t = L_model(1..t) / L_uniform(1..t)
```

gde je L kumulativni log-gubitak (§2.3) mešavine odnosno uniformnog modela.

- K = 1,00 → model ne zna više od slučajnosti
- K < 1 → model izvlači informaciju
- K > 1 → model je preučen

Uz K se računa **pojas ±2σ** oko 1,00, izveden iz varijanse log-gubitka pojedinačnog kola pod uniformnom hipotezom (analitički ili bootstrap nad slučajnim kolima, seed fiksan). Isti K se računa i po ekspertu.

Sekundarne mere (samo za detalj-panel): prosečno preklapanje predloga sa stvarnim kolom (μ = 1,256), pogodak ≥ 3 (teorijska verovatnoća).

### 2.6. Bez curenja

Sve se računa u petlji retro-bektesta iz `prognoza_u_tacki`: u koraku *t* model vidi samo kola ≤ t, predlaže za t+1, tek onda dobija t+1 i ažurira. Težine se **ne inicijalizuju** iz cele istorije. Isti anti-leakage test kao za istoriju i ensemble.

---

## 3. Arhitektura

```
webapp/
  core/
    prelazi.py           # NOVO: eksperti prelaza (povratak, zadrzavanje, prelaz_prekl, pomak_zbira)
    sekvencijalni.py     # NOVO: omotači postojećih prediktora u raspodele, Hedge mešanje,
                         #       koeficijent K, pojas, stanje modela (serijalizabilno)
    prediktori_komb.py   # + k_sekv (registracija: predlog = top-7 iz sekvencijalni)
    konfig.py            # + ETA_HEDGE, TEMPERATURA_SOFTMAX
    baza.py              # + tabela sekv_stanje (težine i K po kolu) + hook na unos kola
  api/app.py             # + /api/sekv/*
  static/                # + panel „Sekvencijalni prediktor“ (Prognoza) + kartica na Dashboardu
  tests/
    test_sekv.py         # NOVO
```

---

## 4. API

| Endpoint | Vraća |
|---|---|
| `GET /api/sekv/stanje` | trenutne težine eksperata, K, pojas, predlog za sledeće kolo, n kola |
| `GET /api/sekv/istorija` | K_t kroz vreme (za krivulju), K po ekspertu kroz vreme, težine kroz vreme |
| `GET /api/sekv/korak?granica=` | vremeplov: stanje u tački (težine, predlog, K) + stvarno kolo + gubitak koraka |
| `POST /api/sekv/rekonstruisi` | ponovo prolazi celu istoriju od nule (~ retro-bektest), upisuje u `sekv_stanje` |

---

## 5. UI

### 5.1. Kartica na Dashboardu

```
┌────────────────────────────────────────────┐
│ Koeficijent nepredvidivosti   1,003        │
│ posle 1.422 kola  (pojas 0,988 – 1,012)    │
│ Sistem uči iz svakog kola. Do sada nije    │
│ naučio ništa što slučajnost ne zna.        │
│                          [Detalji →]       │
└────────────────────────────────────────────┘
```

Tekst se generiše iz K i pojasa (tri ishoda, kao u Sintezi): unutar pojasa / ispod / iznad.

### 5.2. Panel u Prognozi: „Sekvencijalni prediktor“

- Predlog za sledeće kolo (7 brojeva) — **odmah ispod**: „Koeficijent 1,003: ovaj predlog ima istu šansu kao bilo koja druga kombinacija.“
- Krivulja K_t kroz vreme sa pojasom ±2σ; ispod nje krivulje K po ekspertu (tanke).
- Trakasti dijagram trenutnih težina eksperata; uniformni istaknut.
- „Šta je model naučio iz poslednjeg kola“: gubitak po ekspertu, ko je dobio/izgubio težinu.

### 5.3. Vremeplov (Istraži istoriju)

Uz postojeći „Niz razlika“ panel: stanje modela u izabranoj tački — predlog koji je tada dao, težine, K — i dugme „Prikaži stvarni ishod“. Korak napred/nazad preračunava.

### 5.4. Red u Sintezi

`k_sekv` se pojavljuje kao običan red (prosek preklapanja, μ, z, p, p_kor). Plus red tipa `test`: „K ≠ 1?“ sa p-vrednošću iz pojasa.

---

## 6. Faze

### Faza 1 — Jezgro bez UI-ja
1. `sekvencijalni.py`: raspodele, log-gubitak, Hedge, K, pojas. Uniformni ekspert + omotači 6 postojećih prediktora.
2. Petlja kroz istoriju preko `prognoza_u_tacki`; upis u `sekv_stanje`.
3. `test_sekv.py` osnovni testovi (§7).

**Gotovo kad:** rekonstrukcija cele istorije radi determinstički za < 20 s i K_t je izračunat za svako kolo.

### Faza 2 — Eksperti prelaza
1. `prelazi.py` sa 4 eksperta, online ažuriranje.
2. Uključeni u mešavinu.
3. Test: na sintetičkoj uniformnoj bazi svaki ekspert prelaza konvergira ka svom teorijskom očekivanju.

**Gotovo kad:** 11 eksperata u mešavini, K i dalje ≈ 1 na sintetici.

### Faza 3 — Registracija i Sinteza
1. `k_sekv` u `PREDIKTORI_KOMB`.
2. Redovi u Sintezi.

**Gotovo kad:** Sinteza prikazuje `k_sekv` i test K bez izmene UI-ja Sinteze.

### Faza 4 — UI
1. Kartica na Dashboardu.
2. Panel u Prognozi sa krivuljama.
3. Vremeplov integracija.
4. Hook: unos novog kola ažurira stanje inkrementalno (jedan korak, ne rekonstrukcija).

**Gotovo kad:** unos kola u Podacima menja K na Dashboardu bez ručne akcije.

---

## 7. Testovi — `test_sekv.py`

| Test | Proverava |
|---|---|
| `test_raspodela_suma_7` | svaki ekspert vraća p ≥ 0, Σp = 7, za sve tačke |
| `test_uniformni_K_jednak_1` | mešavina samo sa uniformnim ekspertom → K = 1,000 tačno |
| `test_K_na_sintetici` | 5.000 slučajnih kola: K unutar pojasa ±2σ, težina uniformnog ≥ svake druge na kraju |
| `test_K_na_pristrasnoj_sintetici` | sintetika gde broj 7 izlazi 30% češće: K < 1 značajno, ekspert `hot` dobija najveću težinu — **dokaz da bi model prepoznao signal kad bi ga bilo** |
| `test_bez_curenja` | stanje u tački g zavisi samo od kola ≤ g (mutacija budućih kola u temp bazi) |
| `test_determinizam` | dve rekonstrukcije daju identične K_t i težine |
| `test_inkrementalno_jednako_rekonstrukciji` | dodavanje jednog kola korakom == rekonstrukcija sa tim kolom |
| `test_k_sekv_u_registru` | `k_sekv` prolazi kroz postojeći retro-bektest kao svaki drugi prediktor |

Test na **pristrasnoj sintetici** je ključan: on pokazuje da K ≈ 1 na pravim podacima nije mana modela nego osobina podataka.

---

## 8. Šta NE raditi

- Ne prikazivati predlog bez koeficijenta.
- Ne izbacivati uniformnog eksperta iz mešavine.
- Ne podešavati η ili temperaturu „dok K ne padne ispod 1“ na istorijskim podacima — to je preučavanje; parametri se biraju na sintetici i fiksiraju.
- Ne uvoditi neuronske mreže, VAE, Gemini. Sve su brojači i eksponencijalne težine.
- Ne inicijalizovati težine iz cele istorije.
- Ne tumačiti K < 1 u jednom kratkom periodu kao signal — samo izlazak iz pojasa koji se **održava** vodi na proveru curenja, pa tek onda dalje.

---

## 9. Commit redosled

```
1. feat(sekv): raspodele, Hedge mesanje, koeficijent K i pojas
2. feat(sekv): rekonstrukcija istorije i tabela sekv_stanje
3. feat(prelazi): eksperti povratak/zadrzavanje/prelaz_prekl/pomak_zbira
4. feat(prediktori): k_sekv u registru + redovi u Sintezi
5. feat(ui): kartica K na Dashboardu, panel u Prognozi, vremeplov
6. feat(baza): inkrementalni korak na unos kola
7. test: sekv (sintetika, pristrasna sintetika, anti-curenje, determinizam)
8. docs: README i FUNKCIJE sekcija „Sekvencijalni prediktor“
```

---

## 10. Kriterijum uspeha

Korisnik na Dashboardu vidi jedan broj koji se menja sa svakim kolom i koji kaže, iz podataka a ne iz teksta, koliko se iz istorije može naučiti. Model predlaže kombinaciju, uči iz greške, i sam izveštava da to učenje ne donosi ništa — a test na pristrasnoj sintetici dokazuje da bi doneo kad bi imalo šta da se nauči. Ako K ikad trajno izađe iz pojasa, to je jedini crveni signal u aplikaciji i vodi na proveru, ne na tiket.
