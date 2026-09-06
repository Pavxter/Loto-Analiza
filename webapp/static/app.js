// Loto Analizator — frontend logika (Alpine + ECharts)

const BOJE = {
  vruc: '#ff6b4a', hladan: '#4f8cff', neutralan: '#6b7686', svez: '#35d07f',
  accent: '#4f8cff', tekst: '#9aa7b5', mreza: '#262d3a', pozadina: 'transparent',
};

const grafikoni = {};   // id -> echarts instanca
let mapaL = null, mapaSlojL = null, mapaOznakaL = null, mapaGranice = null;   // Leaflet (Mapa kombinacija)
let mapaSadL = null, mapaTajmer = null;                 // prsten oko tekućeg kola i tajmer animacije
let mapaPutCrtac = null;                                // zaseban canvas: putanja uvek ide ispod tačaka
// Markeri i segmenti se prave jednom po setu; vremenski slajder samo menja koji
// je opseg na mapi. Ponovno pravljenje 1.400 slojeva na svaki korak bi seklo animaciju.
const mapaSetovi = { stvarno: null, slucajno: null };

// Boja segmenta putanje = koliko brojeva kolo deli sa prethodnim (0..7), isti pojam
// preklapanja kao na strani „Različitost". Gola linija se nikad ne crta.
const BOJE_PREKLAPANJA = ['#2f6fd0', '#3fa0d8', '#3fc4b0', '#5fd07f', '#c9d24a', '#f0a83a', '#f2662a', '#e02020'];

function bojaPreklapanja(k) {
  return BOJE_PREKLAPANJA[Math.max(0, Math.min(BOJE_PREKLAPANJA.length - 1, k || 0))];
}

// Pomera prikazani opseg sloja na [lo, hi] dodajući i sklanjajući samo razliku.
// st = { grupa, elem, lo, hi }; prazan opseg je hi < lo.
function mapaOpseg(st, lo, hi) {
  if (!st) return;
  if (hi < lo) { lo = 0; hi = -1; }
  if (st.hi < st.lo || lo > st.hi + 1 || hi < st.lo - 1) {   // nema dodira: lakše je sve skloniti
    for (let i = st.lo; i <= st.hi; i++) st.grupa.removeLayer(st.elem[i]);
    st.lo = lo; st.hi = lo - 1;
  }
  for (let i = st.lo; i < lo; i++) st.grupa.removeLayer(st.elem[i]);
  for (let i = st.hi; i > hi; i--) st.grupa.removeLayer(st.elem[i]);
  st.lo = Math.max(st.lo, lo); st.hi = Math.min(st.hi, hi);
  for (let i = st.lo - 1; i >= lo; i--) st.grupa.addLayer(st.elem[i]);
  for (let i = st.hi + 1; i <= hi; i++) st.grupa.addLayer(st.elem[i]);
  st.lo = lo; st.hi = hi;
}

function bazaOpcija() {
  return {
    backgroundColor: BOJE.pozadina,
    textStyle: { color: BOJE.tekst, fontFamily: 'Segoe UI, system-ui, sans-serif' },
    grid: { left: 44, right: 18, top: 24, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#1c2330', borderColor: '#262d3a', textStyle: { color: '#e6edf3' } },
  };
}

function crtaj(id, opcija) {
  const el = document.getElementById(id);
  if (!el) return;
  // Ako je stara instanca vezana za DOM koji je x-if u međuvremenu zamenio, oslobodi je.
  const stara = grafikoni[id];
  if (stara && !stara.isDisposed() && stara.getDom() !== el) stara.dispose();
  let g = echarts.getInstanceByDom(el) || echarts.init(el, null, { renderer: 'canvas' });
  grafikoni[id] = g;
  g.setOption(opcija, true);
  requestAnimationFrame(() => g.resize());
}

window.addEventListener('resize', () => Object.values(grafikoni).forEach(g => { if (!g.isDisposed()) g.resize(); }));

async function jget(url) { const r = await fetch(url); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); }
async function jsend(url, method, body) {
  const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

function app() {
  return {
    strane: [
      { id: 'dashboard', naziv: 'Dashboard', ico: '🏠', opis: 'Ključni pokazatelji i predlog bazena', period: true },
      { id: 'istorija', naziv: 'Istraži istoriju', ico: '🕰️', opis: 'Vremeplov kroz kola — šta je sistem znao u svakom trenutku', period: false },
      { id: 'mapa', naziv: 'Mapa kombinacija', ico: '🗺️', opis: 'Ceo prostor od 15.380.937 kombinacija — gde je koja i gde je tvoj tiket', period: false },
      { id: 'statistika', naziv: 'Statistika', ico: '📊', opis: 'Frekvencija, srednje vrednosti, dekade, poziciona analiza', period: true },
      { id: 'razlicitost', naziv: 'Različitost', ico: '🧬', opis: 'Koliko se izvučene kombinacije razlikuju — poređenje sa čistom slučajnošću', period: true },
      { id: 'rangiranje', naziv: 'Rangiranje', ico: '🎯', opis: 'Rangiranje brojeva: Frekvencija / Bajes / Hibrid', period: false },
      { id: 'prognoza', naziv: 'Prognoza', ico: '🔮', opis: 'Predviđanje jednog broja — statistički eksperiment sa kontrolnom grupom', period: true },
      { id: 'generator', naziv: 'Generator', ico: '⚙️', opis: 'Generiši kombinacije po filterima i bodovanju', period: true },
      { id: 'bektest', naziv: 'Bektest', ico: '🧪', opis: 'Uspešnost sačuvanih strategija', period: false },
      { id: 'tiketi', naziv: 'Moji tiketi', ico: '🎟️', opis: 'Evidencija odigranih tiketa', period: false },
      { id: 'podaci', naziv: 'Podaci', ico: '🗄️', opis: 'Unos kola i uvoz istorije', period: false },
    ],
    strana: 'dashboard',
    period: 0,
    loading: false,
    brojKola: '…',
    toasts: [],

    // podaci strana
    dash: null, stat: null, hik: null,
    metoda: 'frekvencija', rang: [],
    gen: { koristiBazen: false, bazenText: '', min_sv: 1, max_sv: 39,
           f_parni: false, parni: 3, f_vruci: false, vruci: 4, f_hladni: false, hladni: 1,
           f_uzastopni: false, uzastopni: 0, f_dekada: false, dekada_max: 3,
           strategija: 'favorizuj', pristrasnost: true, unikati: false,
           diverzitet: false, max_slicnost: 4, radi: false, rezultati: [], rezime: '', razlicitost: null,
           granica: null },
    bektestovi: [], tiketi: [], noviTiket: '',
    istorija: [], unos: { kolo: null, datum: new Date().toISOString().slice(0, 10), brojevi: '' }, fajl: null, uvozZameni: false,
    prog: { tab: 'broj', ciljnoKolo: null, predlozi: [], izvor: 'uzivo', statistika: [], istorija: [],
            filterMetod: '', prag: 0.007, brojMetoda: 7, radi: false },
    progK: { ciljnoKolo: null, predlozi: [], izvor: 'uzivo', statistika: [], istorija: [],
             filterMetod: '', histMetod: '', hist: null, prag: 0.0036, brojMetoda: 14,
             ocekivano: 1.256, sigma: 0.9317, ucitano: false },
    razl: { podaci: null, profilTip: 'sredina', prikaziParove: false, detaljPar: null },
    mapa: { info: null, sloj: '', detalj: null, tiket: '', zum: 0, uklopljeno: false,
            prikaz: 'stvarno', stvarno: null, slucajno: null, seed: null, radiUzorak: false,
            korak: 0, rep: 50, putanja: true, animira: false,
            test: { otvoren: false, podaci: null, radi: false, granica: null } },
    ist: { granica: null, cilj: null, prozor: 100, broj: null, loading: false, kontekst: null, detalj: null,
           otvori: { sazetak: false, razl: false, rang: false, prog: false }, razl: null, rang: null,
           vremeplov: { podaci: null, ishod: null, radi: false } },

    aktivna() { return this.strane.find(s => s.id === this.strana) || this.strane[0]; },

    async init() {
      try { const d = await jget('/api/dashboard?period=0'); this.brojKola = d.broj_kola; } catch (e) {}
      this.ucitajStranu();
    },

    idi(id) {
      if (this.strana === 'mapa' && id !== 'mapa') this.mapaStani();   // animacija ne sme da radi u pozadini
      this.strana = id;
      this.ucitajStranu();
    },

    ucitajStranu() {
      const f = {
        dashboard: () => this.ucitajDashboard(),
        istorija: () => this.ucitajIstorijskiPregled(),
        mapa: () => this.ucitajMapu(),
        statistika: () => this.ucitajStatistiku(),
        razlicitost: () => this.ucitajRazlicitost(),
        rangiranje: () => this.ucitajRang(),
        prognoza: () => this.ucitajPrognozu(),
        generator: () => {},
        bektest: () => this.ucitajBektest(),
        tiketi: () => this.ucitajTikete(),
        podaci: () => this.ucitajIstoriju(),
      }[this.strana];
      if (f) f();
    },

    toast(tekst, tip = '') {
      const id = Date.now() + Math.random();
      this.toasts.push({ id, tekst, tip });
      setTimeout(() => { this.toasts = this.toasts.filter(t => t.id !== id); }, 3600);
    },

    klasaBroja(b) {
      if (!this.dash) return 'neutralan';
      if (this.dash.vruci.includes(b)) return 'vruc';
      if (this.dash.hladni.includes(b)) return 'hladan';
      return 'neutralan';
    },
    formatSkor(s) { return s < 1 ? Number(s).toExponential(3) : Number(s).toFixed(2); },
    redBrojevi(r) { return [r.b1, r.b2, r.b3, r.b4, r.b5, r.b6, r.b7]; },
    formatKolo(k) { if (k == null) return '—'; const s = String(k); return s.length > 4 ? s.slice(0, 4) + '-' + s.slice(4) : s; },

    // ---------- ISTRAŽI ISTORIJU ----------
    istKolaObrnuto() { return this.ist.kontekst ? [...this.ist.kontekst.kola].reverse() : []; },

    async ucitajIstorijskiPregled() {
      this.ist.loading = true;
      try {
        if (this.ist.granica == null) {
          const g = await jget('/api/istorija/granice');
          this.ist.granica = g.najnovije;   // start = najnovije poznato kolo
        }
        if (this.ist.granica == null) { this.ist.kontekst = null; return; }  // prazna baza
        const k = await jget(`/api/istorija/kontekst?granica=${this.ist.granica}&prozor=${this.ist.prozor}`);
        this.ist.kontekst = k;
        this.ist.cilj = k.cilj;
        if (this.ist.broj != null) await this.istorijaBroj(this.ist.broj);   // osveži otvoreni detalj
        // osveži otvorene collapsible sekcije (Faza 3) da prate granicu/prozor
        if (this.ist.otvori.razl) await this.ucitajIstRazlicitost();
        if (this.ist.otvori.rang) await this.ucitajIstRangiranje();
        // vremeplov (Faza 4): ako je prognoza već izračunata, preračunaj za novu granicu
        if (this.ist.otvori.prog && this.ist.vremeplov.podaci) await this.vremeplovIzracunaj();
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
      this.ist.loading = false;
    },

    istToggle(sekcija) {
      this.ist.otvori[sekcija] = !this.ist.otvori[sekcija];
      if (!this.ist.otvori[sekcija]) return;
      if (sekcija === 'razl' && !this.ist.razl) this.ucitajIstRazlicitost();
      if (sekcija === 'rang' && !this.ist.rang) this.ucitajIstRangiranje();
    },

    async ucitajIstRazlicitost() {
      this.ist.razl = null;
      if (this.ist.cilj == null) return;   // granica = najnovije → nema cilja
      try {
        this.ist.razl = await jget(`/api/istorija/razlicitost?cilj=${this.ist.cilj}&prozor=${this.ist.prozor}`);
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    async ucitajIstRangiranje() {
      try {
        this.ist.rang = await jget(`/api/istorija/rangiranje?granica=${this.ist.granica}&prozor=${this.ist.prozor}`);
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    istSazetakSort() {
      const s = this.ist.kontekst && this.ist.kontekst.sazetak;
      return s ? [...s.brojevi].sort((a, b) => b.pojavljivanja - a.pojavljivanja || a.broj - b.broj) : [];
    },
    istRangSort() {
      const r = this.ist.rang;
      return r ? [...r.tabela].sort((a, b) => a.hibrid.rang - b.hibrid.rang) : [];
    },

    // Vremeplov prognoze (Faza 4)
    async vremeplovIzracunaj() {
      this.ist.vremeplov.ishod = null;            // stari ishod se ne prenosi na novu tačku
      this.ist.vremeplov.radi = true;
      try {
        this.ist.vremeplov.podaci = await jget(`/api/istorija/prognoza?granica=${this.ist.granica}`);
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
      this.ist.vremeplov.radi = false;
    },
    async vremeplovIshod() {
      this.ist.vremeplov.radi = true;
      try {
        this.ist.vremeplov.ishod = await jget(`/api/istorija/prognoza/ishod?granica=${this.ist.granica}`);
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
      this.ist.vremeplov.radi = false;
    },
    vremeplovBroj(red) {
      // jednobrojna prognoza: kad postoji ishod, oboji po pogotku
      const i = this.ist.vremeplov.ishod;
      if (!i) return 'neutralan';
      const o = i.jedan_broj.find(x => x.metod === red.metod);
      return o && o.pogodak ? 'vruc' : 'hladan';
    },
    vremeplovPreklapanje(metod) {
      const i = this.ist.vremeplov.ishod;
      if (!i) return '—';
      const o = i.kombinacija.find(x => x.metod === metod);
      return o ? o.preklapanje : '—';
    },

    async istorijaBroj(broj) {
      this.ist.broj = broj;
      try {
        const d = await jget(`/api/istorija/broj/${broj}?granica=${this.ist.granica}&prozor=${this.ist.prozor}`);
        this.ist.detalj = d;
        this.$nextTick(() => {
          const t = d.timeline;
          crtaj('ist-timeline', {
            ...bazaOpcija(),
            grid: { left: 8, right: 12, top: 10, bottom: 20 },
            tooltip: { trigger: 'axis', backgroundColor: '#1c2330', borderColor: '#262d3a', textStyle: { color: '#e6edf3' },
              formatter: p => { const i = p[0].dataIndex; return this.formatKolo(t[i].kolo) + (t[i].pojavio ? ' · izašao' : ' · nije'); } },
            xAxis: { type: 'category', data: t.map(x => x.kolo), show: false },
            yAxis: { type: 'value', max: 1, min: 0, show: false },
            series: [{ type: 'bar', barWidth: '70%',
              data: t.map(x => ({ value: x.pojavio ? 1 : 0.06, itemStyle: { color: x.pojavio ? BOJE.accent : BOJE.mreza, borderRadius: [2, 2, 0, 0] } })) }],
          });
        });
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    brojRecenica() {
      const d = this.ist.detalj; if (!d) return '';
      const veza = d.znacajno ? 'razlika je značajna (proveriti)' : 'razlika nije značajna';
      return `Broj ${d.broj} se pojavio ${d.u_prozoru} puta u poslednjih ${d.prozor_n} kola; ` +
             `očekivanje ≈ ${d.ocekivano}; ${veza}.`;
    },

    istGranica(kolo) {
      if (kolo == null) return;
      this.ist.granica = kolo;
      this.ucitajIstorijskiPregled();
    },

    async izaberiKolo(kolo) {
      // Klik na kolo u tabeli: to kolo postaje cilj -> granica je kolo pre njega.
      try {
        const d = await jget(`/api/istorija/kolo/${kolo}`);
        this.ist.granica = d.prethodno != null ? d.prethodno : kolo;  // najstarije: ostaje granica
        this.ucitajIstorijskiPregled();
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    // ---------- MAPA KOMBINACIJA ----------
    mapaSloj() {
      const prazan = { sloj: '', opis: '', min: 0, max: 0 };
      if (!this.mapa.info) return prazan;
      return this.mapa.info.slojevi.find(s => s.sloj === this.mapa.sloj) || prazan;
    },

    mapaSkala(v) {
      // Opsezi osobina su celi brojevi, a ocena nije — legenda ne treba decimale.
      return v == null ? '' : Math.round(v).toLocaleString('sr-RS');
    },

    mapaRecenicaSloja() {
      const s = this.mapaSloj();
      if (!s.sloj) return '';
      if (!s.iz_baze) {
        return `Boja je osobina same kombinacije (${s.opis.toLowerCase()}), `
          + 'ne verovatnoća i ne učestalost izvlačenja.';
      }
      const strategija = s.strategija_svezine === 'favorizuj' ? '„favorizuj sveže"' : s.strategija_svezine;
      return `Boja je ocena Generatora — jedini sloj koji zavisi od baze. Pločice su ispečene za `
        + `stanje do kola ${this.formatKolo(s.granica)} (${(s.broj_kola || 0).toLocaleString('sr-RS')} kola), `
        + `strategijom ${strategija} i bez pristrasnosti, i ne menjaju se pri unosu novog kola. `
        + 'Ocena je pravilo po kom Generator bira, a ne verovatnoća: sve kombinacije imaju istu šansu.';
    },

    async ucitajMapu() {
      try {
        if (!this.mapa.info) {
          this.mapa.info = await jget('/api/mapa/info');
          if (!this.mapa.sloj && this.mapa.info.slojevi.length) this.mapa.sloj = this.mapa.info.slojevi[0].sloj;
        }
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); return; }
      if (!this.mapa.info.slojevi.length) return;      // pločice nisu generisane
      if (!this.mapa.stvarno) await this.mapaUcitajTacke();
      this.$nextTick(() => this.mapaPostavi());
    },

    mapaPostavi(pokusaj = 0) {
      const el = document.getElementById('mapa-platno');
      if (!el) {
        if (this.strana === 'mapa' && pokusaj < 100) setTimeout(() => this.mapaPostavi(pokusaj + 1), 60);
        return;
      }
      // Alpine-ov x-if ume da zameni platno; tada je stara instanca vezana za DOM
      // koji više ne postoji i javlja veličinu 0 (isti problem kao kod ECharts-a).
      if (mapaL && mapaL.getContainer() !== el) {
        mapaL.remove();
        mapaL = mapaSlojL = mapaOznakaL = null;
        this.mapa.uklopljeno = false;
      }
      const MAXZ = this.mapa.info.max_zoom, DIM = this.mapa.info.dimenzija;
      const novoPlatno = !mapaL;
      if (!mapaL) {
        mapaL = L.map(el, {
          crs: L.CRS.Simple, minZoom: 0, maxZoom: MAXZ + 3,
          attributionControl: false, zoomControl: true,
          preferCanvas: true,               // hiljade tačaka idu na canvas, ne u SVG

          // sitniji korak da mapa popuni platno; dugmad i dalje idu po ceo nivo
          zoomSnap: 0.25, zoomDelta: 1,
          // točak miša ostaje stranici (mapa je usred duge strane); Ctrl+točak zumira
          scrollWheelZoom: false,
        });
        el.addEventListener('wheel', e => {
          if (!e.ctrlKey) return;
          e.preventDefault();
          if (e.deltaY < 0) mapaL.zoomIn(1); else mapaL.zoomOut(1);
        }, { passive: false });
        // Putanja u svom sloju iznad pločica, a ispod tačaka (overlayPane je 400).
        mapaL.createPane('putanja');
        mapaL.getPane('putanja').style.zIndex = 390;
        mapaPutCrtac = L.canvas({ pane: 'putanja' });
        // Koordinate mape su pikseli najvišeg zuma, pa je (x, y) uvek ćelija kombinacije.
        mapaGranice = L.latLngBounds(mapaL.unproject([0, 0], MAXZ), mapaL.unproject([DIM, DIM], MAXZ));
        mapaL.setMaxBounds(mapaGranice.pad(0.1));
        mapaL.on('click', e => this.mapaKlik(e));
        mapaL.on('zoomend', () => { this.mapa.zum = mapaL.getZoom(); this.mapaAzurirajTacke(); });
        // Prvi fitBounds mora odmah: dok mapa nema postavljen pogled, Leaflet
        // ignoriše invalidateSize i veličina zauvek ostane 0.
        mapaL.fitBounds(mapaGranice);
        // Platno je skriveno dok se ne otvori tab, a menja veličinu i sa prozorom;
        // Leaflet mora da sazna za svaku takvu promenu, inače ostane na veličini 0.
        new ResizeObserver(() => this.mapaOsveziVelicinu()).observe(el);
      }
      if (novoPlatno) this.mapa.uklopljeno = false;
      this.mapaPostaviSloj();
      if (novoPlatno || !mapaSetovi.stvarno) this.mapaGradiSetove();
      this.mapaCrtajTacke();
      this.mapaOsveziVelicinu();
    },

    mapaOsveziVelicinu() {
      if (!mapaL) return;
      mapaL.invalidateSize();
      const v = mapaL.getSize();
      if (!v.x || !v.y) return;             // platno još nije vidljivo
      if (!this.mapa.uklopljeno) {          // prvi put kad se zna prava veličina
        mapaL.fitBounds(mapaGranice);
        this.mapa.uklopljeno = true;
        this.mapaOznaci();
      }
      this.mapa.zum = mapaL.getZoom();
    },

    mapaPostaviSloj() {
      if (!mapaL) return;
      const MAXZ = this.mapa.info.max_zoom;
      if (mapaSlojL) mapaL.removeLayer(mapaSlojL);
      mapaSlojL = L.tileLayer(`/mapa/${this.mapa.sloj}/{z}/{x}/{y}.png`, {
        tileSize: this.mapa.info.velicina_plocice, minZoom: 0, maxZoom: MAXZ + 3,
        maxNativeZoom: MAXZ, noWrap: true, bounds: mapaGranice,
      }).addTo(mapaL);
    },

    async mapaUcitajTacke(seed = null) {
      try {
        if (!this.mapa.stvarno) {
          const s = await jget('/api/mapa/tacke');
          this.mapa.stvarno = s.tacke;
          this.mapa.korak = Math.max(0, s.tacke.length - 1);   // start: cela istorija
        }
        const k = await jget('/api/mapa/slucajno' + (seed === null ? '' : `?seed=${seed}`));
        this.mapa.slucajno = k.tacke;
        this.mapa.seed = k.seed;
        // podaci mogu stići i pošto je mapa već gore
        if (mapaL) { this.mapaGradiSetove(); this.mapaCrtajTacke(); }
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    async mapaNoviUzorak() {
      // Isti broj tačaka, drugi seed — poenta je da slika svaki put izgleda isto tako.
      this.mapa.radiUzorak = true;
      this.mapaStani();
      try {
        await this.mapaUcitajTacke(Math.floor(Math.random() * 1e9));
        if (this.mapa.prikaz === 'stvarno') this.mapa.prikaz = 'slucajno';
        this.mapaStilSetova();
        this.mapaCrtajTacke();
      } finally { this.mapa.radiUzorak = false; }
    },

    mapaPrikazi(prikaz) { this.mapa.prikaz = prikaz; this.mapaStilSetova(); this.mapaCrtajTacke(); },

    mapaPoluprecnik() {
      // Na malom zumu tačke moraju biti sitne: 1.422 krupnih kružića prekriju mapu
      // i slika bi lagala da je prostor pun. Na velikom zumu smeju da porastu.
      const z = mapaL ? mapaL.getZoom() : 0;
      return Math.max(1.2, Math.min(4.5, 1.4 + (z - 1) * 0.7));
    },

    mapaAzurirajTacke() {
      const r = this.mapaPoluprecnik();
      Object.values(mapaSetovi).forEach(set => set && set.tacke.elem.forEach(m => m.setRadius(r)));
    },

    mapaBojaTacaka(ime) {
      // Kad se gleda jedan set, oba imaju IDENTIČAN stil — u tome je poenta kontrole.
      // Boje se razlikuju samo kad su oba seta na mapi istovremeno.
      if (this.mapa.prikaz !== 'oba') return '#f2f6fb';
      return ime === 'stvarno' ? BOJE.vruc : '#f2f6fb';
    },

    mapaCrtaSlucajne() {
      // Putanja je i za stvarno i za slučajno obojena preklapanjem; kad su oba na
      // mapi, isprekidana linija kaže samo KOJI je set, ne i nešto o podacima.
      return this.mapa.prikaz === 'oba' ? '3,4' : null;
    },

    mapaGradiSetove() {
      // Pravi sve markere i sve segmente za oba seta. Poziva se pri prvom crtanju,
      // pri novom platnu i pri novom slučajnom uzorku — nikad na korak vremena.
      if (!mapaL || !this.mapa.info) return;
      const MAXZ = this.mapa.info.max_zoom;
      for (const ime of ['stvarno', 'slucajno']) {
        const stari = mapaSetovi[ime];
        if (stari) { mapaL.removeLayer(stari.tacke.grupa); mapaL.removeLayer(stari.put.grupa); }
        mapaSetovi[ime] = null;
        const niz = this.mapa[ime];
        if (!niz || !niz.length) continue;
        const boja = this.mapaBojaTacaka(ime);
        const crta = ime === 'slucajno' ? this.mapaCrtaSlucajne() : null;
        const tacke = [], segmenti = [null];      // segment i spaja tačku i-1 sa tačkom i
        let prethodna = null;
        niz.forEach((t, i) => {
          // +0.5 da tačka stoji na sredini svoje ćelije, a ne u njenom uglu
          const p = mapaL.unproject([t.x + 0.5, t.y + 0.5], MAXZ);
          const m = L.circleMarker(p, { radius: this.mapaPoluprecnik(), weight: 0,
                                        fillColor: boja, fillOpacity: 0.9 });
          m.on('click', e => {
            L.DomEvent.stop(e);                   // da klik ne ode i mapi ispod
            this.mapaKlikTacka(ime, i);
          });
          m.bindTooltip(this.mapaOpisTacke(ime, i), { direction: 'top', opacity: 0.95 });
          tacke.push(m);
          if (prethodna) {
            segmenti.push(L.polyline([prethodna, p], {
              color: bojaPreklapanja(t.preklapanje_sa_prethodnim),
              weight: 1.6, opacity: 0.8, dashArray: crta, interactive: false,
              pane: 'putanja', renderer: mapaPutCrtac,
            }));
          }
          prethodna = p;
        });
        mapaSetovi[ime] = {
          tacke: { grupa: L.layerGroup(), elem: tacke, lo: 0, hi: -1 },
          put: { grupa: L.layerGroup(), elem: segmenti, lo: 1, hi: 0 },
        };
      }
    },

    mapaStilSetova() {
      // Promena prikaza (Stvarno/Slučajno/Oba) menja samo boju, ne i geometriju.
      const r = this.mapaPoluprecnik();
      for (const ime of ['stvarno', 'slucajno']) {
        const set = mapaSetovi[ime];
        if (!set) continue;
        const boja = this.mapaBojaTacaka(ime);
        set.tacke.elem.forEach(m => m.setStyle({ fillColor: boja, radius: r }));
        if (ime === 'slucajno') {
          const crta = this.mapaCrtaSlucajne();
          set.put.elem.forEach(l => l && l.setStyle({ dashArray: crta }));
        }
      }
    },

    mapaCrtajTacke() {
      if (!mapaL || !this.mapa.info) return;
      const MAXZ = this.mapa.info.max_zoom;
      const korak = this.mapaKorak();
      if (mapaSadL) { mapaL.removeLayer(mapaSadL); mapaSadL = null; }
      const sada = [];
      for (const ime of ['stvarno', 'slucajno']) {
        const set = mapaSetovi[ime];
        if (!set) continue;
        if (this.mapa.prikaz !== ime && this.mapa.prikaz !== 'oba') {
          mapaOpseg(set.tacke, 0, -1);
          mapaOpseg(set.put, 1, 0);
          mapaL.removeLayer(set.tacke.grupa);
          mapaL.removeLayer(set.put.grupa);
          continue;
        }
        if (!mapaL.hasLayer(set.put.grupa)) set.put.grupa.addTo(mapaL);      // putanja ide ispod tačaka
        if (!mapaL.hasLayer(set.tacke.grupa)) set.tacke.grupa.addTo(mapaL);
        const hi = Math.min(korak, set.tacke.elem.length - 1);
        mapaOpseg(set.tacke, 0, hi);
        const rep = Number(this.mapa.rep) || 0;                             // 0 = ceo put
        if (this.mapa.putanja && hi >= 1) mapaOpseg(set.put, rep ? Math.max(1, hi - rep + 1) : 1, hi);
        else mapaOpseg(set.put, 1, 0);
        if (hi >= 0 && !this.mapaNaKrajuVremena()) {
          const t = this.mapa[ime][hi];
          sada.push(L.circleMarker(mapaL.unproject([t.x + 0.5, t.y + 0.5], MAXZ),
            { radius: 7, color: '#f2f6fb', weight: 2, fill: false, interactive: false }));
        }
      }
      if (sada.length) mapaSadL = L.layerGroup(sada).addTo(mapaL);
    },

    // ---------- vreme na mapi: isti pojam „granica" kao u Istraži istoriju ----------
    mapaBrojKola() { return (this.mapa.stvarno || []).length; },

    mapaKorak() {
      const n = this.mapaBrojKola();
      if (!n) return -1;
      return Math.max(0, Math.min(Number(this.mapa.korak) || 0, n - 1));
    },

    mapaNaKrajuVremena() { return this.mapaKorak() >= this.mapaBrojKola() - 1; },

    mapaTekuceKolo() {
      const t = (this.mapa.stvarno || [])[this.mapaKorak()];
      return t ? t.kolo : null;
    },

    mapaVreme() { this.mapaStani(); this.mapaCrtajTacke(); },

    mapaPomeri(d) {
      const n = this.mapaBrojKola();
      if (!n) return;
      this.mapaStani();
      this.mapa.korak = Math.max(0, Math.min(this.mapaKorak() + d, n - 1));
      this.mapaCrtajTacke();
    },

    mapaVremeNaKolo(kolo) {
      const i = (this.mapa.stvarno || []).findIndex(t => t.kolo === kolo);
      if (i < 0) return;
      this.mapaStani();
      this.mapa.korak = i;
      this.mapaCrtajTacke();
    },

    mapaKorakAnimacije() {
      // Cela istorija prođe za oko pola minuta bez obzira na to koliko ima kola.
      return Math.max(1, Math.round(this.mapaBrojKola() / 400));
    },

    mapaPusti() {
      if (mapaTajmer) { this.mapaStani(); return; }
      const n = this.mapaBrojKola();
      if (!n) return;
      if (this.mapaNaKrajuVremena()) this.mapa.korak = 0;    // sa kraja kreni ispočetka
      this.mapa.animira = true;
      const d = this.mapaKorakAnimacije();
      mapaTajmer = setInterval(() => {
        if (this.strana !== 'mapa' || this.mapaNaKrajuVremena()) { this.mapaStani(); return; }
        this.mapa.korak = Math.min(this.mapaKorak() + d, this.mapaBrojKola() - 1);
        this.mapaCrtajTacke();
      }, 70);
      this.mapaCrtajTacke();
    },

    mapaStani() {
      if (mapaTajmer) { clearInterval(mapaTajmer); mapaTajmer = null; }
      this.mapa.animira = false;
    },

    mapaSveVreme() {
      this.mapaStani();
      this.mapa.korak = Math.max(0, this.mapaBrojKola() - 1);
      this.mapaCrtajTacke();
    },

    mapaOpisTacke(ime, i) {
      const t = this.mapa[ime][i];
      const glava = t.kolo != null ? `kolo ${this.formatKolo(t.kolo)}` : `slučajna tačka ${i + 1}`;
      if (t.preklapanje_sa_prethodnim == null) return glava + ' · prva u nizu';
      return `${glava} · deli ${t.preklapanje_sa_prethodnim} od 7 sa prethodnom`;
    },

    async mapaKlikTacka(ime, i) {
      const t = this.mapa[ime][i];
      await this.mapaUcitajCeliju(`/api/mapa/komb?x=${t.x}&y=${t.y}`, false);
    },

    mapaZaboraviPodatke() {
      // Novo kolo menja i broj tačaka i veličinu kontrolnog seta, pa se sve učitava
      // ponovo pri sledećem otvaranju mape. Bez ovoga bi slajder ostao na starom broju kola.
      this.mapaStani();
      for (const ime of ['stvarno', 'slucajno']) {
        const set = mapaSetovi[ime];
        if (set && mapaL) { mapaL.removeLayer(set.tacke.grupa); mapaL.removeLayer(set.put.grupa); }
        mapaSetovi[ime] = null;
      }
      if (mapaSadL && mapaL) { mapaL.removeLayer(mapaSadL); mapaSadL = null; }
      this.mapa.info = null;
      this.mapa.stvarno = null;
      this.mapa.slucajno = null;
      this.mapa.detalj = null;
      this.mapa.korak = 0;
      this.mapaOznaci();
      if (this.strana === 'mapa') this.ucitajMapu();
    },

    mapaRecenicaRazmere() {
      if (!this.mapa.info || !this.mapa.stvarno) return '';
      const n = this.mapa.stvarno.length, uk = this.mapa.info.ukupno_kombinacija;
      const vid = this.mapaKorak() + 1;
      const procenat = (100 * vid / uk).toFixed(3).replace('.', ',');
      const dokle = vid < n
        ? `sve što je izvučeno do kola ${this.formatKolo(this.mapaTekuceKolo())}, od ukupno ${n.toLocaleString('sr-RS')} izvučenih.`
        : 'sve što je ikad izvučeno.';
      return `Na mapi je ${vid.toLocaleString('sr-RS')} od ${uk.toLocaleString('sr-RS')} `
        + `mogućih kombinacija (${procenat}%) — ${dokle} Tačke stoje onako kako teorija predviđa `
        + `za slučajno izvlačenje; uključi „Slučajno" i uporedi dve slike.`;
    },

    async mapaKlik(e) {
      const MAXZ = this.mapa.info.max_zoom, DIM = this.mapa.info.dimenzija;
      const p = mapaL.project(e.latlng, MAXZ);
      const x = Math.floor(p.x), y = Math.floor(p.y);
      if (x < 0 || y < 0 || x >= DIM || y >= DIM) return;
      await this.mapaUcitajCeliju(`/api/mapa/komb?x=${x}&y=${y}`, false);
    },

    async mapaUcitajCeliju(url, centriraj) {
      try {
        this.mapa.detalj = await jget(url);
        this.mapaOznaci(centriraj);
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    mapaOznaci(centriraj = false) {
      const d = this.mapa.detalj;
      if (!mapaL) return;
      if (mapaOznakaL) { mapaL.removeLayer(mapaOznakaL); mapaOznakaL = null; }
      if (!d || d.rang === null) return;
      const MAXZ = this.mapa.info.max_zoom;
      const okvir = L.latLngBounds(mapaL.unproject([d.x, d.y], MAXZ),
                                   mapaL.unproject([d.x + 1, d.y + 1], MAXZ));
      const sredina = okvir.getCenter();
      mapaOznakaL = L.layerGroup([
        L.rectangle(okvir, { color: '#ff6b4a', weight: 2, fill: false }),
        L.circleMarker(sredina, { radius: 9, color: '#ff6b4a', weight: 2, fill: false }),
      ]).addTo(mapaL);
      if (centriraj) mapaL.setView(sredina, Math.max(mapaL.getZoom(), MAXZ + 2));
    },

    mapaZatvori() { this.mapa.detalj = null; this.mapaOznaci(); },

    mapaCelaSlika() { if (mapaL && mapaGranice) mapaL.fitBounds(mapaGranice); },

    async mapaPronadjiTiket() {
      const t = (this.mapa.tiket || '').trim();
      if (!t) { this.toast('Unesi 7 brojeva.', 'err'); return; }
      await this.mapaUcitajCeliju(`/api/mapa/rang?brojevi=${encodeURIComponent(t)}`, true);
    },

    async mapaSlucajna() {
      // Nasumična ćelija; prazan deo krive je oko 8%, pa je nekoliko pokušaja dovoljno.
      const DIM = this.mapa.info.dimenzija;
      for (let i = 0; i < 6; i++) {
        const x = Math.floor(Math.random() * DIM), y = Math.floor(Math.random() * DIM);
        await this.mapaUcitajCeliju(`/api/mapa/komb?x=${x}&y=${y}`, true);
        if (this.mapa.detalj && this.mapa.detalj.brojevi) {
          this.mapa.tiket = this.mapa.detalj.brojevi.join(' ');
          return;
        }
      }
    },

    mapaRazmera() {
      if (!this.mapa.info) return '';
      const mz = this.mapa.info.max_zoom, z = this.mapa.zum;
      if (z >= mz) {
        const px = Math.round(Math.pow(2, z - mz) * 10) / 10;
        return px === 1 ? '1 piksel = 1 kombinacija' : `1 kombinacija = ${px}×${px} piksela`;
      }
      const n = Math.round(Math.pow(4, mz - z));
      return `1 piksel = prosek ${n.toLocaleString('sr-RS')} kombinacija`;
    },

    mapaOsobineTekst() {
      const o = this.mapa.detalj && this.mapa.detalj.osobine;
      if (!o) return '—';
      return `zbir ${o.zbir} · raspon ${o.raspon} · parnih ${o.parni} · dekada ${o.dekade}`;
    },

    mapaKoloZaOtvaranje() {
      const d = this.mapa.detalj;
      if (!d || !d.preklapanje) return null;
      return d.izvucena.length ? d.izvucena[0] : (d.preklapanje.maks ? d.preklapanje.maks.kolo : null);
    },

    mapaPreklapanjeRecenica() {
      const d = this.mapa.detalj, p = d && d.preklapanje;
      if (!p) return '';
      const delovi = [];
      if (d.izvucena.length) {
        delovi.push(`Ova kombinacija je izvučena: ${d.izvucena.map(this.formatKolo).join(', ')}.`);
      } else {
        delovi.push('Ova kombinacija nikad nije izvučena.');
        if (p.maks) delovi.push(`Najviše se poklopila sa kolom ${this.formatKolo(p.maks.kolo)} — ${p.maks.k} od 7 brojeva.`);
      }
      // ako je baš to poslednje kolo, „deli 7 sa samim sobom" ne govori ništa
      if (p.sa_prethodnim && !d.izvucena.includes(p.sa_prethodnim.kolo)) {
        delovi.push(`Sa poslednjim kolom (${this.formatKolo(p.sa_prethodnim.kolo)}) deli ${p.sa_prethodnim.k}.`);
      }
      delovi.push(`Za slučajnu kombinaciju očekuje se ${p.teorija.ocekivano.toFixed(2)} zajedničkih brojeva (σ ${p.teorija.sigma.toFixed(2)}).`);
      return delovi.join(' ');
    },

    // ---------- sekcija „Test": dužine skokova po mapi ----------
    mapaTestToggle() {
      this.mapa.test.otvoren = !this.mapa.test.otvoren;
      if (this.mapa.test.otvoren && !this.mapa.test.podaci) this.mapaUcitajSkokove();
      else if (this.mapa.test.otvoren) this.$nextTick(() => this.crtajSkokove());
    },

    async mapaUcitajSkokove() {
      const granica = this.mapaNaKrajuVremena() ? null : this.mapaTekuceKolo();
      this.mapa.test.radi = true;
      try {
        this.mapa.test.podaci = await jget('/api/mapa/skokovi' + (granica ? `?granica=${granica}` : ''));
        this.mapa.test.granica = granica;
        this.$nextTick(() => this.crtajSkokove());
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
      finally { this.mapa.test.radi = false; }
    },

    mapaTestZastareo() {
      // Slajder je pomeren posle učitavanja: histogram više ne opisuje ono što se vidi.
      if (!this.mapa.test.podaci) return false;
      const sada = this.mapaNaKrajuVremena() ? null : this.mapaTekuceKolo();
      return sada !== this.mapa.test.granica;
    },

    crtajSkokove(pokusaj = 0) {
      const d = this.mapa.test.podaci;
      if (!d) return;
      // x-show otkriva sekciju tek u sledećem frejmu, pa platno ume da ima visinu 0
      // baš u trenutku prvog crtanja; ECharts bi tada ostao prazan zauvek.
      const el = document.getElementById('mapa-skokovi');
      if (!el || !el.clientHeight || !el.clientWidth) {
        if (this.mapa.test.otvoren && pokusaj < 40) setTimeout(() => this.crtajSkokove(pokusaj + 1), 40);
        return;
      }
      crtaj('mapa-skokovi', {
        ...bazaOpcija(), grid: { left: 48, right: 18, top: 30, bottom: 44 },
        legend: { top: 2, textStyle: { color: BOJE.tekst, fontSize: 11 },
                  data: ['stvarna kola', 'slučajno (kontrola)'] },
        tooltip: { trigger: 'axis', backgroundColor: '#1c2330', borderColor: '#262d3a',
          textStyle: { color: '#e6edf3' },
          formatter: p => `skok oko ${Number(p[0].axisValue).toLocaleString('sr-RS')} ćelija<br>`
            + p.map(x => `${x.marker}${x.seriesName}: ${x.value}%`).join('<br>') },
        xAxis: { type: 'category', data: d.sredine, name: 'dužina skoka (ćelija)',
          nameLocation: 'middle', nameGap: 30, axisLabel: { fontSize: 9, interval: 3 },
          axisLine: { lineStyle: { color: BOJE.mreza } } },
        yAxis: { type: 'value', axisLabel: { formatter: '{value}%' },
          splitLine: { lineStyle: { color: BOJE.mreza } } },
        series: [
          { name: 'stvarna kola', type: 'bar', data: d.stvarno.udeli,
            itemStyle: { color: BOJE.accent, borderRadius: [3, 3, 0, 0] } },
          { name: 'slučajno (kontrola)', type: 'line', data: d.slucajno.udeli,
            symbol: 'circle', symbolSize: 6, smooth: true,
            lineStyle: { color: BOJE.vruc, width: 2 }, itemStyle: { color: BOJE.vruc } },
        ],
      });
    },

    mapaSkokoviRecenica() {
      const d = this.mapa.test.podaci;
      if (!d || !d.stvarno.broj) return '';
      const br = v => Math.round(v).toLocaleString('sr-RS');
      return `Prosečan skok je ${br(d.stvarno.prosek)} ćelija za stvarna kola i `
        + `${br(d.slucajno.prosek)} za kontrolni set (medijane ${br(d.stvarno.medijana)} i `
        + `${br(d.slucajno.medijana)}), na ${d.stvarno.broj.toLocaleString('sr-RS')} koraka. `
        + 'Dužina skoka je druga mera razlike dva uzastopna kola, pa je ovo isti test kao na '
        + 'strani „Različitost", samo u drugom obliku: dok se dve raspodele poklapaju, kola se '
        + 'po mapi pomeraju isto kao slučajan izbor.';
    },

    mapaOtvoriKolo(kolo) {
      if (kolo == null) return;
      this.ist.granica = kolo;
      this.ist.broj = null;
      this.ist.detalj = null;
      this.idi('istorija');
    },

    // ---------- DASHBOARD ----------
    async ucitajDashboard() {
      this.loading = true;
      try {
        this.dash = await jget(`/api/dashboard?period=${this.period}`);
        this.brojKola = this.dash.broj_kola;
        const stat = await jget(`/api/statistika?period=${this.period}`);
        this.$nextTick(() => {
          const f = stat.frekvencija;
          crtaj('dash-freq', {
            ...bazaOpcija(),
            xAxis: { type: 'category', data: f.map(x => x.broj), axisLine: { lineStyle: { color: BOJE.mreza } }, axisLabel: { fontSize: 9 } },
            yAxis: { type: 'value', splitLine: { lineStyle: { color: BOJE.mreza } } },
            series: [{ type: 'bar', data: f.map(x => ({ value: x.frekvencija, itemStyle: { color: BOJE[x.kategorija], borderRadius: [3, 3, 0, 0] } })) }],
          });
        });
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
      this.loading = false;
    },

    // ---------- STATISTIKA ----------
    async ucitajStatistiku() {
      this.loading = true; this.hik = null;
      try {
        this.stat = await jget(`/api/statistika?period=${this.period}`);
        this.$nextTick(() => this.crtajStatistiku());
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
      this.loading = false;
    },

    crtajStatistiku() {
      const s = this.stat;
      const f = s.frekvencija;
      crtaj('st-freq', {
        ...bazaOpcija(), grid: { left: 44, right: 18, top: 16, bottom: 30 },
        xAxis: { type: 'category', data: f.map(x => x.broj), axisLine: { lineStyle: { color: BOJE.mreza } }, axisLabel: { fontSize: 9 } },
        yAxis: { type: 'value', splitLine: { lineStyle: { color: BOJE.mreza } } },
        series: [{ type: 'bar', data: f.map(x => ({ value: x.frekvencija, itemStyle: { color: BOJE[x.kategorija], borderRadius: [3, 3, 0, 0] } })) }],
      });

      // Histogram srednjih vrednosti
      const sv = s.srednje_vrednosti;
      const binN = 20, minv = Math.min(...sv), maxv = Math.max(...sv), sirina = (maxv - minv) / binN || 1;
      const binovi = new Array(binN).fill(0), labele = [];
      sv.forEach(v => { let i = Math.min(binN - 1, Math.floor((v - minv) / sirina)); binovi[i]++; });
      for (let i = 0; i < binN; i++) labele.push((minv + i * sirina).toFixed(1));
      crtaj('st-sv', {
        ...bazaOpcija(),
        xAxis: { type: 'category', data: labele, axisLine: { lineStyle: { color: BOJE.mreza } }, axisLabel: { fontSize: 9, interval: 3 } },
        yAxis: { type: 'value', splitLine: { lineStyle: { color: BOJE.mreza } } },
        series: [{ type: 'bar', data: binovi, itemStyle: { color: BOJE.accent, borderRadius: [3, 3, 0, 0] } }],
      });

      // Trend
      const ts = s.vremenska_serija;
      crtaj('st-trend', {
        ...bazaOpcija(),
        xAxis: { type: 'category', data: ts.kola, axisLine: { lineStyle: { color: BOJE.mreza } }, axisLabel: { fontSize: 9 } },
        yAxis: { type: 'value', splitLine: { lineStyle: { color: BOJE.mreza } } },
        dataZoom: [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 6 }],
        series: [{ type: 'line', data: ts.vrednosti, smooth: true, symbol: 'none', lineStyle: { color: BOJE.svez, width: 1 }, areaStyle: { color: 'rgba(53,208,127,.08)' } }],
      });

      // Ritam
      const ritam = Object.entries(s.ritam).sort((a, b) => a[0] - b[0]);
      crtaj('st-ritam', {
        ...bazaOpcija(),
        xAxis: { type: 'category', data: ritam.map(r => r[0]), axisLine: { lineStyle: { color: BOJE.mreza } }, axisLabel: { fontSize: 8 } },
        yAxis: { type: 'value', splitLine: { lineStyle: { color: BOJE.mreza } } },
        series: [{ type: 'bar', data: ritam.map(r => r[1]), itemStyle: { color: '#8a5cff', borderRadius: [3, 3, 0, 0] } }],
      });

      // Uzastopni
      const uz = Object.entries(s.uzastopni).sort((a, b) => a[0] - b[0]);
      crtaj('st-uzastopni', {
        ...bazaOpcija(),
        xAxis: { type: 'category', data: uz.map(u => u[0]), axisLine: { lineStyle: { color: BOJE.mreza } } },
        yAxis: { type: 'value', splitLine: { lineStyle: { color: BOJE.mreza } } },
        series: [{ type: 'bar', data: uz.map(u => u[1]), itemStyle: { color: BOJE.hladan, borderRadius: [3, 3, 0, 0] } }],
      });

      // Dekade
      const dek = Object.entries(s.dekade);
      crtaj('st-dekade', {
        ...bazaOpcija(), tooltip: { trigger: 'item' },
        xAxis: { type: 'category', data: dek.map(d => d[0]), axisLine: { lineStyle: { color: BOJE.mreza } } },
        yAxis: { type: 'value', splitLine: { lineStyle: { color: BOJE.mreza } } },
        series: [{ type: 'bar', data: dek.map(d => d[1]), itemStyle: { color: BOJE.vruc, borderRadius: [3, 3, 0, 0] } }],
      });

      // Heatmap poziciona
      const pz = s.poziciona;
      const podaci = [];
      pz.vrednosti.forEach((red, bi) => red.forEach((v, pi) => podaci.push([pi, bi, v])));
      const maxv2 = Math.max(...podaci.map(d => d[2]));
      crtaj('st-heat', {
        backgroundColor: BOJE.pozadina, textStyle: { color: BOJE.tekst },
        tooltip: { position: 'top', formatter: p => `Broj ${pz.brojevi[p.data[1]]}, poz. ${pz.pozicije[p.data[0]]}: ${p.data[2]}` },
        grid: { left: 40, right: 20, top: 10, bottom: 50 },
        xAxis: { type: 'category', data: pz.pozicije, name: 'Pozicija', axisLine: { lineStyle: { color: BOJE.mreza } } },
        yAxis: { type: 'category', data: pz.brojevi, axisLabel: { fontSize: 8, interval: 1 }, axisLine: { lineStyle: { color: BOJE.mreza } } },
        visualMap: { min: 0, max: maxv2, calculable: true, orient: 'horizontal', left: 'center', bottom: 6,
          inRange: { color: ['#161b22', '#1e2b45', '#4f8cff', '#ff6b4a'] }, textStyle: { color: BOJE.tekst } },
        series: [{ type: 'heatmap', data: podaci }],
      });
    },

    async pokreniHiKvadrat() {
      try { this.hik = await jget('/api/hi-kvadrat'); } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    // ---------- RAZLIČITOST ----------
    async ucitajRazlicitost() {
      this.loading = true; this.razl.detaljPar = null;
      try {
        if (!this.dash) { try { this.dash = await jget('/api/dashboard?period=0'); } catch (e) {} }
        this.razl.podaci = await jget(`/api/razlicitost?period=${this.period}`);
        this.$nextTick(() => this.crtajRazlicitost());
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
      this.loading = false;
    },

    testTekst(t) {
      if (!t || t.p == null) return 'Premalo podataka za test (očekivana frekvencija < 5).';
      return t.p < 0.05
        ? `χ² = ${t.chi2}, df = ${t.df}, p = ${t.p} → odstupa od slučajnosti (proveriti podatke!).`
        : `χ² = ${t.chi2}, df = ${t.df}, p = ${t.p} → nerazlučivo od čiste slučajnosti.`;
    },

    crtajRazlicitost() {
      const d = this.razl.podaci;
      if (!d || !d.dovoljno_podataka) return;
      this.crtHistogram('razl-uzastopna', d.uzastopna.histogram);
      this.crtHistogram('razl-parovi', d.svi_parovi.histogram);
      this.crtScatter();
      this.crtHeat();
    },

    crtHistogram(id, h) {
      crtaj(id, {
        ...bazaOpcija(), grid: { left: 48, right: 18, top: 30, bottom: 34 },
        legend: { top: 2, textStyle: { color: BOJE.tekst, fontSize: 11 }, data: ['posmatrano', 'slučajnost (teorija)'] },
        tooltip: { trigger: 'axis', backgroundColor: '#1c2330', borderColor: '#262d3a', textStyle: { color: '#e6edf3' },
          formatter: p => `k = ${p[0].axisValue}<br>` + p.map(s => `${s.marker}${s.seriesName}: ${s.value}%`).join('<br>') },
        xAxis: { type: 'category', data: h.k, name: 'zajedničkih brojeva (k)', nameLocation: 'middle', nameGap: 24,
          axisLine: { lineStyle: { color: BOJE.mreza } } },
        yAxis: { type: 'value', axisLabel: { formatter: '{value}%' }, splitLine: { lineStyle: { color: BOJE.mreza } } },
        series: [
          { name: 'posmatrano', type: 'bar', data: h.posmatrano_udeo,
            itemStyle: { color: BOJE.accent, borderRadius: [3, 3, 0, 0] } },
          { name: 'slučajnost (teorija)', type: 'line', data: h.teorija_udeo, symbol: 'circle', symbolSize: 7,
            lineStyle: { color: BOJE.vruc, width: 2 }, itemStyle: { color: BOJE.vruc } },
        ],
      });
    },

    async promeniProfil() {
      try {
        const p = await jget(`/api/razlicitost/profil?period=${this.period}&tip=${this.razl.profilTip}`);
        this.razl.podaci.profil = p;
        this.$nextTick(() => this.crtScatter());
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    crtScatter() {
      const pf = this.razl.podaci.profil;
      crtaj('razl-scatter', {
        ...bazaOpcija(), grid: { left: 48, right: 18, top: 20, bottom: 40 },
        tooltip: { trigger: 'item', backgroundColor: '#1c2330', borderColor: '#262d3a', textStyle: { color: '#e6edf3' },
          formatter: p => p.seriesName === 'prosek po binu'
            ? `Δprofil ≈ ${p.data[0]}: prosek ${p.data[1]}` : `Δprofil ${p.data[0]}, preklapanje ${Math.round(p.data[1])}` },
        xAxis: { type: 'value', name: pf.profili[pf.tip], nameLocation: 'middle', nameGap: 26,
          axisLine: { lineStyle: { color: BOJE.mreza } }, splitLine: { show: false } },
        yAxis: { type: 'value', name: 'zajedničkih', min: -0.5, max: 7.5,
          splitLine: { lineStyle: { color: BOJE.mreza } } },
        series: [
          { name: 'parovi', type: 'scatter', data: pf.tacke, symbolSize: 3,
            itemStyle: { color: 'rgba(79,140,255,.35)' } },
          { name: 'prosek po binu', type: 'line', data: pf.linija_proseka, symbol: 'circle', symbolSize: 5,
            lineStyle: { color: BOJE.vruc, width: 2 }, itemStyle: { color: BOJE.vruc },
            markLine: { silent: true, symbol: 'none', lineStyle: { color: BOJE.svez, type: 'dashed' },
              data: [{ yAxis: pf.referenca, label: { formatter: 'μ = ' + pf.referenca, color: BOJE.svez, position: 'insideEndTop' } }] } },
        ],
      });
    },

    crtHeat() {
      const ko = this.razl.podaci.ko_okurencija;
      const N = ko.matrica_z.length - 1;
      const podaci = [];
      let apsmax = 0.001;
      for (let a = 1; a <= N; a++) for (let b = 1; b <= N; b++) {
        if (a === b) continue;
        const z = ko.matrica_z[a][b];
        podaci.push([b - 1, a - 1, z]);
        apsmax = Math.max(apsmax, Math.abs(z));
      }
      const brojevi = Array.from({ length: N }, (_, i) => i + 1);
      crtaj('razl-heat', {
        backgroundColor: BOJE.pozadina, textStyle: { color: BOJE.tekst },
        tooltip: { position: 'top', formatter: p => `Par ${p.data[1] + 1} & ${p.data[0] + 1}<br>z-skor: ${p.data[2]}<br><small>klik za detalje</small>` },
        grid: { left: 34, right: 20, top: 10, bottom: 60 },
        xAxis: { type: 'category', data: brojevi, axisLabel: { fontSize: 7, interval: 1 }, axisLine: { lineStyle: { color: BOJE.mreza } } },
        yAxis: { type: 'category', data: brojevi, axisLabel: { fontSize: 7, interval: 1 }, axisLine: { lineStyle: { color: BOJE.mreza } } },
        visualMap: { min: -apsmax, max: apsmax, calculable: true, orient: 'horizontal', left: 'center', bottom: 8,
          inRange: { color: ['#4f8cff', '#161b22', '#ff6b4a'] }, textStyle: { color: BOJE.tekst },
          text: ['češće (crveno)', 'ređe (plavo)'] },
        series: [{ type: 'heatmap', data: podaci, progressive: 2000,
          emphasis: { itemStyle: { borderColor: '#e6edf3', borderWidth: 1 } } }],
      });
      const el = document.getElementById('razl-heat');
      const g = echarts.getInstanceByDom(el);
      if (g) { g.off('click'); g.on('click', p => this.klikCelija(p.data[1] + 1, p.data[0] + 1)); }
    },

    async klikCelija(a, b) {
      try { this.razl.detaljPar = await jget(`/api/razlicitost/par?a=${a}&b=${b}&period=${this.period}`); }
      catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    // ---------- RANGIRANJE ----------
    async ucitajRang() {
      try {
        const d = await jget(`/api/rangiranje?metoda=${this.metoda}`);
        this.rang = d.rang;
        this.$nextTick(() => {
          const r = [...this.rang].sort((a, b) => a.broj - b.broj);
          crtaj('rang-chart', {
            ...bazaOpcija(),
            tooltip: { trigger: 'axis', backgroundColor: '#1c2330', borderColor: '#262d3a', textStyle: { color: '#e6edf3' } },
            xAxis: { type: 'category', data: r.map(x => x.broj), axisLine: { lineStyle: { color: BOJE.mreza } }, axisLabel: { fontSize: 9 } },
            yAxis: { type: 'value', splitLine: { lineStyle: { color: BOJE.mreza } } },
            series: [{ type: 'bar', data: r.map(x => x.skor), itemStyle: { color: BOJE.accent, borderRadius: [3, 3, 0, 0] } }],
          });
        });
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    // ---------- GENERATOR ----------
    async generisi() {
      this.gen.radi = true; this.gen.rezultati = []; this.gen.razlicitost = null;
      try {
        const filteri = {
          min_sv: this.gen.min_sv, max_sv: this.gen.max_sv,
          strategija_svezine: this.gen.strategija, primeni_pristrasnost: this.gen.pristrasnost,
          filtriraj_unikate: this.gen.unikati,
        };
        if (this.gen.f_parni) filteri.parni = this.gen.parni;
        if (this.gen.f_vruci) filteri.vruci = this.gen.vruci;
        if (this.gen.f_hladni) filteri.hladni = this.gen.hladni;
        if (this.gen.f_uzastopni) filteri.uzastopni = this.gen.uzastopni;
        if (this.gen.f_dekada) filteri.dekada_max = this.gen.dekada_max;
        if (this.gen.diverzitet) { filteri.diverzitet = true; filteri.max_slicnost = this.gen.max_slicnost; }

        let bazen = null;
        if (this.gen.koristiBazen && this.gen.bazenText.trim()) {
          bazen = this.gen.bazenText.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x));
        }
        const granica = this.gen.granica || null;
        const d = await jsend('/api/generator', 'POST', { period: this.period, bazen, filteri, granica });
        this.gen.rezultati = d.kombinacije;
        this.gen.razlicitost = d.razlicitost;
        const doKola = d.granica ? ` · analiza do ${this.formatKolo(d.granica)} (${d.broj_kola} kola)` : '';
        this.gen.rezime = `Validnih: ${d.ukupno_validnih} · posle diverziteta: ${d.posle_diverziteta} · prikazano ${d.kombinacije.length}${doKola}`;
        if (!d.kombinacije.length) this.toast('Nijedna kombinacija ne zadovoljava filtere.', 'warn');
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
      this.gen.radi = false;
    },

    bazenUGenerator(bazen) {
      this.gen.koristiBazen = true;
      this.gen.bazenText = bazen.join(', ');
      this.idi('generator');
      this.toast('Bazen prebačen u Generator.', 'ok');
    },

    // Vremeplov → Generator: generiši sa znanjem samo do granice (Faza 5)
    generisiDoKola() {
      this.gen.granica = this.ist.granica;
      this.idi('generator');
      this.toast(`Generator će analizirati samo do kola ${this.formatKolo(this.ist.granica)}.`, 'ok');
    },

    // BrojDetalj → skok na drugi tab sa istaknutim brojem (Faza 5)
    skociNaBroj(broj, tab) {
      this.idi(tab);
      // istakni stubić broja u grafikonu ciljnog taba (indeks = broj - 1, brojevi 1..39)
      this.$nextTick(() => setTimeout(() => {
        const id = tab === 'statistika' ? 'st-freq' : 'rang-chart';
        const g = window.echarts && echarts.getInstanceByDom(document.getElementById(id));
        if (g) {
          g.dispatchAction({ type: 'highlight', seriesIndex: 0, dataIndex: broj - 1 });
          g.dispatchAction({ type: 'showTip', seriesIndex: 0, dataIndex: broj - 1 });
        }
      }, 350));
      this.toast(`Broj ${broj} istaknut u tabu.`, 'ok');
    },

    async dodajTiketIz(brojevi) {
      const s = '(' + brojevi.join(', ') + ')';
      try { const d = await jsend('/api/tiketi', 'POST', { kombinacija: s }); this.toast(d.dodato ? 'Tiket dodat.' : 'Tiket već postoji.', d.dodato ? 'ok' : 'warn'); }
      catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    // ---------- PROGNOZA ----------
    async ucitajPrognozu() {
      try {
        if (!this.dash) { try { this.dash = await jget('/api/dashboard?period=0'); } catch (e) {} }
        const d = await jget(`/api/prognoza/predlozi?period=${this.period}`);
        this.prog.ciljnoKolo = d.ciljno_kolo;
        this.prog.predlozi = d.predlozi;
        await this.progUcitajRezultate();
        await this.progUcitajIstoriju();
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    progNaziv(metod) {
      const p = this.prog.predlozi.find(x => x.metod === metod);
      return p ? p.naziv : metod;
    },

    async progPreracunaj() {
      try {
        const d = await jsend(`/api/prognoza/preracunaj?period=${this.period}`, 'POST');
        this.prog.ciljnoKolo = d.ciljno_kolo;
        this.prog.predlozi = d.predlozi;
        this.toast('Predlozi preračunati (ocenjeni ostaju zaključani).', 'ok');
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    async progRetro() {
      if (!confirm('Retro-bektest briše postojeće retro rezultate i računa iznova nad celom istorijom.\nNastaviti?')) return;
      this.prog.radi = true;
      try {
        const d = await jsend('/api/prognoza/retro', 'POST');
        this.toast(`Retro-bektest gotov: ${d.kola_ocenjeno} kola, ${d.redova} prognoza za ${d.trajanje_s}s.`, 'ok');
        this.prog.izvor = 'retro';
        await this.progUcitajRezultate();
        await this.progUcitajIstoriju();
        // retro računa i kombinacijske — osveži i taj tab ako je učitan
        this.progK.izvor = 'retro';
        if (this.progK.ucitano) {
          await this.progKUcitajRezultate();
          await this.progKUcitajHistogram();
          await this.progKUcitajIstoriju();
        }
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
      this.prog.radi = false;
    },

    async progUcitajRezultate() {
      try {
        const d = await jget(`/api/prognoza/rezultati?izvor=${this.prog.izvor}`);
        this.prog.statistika = d.statistika.metode;
        this.prog.prag = Math.round(d.statistika.prag * 10000) / 10000;
        this.prog.brojMetoda = d.statistika.broj_metoda;
        this.$nextTick(() => this.progCrtaj(d.grafikon));
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    progCrtaj(g) {
      const bojeMetoda = {
        hot: '#ff6b4a', cold: '#4f8cff', bayes: '#8a5cff', hybrid: '#35d07f',
        rhythm: '#f2b955', fresh: '#ff5c9d', random: '#9aa7b5',
      };
      const x = Array.from({ length: g.n_max }, (_, i) => i + 1);
      const serije = [
        // pojas pouzdanosti (stack trik: donja transparentna + razlika osenčena)
        { name: 'pojas-donja', type: 'line', data: g.pojas_donja, stack: 'pojas', symbol: 'none',
          lineStyle: { opacity: 0 }, silent: true, tooltip: { show: false }, showInLegend: false },
        { name: 'pojas 95%', type: 'line', data: g.pojas_gornja.map((v, i) => Math.round((v - g.pojas_donja[i]) * 100) / 100),
          stack: 'pojas', symbol: 'none', lineStyle: { opacity: 0 },
          areaStyle: { color: 'rgba(154,167,181,.10)' }, silent: true, tooltip: { show: false } },
        // baseline
        { name: 'slučajnost 17,95%', type: 'line', symbol: 'none',
          data: x.map(() => g.baseline), lineStyle: { color: '#9aa7b5', type: 'dashed', width: 1.5 } },
      ];
      for (const [metod, podaci] of Object.entries(g.serije)) {
        if (!podaci.length) continue;
        serije.push({
          name: g.nazivi[metod], type: 'line', data: podaci, symbol: 'none', smooth: false,
          lineStyle: { color: bojeMetoda[metod] || '#4f8cff', width: metod === 'random' ? 1.5 : 2,
                       type: metod === 'random' ? 'dotted' : 'solid' },
        });
      }
      crtaj('prog-chart', {
        ...bazaOpcija(),
        grid: { left: 44, right: 18, top: 46, bottom: 34 },
        legend: { top: 4, textStyle: { color: '#9aa7b5', fontSize: 11 },
                  data: serije.filter(s => !s.silent).map(s => s.name) },
        xAxis: { type: 'category', data: x, name: 'ocenjeno kola',
                 axisLine: { lineStyle: { color: BOJE.mreza } }, axisLabel: { fontSize: 9 } },
        yAxis: { type: 'value', axisLabel: { formatter: '{value}%' }, max: 60,
                 splitLine: { lineStyle: { color: BOJE.mreza } } },
        series: serije,
      });
    },

    async progUcitajIstoriju() {
      try {
        this.prog.istorija = await jget(`/api/prognoza/istorija?izvor=${this.prog.izvor}&metod=${this.prog.filterMetod}&limit=50`);
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    // ---------- PROGNOZA / KOMBINACIJA ----------
    async ucitajPrognozuKomb() {
      try {
        if (!this.dash) { try { this.dash = await jget('/api/dashboard?period=0'); } catch (e) {} }
        const d = await jget(`/api/prognoza/komb/predlozi?period=${this.period}`);
        this.progK.ciljnoKolo = d.ciljno_kolo;
        this.progK.predlozi = d.predlozi;
        if (!this.progK.histMetod && d.predlozi.length) this.progK.histMetod = d.predlozi[0].metod;
        this.progK.ucitano = true;
        await this.progKUcitajRezultate();
        await this.progKUcitajHistogram();
        await this.progKUcitajIstoriju();
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    progKNaziv(metod) { const p = this.progK.predlozi.find(x => x.metod === metod); return p ? p.naziv : metod; },

    async progKPreracunaj() {
      try {
        const d = await jsend(`/api/prognoza/komb/preracunaj?period=${this.period}`, 'POST');
        this.progK.ciljnoKolo = d.ciljno_kolo;
        this.progK.predlozi = d.predlozi;
        this.toast('Kombinacijski predlozi preračunati (ocenjeni ostaju zaključani).', 'ok');
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    async progKUcitajRezultate() {
      try {
        const d = await jget(`/api/prognoza/komb/rezultati?izvor=${this.progK.izvor}`);
        this.progK.statistika = d.statistika.metode;
        this.progK.prag = Math.round(d.statistika.prag * 100000) / 100000;
        this.progK.brojMetoda = d.statistika.broj_metoda;
        this.progK.ocekivano = d.statistika.ocekivano;
        this.progK.sigma = d.statistika.sigma;
        this.$nextTick(() => this.progKCrtaj(d.grafikon));
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    progKCrtaj(g) {
      const boje = {
        k_hot7: '#ff6b4a', k_cold7: '#4f8cff', k_bayes7: '#8a5cff', k_hybrid7: '#35d07f',
        k_rhythm7: '#f2b955', k_cooc: '#ff5c9d', k_random: '#9aa7b5',
      };
      const x = Array.from({ length: g.n_max }, (_, i) => i + 1);
      const serije = [
        { name: 'pojas-donja', type: 'line', data: g.pojas_donja, stack: 'pojas', symbol: 'none',
          lineStyle: { opacity: 0 }, silent: true, tooltip: { show: false }, showInLegend: false },
        { name: 'pojas 95%', type: 'line', data: g.pojas_gornja.map((v, i) => Math.round((v - g.pojas_donja[i]) * 1000) / 1000),
          stack: 'pojas', symbol: 'none', lineStyle: { opacity: 0 },
          areaStyle: { color: 'rgba(154,167,181,.10)' }, silent: true, tooltip: { show: false } },
        { name: 'slučajnost μ=' + g.baseline, type: 'line', symbol: 'none',
          data: x.map(() => g.baseline), lineStyle: { color: '#9aa7b5', type: 'dashed', width: 1.5 } },
      ];
      for (const [metod, podaci] of Object.entries(g.serije)) {
        if (!podaci.length) continue;
        serije.push({
          name: g.nazivi[metod], type: 'line', data: podaci, symbol: 'none',
          lineStyle: { color: boje[metod] || '#4f8cff', width: metod === 'k_random' ? 1.5 : 2,
                       type: metod === 'k_random' ? 'dotted' : 'solid' },
        });
      }
      crtaj('progk-chart', {
        ...bazaOpcija(),
        grid: { left: 44, right: 18, top: 46, bottom: 34 },
        legend: { top: 4, textStyle: { color: '#9aa7b5', fontSize: 11 }, data: serije.filter(s => !s.silent).map(s => s.name) },
        xAxis: { type: 'category', data: x, name: 'ocenjeno kola', axisLine: { lineStyle: { color: BOJE.mreza } }, axisLabel: { fontSize: 9 } },
        yAxis: { type: 'value', name: 'prosek preklapanja', min: 0,
                 splitLine: { lineStyle: { color: BOJE.mreza } } },
        series: serije,
      });
    },

    async progKUcitajHistogram() {
      if (!this.progK.histMetod) return;
      try {
        this.progK.hist = await jget(`/api/prognoza/komb/histogram?izvor=${this.progK.izvor}&metod=${this.progK.histMetod}`);
        this.$nextTick(() => this.crtHistogram('progk-hist', this.progK.hist.histogram));
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    async progKUcitajIstoriju() {
      try {
        this.progK.istorija = await jget(`/api/prognoza/komb/istorija?izvor=${this.progK.izvor}&metod=${this.progK.filterMetod}&limit=50`);
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },

    // ---------- BEKTEST ----------
    async ucitajBektest() { try { this.bektestovi = await jget('/api/bektest'); } catch (e) { this.toast('Greška: ' + e.message, 'err'); } },
    opisBektesta(b) { try { return JSON.parse(b.filter_podesavanja).opis || JSON.parse(b.filter_podesavanja).tip || '—'; } catch { return (b.filter_podesavanja || '—').slice(0, 40); } },
    async obrisiBektest(id) { try { await jsend('/api/bektest/' + id, 'DELETE'); this.ucitajBektest(); this.toast('Obrisano.', 'ok'); } catch (e) { this.toast('Greška: ' + e.message, 'err'); } },

    // ---------- TIKETI ----------
    async ucitajTikete() { try { this.tiketi = await jget('/api/tiketi'); } catch (e) { this.toast('Greška: ' + e.message, 'err'); } },
    async dodajTiket() {
      let t = this.noviTiket.trim();
      if (!t) return;
      if (!t.startsWith('(')) t = '(' + t.split(',').map(x => x.trim()).join(', ') + ')';
      try { const d = await jsend('/api/tiketi', 'POST', { kombinacija: t }); this.noviTiket = ''; this.ucitajTikete(); this.toast(d.dodato ? 'Tiket dodat.' : 'Već postoji.', d.dodato ? 'ok' : 'warn'); }
      catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },
    async obrisiTiket(id) { try { await jsend('/api/tiketi/' + id, 'DELETE'); this.ucitajTikete(); this.toast('Obrisano.', 'ok'); } catch (e) { this.toast('Greška: ' + e.message, 'err'); } },

    // ---------- PODACI ----------
    async ucitajIstoriju() { try { this.istorija = await jget('/api/istorija?limit=60'); } catch (e) { this.toast('Greška: ' + e.message, 'err'); } },
    async dodajKolo() {
      try {
        const brojevi = this.unos.brojevi.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x));
        const d = await jsend('/api/istorija', 'POST', { kolo: this.unos.kolo, datum: this.unos.datum, brojevi });
        this.toast(d.dodato ? `Kolo ${d.kolo} dodato. Provereno tiketa: ${d.provereno_tiketa}, bektestova: ${d.provereno_bektestova}, ocenjeno prognoza: ${d.ocenjeno_prognoza}.` : `Kolo već postoji; provere ažurirane.`, d.dodato ? 'ok' : 'warn');
        this.unos.brojevi = '';
        this.mapaZaboraviPodatke();
        this.ucitajIstoriju();
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },
    izabranFajl(ev) { this.fajl = ev.target.files[0] || null; },
    async uvezi() {
      if (!this.fajl) return;
      if (this.uvozZameni && !confirm(`Ovo će obrisati svih ${this.brojKola} kola i uvesti nova iz fajla.\nPre brisanja se pravi rezervna kopija.\n\nNastaviti?`)) return;
      const fd = new FormData(); fd.append('fajl', this.fajl);
      try {
        const r = await fetch('/api/uvoz?zameni=' + (this.uvozZameni ? 'true' : 'false'), { method: 'POST', body: fd });
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || 'Greška');
        const poruka = d.zamenjeno
          ? `Istorija zamenjena: obrisano ${d.obrisano}, uvezeno ${d.uvezeno}. Kopija: ${d.backup}`
          : `Uvezeno ${d.uvezeno} od ${d.ukupno_u_fajlu} kola.`;
        this.toast(poruka, 'ok');
        this.fajl = null; this.uvozZameni = false;
        const dd = await jget('/api/dashboard?period=0'); this.brojKola = dd.broj_kola;
        this.mapaZaboraviPodatke();
        this.ucitajIstoriju();
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },
  };
}
