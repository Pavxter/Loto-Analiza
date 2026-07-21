// Loto Analizator — frontend logika (Alpine + ECharts)

const BOJE = {
  vruc: '#ff6b4a', hladan: '#4f8cff', neutralan: '#6b7686', svez: '#35d07f',
  accent: '#4f8cff', tekst: '#9aa7b5', mreza: '#262d3a', pozadina: 'transparent',
};

const grafikoni = {};   // id -> echarts instanca

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
      { id: 'statistika', naziv: 'Statistika', ico: '📊', opis: 'Frekvencija, srednje vrednosti, dekade, poziciona analiza', period: true },
      { id: 'rangiranje', naziv: 'Rangiranje', ico: '🎯', opis: 'Rangiranje brojeva: Frekvencija / Bajes / Hibrid', period: false },
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
           diverzitet: false, max_slicnost: 4, radi: false, rezultati: [], rezime: '' },
    bektestovi: [], tiketi: [], noviTiket: '',
    istorija: [], unos: { kolo: null, datum: new Date().toISOString().slice(0, 10), brojevi: '' }, fajl: null, uvozZameni: false,

    aktivna() { return this.strane.find(s => s.id === this.strana) || this.strane[0]; },

    async init() {
      try { const d = await jget('/api/dashboard?period=0'); this.brojKola = d.broj_kola; } catch (e) {}
      this.ucitajStranu();
    },

    idi(id) { this.strana = id; this.ucitajStranu(); },

    ucitajStranu() {
      const f = {
        dashboard: () => this.ucitajDashboard(),
        statistika: () => this.ucitajStatistiku(),
        rangiranje: () => this.ucitajRang(),
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
      this.gen.radi = true; this.gen.rezultati = [];
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
        const d = await jsend('/api/generator', 'POST', { period: this.period, bazen, filteri });
        this.gen.rezultati = d.kombinacije;
        this.gen.rezime = `Validnih: ${d.ukupno_validnih} · posle diverziteta: ${d.posle_diverziteta} · prikazano ${d.kombinacije.length}`;
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

    async dodajTiketIz(brojevi) {
      const s = '(' + brojevi.join(', ') + ')';
      try { const d = await jsend('/api/tiketi', 'POST', { kombinacija: s }); this.toast(d.dodato ? 'Tiket dodat.' : 'Tiket već postoji.', d.dodato ? 'ok' : 'warn'); }
      catch (e) { this.toast('Greška: ' + e.message, 'err'); }
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
        this.toast(d.dodato ? `Kolo ${d.kolo} dodato. Provereno tiketa: ${d.provereno_tiketa}, bektestova: ${d.provereno_bektestova}.` : `Kolo već postoji; provere ažurirane.`, d.dodato ? 'ok' : 'warn');
        this.unos.brojevi = '';
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
        this.ucitajIstoriju();
      } catch (e) { this.toast('Greška: ' + e.message, 'err'); }
    },
  };
}
