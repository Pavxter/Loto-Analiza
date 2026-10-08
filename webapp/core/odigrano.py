"""Dnevnik odigranih kombinacija (PLAN_TIKETI_GRADITELJ §3).

Za razliku od stare tabele `odigrani_tiketi` (stalni tiketi bez kola, rezultat se
prepisuje svako kolo), ovde je svaki red jedna kombinacija odigrana u JEDNOM kolu.
Korisnik je upisuje pre izvlačenja; ocena dolazi kad se kolo unese (faza 2).

Kombinacija se gleda kao skup: čuva se sortirana, kao CSV (isti format kao
`prognoze.kombinacija`).
"""

from datetime import datetime

import numpy as np

from . import konfig, mapa
from . import razlicitost_teorija as T

# Upozorenje o sličnosti sa istorijom od ovoliko zajedničkih brojeva (§3.3). Na ~1.400
# kola nasumična kombinacija u proseku ima jedno kolo sa 5 poklapanja, pa 5 nije znak
# lošeg izbora; 6 se očekuje tek u ~2% slučajeva.
PRAG_UPOZORENJA = 6
# Najmanje poklapanje koje se prikazuje u raspodeli i u spisku kola.
MIN_PRIKAZ = 4
# Koliko najbližih kola se navodi poimence.
NAJBLIZIH = 5

IZVORI = ("rucno", "generator", "sekv", "graditelj", "uvoz")


def normalizuj(brojevi):
    """Sortirana 7-torka; ValueError ako kombinacija nije ispravna (iste provere kao Mapa)."""
    return mapa.proveri_kombinaciju(brojevi)


def u_csv(komb):
    return ",".join(str(b) for b in komb)


def iz_csv(s):
    return [int(x) for x in s.split(",")] if s else []


def ispravan_izvor(izvor):
    """`prognoza:<metod>` je dozvoljen za bilo koji metod Prognoze."""
    return izvor in IZVORI or (izvor or "").startswith("prognoza:")


def ispravno_kolo(kolo):
    """Numeracija godina*1000 + broj, broj 1..999."""
    godina, broj = divmod(int(kolo), 1000)
    return 2000 <= godina <= 2100 and broj >= 1


def poslednje_kolo(conn):
    r = conn.execute("SELECT kolo FROM istorijski_rezultati ORDER BY id DESC LIMIT 1").fetchone()
    return int(r[0]) if r else None


def sledece_kolo(conn):
    """Poslednje uneto + 1 — ista konvencija kao `prognoza.ciljno_kolo`.

    Prelaz godine se ne pogađa (broj kola po godini varira); korisnik menja kolo ručno.
    """
    p = poslednje_kolo(conn)
    return p + 1 if p is not None else None


def izvucena_kola(conn):
    """{kolo: frozenset brojeva} za sva uneta kola."""
    redovi = conn.execute("SELECT kolo, b1, b2, b3, b4, b5, b6, b7 FROM istorijski_rezultati").fetchall()
    return {int(r[0]): frozenset(int(x) for x in r[1:8]) for r in redovi}


def slicnost(istorija, brojevi):
    """Kako se kombinacija poklapa sa SVIM izvučenim kolima (§3.3).

    istorija: lista (kolo, brojevi), hronološki. Vraća najveće poklapanje, kola u
    kojima je izvučena ista kombinacija, raspodelu poklapanja 4..7 uz očekivanje za
    nasumičnu kombinaciju na istom broju kola, i najbliža kola poimence.
    """
    komb = normalizuj(brojevi)
    cm = T.maska(komb)
    n = len(istorija)
    poklapanja = [(kolo, T.preklapanje(cm, T.maska(br))) for kolo, br in istorija]

    raspodela = []
    for k in range(T.K, MIN_PRIKAZ - 1, -1):
        raspodela.append({
            "k": k,
            "broj": sum(1 for _kolo, p in poklapanja if p == k),
            "ocekivano": T.hipergeom_pmf(k) * n,
        })

    # Najbliža: najveće poklapanje, pa novije kolo.
    najbliza = sorted((x for x in poklapanja if x[1] >= MIN_PRIKAZ), key=lambda x: (-x[1], -x[0]))
    maks = max((p for _kolo, p in poklapanja), default=0)
    return {
        "kombinacija": list(komb),
        "broj_kola": n,
        "maks": maks,
        "ista": [kolo for kolo, p in poklapanja if p == T.K],
        "raspodela": raspodela,
        "najbliza": [{"kolo": kolo, "k": p} for kolo, p in najbliza[:NAJBLIZIH]],
        "upozorenje": maks >= PRAG_UPOZORENJA,
        "prag": PRAG_UPOZORENJA,
    }


# ----------------------------------------------------------------------------
# Ocena tiketa posle izvlačenja (§4)
# ----------------------------------------------------------------------------

def rastojanje(a, b):
    """D = Σ |a₍ᵢ₎ − b₍ᵢ₎| nad sortiranim kombinacijama.

    U 1D je sortirano uparivanje optimalno 1-na-1 uparivanje (Earth mover's), pa
    jedan broj tiketa ne može „pokriti" dva dobitna kao u `promasaj_kombinacije`.
    """
    return sum(abs(x - y) for x, y in zip(sorted(a), sorted(b)))


def skoro_pogoci(tiket, izvuceni):
    """Koliko izvučenih brojeva je promašeno za tačno ±1 (pogođeni se ne broje)."""
    t = set(tiket)
    return sum(1 for d in izvuceni if d not in t and (d - 1 in t or d + 1 in t))


def raspodela_rastojanja(izvuceni, n=konfig.MAX_BROJ):
    """Broj k-podskupova od 1..n za svako rastojanje D do `izvuceni` (k = len).

    Dinamičko programiranje po sortiranim pozicijama: f[v, s] = broj načina da se
    izaberu x₁ < … < xᵢ sa xᵢ = v i delimičnim zbirom s. Tačno, bez simulacije.
    Zbir rezultata je C(n, k).
    """
    d = sorted(izvuceni)
    k = len(d)
    maks = k * (n - k)                       # |xᵢ − dᵢ| ≤ n − k za svaku poziciju
    f = np.zeros((n + 1, maks + 1), dtype=np.int64)
    for v in range(1, n + 1):
        f[v, abs(v - d[0])] = 1
    for i in range(1, k):
        pref = np.cumsum(f, axis=0)          # pref[v] = Σ_{u ≤ v} f[u]
        g = np.zeros_like(f)
        for w in range(2, n + 1):
            c = abs(w - d[i])
            g[w, c:] = pref[w - 1, :maks + 1 - c]
        f = g
    return f.sum(axis=0)


def percentil(D, raspodela):
    """Udeo kombinacija DALJIH od izvučene + pola izjednačenih (srednji rang).

    Sa srednjim rangom nasumičan tiket ima očekivani percentil tačno 0,5.
    """
    ukupno = int(raspodela.sum())
    dalje = int(raspodela[D + 1:].sum())
    return (dalje + 0.5 * int(raspodela[D])) / ukupno


def oceni(tiket, izvuceni, raspodela=None):
    """Sve mere jednog tiketa za jedno izvlačenje."""
    if raspodela is None:
        raspodela = raspodela_rastojanja(izvuceni)
    D = rastojanje(tiket, izvuceni)
    return {
        "pogoci": len(set(tiket) & set(izvuceni)),
        "skoro": skoro_pogoci(tiket, izvuceni),
        "rastojanje": D,
        "percentil": percentil(D, raspodela),
    }


def oceni_sve(conn):
    """Preračunava ocene svih redova iz TRENUTNIH izvučenih kola.

    Poziva se pri startu i posle svakog dodavanja, izmene ili brisanja kola. Ceo
    prolaz je jeftin (jedan DP po kolu sa tiketima), a zato ocena ne može da
    zastari kad se staro kolo ispravi. Redovi čije kolo nije (više) izvučeno se
    vraćaju na neocenjeno. Vraća broj ocenjenih redova.
    """
    izvuceno = izvucena_kola(conn)
    sad = datetime.now().isoformat(timespec="seconds")
    raspodele = {}
    ocenjeno = 0
    for r in conn.execute("SELECT id, kolo, kombinacija FROM odigrano WHERE kolo IS NOT NULL").fetchall():
        kolo = r["kolo"]
        if kolo not in izvuceno:
            conn.execute("UPDATE odigrano SET pogoci=NULL, skoro=NULL, rastojanje=NULL, "
                         "percentil=NULL, ocenjeno=NULL WHERE id=?", (r["id"],))
            continue
        if kolo not in raspodele:
            raspodele[kolo] = raspodela_rastojanja(izvuceno[kolo])
        m = oceni(iz_csv(r["kombinacija"]), izvuceno[kolo], raspodele[kolo])
        conn.execute("UPDATE odigrano SET pogoci=?, skoro=?, rastojanje=?, percentil=?, ocenjeno=? "
                     "WHERE id=?", (m["pogoci"], m["skoro"], m["rastojanje"], m["percentil"], sad, r["id"]))
        ocenjeno += 1
    conn.commit()
    return ocenjeno


def broj_za_kolo(conn, kolo):
    return conn.execute("SELECT COUNT(*) FROM odigrano WHERE kolo=?", (kolo,)).fetchone()[0]


def lista(conn):
    """Dnevnik, najnovije kolo prvo; uvezeni (bez kola) na kraju."""
    izvuceno = izvucena_kola(conn)
    redovi = conn.execute(
        "SELECT * FROM odigrano ORDER BY kolo IS NULL, kolo DESC, id ASC").fetchall()
    out = []
    for r in redovi:
        d = dict(r)
        d["brojevi"] = iz_csv(d["kombinacija"])
        if d["kolo"] is None:
            d["status"] = "bez_kola"
        elif d["kolo"] in izvuceno:
            d["status"] = "izvuceno"
            d["izvuceni"] = sorted(izvuceno[d["kolo"]])
        else:
            d["status"] = "ceka"
        out.append(d)
    return out
