"""Dnevnik odigranih kombinacija (PLAN_TIKETI_GRADITELJ §3).

Za razliku od stare tabele `odigrani_tiketi` (stalni tiketi bez kola, rezultat se
prepisuje svako kolo), ovde je svaki red jedna kombinacija odigrana u JEDNOM kolu.
Korisnik je upisuje pre izvlačenja; ocena dolazi kad se kolo unese (faza 2).

Kombinacija se gleda kao skup: čuva se sortirana, kao CSV (isti format kao
`prognoze.kombinacija`).
"""

import math
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


# ----------------------------------------------------------------------------
# Kumulativni pregled (§5)
# ----------------------------------------------------------------------------
# H₀: izbor tiketa nema veze sa ishodom, tj. izvlačenje je slučajno nezavisno od
# tiketa. Očekivanje NIJE 50% percentila za svaki tiket: D zavisi od položaja
# brojeva, pa kombinacija sa ivice (1–7) skoro uvek ispada daleko. Zato se svaki
# tiket poredi sa SVOJIM očekivanjem — ocenom naspram svih istorijskih izvlačenja,
# koja su uzorak raspodele izvlačenja pod H₀. Tiketi istog kola dele izvlačenje,
# pa se po kolu sabira ceo vektor i varijansa uzima sa kovarijansama.

MERE = ("pogoci", "skoro", "percentil")
MIN_TIKETA = 10        # ispod ovoga se pregled označava kao premali uzorak

_osnova_kes = {"kljuc": None, "vrednost": None}


def _null_osnova(izvuceno):
    """Matrice nad svim izvučenim kolima, keširane dok se istorija ne promeni.

    S: sortirana izvlačenja (H×7); M: maska prisutnosti (H×(n+2)); PCT[h, D]: percentil
    rastojanja D u kolu h. PCT traži jedan DP po kolu (~2 ms) — ~3 s za celu istoriju,
    zato keš.
    """
    kljuc = hash(tuple(sorted((k, tuple(sorted(v))) for k, v in izvuceno.items())))
    if _osnova_kes["kljuc"] == kljuc:
        return _osnova_kes["vrednost"]
    kola = sorted(izvuceno)
    n = konfig.MAX_BROJ
    S = np.array([sorted(izvuceno[k]) for k in kola], dtype=np.int64)
    M = np.zeros((len(kola), n + 2), dtype=np.int64)
    for i, k in enumerate(kola):
        M[i, list(izvuceno[k])] = 1
    PCT = []
    for k in kola:
        r = raspodela_rastojanja(izvuceno[k]).astype(np.float64)
        dalje = np.concatenate([np.cumsum(r[::-1])[::-1][1:], [0.0]])   # Σ r[D+1:]
        PCT.append((dalje + 0.5 * r) / r.sum())
    osnova = {"S": S, "M": M, "PCT": np.array(PCT)}
    _osnova_kes.update(kljuc=kljuc, vrednost=osnova)
    return osnova


def null_vektori(tiket, osnova):
    """Vrednost svake mere za `tiket` u svakom istorijskom kolu (raspodela pod H₀)."""
    t = sorted(tiket)
    M = osnova["M"]
    susedi = sorted(({b - 1 for b in t} | {b + 1 for b in t}) - set(t))
    D = np.abs(osnova["S"] - np.array(t)).sum(axis=1)
    return {
        "pogoci": M[:, t].sum(axis=1).astype(np.float64),
        "skoro": M[:, susedi].sum(axis=1).astype(np.float64),
        "percentil": osnova["PCT"][np.arange(len(D)), D],
    }


def _test(redovi, osnova):
    """z-test zbira odstupanja od sopstvenog očekivanja, po meri.

    redovi: ocenjeni redovi (dict sa kolo, brojevi i merama). Za svako kolo se
    centrirani null-vektori tiketa saberu; varijansa tog zbira nad istorijom je
    Σ kovarijansi tiketa tog kola. Kola su međusobno nezavisna pod H₀.
    """
    po_kolu = {}
    for r in redovi:
        po_kolu.setdefault(r["kolo"], []).append(r)
    out = {}
    for mera in MERE:
        odstupanje, varijansa, posmatrano, ocekivano = 0.0, 0.0, 0.0, 0.0
        for grupa in po_kolu.values():
            zbir = None
            for r in grupa:
                v = null_vektori(r["brojevi"], osnova)[mera]
                mu = float(v.mean())
                posmatrano += r[mera]
                ocekivano += mu
                odstupanje += r[mera] - mu
                zbir = v - mu if zbir is None else zbir + (v - mu)
            varijansa += float((zbir ** 2).mean())
        n = len(redovi)
        z = odstupanje / math.sqrt(varijansa) if varijansa > 0 else 0.0
        out[mera] = {
            "prosek": posmatrano / n,
            "ocekivano": ocekivano / n,
            "z": z,
            "p": math.erfc(abs(z) / math.sqrt(2)),     # dvostrano, normalna aproksimacija
        }
    return out


def _grupa_izvora(izvor):
    """Sve metode Prognoze su jedna grupa — inače bi Bonferroni rastao sa brojem metoda."""
    return "prognoza" if (izvor or "").startswith("prognoza:") else (izvor or "rucno")


def pregled(conn):
    """Kumulativni pregled ocenjenih tiketa: ukupno, po izvoru i tačke za grafikon."""
    redovi = [r for r in lista(conn) if r["kolo"] is not None and r["percentil"] is not None]
    rezultat = {"n_tiketa": len(redovi), "n_kola": len({r["kolo"] for r in redovi}),
                "min_tiketa": MIN_TIKETA, "ukupno": None, "po_izvoru": [], "tacke": []}
    if not redovi:
        return rezultat
    osnova = _null_osnova(izvucena_kola(conn))

    rezultat["ukupno"] = _test(redovi, osnova)

    grupe = {}
    for r in redovi:
        grupe.setdefault(_grupa_izvora(r["izvor"]), []).append(r)
    prag = 0.05 / len(grupe)                          # Bonferroni po broju grupa
    for izvor, gr in sorted(grupe.items(), key=lambda x: -len(x[1])):
        t = _test(gr, osnova)["percentil"]
        rezultat["po_izvoru"].append({
            "izvor": izvor, "n": len(gr), "kola": len({r["kolo"] for r in gr}),
            **t, "znacajno": t["p"] < prag,
        })
    rezultat["prag"] = prag

    # Tačke hronološki, sa očekivanjem svakog tiketa — UI crta kumulativne proseke.
    for r in sorted(redovi, key=lambda r: (r["kolo"], r["id"])):
        rezultat["tacke"].append({
            "kolo": r["kolo"], "percentil": r["percentil"], "pogoci": r["pogoci"],
            "izvor": r["izvor"], "brojevi": r["brojevi"],
            "ocekivano": float(null_vektori(r["brojevi"], osnova)["percentil"].mean()),
        })
    return rezultat


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
