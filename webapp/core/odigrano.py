"""Dnevnik odigranih kombinacija (PLAN_TIKETI_GRADITELJ §3).

Za razliku od stare tabele `odigrani_tiketi` (stalni tiketi bez kola, rezultat se
prepisuje svako kolo), ovde je svaki red jedna kombinacija odigrana u JEDNOM kolu.
Korisnik je upisuje pre izvlačenja; ocena dolazi kad se kolo unese (faza 2).

Kombinacija se gleda kao skup: čuva se sortirana, kao CSV (isti format kao
`prognoze.kombinacija`).
"""

from . import mapa
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


def lista(conn):
    """Dnevnik, najnovije kolo prvo; uvezeni (bez kola) na kraju.

    Dok faza 2 ne upiše trajne mere, `pogoci` se za izvučena kola računa ovde —
    jeftino je, a korisnik odmah vidi rezultat.
    """
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
            if d["pogoci"] is None:
                d["pogoci"] = len(izvuceno[d["kolo"]] & set(d["brojevi"]))
        else:
            d["status"] = "ceka"
        out.append(d)
    return out
