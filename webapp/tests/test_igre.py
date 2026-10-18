"""Dve igre (Loto / Loto plus): svaka ima svoju bazu i svoj keš analize, a dnevnik
tiketa je zajednički (u bazi Lota) — Loto plus je doplata na isti tiket.

Zahtev bira igru zaglavljem X-Igra; bez zaglavlja važi Loto. Ovde se proverava da
kolo uneto u jednu igru ne curi u drugu, ni kroz bazu ni kroz keš `app.py`, i da
middleware zaista prebacuje bazu za ceo zahtev (poziv ide kroz pravi ASGI stek).

Radi nad privremenim bazama (ne dira loto_baza.db ni loto_plus.db).

Pokretanje:  python -X utf8 -m webapp.tests.test_igre
"""

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from webapp.core import konfig  # noqa: E402

# Pre uvoza aplikacije: obe igre idu u isti privremeni folder.
konfig.PUTANJA_BAZE = os.path.join(tempfile.mkdtemp(), "loto_baza.db")

from webapp.api import app as api  # noqa: E402
from webapp.core import baza  # noqa: E402

KOLO = 2026080
BROJEVI = [12, 3, 39, 21, 7, 30, 18]     # redosled izvlačenja, ne sortirano


def _zahtev(metod, putanja, igra=None, telo=None):
    """Jedan HTTP zahtev kroz ceo ASGI stek (middleware + ruta), bez httpx-a."""
    zaglavlja = [(b"content-type", b"application/json")]
    if igra is not None:
        zaglavlja.append((b"x-igra", igra.encode()))
    putanja, _, upit = putanja.partition("?")
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": metod, "scheme": "http", "path": putanja, "raw_path": putanja.encode(),
             "query_string": upit.encode(), "root_path": "", "headers": zaglavlja,
             "client": ("127.0.0.1", 1), "server": ("test", 80)}
    sadrzaj = json.dumps(telo).encode() if telo is not None else b""
    poslato = {"status": None, "telo": b""}

    async def receive():
        return {"type": "http.request", "body": sadrzaj, "more_body": False}

    async def send(poruka):
        if poruka["type"] == "http.response.start":
            poslato["status"] = poruka["status"]
        elif poruka["type"] == "http.response.body":
            poslato["telo"] += poruka.get("body", b"")

    asyncio.run(api.app(scope, receive, send))
    return poslato["status"], json.loads(poslato["telo"] or b"null")


def test_putanje_igara():
    assert konfig.putanja_baze("loto") == konfig.PUTANJA_BAZE
    plus = konfig.putanja_baze("loto_plus")
    assert os.path.dirname(plus) == os.path.dirname(konfig.PUTANJA_BAZE)
    assert os.path.basename(plus) == "loto_plus.db"
    print("  Loto plus ima zasebnu bazu pored glavne")


def test_kolo_ne_curi_medju_igrama():
    api._startup()
    # Keš Lota se puni PRE unosa u Loto plus — posle unosa ne sme da pokaže tuđe kolo.
    status, d = _zahtev("GET", "/api/dashboard?period=0")
    assert status == 200 and d["broj_kola"] == 0

    status, _ = _zahtev("POST", "/api/istorija", "loto_plus",
                        {"kolo": KOLO, "datum": "2026-10-16", "brojevi": BROJEVI})
    assert status == 200

    status, plus = _zahtev("GET", "/api/istorija", "loto_plus")
    assert status == 200 and len(plus) == 1
    assert [plus[0][f"b{i}"] for i in range(1, 8)] == BROJEVI     # redosled sačuvan
    assert _zahtev("GET", "/api/istorija")[1] == []
    assert _zahtev("GET", "/api/istorija", "loto")[1] == []

    assert _zahtev("GET", "/api/dashboard?period=0", "loto_plus")[1]["broj_kola"] == 1
    assert _zahtev("GET", "/api/dashboard?period=0", "loto")[1]["broj_kola"] == 0

    igre = {g["id"]: g["broj_kola"] for g in _zahtev("GET", "/api/igre")[1]}
    assert igre == {"loto": 0, "loto_plus": 1}
    print("  kolo uneto u Loto plus ne vidi se u Lotu (ni baza ni keš)")


def test_nepoznata_igra_je_400():
    status, d = _zahtev("GET", "/api/istorija", "keno")
    assert status == 400 and "keno" in d["detail"]
    print("  nepoznata igra → 400")


def test_igra_vracena_posle_zahteva():
    _zahtev("GET", "/api/istorija", "loto_plus")
    assert konfig.aktivna_igra() == konfig.PODRAZUMEVANA_IGRA
    print("  posle zahteva važi podrazumevana igra")


def _tiketi(igra=None):
    status, d = _zahtev("GET", "/api/odigrano", igra)
    assert status == 200
    return {tuple(r["brojevi"]): r for r in d["redovi"]}


def test_tiket_igra_u_oba_izvlacenja():
    # Loto plus kolo KOLO je već uneto (BROJEVI), Loto kolo još nije.
    sa_plusom = [3, 7, 12, 13, 14, 15, 17]     # Loto plus: 3, 7, 12 pogođeni; 18 promašen za 1
    bez_plusa = [1, 2, 4, 5, 6, 8, 9]
    # Upis dok je izabran Loto plus mora da završi u ISTOM dnevniku (baza Lota).
    status, d = _zahtev("POST", "/api/odigrano", "loto_plus",
                        {"brojevi": sa_plusom, "kolo": KOLO, "loto_plus": True})
    assert status == 200 and d["dodato"]
    status, _ = _zahtev("POST", "/api/odigrano", None, {"brojevi": bez_plusa, "kolo": KOLO})
    assert status == 200

    t = _tiketi()
    assert _tiketi("loto_plus").keys() == t.keys() == {tuple(sa_plusom), tuple(bez_plusa)}
    r = t[tuple(sa_plusom)]
    assert r["loto_plus"] == 1 and r["status"] == "ceka"      # Loto još nije izvučen
    assert r["pogoci_plus"] == 3 and r["skoro_plus"] == 1
    assert r["izvuceni_plus"] == sorted(BROJEVI)
    r = t[tuple(bez_plusa)]
    assert r["loto_plus"] == 0 and r["pogoci_plus"] is None

    # Kad stigne i Loto izvlačenje, tiket ima obe ocene.
    status, _ = _zahtev("POST", "/api/istorija", None,
                        {"kolo": KOLO, "datum": "2026-10-16", "brojevi": [14, 1, 22, 33, 15, 2, 38]})
    assert status == 200
    r = _tiketi()[tuple(sa_plusom)]
    assert r["status"] == "izvuceno" and r["pogoci"] == 2 and r["pogoci_plus"] == 3
    assert _tiketi()[tuple(bez_plusa)]["pogoci"] == 2

    # Naknadno uključivanje/isključivanje Loto plusa odmah menja ocenu.
    id_bez = _tiketi()[tuple(bez_plusa)]["id"]
    assert _zahtev("PUT", f"/api/odigrano/{id_bez}/loto-plus", None, {"loto_plus": True})[0] == 200
    r = _tiketi()[tuple(bez_plusa)]
    assert r["loto_plus"] == 1 and r["pogoci_plus"] == 0 and r["skoro_plus"] == 2   # 3 i 7 su susedi
    assert _zahtev("PUT", f"/api/odigrano/{id_bez}/loto-plus", None, {"loto_plus": False})[0] == 200
    assert _tiketi()[tuple(bez_plusa)]["pogoci_plus"] is None
    assert _zahtev("PUT", "/api/odigrano/999999/loto-plus", None, {"loto_plus": True})[0] == 404

    # Baza Loto plusa nema svoj dnevnik.
    conn = baza.konekcija(konfig.putanja_baze("loto_plus"))
    try:
        assert conn.execute("SELECT COUNT(*) FROM odigrano").fetchone()[0] == 0
    finally:
        conn.close()
    print("  tiket sa Loto plusom ocenjen u oba izvlačenja; dnevnik zajednički")


def test_brisanje_plus_kola_vraca_tiket_na_cekanje():
    plus = {r["kolo"]: r for r in _zahtev("GET", "/api/istorija", "loto_plus")[1]}
    assert _zahtev("DELETE", f"/api/istorija/{plus[KOLO]['id']}", "loto_plus")[0] == 200
    r = _tiketi()[(3, 7, 12, 13, 14, 15, 17)]
    assert r["pogoci_plus"] is None and "izvuceni_plus" not in r
    assert r["pogoci"] == 2                     # Loto ocena ostaje
    print("  brisanje Loto plus kola vraća Loto plus ocenu na čekanje")


def test_backup_nosi_ime_igre():
    token = konfig.postavi_igru("loto_plus")
    try:
        put = baza.napravi_backup()
    finally:
        konfig.vrati_igru(token)
    try:
        assert os.path.basename(put).startswith("loto_plus_backup_")
    finally:
        os.remove(put)
    print("  rezervna kopija Loto plus baze nosi ime igre")


def main():
    test_putanje_igara()
    test_kolo_ne_curi_medju_igrama()
    test_nepoznata_igra_je_400()
    test_igra_vracena_posle_zahteva()
    test_tiket_igra_u_oba_izvlacenja()
    test_brisanje_plus_kola_vraca_tiket_na_cekanje()
    test_backup_nosi_ime_igre()
    print("\nSVI TESTOVI IGARA PROSLI [OK]")


if __name__ == "__main__":
    main()
