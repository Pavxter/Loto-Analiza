"""Prazna baza: nijedna GET ruta ne sme da padne na serijalizaciji (NaN/inf u JSON-u).

Starlette serijalizuje sa allow_nan=False, pa NaN iz proseka prazne kolone daje 500
umesto praznog prikaza. Ovde se svaka GET ruta bez obaveznih parametara poziva nad
praznom bazom i odgovor serijalizuje na isti način. HTTPException (npr. 400 „premalo
kola") je ispravan odgovor; svaki drugi izuzetak je greška.

Radi nad privremenom bazom (ne dira loto_baza.db).

Pokretanje:  python -X utf8 -m webapp.tests.test_prazna_baza
"""

import inspect
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from webapp.core import konfig  # noqa: E402

# Pre uvoza aplikacije: sve konekcije bez eksplicitne putanje idu u privremenu bazu.
konfig.PUTANJA_BAZE = os.path.join(tempfile.mkdtemp(), "prazna.db")

from fastapi import HTTPException  # noqa: E402
from fastapi.encoders import jsonable_encoder  # noqa: E402

from webapp.api import app as api  # noqa: E402


def _get_rute_bez_obaveznih():
    for r in api.app.routes:
        if "GET" not in (getattr(r, "methods", None) or ()) or not r.path.startswith("/api"):
            continue
        param = inspect.signature(r.endpoint).parameters.values()
        if all(p.default is not inspect.Parameter.empty for p in param):
            yield r


def test_get_rute_nad_praznom_bazom():
    api._startup()
    rute = sorted(_get_rute_bez_obaveznih(), key=lambda r: r.path)
    assert any(r.path == "/api/statistika" for r in rute)
    padovi = []
    for r in rute:
        try:
            json.dumps(jsonable_encoder(r.endpoint()), allow_nan=False)
        except HTTPException as e:
            assert e.status_code < 500, f"{r.path}: {e.status_code} {e.detail}"
        except Exception as e:
            padovi.append(f"{r.path}: {type(e).__name__}: {e}")
    assert not padovi, "\n".join(padovi)
    print(f"  {len(rute)} GET ruta nad praznom bazom — bez NaN i bez pada")


def test_statistika_za_period_nad_praznom_bazom():
    # Period > broja kola je isti slučaj kao prazna baza (prazan „analizirani" deo).
    for period in (0, 50):
        json.dumps(jsonable_encoder(api.statistika(period)), allow_nan=False)
        json.dumps(jsonable_encoder(api.dashboard(period)), allow_nan=False)
    print("  statistika i dashboard za period 0 i 50 — ispravan JSON")


def main():
    test_get_rute_nad_praznom_bazom()
    test_statistika_za_period_nad_praznom_bazom()
    print("\nSVI TESTOVI PRAZNE BAZE PROSLI [OK]")


if __name__ == "__main__":
    main()
