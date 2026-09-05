"""Pokretač web aplikacije Loto Analizator.

Upotreba:
    python pokreni.py            # otvara http://127.0.0.1:8000
    python pokreni.py --port 9000

Zahtevi:  pip install fastapi "uvicorn[standard]" python-multipart pandas scipy openpyxl
"""

import argparse
import webbrowser

import uvicorn

from webapp.core import baza


def main():
    parser = argparse.ArgumentParser(description="Loto Analizator — web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bez-otvaranja", action="store_true", help="Ne otvaraj automatski browser")
    args = parser.parse_args()

    baza.postavi_bazu()  # osigurava da tabele postoje

    url = f"http://{args.host}:{args.port}"
    print(f"Loto Analizator radi na: {url}")
    if not args.bez_otvaranja:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run("webapp.api.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
