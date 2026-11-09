#!/usr/bin/env bash
# Pokretac Loto Analizatora (web) za Linux/Ubuntu.
#
#   bash pokreni_loto.sh            # pokrece server i otvara browser
#   bash pokreni_loto.sh --precica  # dodaje "Loto Analizator" u meni aplikacija
#
# Prvi put sam pravi .venv i instalira requirements-web.txt.
# Ostali argumenti idu pravo u pokreni.py (npr. --port 9000 --bez-otvaranja).

set -e
cd "$(dirname "$(readlink -f "$0")")"
KOREN="$(pwd)"

if [ "$1" = "--precica" ]; then
    DIR="$HOME/.local/share/applications"
    mkdir -p "$DIR"
    cat > "$DIR/loto-analizator.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Loto Analizator
Comment=Loto Analizator - web server
Exec=bash "$KOREN/pokreni_loto.sh"
Path=$KOREN
Icon=applications-games
Terminal=true
Categories=Utility;
EOF
    echo "Precica dodata: $DIR/loto-analizator.desktop"
    echo "Potrazi 'Loto Analizator' u meniju aplikacija."
    exit 0
fi

if [ ! -f .venv/.instalirano ]; then
    echo "Prvo pokretanje: pravim virtuelno okruzenje (.venv)..."
    if ! python3 -m venv .venv; then
        echo "Nije uspelo. Instaliraj ga sa: sudo apt install python3-venv"
        exit 1
    fi
    .venv/bin/pip install -r requirements-web.txt
    touch .venv/.instalirano   # tek posle uspesne instalacije; inace se ponavlja
fi

echo "============================================"
echo "  Loto Analizator - pokrecem server..."
echo "  Aplikacija ce se otvoriti u browseru."
echo "  Za gasenje: zatvori ovaj prozor ili Ctrl+C"
echo "============================================"
echo

.venv/bin/python pokreni.py "$@" || true

echo
echo "Server je zaustavljen."
read -r -p "Pritisni Enter za zatvaranje..." _
