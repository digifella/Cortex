#!/usr/bin/env bash
# Deploy the photo_archive package to a Windows-local directory.
#
# The package lives in WSL but must execute under Windows Python to read P: at
# usable speed. Running it straight off the \\wsl$ UNC path works but crosses
# the 9p bridge on every import, and that bridge has been flaky on this machine
# under sustained load. So the WSL git repo stays the source of truth and C: is
# a deployment target.
#
# Layout after deploy:
#   C:\photo_archive\photo_archive\*.py    <- the package
# Run from C:\photo_archive with:
#   C:\Python311\python.exe -m photo_archive.cli <stage> --db C:\pindex.db
#
# Re-run this after every code change; it overwrites and prunes .pyc caches.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="/mnt/c/photo_archive/photo_archive"

mkdir -p "$DEST"
rm -rf "$DEST"/__pycache__
cp -f "$SRC"/*.py "$DEST"/
cp -f "$SRC"/README.md "$DEST"/ 2>/dev/null || true

echo "deployed $(ls -1 "$DEST"/*.py | wc -l) modules -> C:\\photo_archive\\photo_archive\\"
ls -1 "$DEST"/*.py | xargs -n1 basename | sed 's/^/  /'
echo
echo "run with:"
echo '  cd /mnt/c/photo_archive && /mnt/c/Python311/python.exe -m photo_archive.cli --help'
