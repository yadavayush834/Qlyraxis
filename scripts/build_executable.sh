#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if ! .venv/bin/python -m PyInstaller --version >/dev/null 2>&1; then
  echo "Install the packaging extra first: .venv/bin/pip install -e '.[packaging]'" >&2
  exit 2
fi

.venv/bin/python -m PyInstaller --clean --noconfirm packaging/qlyraxis.spec
echo "Executable created at dist/Qlyraxis/Qlyraxis"
