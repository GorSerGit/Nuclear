#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "$0")"
export PYTHONUTF8=1
if [ ! -x .venv/bin/python ]; then
    python3.13 -m venv .venv
fi
.venv/bin/python -c 'import sys, struct; sys.exit(sys.version_info[:2] != (3, 13) or struct.calcsize("P") != 8)'
.venv/bin/python scripts/office_dependencies.py --prepare
exec .venv/bin/python scripts/office_web.py "$@"
