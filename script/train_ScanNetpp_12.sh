#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "${PYTHON:-python}" "$project_root/run_dataset.py" --dataset ScanNetpp --views 12 "$@"
