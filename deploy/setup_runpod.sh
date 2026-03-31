#!/bin/bash
# Compatibility wrapper: the v2 setup is portable across Kaggle and Runpod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/v2/setup.sh" "$@"
