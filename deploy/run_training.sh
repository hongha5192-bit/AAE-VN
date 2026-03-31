#!/bin/bash
# Compatibility wrapper: use the v2 launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../start_training.sh" "$@"
