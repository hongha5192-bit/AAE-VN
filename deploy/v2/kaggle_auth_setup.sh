#!/bin/bash
# Create ~/.kaggle/kaggle.json from env vars or prompts.
set -euo pipefail

KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
KAGGLE_TOKEN_FILE="$KAGGLE_CONFIG_DIR/kaggle.json"

username="${KAGGLE_USERNAME:-}"
key="${KAGGLE_KEY:-}"

if [ -z "$username" ]; then
    read -r -p "Kaggle username: " username
fi
if [ -z "$key" ]; then
    read -r -s -p "Kaggle key: " key
    echo
fi

[ -n "$username" ] || { echo "ERROR: empty username" >&2; exit 1; }
[ -n "$key" ] || { echo "ERROR: empty key" >&2; exit 1; }

mkdir -p "$KAGGLE_CONFIG_DIR"
cat > "$KAGGLE_TOKEN_FILE" <<EOF
{"username":"$username","key":"$key"}
EOF
chmod 600 "$KAGGLE_TOKEN_FILE"

# For Kaggle access tokens (e.g. KGAT_*), also write ~/.kaggle/access_token
if [[ "$key" == KGAT_* ]]; then
    printf '%s\n' "$key" > "$KAGGLE_CONFIG_DIR/access_token"
    chmod 600 "$KAGGLE_CONFIG_DIR/access_token"
fi

echo "Wrote $KAGGLE_TOKEN_FILE"
echo "Permissions: $(stat -f '%Lp' "$KAGGLE_TOKEN_FILE" 2>/dev/null || stat -c '%a' "$KAGGLE_TOKEN_FILE")"
