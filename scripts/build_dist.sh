#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TMP_LLMS="$(mktemp -t llms.XXXXXX.txt)"
trap 'rm -f "$TMP_LLMS"' EXIT

uv run --group docs python scripts/generate_llms_txt.py --output "$TMP_LLMS"

cp "$TMP_LLMS" "$REPO_ROOT/snapatac2/llms.txt"

maturin "$@"
BUILD_STATUS=$?

if [ "$BUILD_STATUS" -eq 0 ]; then
    rm -f "$REPO_ROOT/snapatac2/llms.txt"
fi

exit "$BUILD_STATUS"
