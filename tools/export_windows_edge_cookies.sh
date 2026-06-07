#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
COOKIE_TARGET=${1:-"$REPO_ROOT/data/youtube_cookies.txt"}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[export_windows_edge_cookies] missing required command: $1" >&2
    exit 1
  fi
}

require_command powershell.exe
require_command curl
require_command wslpath

mkdir -p "$(dirname "$COOKIE_TARGET")"

if powershell.exe -NoProfile -Command 'if (Get-Process msedge -ErrorAction SilentlyContinue) { exit 1 }' >/dev/null 2>&1; then
  :
else
  echo "[export_windows_edge_cookies] Microsoft Edge is currently running." >&2
  echo "[export_windows_edge_cookies] Please close all Edge windows, then rerun this script." >&2
  exit 1
fi

WIN_LOCALAPPDATA=$(powershell.exe -NoProfile -Command '$env:LOCALAPPDATA' | tr -d '\r')
if [[ -z "$WIN_LOCALAPPDATA" ]]; then
  echo "[export_windows_edge_cookies] unable to resolve Windows LOCALAPPDATA." >&2
  exit 1
fi

WIN_YTDLP="$WIN_LOCALAPPDATA\\Temp\\yt-dlp.exe"
WIN_COOKIE="$WIN_LOCALAPPDATA\\Temp\\youtube_cookies.txt"
WSL_YTDLP=$(wslpath "$WIN_YTDLP")
WSL_COOKIE_TEMP=$(wslpath "$WIN_COOKIE")

if [[ ! -f "$WSL_YTDLP" ]]; then
  echo "[export_windows_edge_cookies] downloading Windows yt-dlp.exe..." >&2
  curl -L --fail -o "$WSL_YTDLP" \
    https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe
fi

rm -f "$WSL_COOKIE_TEMP"

echo "[export_windows_edge_cookies] exporting cookies from Windows Edge..." >&2
if ! powershell.exe -NoProfile -Command \
  '$exe = Join-Path $env:LOCALAPPDATA "Temp\yt-dlp.exe"; ' \
  '$out = Join-Path $env:LOCALAPPDATA "Temp\youtube_cookies.txt"; ' \
  'Remove-Item $out -ErrorAction SilentlyContinue; ' \
  '& $exe --cookies-from-browser edge --cookies $out'; then
  echo "[export_windows_edge_cookies] Edge automatic export failed." >&2
  echo "[export_windows_edge_cookies] This script only exports a copy of cookies and does not move or modify Edge's original cookie database." >&2
  echo "[export_windows_edge_cookies] A common cause is Windows DPAPI decryption failure." >&2
  echo "[export_windows_edge_cookies] Please switch to manual export:" >&2
  echo "[export_windows_edge_cookies] 1. Open a fresh private/incognito window in Windows Firefox or Edge/Chrome." >&2
  echo "[export_windows_edge_cookies] 2. Log in to YouTube and open https://www.youtube.com/robots.txt in the same tab." >&2
  echo "[export_windows_edge_cookies] 3. Export youtube.com cookies in Netscape format using a browser extension." >&2
  echo "[export_windows_edge_cookies] 4. Save the file to $COOKIE_TARGET" >&2
  exit 1
fi

if [[ ! -s "$WSL_COOKIE_TEMP" ]]; then
  echo "[export_windows_edge_cookies] export did not produce a usable cookies file." >&2
  exit 1
fi

cp "$WSL_COOKIE_TEMP" "$COOKIE_TARGET"
echo "[export_windows_edge_cookies] wrote $COOKIE_TARGET" >&2
echo "[export_windows_edge_cookies] next step:" >&2
echo "uv run python main.py prepare-data -- download \\" >&2
echo "  --json data/MUSICES.json \\" >&2
echo "  --data-root data \\" >&2
echo "  --skip-existing \\" >&2
echo "  --max-videos 1 \\" >&2
echo "  --abort-on-download-error \\" >&2
echo "  --yt-dlp-extra-arg=--cookies \\" >&2
echo "  --yt-dlp-extra-arg=$COOKIE_TARGET" >&2
