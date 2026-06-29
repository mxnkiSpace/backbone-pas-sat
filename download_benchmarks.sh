#!/usr/bin/env bash
# Descarga benchmarks SAT.
#
# ADVERTENCIA: la descarga completa pesa aprox. 10 GB 
# Uso:
#   chmod +x download_benchmarks.sh
#   ./download_benchmarks.sh
#

set -euo pipefail

DATA_DIR="data"
SATLIB_BASE="https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT"
VERBOSE="${VERBOSE:-0}"

log() { [[ "$VERBOSE" == "1" ]] && echo "  $*" || true; }

mkdir -p "$DATA_DIR/satlib/cbs_backbone_controlled"
mkdir -p "$DATA_DIR/satlib/random_3sat"
mkdir -p "$DATA_DIR/satlib/graph_coloring"
mkdir -p "$DATA_DIR/satlib/planning"
mkdir -p "$DATA_DIR/satcomp_holdout/2024"
mkdir -p "$DATA_DIR/satcomp_holdout/2025"
mkdir -p "$DATA_DIR/.gbd"

# SATLIB — Random-3-SAT con TAMAÑO DE BACKBONE CONTROLADO (CBS)
# Backbone proportion conocido por construcción: b10/b30/b50/b70/b90

echo "==> SATLIB CBS (controlled backbone size)"
cbs_ok=0 cbs_skip=0 cbs_fail=0
for m in 403 411 418 423 429 435 441 449; do
  for b in 10 30 50 70 90; do
    fname="CBS_k3_n100_m${m}_b${b}.tar.gz"
    url="${SATLIB_BASE}/CBS/${fname}"
    out="$DATA_DIR/satlib/cbs_backbone_controlled/${fname}"
    marker="${out}.done"
    if [[ -f "$marker" ]]; then
      log "skip (exists): $fname"; ((cbs_skip++)); continue
    fi
    if curl -fsSL "$url" -o "$out"; then
      tar -xzf "$out" -C "$DATA_DIR/satlib/cbs_backbone_controlled"
      touch "$marker"; log "ok: $fname"; ((cbs_ok++))
    else
      log "unavailable: $fname"; rm -f "$out"; ((cbs_fail++))
    fi
  done
done
echo "    downloaded=$cbs_ok skipped=$cbs_skip failed=$cbs_fail"

# SATLIB — instancias adicionales por dominio

echo "==> SATLIB by domain (random / graph coloring / planning)"
declare -A satlib_extra=(
  ["random_3sat/uf100-430.tar.gz"]="RND3SAT/uf100-430.tar.gz"
  ["random_3sat/uuf100-430.tar.gz"]="RND3SAT/uuf100-430.tar.gz"
  ["graph_coloring/flat100-239.tar.gz"]="GCP/flat100-239.tar.gz"
  ["planning/blocksworld.tar.gz"]="PLANNING/BlocksWorld/blocksworld.tar.gz"
  ["planning/logistics.tar.gz"]="PLANNING/Logistics/logistics.tar.gz"
)
dom_ok=0 dom_skip=0
for rel in "${!satlib_extra[@]}"; do
  url="${SATLIB_BASE}/${satlib_extra[$rel]}"
  out="$DATA_DIR/satlib/${rel}"
  marker="${out}.done"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$marker" ]]; then
    log "skip (exists): $rel"; ((dom_skip++)); continue
  fi
  log "downloading: $rel"
  curl -fsSL "$url" -o "$out"
  tar -xzf "$out" -C "$(dirname "$out")"
  touch "$marker"; ((dom_ok++))
done
echo "    downloaded=$dom_ok skipped=$dom_skip"

# SAT Competition main track — holdout sin fuga de datos (2024, 2025)

echo "==> SATCOMP main-track holdout (via GBD)"

if command -v uv >/dev/null 2>&1; then
  gbd() { uvx --from gbd-tools --with rich gbd "$@"; }
elif command -v python3 >/dev/null 2>&1; then
  log "uv not found, falling back to python3 -m pip"
  python3 -m pip install gbd-tools rich --break-system-packages >/dev/null
else
  echo "error: neither uv nor python3 found in PATH" >&2
  exit 1
fi

if [[ -f "$DATA_DIR/.gbd/meta.db" ]]; then
  log "meta.db exists, skipping download"
else
  log "fetching GBD meta.db"
  curl -fsSL https://benchmark-database.de/getdatabase/meta.db -o "$DATA_DIR/.gbd/meta.db"
fi
export GBD_DB="$DATA_DIR/.gbd/meta.db"

[[ "$VERBOSE" == "1" ]] && gbd info || gbd info >/dev/null 2>&1

GBD_FILE_BASE="https://benchmark-database.de/file"

download_track() {
  local track="$1" outdir="$2"
  local ok=0 skip=0 fail=0
  gbd get "track=${track}" -r filename 2>/dev/null | while read -r hash name; do
    [[ -z "$hash" || "$hash" == "hash" ]] && continue
    local out="$outdir/$name"
    if [[ -f "$out.done" ]]; then
      ((skip++)); continue
    fi
    if curl -fsSL "$GBD_FILE_BASE/$hash" -o "$out"; then
      touch "$out.done"; ((ok++)); log "ok: $name"
    else
      log "failed: $name [$hash]"; rm -f "$out"; ((fail++))
    fi
    printf '\r    %s: downloaded=%d skipped=%d failed=%d' "$track" "$ok" "$skip" "$fail"
  done
  echo
}

download_track "main_2024" "$DATA_DIR/satcomp_holdout/2024"
download_track "main_2025" "$DATA_DIR/satcomp_holdout/2025"

echo "==> Done. Data in: $DATA_DIR"
