#!/usr/bin/env bash
# superskillret: first-time setup
#
# - Creates a local venv in .venv/
# - Installs torch (CPU wheel) + sentence-transformers + datasets + numpy
# - Downloads the SKILLRET skill pool (16,783 skills) as skill_pool/skills.jsonl
# - Downloads the SKILLRET embedding model (~1.2GB) via Hugging Face cache
# - Fetches the prebuilt embedding index from Hugging Face Hub; falls back to
#   building it locally if the prebuilt index is unavailable.
#
# Re-running is safe: steps are skipped when their outputs already exist.
# Set FORCE=1 to rebuild from scratch.
# Set SUPERSKILLRET_INDEX_REPO to override the prebuilt-index dataset repo.
# Set SUPERSKILLRET_SKIP_PREBUILT=1 to always build the index locally.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

INDEX_REPO="${SUPERSKILLRET_INDEX_REPO:-youngryankim/superskillret-index-fullcontext}"
ONNX_REPO="${SUPERSKILLRET_ONNX_REPO:-youngryankim/superskillret-onnx-int8}"

log() { printf '[superskillret] %s\n' "$*"; }

# 1. Python
if ! command -v python3 >/dev/null 2>&1; then
  log "ERROR: python3 not found. Install Python 3.10+ first."
  exit 1
fi
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
log "using system python3 $PYVER"

# 2. venv
if [ ! -d "$VENV" ] || [ "${FORCE:-0}" = "1" ]; then
  log "creating venv at $VENV"
  rm -rf "$VENV"
  python3 -m venv "$VENV"
else
  log "venv exists, reusing ($VENV)"
fi

# 3. deps
log "upgrading pip"
"$PIP" install --quiet --upgrade pip

# superskillret is CPU-only — always install the CPU torch wheel.
if ! "$PY" -c "import torch" 2>/dev/null; then
  log "installing torch (CPU wheel)"
  "$PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cpu
else
  log "torch already installed"
fi

log "installing deps: sentence-transformers, onnxruntime, transformers, datasets, numpy, huggingface_hub"
"$PIP" install --quiet \
    "sentence-transformers>=3.0" \
    "onnxruntime>=1.17" \
    "transformers>=4.40" \
    "datasets>=3.0" \
    "numpy>=1.26" \
    "huggingface_hub>=0.24"

# 4. ONNX INT8 encoder (default backend; small/fast)
ONNX_DIR="$ROOT/onnx_model_int8"
if [ ! -s "$ONNX_DIR/model.onnx" ] || [ "${FORCE:-0}" = "1" ]; then
  log "fetching ONNX INT8 encoder from $ONNX_REPO (~598 MB, ~5-30s)"
  mkdir -p "$ONNX_DIR"
  ROOT="$ROOT" REPO="$ONNX_REPO" ONNX_DIR="$ONNX_DIR" "$PY" - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        repo_id=os.environ["REPO"],
        repo_type="model",
        local_dir=os.environ["ONNX_DIR"],
    )
    print("[superskillret] ONNX encoder downloaded")
except Exception as e:
    print(f"[superskillret] ONNX fetch failed: {type(e).__name__}: {e}", file=sys.stderr)
    print("[superskillret] the daemon will fall back to the PyTorch backend", file=sys.stderr)
    sys.exit(0)  # non-fatal; daemon falls back to pytorch
PYEOF
else
  log "ONNX encoder already present at $ONNX_DIR"
fi

# 5. embedding index — try prebuilt first, fall back to local build
EMB="$ROOT/cache/skill_embeddings.npy"
META="$ROOT/cache/skill_metadata.jsonl"
VER="$ROOT/cache/VERSION"
mkdir -p "$ROOT/cache"

need_index=0
if [ ! -s "$EMB" ] || [ ! -s "$META" ] || [ "${FORCE:-0}" = "1" ]; then
  need_index=1
fi

if [ "$need_index" = "1" ] && [ "${SUPERSKILLRET_SKIP_PREBUILT:-0}" != "1" ]; then
  log "trying to fetch prebuilt index from dataset: $INDEX_REPO"
  if ROOT="$ROOT" REPO="$INDEX_REPO" "$PY" - <<'PYEOF'
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError, GatedRepoError

root = Path(os.environ["ROOT"])
repo = os.environ["REPO"]
try:
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(root / "cache"),
        allow_patterns=[
            "skill_embeddings.npy",
            "skill_embeddings_int8.npy",
            "skill_embeddings_scale.npy",
            "skill_metadata.jsonl",
            "VERSION",
            "README.md",
        ],
    )
    print("[superskillret] prebuilt index downloaded")
except (HfHubHTTPError, RepositoryNotFoundError, GatedRepoError) as e:
    print(f"[superskillret] prebuilt fetch failed: {e}", file=sys.stderr)
    sys.exit(2)
except Exception as e:
    print(f"[superskillret] prebuilt fetch failed: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(2)
PYEOF
  then
    need_index=0
    if [ -f "$VER" ]; then
      log "prebuilt index version: $(cat "$VER")"
    fi
  else
    log "prebuilt index not available (private repo? no token? offline?) — will build locally"
  fi
fi

# 5. skill pool — needed either to build index locally, OR so custom workflows
# (scripts/build_index_fullcontext.py, scripts/publish_index_fullcontext.py)
# can re-run. If we already have the prebuilt index and the pool is missing,
# skip the 300MB download.
POOL="$ROOT/skill_pool/skills.jsonl"
if [ "$need_index" = "1" ] && { [ ! -s "$POOL" ] || [ "${FORCE:-0}" = "1" ]; }; then
  log "downloading SKILLRET skill pool (~300MB, required for local index build)"
  mkdir -p "$ROOT/skill_pool"
  ROOT="$ROOT" "$PY" - <<'PYEOF'
from datasets import load_dataset
import json, os
ds = load_dataset("ThakiCloud/SKILLRET", "skills", split="train+test")
out = os.path.join(os.environ["ROOT"], "skill_pool", "skills.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"wrote {len(ds)} skills")
PYEOF
elif [ -s "$POOL" ]; then
  log "skill pool already present ($(wc -l <"$POOL") skills)"
else
  log "skill pool not fetched (prebuilt index is enough for runtime retrieval)"
fi

# 6. build index locally if we still need it
if [ "$need_index" = "1" ]; then
  log "building full-context embedding index (slow on CPU; many hours — prefer the prebuilt index)"
  "$PY" "$ROOT/scripts/build_index_fullcontext.py" --batch-size 1 --max-seq-length 32768 --device cpu --out-dir "$ROOT/cache"
fi

touch "$ROOT/.installed"

log "install complete."
log "  plugin python:  $PY"
log "  socket path:    ${SUPERSKILLRET_SOCKET:-/tmp/superskillret.sock}"
log "  index:          $EMB ($(du -h "$EMB" 2>/dev/null | cut -f1))"
log "  metadata:       $META ($(du -h "$META" 2>/dev/null | cut -f1))"
log ""
log "Claude Code hook is pre-wired in hooks/hooks.json; skill retrieval"
log "will activate automatically on your next user prompt."
