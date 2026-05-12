"""Build full-context embedding index for the SKILLRET skill pool.

Encodes each skill as `name + description + full body`, truncated to the
encoder's max sequence length (default 32k tokens). Replaces the older
short-context build that only encoded `name + description`.

Outputs (default --out-dir cache_fullcontext/):
  - skill_embeddings.npy            (N, dim) float16
  - skill_embeddings_int8.npy       (N, dim) int8     (per-row quantized)
  - skill_embeddings_scale.npy      (N,)     float32  (per-row max abs)
  - skill_metadata.jsonl            one record per line, aligned with embeddings
  - VERSION                         integer version tag
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = ROOT / "skill_pool" / "skills.jsonl"

MODEL_NAME = "ThakiCloud/SkillRet-Embedding-0.6B"


def skill_text(rec: dict) -> str:
    name = rec.get("name", "") or ""
    desc = rec.get("description", "") or ""
    body = rec.get("body", "") or ""
    parts = []
    if name:
        parts.append(name)
    if desc:
        parts.append(desc)
    header = " | ".join(parts)
    if body:
        return f"{header}\n\n{body}"
    return header


def quantize_int8(emb_fp32: np.ndarray):
    """Per-row symmetric INT8 quantization.

    Reconstruction: (int8 / 127) * scale.
    Compatible with daemon.py:175-179.
    """
    scale = np.max(np.abs(emb_fp32), axis=1, keepdims=True)
    scale = np.where(scale == 0, 1.0, scale)
    int8 = np.clip(np.round(emb_fp32 / scale * 127), -127, 127).astype(np.int8)
    return int8, scale.squeeze(1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=1,
                    help="encoder batch size; long context => keep small")
    ap.add_argument("--max-seq-length", type=int, default=32768)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--out-dir", default=str(ROOT / "cache_fullcontext"))
    ap.add_argument("--version", default="1")
    ap.add_argument("--limit", type=int, default=0,
                    help="dev: only encode first N skills (0 = all)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "skill_embeddings.npy"
    emb_int8_path = out_dir / "skill_embeddings_int8.npy"
    emb_scale_path = out_dir / "skill_embeddings_scale.npy"
    meta_path = out_dir / "skill_metadata.jsonl"
    version_path = out_dir / "VERSION"

    if not POOL_PATH.exists():
        print(f"ERROR: {POOL_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    records = []
    with POOL_PATH.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]
    print(f"Loaded {len(records)} skills from {POOL_PATH}")

    texts = [skill_text(r) for r in records]
    char_lens = [len(t) for t in texts]
    print(f"Input chars  min={min(char_lens)} max={max(char_lens)} "
          f"mean={int(sum(char_lens)/len(char_lens))}")

    print(f"Loading model on {args.device} (max_seq_length={args.max_seq_length}, dtype={args.dtype})...")
    t0 = time.time()
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        device=args.device,
        model_kwargs={"torch_dtype": torch_dtype},
    )
    model.max_seq_length = args.max_seq_length
    print(f"Model loaded in {time.time()-t0:.1f}s | max_seq_length={model.max_seq_length}")

    print(f"Encoding {len(texts)} skills (batch={args.batch_size}) ...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    dt = time.time() - t0
    print(f"Encoded in {dt:.1f}s | shape={embeddings.shape} dtype={embeddings.dtype}")

    embeddings_fp32 = embeddings.astype(np.float32)
    embeddings_fp16 = embeddings_fp32.astype(np.float16)
    np.save(emb_path, embeddings_fp16)
    print(f"Saved FP16 embeddings to {emb_path} "
          f"({emb_path.stat().st_size/1e6:.1f} MB)")

    int8, scale = quantize_int8(embeddings_fp32)
    np.save(emb_int8_path, int8)
    np.save(emb_scale_path, scale)
    print(f"Saved INT8 embeddings to {emb_int8_path} "
          f"({emb_int8_path.stat().st_size/1e6:.1f} MB)")
    print(f"Saved scales to {emb_scale_path} "
          f"({emb_scale_path.stat().st_size/1e6:.1f} MB)")

    # Reconstruction error (sanity)
    recon = (int8.astype(np.float32) / 127.0) * scale[:, None]
    err = np.abs(recon - embeddings_fp32)
    print(f"INT8 reconstruction error  mean={err.mean():.2e}  max={err.max():.2e}")

    with meta_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "id": r.get("id"),
                "name": r.get("name"),
                "namespace": r.get("namespace"),
                "description": r.get("description"),
                "repo": r.get("repo"),
                "source_url": r.get("source_url"),
                "body": r.get("body"),
            }, ensure_ascii=False) + "\n")
    print(f"Saved metadata to {meta_path} ({meta_path.stat().st_size/1e6:.1f} MB)")

    version_path.write_text(args.version + "\n")
    print(f"Wrote version {args.version} to {version_path}")


if __name__ == "__main__":
    main()
