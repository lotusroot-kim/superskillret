"""Publish the full-context prebuilt embedding index to a Hugging Face dataset repo.

Usage:
    HF_TOKEN=hf_xxx python scripts/publish_index_fullcontext.py \
        --repo youngryankim/superskillret-index-fullcontext

Uploads cache_fullcontext/{skill_embeddings.npy, skill_embeddings_int8.npy,
skill_embeddings_scale.npy, skill_metadata.jsonl, VERSION, README.md}.
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent


def dataset_readme(version: str, n_skills: int, dim: int, max_seq_length: int) -> str:
    return f"""---
license: mit
tags:
  - embeddings
  - skill-retrieval
  - claude-code
---

# superskillret prebuilt index — full-context

Prebuilt embedding index for the [superskillret](https://github.com/lotusroot-kim/superskillret) Claude Code plugin.

Unlike the default index (which embeds only `name + description`), this build
encodes the **full skill body** (`name + description + body`) up to
`max_seq_length={max_seq_length}` tokens. Larger index, much higher recall on
skills whose name/description don't capture every keyword in the body.

- **Version:** {version}
- **Corpus:** [`ThakiCloud/SKILLRET`](https://huggingface.co/datasets/ThakiCloud/SKILLRET) (`train+test`)
- **Encoder:** [`ThakiCloud/SkillRet-Embedding-0.6B`](https://huggingface.co/ThakiCloud/SkillRet-Embedding-0.6B)
- **Skills indexed:** {n_skills}
- **Embedding dim:** {dim}
- **Encoded text:** `name + description + body` (truncated to {max_seq_length} tokens)
- **Normalized:** yes (inner product = cosine similarity)

## Files

| File | Description |
|---|---|
| `skill_embeddings.npy` | FP16 numpy array of shape `({n_skills}, {dim})` |
| `skill_embeddings_int8.npy` | INT8 per-row quantized array of shape `({n_skills}, {dim})` |
| `skill_embeddings_scale.npy` | float32 per-row scale of shape `({n_skills},)` — reconstruct as `(int8 / 127) * scale` |
| `skill_metadata.jsonl` | one JSON record per row, aligned with embeddings (includes `name`, `description`, `body`, `source_url`) |
| `VERSION` | integer version tag; bumped when the corpus or encoder changes |

## Usage

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="youngryankim/superskillret-index-fullcontext",
    repo_type="dataset",
    local_dir="cache/",
)
```

Downstream consumers should check `VERSION` against their cached copy before reusing local files.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. youngryankim/superskillret-index-fullcontext")
    ap.add_argument("--cache-dir", default=str(ROOT / "cache_fullcontext"))
    ap.add_argument("--max-seq-length", type=int, default=32768)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache_dir)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var is required.", file=sys.stderr)
        sys.exit(1)

    emb = cache / "skill_embeddings.npy"
    emb_int8 = cache / "skill_embeddings_int8.npy"
    emb_scale = cache / "skill_embeddings_scale.npy"
    meta = cache / "skill_metadata.jsonl"
    version_path = cache / "VERSION"
    for f in (emb, emb_int8, emb_scale, meta, version_path):
        if not f.exists():
            print(f"ERROR: missing {f}.", file=sys.stderr)
            sys.exit(1)
    version = version_path.read_text().strip()

    import numpy as np
    arr = np.load(emb, mmap_mode="r")
    n_skills, dim = int(arr.shape[0]), int(arr.shape[1])

    api = HfApi(token=token)
    print(f"creating repo {args.repo} (private={args.private})")
    create_repo(args.repo, repo_type="dataset", private=args.private,
                token=token, exist_ok=True)

    readme = cache / "README.md"
    readme.write_text(dataset_readme(version, n_skills, dim, args.max_seq_length))

    print(f"uploading {cache}/ -> {args.repo}")
    api.upload_folder(
        folder_path=str(cache),
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=[
            "skill_embeddings.npy",
            "skill_embeddings_int8.npy",
            "skill_embeddings_scale.npy",
            "skill_metadata.jsonl",
            "VERSION",
            "README.md",
        ],
        commit_message=f"Publish full-context index v{version} ({n_skills} skills, dim {dim}, max_seq_len {args.max_seq_length})",
    )
    print(f"done -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
