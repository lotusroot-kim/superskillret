---
license: mit
tags:
  - embeddings
  - skill-retrieval
  - claude-code
---

# superskillret prebuilt index

Prebuilt embedding index for the [superskillret](../superskillret) Claude Code plugin.

- **Version:** 1
- **Corpus:** [`ThakiCloud/SKILLRET`](https://huggingface.co/datasets/ThakiCloud/SKILLRET) (`train+test`)
- **Encoder:** [`ThakiCloud/SkillRet-Embedding-0.6B`](https://huggingface.co/ThakiCloud/SkillRet-Embedding-0.6B)
- **Skills indexed:** 16783
- **Embedding dim:** 1024
- **Normalized:** yes (inner product = cosine similarity)

## Files

| File | Description |
|---|---|
| `skill_embeddings.npy` | FP16 numpy array of shape `(16783, 1024)` |
| `skill_metadata.jsonl` | one JSON record per row, aligned with embeddings (includes `name`, `description`, `body`, `source_url`) |
| `VERSION` | integer version tag; bumped when the corpus or encoder changes |

## Usage

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="youngryankim/superskillret-index-fullcontext",
                  repo_type="dataset",
                  local_dir="cache/")
```

Downstream consumers should check `VERSION` against their cached copy before reusing local files.
