---
name: install
description: Run the one-time superskillret setup (creates the venv, downloads the ONNX INT8 encoder and the prebuilt skill index from Hugging Face). Idempotent — re-runs are no-ops once the .installed marker is in place. Use this if you installed via /plugin install and don't want to restart Claude Code.
disable-model-invocation: true
---

## Run superskillret one-time setup

```!
bash "${CLAUDE_PLUGIN_ROOT}/scripts/install.sh"
```

If the script printed `install complete.` (or it was already installed), skill retrieval will activate automatically on your next user prompt — no restart needed. If it failed, the error is at the bottom of the output above; the most common causes are HF unreachable and pip install failures.
