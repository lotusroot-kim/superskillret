# superskillret

> Embedding-based **skill retrieval plugin for Claude Code**. On every user prompt it silently picks the top‑K most relevant skills from a pool of 16,783 public skills and injects them as context — so Claude gets the right "how to" reference without you preloading every skill in the system prompt.

Built on [`ThakiCloud/SkillRet-Embedding-0.6B`](https://huggingface.co/ThakiCloud/SkillRet-Embedding-0.6B) (fine‑tuned from Qwen3‑Embedding‑0.6B) over the [`ThakiCloud/SKILLRET`](https://huggingface.co/datasets/ThakiCloud/SKILLRET) 16,783‑skill corpus. Ships with an INT8‑quantized ONNX encoder ([`youngryankim/superskillret-onnx-int8`](https://huggingface.co/youngryankim/superskillret-onnx-int8), 598 MB) plus a prebuilt, INT8‑quantized **full‑context** embedding index ([`youngryankim/superskillret-index-fullcontext`](https://huggingface.co/datasets/youngryankim/superskillret-index-fullcontext)) that encodes each skill's `name + description + full body` up to 32k tokens. Warm retrieval is ~0.3 s end‑to‑end on CPU.

## Quickstart

In a Claude Code session:

```
/plugin marketplace add lotusroot-kim/superskillret
/plugin install superskillret@lotusroot-kim
/superskillret:install
```

`/superskillret:install` runs `scripts/install.sh` once: creates a local venv, downloads the ONNX INT8 encoder and the prebuilt skill index from Hugging Face, and writes a `.installed` marker when it finishes — typically **1–2 minutes** on a reasonable connection. (If you restart Claude Code instead, a `SessionStart` bootstrap hook fires the same script automatically; `/reload-plugins` alone does not, hence the explicit command.)

While setup is running your first user prompts get a short English notice asking you to wait. Once `.installed` lands, skill retrieval activates automatically on every subsequent prompt. No manual step required.

Follow progress with `tail -f /tmp/superskillret-install.log`. Inspect or reset the daemon at any time with `/superskillret:status` and `/superskillret:stop`.

## How it works

![superskillret end-to-end flow](figure/superskillret.png)

Every user prompt flows through the same pipeline:

1. **User** sends a prompt.
2. The **`UserPromptSubmit` hook** (`scripts/retrieve.py`) intercepts it before Claude sees it.
3. The hook talks to the **local retrieval daemon** over a Unix socket. The daemon lazy‑starts on the first request and then stays warm — model and index are loaded **once**.
4. The daemon encodes the prompt (ONNX INT8, ~0.1 s on CPU) and runs cosine similarity against the **pre‑embedded skill pool** (16,783 public skills, prebuilt index).
5. The hook formats the top‑K `SKILL.md` bodies as an `additionalContext` JSON and emits it back to Claude Code.
6. **Claude** gets the original prompt plus the injected skills and answers with that extra reference material.

Two details not shown in the diagram but live in the code:

- **First‑time setup is automatic.** A `SessionStart` hook (`scripts/bootstrap.sh`) forks `scripts/install.sh` in the background on first load to create the venv and download the encoder + index from Hugging Face, writing a `.installed` marker when done. While setup runs, the hook shows a polite English wait‑notice instead of retrieval.
- **Per‑session dedup.** The daemon tracks, per `session_id`, which skills it already returned and skips them in future retrievals for that session, so the same `SKILL.md` isn't re‑injected into every turn. Clear with `/superskillret:reset`.

## Configuration

### Retrieval hyperparameters

These two knobs control the **quality vs. token‑cost** tradeoff. Every prompt injects up to `TOP_K` full `SKILL.md` bodies (~2–5 KB each) into Claude's context.

| Variable | Default | Meaning |
|---|---|---|
| `SUPERSKILLRET_TOP_K` | **`3`** | How many skills to inject per prompt. Higher = more context, more input tokens. |
| `SUPERSKILLRET_MIN_SCORE` | **`0.30`** | Drop hits below this cosine score. Higher = stricter, fewer (sometimes zero) hits. |

Token‑cost calibration (empirical, ~42 KB average additional context at `TOP_K=5 / MIN_SCORE=0.25`):

| Setting | Extra input tokens / prompt | Relative cost |
|---|---|---|
| `TOP_K=5`, `MIN_SCORE=0.25` | ~10,000 (median) | baseline |
| **`TOP_K=3`, `MIN_SCORE=0.30`** (default) | **~5,000–7,000** | **~40–50 % off** — recall‑leaning |
| `TOP_K=3`, `MIN_SCORE=0.40` | ~3,000–6,000 | ~60 % off — stricter |
| `TOP_K=1`, `MIN_SCORE=0.40` | ~1,500–2,500 | ~80 % off |
| `TOP_K=3`, `MIN_SCORE=0.55` | 0–3,000 (often 0) | aggressive; most off‑topic prompts inject nothing |

Bump `MIN_SCORE` if you want fewer, more relevant hits; lower it if you want more recall.

### Runtime settings

| Variable | Default | Meaning |
|---|---|---|
| `SUPERSKILLRET_DISABLE` | unset | set to `1` to turn the hook into a no‑op (retrieval is silently skipped) |
| `SUPERSKILLRET_SPAWN_WAIT` | `180` | seconds the hook waits for a lazy‑spawned daemon to come up. Raise on slow networks / first‑time installs. |
| `SUPERSKILLRET_SEEN_TRACKING` | `1` | per‑session dedup: skip skills already returned in the same Claude Code session. Set to `0` to always return the absolute top‑K. |
| `SUPERSKILLRET_ONNX_REPO` | `youngryankim/superskillret-onnx-int8` | HF repo the encoder is fetched from. Override to use your own fine‑tuned encoder. |
| `SUPERSKILLRET_INDEX_REPO` | `youngryankim/superskillret-index-fullcontext` | HF repo the prebuilt index is fetched from. Override if you publish your own skill corpus. |

Variables can be set in the shell, in `~/.claude/settings.json` under `"env": {...}`, or in the hook `command` itself. A handful of lower‑level knobs (socket/pid/log paths, ONNX dir override, session‑dedup internals, `BACKEND=pytorch` fallback) live in the daemon docstring if you need them.

### Forcing a re‑install / re‑fetch

| Command | Effect |
|---|---|
| `FORCE=1 bash scripts/install.sh` | wipe `.venv/` and re‑run every step from scratch |
| `SUPERSKILLRET_SKIP_PREBUILT=1 bash scripts/install.sh` | build the embedding index locally instead of fetching it from HF (useful when customizing the skill pool) |

### CPU‑only

superskillret runs on CPU by design. A single ONNX INT8 forward pass per prompt is ~0.1 s, which fits inside Claude's own answer latency — there is no GPU path to configure.

## Latency

End‑to‑end, hook invocation to `additionalContext` emitted.

| Call | Latency | Notes |
|---|---|---|
| First prompt in a session (daemon cold; loads ONNX encoder + index) | ~15–30 s | one‑time per session |
| Warm prompt | **~0.3 s** | ~0.1 s hook Python startup + ~0.2 s daemon work |

Retrieval quality vs. the un‑quantized FP32 reference: top‑1 skill identical across the smoke‑test queries; top‑5 overlap ~80 % (same topic, minor reshuffles between near‑duplicate skills in the corpus).

## Usage

You don't interact with superskillret directly — it just runs on every user prompt via the `UserPromptSubmit` hook.

| Slash command | What it does |
|---|---|
| `/superskillret:status` | daemon pid, socket state, last 20 log lines |
| `/superskillret:stop` | kill the daemon; it lazy‑starts again on the next prompt |
| `/superskillret:reset` | clear the per‑session "already seen" skill memory so the next prompt can surface any skill again |

## Files

```
superskillret/
├── .claude-plugin/
│   ├── plugin.json                # plugin manifest
│   └── marketplace.json           # self‑hosted marketplace entry (HTTPS URL source)
├── hooks/hooks.json               # registers SessionStart (bootstrap) + UserPromptSubmit (retrieve)
├── commands/
│   ├── status.md                  # /superskillret:status
│   ├── stop.md                    # /superskillret:stop
│   └── reset.md                   # /superskillret:reset (clear per-session dedup memory)
├── scripts/
│   ├── bootstrap.sh               # SessionStart: fork install.sh in background, exit in ms
│   ├── install.sh                 # one‑shot setup (venv, ONNX encoder, embedding index)
│   ├── daemon.py                  # long‑running retrieval server (Unix socket)
│   ├── retrieve.py                # UserPromptSubmit hook: thin socket client + wait‑notice
│   ├── build_index_fullcontext.py # (re)build the full‑context embedding index from a skill pool
│   ├── quantize_onnx.py           # INT8‑quantize a fresh ONNX export
│   ├── publish_index_fullcontext.py # upload cache_fullcontext/ to HF dataset repo (maintainer only)
│   ├── compare_backends.py        # PyTorch vs ONNX FP32 vs INT8 parity benchmark (dev)
│   └── smoke_test.py              # small retrieval sanity test (dev)
├── figure/
│   ├── superskillret.pdf          # flow diagram source
│   └── superskillret.png          # rendered for README inline display (300 dpi)
├── onnx_model_int8/               # (downloaded) model.onnx + tokenizer files
├── cache/                         # (downloaded) skill_embeddings(_int8|_scale).npy + metadata.jsonl
├── skill_pool/skills.jsonl        # (optional) full skill corpus, only needed for local index rebuild
└── .installed                     # marker written by install.sh on success; drives wait‑notice logic
```

## Retrieval pipeline internals

1. Each skill's `(name | description)` is encoded at build time with the SKILLRET model; embeddings are L2‑normalized.
2. The embedding index is stored as `skill_embeddings_int8.npy` + per‑vector `skill_embeddings_scale.npy` (75 % smaller than float32 with negligible quality loss); the daemon falls back to `skill_embeddings.npy` if only the float32 copy is present.
3. At query time the daemon encodes `"Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: <user prompt>"` (the query‑side prompt SKILLRET was trained with) and ranks skills by inner product.
4. Hits below `MIN_SCORE` are dropped so off‑topic prompts (small talk, meta questions) emit an empty `additionalContext` and cost zero extra tokens.
5. **Per‑session dedup**: the daemon remembers, per `session_id`, which skills it already returned in the current Claude Code session, and skips them next time. This stops the same `SKILL.md` from being re‑injected into every turn and surfaces fresh, related skills instead. The set of remembered sessions is LRU‑capped (`SUPERSKILLRET_MAX_SESSIONS`), reset per daemon restart, or clearable via `/superskillret:reset`.

Model card eval (FP32): NDCG@15 = 0.7887, Recall@10 = 0.8542. The INT8 pipeline shipped here hasn't been re‑benchmarked against the official SKILLRET eval splits — see Roadmap.

## Troubleshooting

- **First prompt just shows a "setup is running" notice.** Expected. `install.sh` is still downloading in the background. `tail -f /tmp/superskillret-install.log` to watch progress; retry the prompt in a minute.
- **Hook times out.** `install.sh` is still going but exceeded `SUPERSKILLRET_SPAWN_WAIT` (default 180 s) from the hook's point of view. Usually harmless — the install continues; try another prompt. To raise the window: export `SUPERSKILLRET_SPAWN_WAIT=300`.
- **No skills ever get injected.** Run `/superskillret:status`. If the socket is missing and ping fails, try `bash scripts/install.sh` directly in a terminal to see the full error. Common causes: HF repo unreachable, pip install failed.
- **Retrieved skills feel off.** Raise `SUPERSKILLRET_MIN_SCORE` to `0.40`–`0.45` so only strongly matching hits survive, and/or drop `SUPERSKILLRET_TOP_K` to 1–2.
- **Context window fills up too fast.** Each hit is ~2–5 KB of `SKILL.md`; lower `TOP_K` and raise `MIN_SCORE`. See the token‑cost table above.
- **Want to use a custom skill pool.** Replace `skill_pool/skills.jsonl` (one JSON record per line with `name`, `description`, `body`), run `python scripts/build_index_fullcontext.py`, then restart the daemon via `/superskillret:stop`.

## Status & roadmap

Production‑ready and installed via the `lotusroot-kim` marketplace. End‑to‑end verified in a live Claude Code session. Default backend is ONNX INT8, default embedding index is INT8‑quantized, default install path (HF prebuilt fetch) takes ~45 s on a healthy connection.

### What's shipped

- **ONNX INT8 encoder** (598 MB, ~0.1 s CPU inference) replaces the 2.4 GB PyTorch path. ~18× faster than the original 5.5 s warm latency. Published at [`youngryankim/superskillret-onnx-int8`](https://huggingface.co/youngryankim/superskillret-onnx-int8).
- **INT8‑quantized embedding index** (17 MB + 67 KB scale vs. 34 MB FP32), auto‑selected by the daemon when present. Reconstruction error mean 2e‑4 / max 8e‑4.
- **Prebuilt index** at [`youngryankim/superskillret-index-fullcontext`](https://huggingface.co/datasets/youngryankim/superskillret-index-fullcontext) (public; FP16 + INT8 + scale + metadata, ~210 MB). Encodes each skill's `name + description + full body` up to 32k tokens. `install.sh` downloads in ~5 s, falls back to a local rebuild (`scripts/build_index_fullcontext.py`) only if HF is unreachable.
- **Self‑hosted marketplace** in the same repo (`.claude-plugin/marketplace.json`, HTTPS source so SSH‑keyless installs work).
- **Auto‑bootstrap**: `SessionStart` hook forks `install.sh` in the background; `retrieve.py` shows a polite English wait‑notice until the `.installed` marker appears. No manual `bash scripts/install.sh` required for regular users.

### Known limitations

- **No idle timeout on the daemon.** It stays resident (~1.4 GB RAM) until `/superskillret:stop` or a kill.
- **Quality is spot‑checked, not formally benchmarked.** Parity vs. the FP32 PyTorch reference is top‑1 100 % / top‑5 80 % on a 10‑query smoke test. NDCG@15 / Recall@10 against the official SKILLRET eval splits has not been re‑run for the INT8 pipeline.
- **Token cost is real.** At defaults each on‑topic prompt costs ~5–7 K extra input tokens. See Configuration for the cost table.

### Roadmap — pick‑one, pick‑none

1. **Daemon idle timeout** — add `last_request_at` tracking in `scripts/daemon.py` plus a watchdog thread that `SIGTERM`s itself after N idle seconds (`SUPERSKILLRET_IDLE_TIMEOUT`, e.g. 1800 s). The socket client already lazy‑spawns, so reaping the daemon is free — the only cost is one cold‑start after idle.
2. **Regression eval against the SKILLRET benchmark** — load `queries` + `qrels` test splits, compute NDCG@15 and Recall@10 for PyTorch / ONNX FP32 / ONNX INT8. Commit as `scripts/eval.py` and run it as a gate before any encoder or index swap.
3. **Submit to the official Anthropic marketplace** — https://platform.claude.com/plugins/submit. Gets the plugin listed under `claude-plugins-official` in the `/plugin` Discover tab. Anthropic‑curated, takes days‑to‑weeks.

### Maintainer‑only — refreshing the prebuilt artefacts

Bump the index:

```bash
# bump cache_fullcontext/VERSION first, then:
python scripts/build_index_fullcontext.py --batch-size 8 --max-seq-length 32768
HF_TOKEN=... python scripts/publish_index_fullcontext.py --repo youngryankim/superskillret-index-fullcontext
```

Refresh the ONNX encoder:

```bash
optimum-cli export onnx --model ThakiCloud/SkillRet-Embedding-0.6B ./onnx_model
python scripts/quantize_onnx.py --src onnx_model --dst onnx_model_int8
# then upload onnx_model_int8/ to youngryankim/superskillret-onnx-int8 via the HF web UI or hf upload
```

## License

MIT.
