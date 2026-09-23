# Llama Studio — invariants

1. **`web/index.html` stays one file, no build step.** Same shell as the other
   four apps.
2. **The fit calculation never guesses a number it could measure.**
   - weights: real GGUF size from the HuggingFace API, never a table
   - KV cache: from the model's own config.json; when that cannot be read the
     result is flagged as estimated in the UI text
   - speed: bandwidth ÷ bytes-per-token, only for cards in `GPUS`; otherwise no
     number at all. Never invent a benchmark.
3. **Speed guard:** files under 50 MB get no speed estimate, and anything over
   2000 tok/s is suppressed. A stub or truncated download otherwise reports
   fantasy numbers.
4. **Memory is displayed in GiB** (1024³) everywhere it is compared against
   VRAM, because that is the unit a GPU reports. HuggingFace's decimal sizes
   will look ~7% larger; that is expected and documented in the README.
5. **The user's message is persisted before generation starts**, and the stream
   generator saves partial replies in a `finally:` block so a disconnect or a
   Stop press keeps what was written. Losing the whole exchange on Stop was the
   original bug.
6. **`-ngl` comes from the fit calculation**, not from a fixed 999. That is what
   makes an oversized model degrade gracefully instead of failing to load.
7. **Plan-for mode is clearly labelled.** `hardware()` marks it
   `planned: True` and the name carries "(planned, not measured)". Never let a
   planned card be mistaken for the real one.
8. llama.cpp comes from an official GitHub release asset chosen by GPU vendor
   and OS. Never build from source, never vendor binaries into the repo.

9. **Hybrid attention models** (Qwen3-Next, Qwen3.8 …) keep a KV cache only on
   full-attention layers. `model_config` reads `layer_types` /
   `full_attention_interval` and sets `kv_layers` separately from `layers`.
   Never compute KV from the raw layer count for these.
10. **mmproj files are companions, not models.** Exclude them from the
   recommendation and from fit verdicts; they load with `--mmproj`.
11. **Catalogue entries may declare `runtime`.** Anything other than
   `llama.cpp` returns an `unsupported` payload with a reason and a `gguf`
   alternative instead of an error — the model stays visible, it just says
   what it needs. Never pretend such a build is loadable.
12. **Non-GGUF repos fail with a named reason** (FP8/NVFP4/INT8/MLX are other
   runtimes), not an empty list.
13. **A reply always says why it ended.** `chat_stream` yields a final
   `{"stop": reason}` carrying llama.cpp's `finish_reason`; the done event
   and the stored message keep it. Never let a length-capped reply look like
   a finished one — that is what made truncated answers pass for whole ones.
14. **The reply limit defaults to automatic** (`max_tokens: 0` = whatever the
   context has left after the prompt). Never reintroduce a fixed cap as the
   default.
15. **The prompt is budgeted before it is sent**, counted with the model's own
   `/tokenize`, and old messages are dropped here with the count reported.
   Never leave llama.cpp to context-shift silently — it drops the system
   prompt first.
16. **Sampling comes from the model's own `generation_config.json`**, read
   like config.json (`fit.model_sampling`) and saved beside the architecture
   as `last_model_sampling`. Keys it leaves out take transformers' defaults;
   min-p is 0. Never a table of per-model numbers, and never a fixed
   temperature forced by a page — the Code page's old 0.2 was near-greedy,
   which thinking models are explicitly warned off. `sampling: "manual"` is
   the person's override; a model with no profile uses the sliders.
17. **Reasoning is shown, never sent back.** `chat_stream` yields
   `{"think": …}` for `reasoning_content`; it is stored as `reasoning`, and
   history is built from `content` only. Continuations run with
   `chat_template_kwargs: {"enable_thinking": false}`. A reply that is only
   reasoning (the window ran out mid-thought) is stored and carried on with
   `ANSWER_NUDGE`, not dropped.
18. **Context defaults to Auto (`ctx: 0`)**: `fit.auto_ctx` takes the longest
   of 8k/16k/32k that costs no GPU layer, capped by the model's
   `max_position_embeddings`, judged at load against free VRAM measured
   after the old model is stopped. Auto stays the setting; the window it
   resolved to is `last_ctx`. Never reintroduce a fixed 8k default — a
   thinking model can spend all of it before writing any code.
19. **The Office-file convention rides only with Office requests**
   (`wants_documents`), never with every coding question.

## Validation gate

```bash
python -m py_compile server.py engine.py fit.py
python - <<'PY'
import re, pathlib
src = pathlib.Path('web/index.html').read_text()
pathlib.Path('/tmp/ls.js').write_text('\n'.join(re.findall(r'<script>(.*?)</script>', src, re.S)))
PY
node --check /tmp/ls.js
```

Then check element ids referenced by the JS against ids present in the markup —
that mismatch has broken a sibling app twice.
