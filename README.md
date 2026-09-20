# Llama Studio

A local chat frontend for **llama.cpp**. It works out which models your card can
actually hold, downloads the right GGUF, runs `llama-server`, and talks to it.

Port 7806. `run.bat` on Windows, `./run.sh` elsewhere.

---

## The fit check

The same job the canirun.ai page does with dropdowns, except nothing is assumed:

| What | Where it comes from |
|---|---|
| Weights | the real GGUF byte size, read from the HuggingFace API |
| KV cache | the model's own `config.json` — layers × KV heads × head dim × context |
| Fit | weights + KV + ~0.6 GB working space against your measured VRAM |
| Speed | memory bandwidth ÷ bytes read per token, at 80% efficiency |

That last one is why quant choice matters: a token reads the whole weight file
once, so halving the file roughly doubles the speed. It is an estimate and the
app says so; where your card is not in the bandwidth table, no number is shown
rather than a guessed one.

On an **RTX 4060 8 GB at 8k context**, Qwen3 8B comes out:

| Quant | Weights | Verdict | Estimate |
|---|---|---|---|
| Q4_K_M | 4.6 GB | fits, all 36 layers on the GPU | ~44 tok/s |
| Q5_K_M | 5.3 GB | fits | ~38 tok/s |
| Q6_K | 6.3 GB | tight — every layer fits but nothing is spare | ~32 tok/s |
| Q8_0 | 8.1 GB | 27 of 36 layers on the GPU, rest on the CPU | ~13 tok/s |

Which is the useful correction to "recommended: Q6_K" — that recommendation
ignores the 1.1 GB of KV cache sitting beside the weights.

Sizes are shown in GiB, the unit a GPU reports, so they read slightly smaller
than HuggingFace's decimal figures. Same bytes.

**Plan for another card** on the Engine page re-runs every judgement against a
GPU you do not own, so you can see what an upgrade buys before buying it.
Loading a model always uses the real hardware.

---

## Running a model

1. **Engine → Install or update** fetches an official llama.cpp release build
   matching your GPU (CUDA, HIP, Metal or CPU). No compiler.
2. **Models** lists models worth running here. Open one, read the fit column,
   download the quant you want — resumable, with progress.
3. **Load** starts `llama-server` with `-ngl` set from the fit calculation, so
   as many layers as will fit go on the GPU and no more.
4. Chat. Replies stream, and each one reports its measured tokens per second.

Any GGUF repo can be checked by pasting `user/Model-GGUF` into the box on the
Models page.

### Two engines

On first run the app downloads two models into their own folders and keeps
them side by side:

- **Qwen3 8B** → `models/qwen/` — the primary engine, loaded by default.
- **Gemma 4 E4B Uncensored** (`HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive`)
  → `models/gemma/` — the second engine.

The quant is chosen by the fit calculation against the machine's real
hardware (Q4_K_M when it fits, which it does on an 8 GB card). A
**Qwen / Gemma** switcher in the top bar swaps the running engine with one
click; `-ngl` is recomputed on every switch. The smaller Gemma 4 E2B build
is in the catalogue for manual download. "Set up later" skips the whole
thing.

Parameters (context, temperature, top-p, reply limit, KV cache precision) are in
the prompt bar. Context and KV precision only take effect on reload — the button
does the reload for you. Setting KV cache to q8 halves that 1.1 GB, which is
often what moves a model from *tight* to *fits*.

---

## Big models on a small card

A 27B like `orcarouter/Qwen3.8-27B-Uncensored-GGUF` is past an RTX 4060 at every
quant — even Q2_K is ~10 GiB against 8 GB of VRAM, so roughly two thirds of the
layers sit on the GPU and the rest stream from system RAM at a few tokens a
second. The fit column says exactly how many layers land where, so the decision
is yours rather than a surprise after a 10 GB download.

Two details that model family exposes, both handled:

- **Hybrid attention.** Qwen3.8 alternates linear-attention layers with full
  attention, and only the full-attention layers keep a KV cache. Counting all 64
  layers would overstate the cache four-fold and wrongly condemn quants that
  actually fit.
- **mmproj files.** The vision projector in a GGUF repo is not a model; it is
  loaded beside one with `--mmproj`. It is listed as a companion rather than as
  a candidate.

Repos in FP8, NVFP4, INT8 or MLX are for vLLM, TensorRT and Apple MLX — llama.cpp
loads GGUF only. Both builds of that family are in the catalogue: the FP8 one is
listed for reference and says plainly that it cannot run here (and that it wants
~28 GB resident with no CPU offload), with a button through to the GGUF build.
Pasting a safetensors repo by hand does the same thing — the app checks whether
a `-GGUF` sibling actually exists before suggesting it, so the suggestion is
never a dead link.

## Memory

Two kinds, both plain files you can read and edit:

- **Persistent memory** (`data/memory.md`) is sent with every request in
  every conversation. Edit it from the **Memory** pill in the top bar, or
  let the model add to it: any reply line starting with `remember:` is
  appended (de-duplicated, size-capped). Delete lines you don't want kept.
- **Recall**: excerpts of earlier conversations that match the new
  question ride along as context, so "how did we fix that bug last
  week?" actually works. With the tiny **semantic recall** model
  installed (nomic-embed-text, ~150 MB, one click in the Memory panel),
  matching is by *meaning*: a second `llama-server --embeddings`
  instance runs it on the CPU beside the chat model, every finished
  exchange is embedded into `data/embeddings.json`, and questions are
  matched by cosine similarity. Without it, plain keyword overlap over
  `data/chats.json` is used instead. Either way it is local,
  inspectable, and switched off by one toggle.

## Images

The Images page generates pictures locally with **stable-diffusion.cpp**
(the diffusion engine from the same ggml family): one click downloads the
official release build — the CUDA variant for NVIDIA cards, with its
runtime DLLs — plus the Stable Diffusion 1.5 Q8 GGUF (~1.8 GB) into
`models/image/`. Prompt, size and steps in, PNGs out, with progress per
sampling step; everything lands in `data/images` with its prompt saved
beside it, browsable in the gallery. On an RTX 4060 a 512×512 at 20 steps
takes a few seconds. Swap any sd.cpp-compatible GGUF into `models/image/`
to change models.

## Voice

- **Speech in**: the mic button beside Send records, then transcribes
  locally with **whisper.cpp** — the official release binary running as a
  small server, with the `ggml-base` model in `models/voice/` (one-click
  download under Parameters, ~160 MB total). The browser converts your
  recording to 16 kHz WAV itself, so no converter is needed, and no audio
  ever leaves the machine. The browser's built-in cloud speech
  recognition is deliberately not used.
- **Speech out**: the Voice pill in the top bar reads replies aloud with
  the voices built into your operating system (code blocks are skipped,
  links read as "a link"); every reply also has a Speak button.

## Web search (opt-in)

The **Web** pill in the chat composer, when switched on, searches
DuckDuckGo first, hands the top results to the local model, and the reply
cites them as [1], [2]… with clickable links. It is off by default and
per-message, so the app stays fully offline unless you ask otherwise
(`LLAMA_STUDIO_SEARCH_URL` points it at a different search endpoint).

## Working on a folder of code

The Code page can open any folder on this computer: type its path, press
Open, and its files appear in the explorer (common noise like `.git`,
`node_modules` and `__pycache__` is skipped). Click files to attach them —
or **Attach all files** to hand the model the whole project (size-capped) —
and ask. Replies follow a convention where every changed file is its own
fenced block tagged `path=<relative path>`, so each block gets an
**Apply to file** button and a multi-file reply gets **Apply all files**,
which writes every changed file back to disk in one go. Every apply keeps
the previous version beside the file as `.bak`, so nothing is ever
silently lost. Reads are capped at 300 KB per file and writes can only
land inside the opened folder. For this whole-project mode the Qwen3
Coder 30B A3B entry in the catalogue is the model to reach for.

### Run &amp; fix

Set a run command under the file list — `python app.py`, `npm test`,
whatever proves the project works — and **Run** executes it inside the
open folder with the output in a console. **Run &amp; fix** is the
autonomous loop: run; on a non-zero exit, send the error output and the
current files to the model, apply the corrected files it returns (each
with a `.bak` of the previous version), and run again — up to three
rounds, stopping the moment the command passes, the model returns no
file changes, or the rounds run out. The command is always the one you
typed; model output never chooses what gets executed.

## Documents and files

Everything the model writes can leave the chat as a real file, entirely in
the browser — nothing is uploaded anywhere:

- every code block has **Copy**, **Save** (named with the right extension —
  .html, .py, .js, .md and so on) and, for HTML and SVG, **Preview**, which
  opens the generated page rendered in a new tab;
- every reply has **Copy reply**, **Save .md** (the raw text) and
  **Save .html** (the reply wrapped as a clean, printable standalone
  document).

So "write me a status report page" ends as an .html file you can open,
print, host or send.

## Using other agent tools with the engine

The loaded engine is an ordinary OpenAI-compatible server on
`http://127.0.0.1:8080/v1` (shown, with a copy button, on the Engine
page). Any tool that takes a custom OpenAI-compatible provider — DeepSeek
Harness (`dsh`), Aider, Continue and the like — can be pointed at it and
will drive whichever model the switcher has loaded; use the loaded file's
name as the model name. The big DeepSeek V4-Flash model those tools
default to does not fit consumer hardware, but the harness driving this
app's Coder engine does.

## Notes

- Conversations are stored in `data/chats.json`. The user's message is saved
  before generation starts, and a partial reply is kept if you press Stop, so
  nothing is lost mid-answer.
- Models live in `./models`. Delete from the Models page.
- Split GGUF files (`-of-` in the name) are listed but not auto-downloaded;
  they need joining by hand.
- Nothing leaves the machine except the model downloads themselves.

```
server.py       Flask API — chat streaming, fit, downloads, engine control
fit.py          Hardware detection, model catalogue, the fit calculator
engine.py       llama.cpp release install, llama-server process, GGUF downloads
web/index.html  The interface — one file, no build step
```

Port: `LLAMA_STUDIO_PORT`. `LLAMA_STUDIO_NO_BROWSER=1` stops it opening a tab.
