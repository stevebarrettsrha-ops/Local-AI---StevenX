"""
fit.py - what this machine can actually run, and how fast.

The same job the canirun.ai page does with dropdowns, except the numbers come
from the machine in front of you and from the real file sizes on HuggingFace
rather than from a table of assumptions:

  weights   the actual GGUF byte size, read from the HuggingFace API
  KV cache  computed from the model's own config.json (layers, KV heads,
            head dim) at the context length you have chosen
  fit       weights + KV + a working allowance against measured free VRAM
  speed     memory bandwidth / bytes-read-per-token, which is what makes a
            quantised model's speed predictable at all
  context   on Auto, the longest window that costs no GPU layer
  sampling  the model's own generation_config.json, when it has one

Where a number is a guess it says so. Nothing here invents a benchmark.
"""

from __future__ import annotations

import json
import math
import mmap
import os
import platform
import re
import shutil
import struct
import subprocess
from pathlib import Path

import requests

# --------------------------------------------------------------------------- #
# catalogue
# --------------------------------------------------------------------------- #
# Repos that publish GGUF quants directly. Sizes are deliberately NOT stored:
# they are read from the HuggingFace API so they cannot drift out of date.
CATALOGUE = [
    {"id": "qwen3-8b", "name": "Qwen3 8B", "params": 8.2, "family": "qwen",
     "repo": "Qwen/Qwen3-8B-GGUF", "config_repo": "Qwen/Qwen3-8B",
     "note": "Strong general model with a thinking mode. The default pick on "
             "an 8 GB card, and the primary engine here."},
    {"id": "qwen3-4b", "name": "Qwen3 4B", "params": 4.0,
     "repo": "Qwen/Qwen3-4B-GGUF", "config_repo": "Qwen/Qwen3-4B",
     "note": "Half the size, most of the manners. Leaves room for long "
             "context."},
    {"id": "qwen3-14b", "name": "Qwen3 14B", "params": 14.8,
     "repo": "Qwen/Qwen3-14B-GGUF", "config_repo": "Qwen/Qwen3-14B",
     "note": "Noticeably sharper, but spills off an 8 GB card."},
    {"id": "llama-3.1-8b", "name": "Llama 3.1 8B Instruct", "params": 8.0,
     "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
     "config_repo": "meta-llama/Llama-3.1-8B-Instruct",
     "note": "The steady all-rounder. 128k context if you have the memory."},
    {"id": "gemma-3-4b", "name": "Gemma 3 4B", "params": 4.3,
     "repo": "bartowski/google_gemma-3-4b-it-GGUF",
     "config_repo": "google/gemma-3-4b-it",
     "note": "Small, fast, reads images too."},
    {"id": "gemma-3-12b", "name": "Gemma 3 12B", "params": 12.2,
     "repo": "bartowski/google_gemma-3-12b-it-GGUF",
     "config_repo": "google/gemma-3-12b-it",
     "note": "Good writing. Tight on 8 GB even at Q4."},
    {"id": "gemma-4-e4b-unc", "name": "Gemma 4 E4B Uncensored", "params": 4.0,
     "family": "gemma",
     "repo": "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
     "config_repo": "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
     "note": "Community abliterated build, ~4B effective parameters. The "
             "second engine: downloaded on first run beside Qwen and a click "
             "away on the switcher. Comfortable on an 8 GB card at Q4."},
    {"id": "gemma-4-e2b-unc", "name": "Gemma 4 E2B Uncensored", "params": 2.0,
     "family": "gemma",
     "repo": "HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive",
     "config_repo": "HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive",
     "note": "The small sibling of the E4B build — half the memory, quicker "
             "replies, for when the big one is busy or VRAM is short."},
    {"id": "mistral-7b", "name": "Mistral 7B Instruct v0.3", "params": 7.2,
     "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
     "config_repo": "mistralai/Mistral-7B-Instruct-v0.3",
     "note": "Old but fast and undemanding."},
    {"id": "qwen2.5-coder-7b", "name": "Qwen2.5 Coder 7B", "params": 7.6,
     "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
     "config_repo": "Qwen/Qwen2.5-Coder-7B-Instruct",
     "note": "For code. Fill-in-the-middle as well as chat."},
    {"id": "qwen3-coder-30b", "name": "Qwen3 Coder 30B A3B", "params": 30.5,
     "family": "coder",
     "repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
     "config_repo": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
     "note": "The big agentic coder. Mixture of experts — only ~3B of the "
             "30B runs per token — so every layer runs on the card while "
             "the experts that do not fit wait in system RAM: the practical "
             "way to run a big coder on an 8 GB card beside 32 GB of RAM. "
             "Quantise the KV cache (q8) for long contexts."},
    {"id": "glm-4.7-flash", "name": "GLM 4.7 Flash", "params": 30.0,
     "family": "glm",
     "repo": "unsloth/GLM-4.7-Flash-GGUF",
     "config_repo": "zai-org/GLM-4.7-Flash",
     "note": "30B-class mixture of experts tuned for agentic work; the Q4 "
             "builds are notably smaller than Qwen3 Coder's, so more of it "
             "fits on an 8 GB card. A strong second opinion for code."},
    {"id": "qwen38-27b-unc-fp8", "name": "Qwen3.8 27B Uncensored — FP8",
     "params": 27.0, "runtime": "vllm",
     "repo": "orcarouter/Qwen3.8-27B-Uncensored-FP8",
     "config_repo": "orcarouter/Qwen3.8-27B-Uncensored",
     "gguf": "orcarouter/Qwen3.8-27B-Uncensored-GGUF",
     "vram_needed": 30 * 1024 ** 3,
     "note": "Block-FP8 for vLLM serving, 262K context, MTP head intact. "
             "llama.cpp cannot load it, and FP8 has no CPU offload — it wants "
             "the whole ~28 GB resident. Kept here so the family is in one "
             "place; the GGUF build is the one that runs locally."},
    {"id": "qwen38-27b-unc", "name": "Qwen3.8 27B Uncensored — GGUF",
     "params": 27.0,
     "repo": "orcarouter/Qwen3.8-27B-Uncensored-GGUF",
     "config_repo": "orcarouter/Qwen3.8-27B-Uncensored",
     "note": "Abliterated 27B vision model, q2_K to q8_0. Far past an 8 GB "
             "card — even Q2_K is about 11 GB — so most of it reads from "
             "system RAM. Vision needs the separate mmproj file."},
    {"id": "phi-4", "name": "Phi-4 14B", "params": 14.7,
     "repo": "bartowski/phi-4-GGUF", "config_repo": "microsoft/phi-4",
     "note": "Reasons well for its size; heavy for 8 GB."},
    {"id": "deepseek-r1-8b", "name": "DeepSeek-R1 Distill Llama 8B",
     "params": 8.0,
     "repo": "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
     "config_repo": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
     "note": "Shows its working. Slower per answer because it thinks first."},
]

QUANT_ORDER = ["Q2_K", "IQ3_XXS", "IQ3_M", "Q3_K_S", "Q3_K_M", "Q3_K_L",
               "IQ4_XS", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K",
               "Q8_0", "BF16", "F16"]
QUANT_RE = re.compile(r"(IQ\d[A-Z_]*|Q\d_[KS0-9_A-Z]*|BF16|F16|F32)", re.I)


def quant_of(filename: str) -> str:
    m = QUANT_RE.search(Path(filename).stem)
    return m.group(1).upper() if m else ""


def catalogue_match(filename: str) -> dict | None:
    """The catalogue entry a file on disk most likely came from.

    Scores each entry by how much of its name, token by token from the
    left, appears in the filename, and stops at the first missing token —
    so "Gemma-4-E4B-…" lands on Gemma 4 E4B rather than Gemma 3, and a
    file no entry describes matches nothing rather than something."""
    low = re.sub(r"[^a-z0-9.]+", "-", filename.lower())
    best, best_score = None, 0
    for entry in CATALOGUE:
        score = 0
        for tok in re.sub(r"[^a-z0-9.]+", " ", entry["name"].lower()).split():
            if tok in low:
                score += len(tok)
            else:
                break
        if score > best_score:
            best, best_score = entry, score
    return best


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #
# Published memory bandwidth, GB/s. Only used for the speed estimate, and the
# estimate is withheld entirely for a card that is not in here rather than
# guessed at.
# name -> (VRAM GB, published memory bandwidth GB/s). Used for the speed
# estimate, and as the list you can plan against when the card is not in this
# machine. A card that is not here gets no speed estimate rather than a guess.
GPUS = {
    "RTX 5090": (32, 1792), "RTX 5080": (16, 960), "RTX 5070 Ti": (16, 896),
    "RTX 5070": (12, 672), "RTX 5060 Ti": (16, 448), "RTX 5060": (8, 448),
    "RTX 4090": (24, 1008), "RTX 4080 SUPER": (16, 736), "RTX 4080": (16, 717),
    "RTX 4070 Ti SUPER": (16, 672), "RTX 4070 Ti": (12, 504),
    "RTX 4070 SUPER": (12, 504), "RTX 4070": (12, 504),
    "RTX 4060 Ti": (16, 288), "RTX 4060": (8, 272), "RTX 4050": (6, 216),
    "RTX 3090 Ti": (24, 1008), "RTX 3090": (24, 936), "RTX 3080 Ti": (12, 912),
    "RTX 3080": (10, 760), "RTX 3070 Ti": (8, 608), "RTX 3070": (8, 448),
    "RTX 3060 Ti": (8, 448), "RTX 3060": (12, 360), "RTX 3050": (8, 224),
    "RTX 2080 Ti": (11, 616), "RTX 2070": (8, 448), "RTX 2060": (6, 336),
    "GTX 1080 Ti": (11, 484), "GTX 1080": (8, 320), "GTX 1070": (8, 256),
    "GTX 1660": (6, 192), "GTX 1650": (4, 192), "GTX 1060": (6, 192),
    "A100": (80, 1555), "H100": (80, 3350), "L4": (24, 300), "T4": (16, 320),
    "RTX A6000": (48, 768), "RTX 6000 Ada": (48, 960), "RTX A4000": (16, 448),
    "RX 7900 XTX": (24, 960), "RX 7900 XT": (20, 800), "RX 7800 XT": (16, 624),
    "RX 6800 XT": (16, 512), "RX 6700 XT": (12, 384),
    "Arc A770": (16, 560), "Arc A750": (8, 512),
}
BANDWIDTH = {k.lower(): v[1] for k, v in GPUS.items()}

# A CPU-only fallback. Dual-channel DDR4/DDR5 desktop RAM lands here; it is a
# range, not a spec, so the app labels anything derived from it as rough.
SYSTEM_BANDWIDTH = 60


def gpu_info() -> dict:
    """name, total VRAM, free VRAM — via nvidia-smi, then rocm-smi."""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20)
            if out.returncode == 0 and out.stdout.strip():
                name, total, free = [x.strip() for x in
                                     out.stdout.strip().splitlines()[0].split(",")]
                return {"name": name, "vram": int(float(total)) * 1024 * 1024,
                        "vram_free": int(float(free)) * 1024 * 1024,
                        "vendor": "nvidia"}
        except Exception:
            pass
    if shutil.which("rocm-smi"):
        try:
            out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                                 capture_output=True, text=True, timeout=20)
            data = json.loads(out.stdout)
            for card in data.values():
                total = int(card.get("VRAM Total Memory (B)", 0))
                used = int(card.get("VRAM Total Used Memory (B)", 0))
                if total:
                    return {"name": card.get("Card series", "AMD GPU"),
                            "vram": total, "vram_free": total - used,
                            "vendor": "amd"}
        except Exception:
            pass
    if platform.system() == "Darwin":
        ram = ram_bytes()
        # Unified memory: roughly three quarters is addressable by the GPU.
        return {"name": "Apple Silicon (unified memory)",
                "vram": int(ram * 0.75), "vram_free": int(ram * 0.75),
                "vendor": "apple"}
    return {"name": "", "vram": 0, "vram_free": 0, "vendor": ""}


def ram_bytes() -> int:
    import os
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MS()
            stat.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        except Exception:
            return 0
    return 0


def bandwidth_of(name: str) -> int:
    low = (name or "").lower()
    if "apple" in low or "unified" in low:
        return 0
    best, best_len = 0, 0
    for key, value in BANDWIDTH.items():
        if key in low and len(key) > best_len:
            best, best_len = value, len(key)
    return best


def presets() -> list[dict]:
    return [{"name": k, "vram": v[0] * 1024 ** 3, "bandwidth": v[1]}
            for k, v in sorted(GPUS.items(), key=lambda kv: -kv[1][1])]


def hardware(cfg: dict | None = None) -> dict:
    """What is actually in this machine — or, if you have asked to plan for a
    different card, that card with this machine's RAM."""
    plan = (cfg or {}).get("plan_for") or ""
    if plan and plan in GPUS:
        vram_gb, bw = GPUS[plan]
        return {"name": plan + " (planned, not measured)",
                "vram": vram_gb * 1024 ** 3, "vram_free": vram_gb * 1024 ** 3,
                "bandwidth": bw, "vendor": "planned", "planned": True,
                "ram": ram_bytes(), "os": platform.system()}
    gpu = gpu_info()
    return {**gpu, "ram": ram_bytes(), "bandwidth": bandwidth_of(gpu["name"]),
            "planned": False, "os": platform.system()}


# --------------------------------------------------------------------------- #
# model config (for an exact KV cache, not a rule of thumb)
# --------------------------------------------------------------------------- #
_CONFIG_CACHE: dict[str, dict] = {}


def model_config(cfg: dict, repo: str) -> dict:
    """layers / kv heads / head dim from the model's own config.json."""
    if repo in _CONFIG_CACHE:
        return _CONFIG_CACHE[repo]
    endpoint = (cfg.get("hf_endpoint") or "https://huggingface.co").rstrip("/")
    token = (cfg.get("hf_token") or "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.get(f"{endpoint}/{repo}/resolve/main/config.json",
                         headers=headers, timeout=20)
        r.raise_for_status()
        c = r.json()
        text = c.get("text_config") or c
        layers = int(text.get("num_hidden_layers") or 0)
        # Hybrid models (Qwen3-Next / Qwen3.8 and friends) alternate linear
        # attention with full attention, and only the full-attention layers
        # keep a KV cache. Counting every layer overstates it several times
        # over, which would wrongly condemn a model as too big.
        kv_layers = layers
        types = text.get("layer_types") or text.get("layers_block_type")
        if isinstance(types, list) and types:
            full = sum(1 for t in types if "full" in str(t).lower()
                       or str(t).lower() in ("attention", "attn"))
            if full:
                kv_layers = full
        elif text.get("full_attention_interval"):
            step = int(text["full_attention_interval"])
            if step > 1:
                kv_layers = max(1, layers // step)
        heads = int(text.get("num_attention_heads") or 0)
        kv_heads = int(text.get("num_key_value_heads") or heads or 0)
        hidden = int(text.get("hidden_size") or 0)
        head_dim = int(text.get("head_dim") or
                       (hidden // heads if heads else 0))
        # Mixture-of-experts models read only a few experts per token, so
        # "file size ÷ bandwidth" would overstate the read several-fold.
        # The exact bytes per token cannot be measured from config.json
        # (shared weights are always read), so the speed estimate is
        # withheld for these rather than invented.
        experts = int(text.get("num_experts") or text.get("n_routed_experts")
                      or text.get("num_local_experts") or 0)
        if layers and kv_heads and head_dim:
            out = {"layers": layers, "kv_layers": kv_layers,
                   "kv_heads": kv_heads, "head_dim": head_dim,
                   "exact": True,
                   "hybrid": kv_layers != layers,
                   "moe": experts > 1,
                   # the window it was trained on: Auto context never
                   # stretches a model past it
                   "max_ctx": int(text.get("max_position_embeddings")
                                  or c.get("max_position_embeddings") or 0)}
            _CONFIG_CACHE[repo] = out
            return out
    except Exception:
        pass
    return {"layers": 32, "kv_layers": 32, "kv_heads": 8, "head_dim": 128,
            "exact": False, "hybrid": False, "moe": False, "max_ctx": 0}


# --------------------------------------------------------------------------- #
# sampling (the model authors' own settings, not a slider's)
# --------------------------------------------------------------------------- #
_SAMPLING_CACHE: dict[str, dict] = {}

# generation_config.json key -> llama-server request field, with the range a
# sane value falls in. Anything outside it discards the whole profile.
_SAMPLING_KEYS = (("temperature", "temperature", 0.0, 2.0),
                  ("top_p", "top_p", 0.0, 1.0),
                  ("top_k", "top_k", 0, 1000),
                  ("min_p", "min_p", 0.0, 1.0),
                  ("repetition_penalty", "repeat_penalty", 0.5, 2.0))


def model_sampling(cfg: dict, repo: str) -> dict | None:
    """The sampling the model's authors ship in its generation_config.json.

    Thinking models are tuned for a particular temperature / top-k / top-p
    and degrade badly away from it (Qwen3's card: greedy or near-greedy
    decoding "can lead to performance degradation and endless repetitions").
    So the numbers are read from the model's own repo, like config.json is,
    never from a table here. A key the file leaves out takes the value the
    authors' reference stack (transformers) uses when it is absent — which
    is also why min-p is 0: llama.cpp's 0.05 is its own addition.

    {} means the model publishes no sampling: the sliders apply. None means
    the repo could not be reached, which says nothing either way."""
    if not repo:
        return {}
    if repo in _SAMPLING_CACHE:
        return dict(_SAMPLING_CACHE[repo])
    endpoint = (cfg.get("hf_endpoint") or "https://huggingface.co").rstrip("/")
    token = (cfg.get("hf_token") or "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.get(f"{endpoint}/{repo}/resolve/main/"
                         "generation_config.json", headers=headers, timeout=20)
        if r.status_code == 404:
            _SAMPLING_CACHE[repo] = {}
            return {}
        r.raise_for_status()
        g = r.json()
    except Exception:
        return None    # a network blip is not cached as "no settings"
    out: dict = {}
    if isinstance(g, dict) and g.get("do_sample") is not False:
        for src, dst, lo, hi in _SAMPLING_KEYS:
            v = g.get(src)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if not (math.isfinite(v) and lo <= v <= hi):
                out = {}
                break
            out[dst] = int(v) if dst == "top_k" else float(v)
    if out:
        out = {"temperature": 1.0, "top_p": 1.0, "top_k": 50,
               "min_p": 0.0, "repeat_penalty": 1.0, **out,
               "source": repo}
    _SAMPLING_CACHE[repo] = out
    return dict(out)


# --------------------------------------------------------------------------- #
# the GGUF's own tensor table (where a file's bytes actually are)
# --------------------------------------------------------------------------- #
# llama.cpp's LLM_FFN_EXPS_REGEX, anchored to a block: the tensors that
# --n-cpu-moe N keeps in system RAM for blocks 0..N-1. Shared experts
# (ffn_*_shexp) are not among them and stay with the rest of the layer.
EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(?:up|down|gate|gate_up)_(?:ch)?exps")

# GGUF value types with a fixed width, keyed by type id.
_GGUF_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_LAYOUT_CACHE: dict[tuple, dict | None] = {}


def gguf_layout(path: str) -> dict | None:
    """How a model file's bytes divide between expert weights, block by
    block, and everything else — read from the file's own tensor table, not
    estimated from the parameter count.

    Only the header is read (it is memory-mapped, so a 17 GB file costs a
    few MB of reading). Each tensor's size is the gap to the next tensor's
    offset, which includes its alignment padding: a few bytes, never less
    than the truth. None for anything that is not a readable GGUF."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _LAYOUT_CACHE:
        try:
            _LAYOUT_CACHE[key] = _read_layout(path, st.st_size)
        except Exception:
            _LAYOUT_CACHE[key] = None
    return _LAYOUT_CACHE[key]


def _read_layout(path: str, size: int) -> dict | None:
    with open(path, "rb") as fh, \
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        if mm[:4] != b"GGUF" or struct.unpack_from("<I", mm, 4)[0] < 2:
            return None
        n_tensors, n_kv = struct.unpack_from("<QQ", mm, 8)
        pos = 24

        def text(at: int) -> tuple[str, int]:
            n = struct.unpack_from("<Q", mm, at)[0]
            return (mm[at + 8:at + 8 + n].decode("utf-8", "replace"),
                    at + 8 + n)

        def value(kind: int, at: int):
            if kind in _GGUF_FIXED:
                fmt = _GGUF_FIXED[kind]
                return (struct.unpack_from(fmt, mm, at)[0],
                        at + struct.calcsize(fmt))
            if kind == 8:
                return text(at)
            if kind == 9:
                inner, count = struct.unpack_from("<IQ", mm, at)
                at += 12
                if inner in _GGUF_FIXED:          # skipped in one step
                    return None, at + count * struct.calcsize(
                        _GGUF_FIXED[inner])
                for _ in range(count):            # the vocabulary: 150k+
                    if inner == 8:
                        at += 8 + struct.unpack_from("<Q", mm, at)[0]
                    else:
                        _, at = value(inner, at)
                return None, at
            raise ValueError(f"unknown GGUF value type {kind}")

        meta = {}
        for _ in range(n_kv):
            name, pos = text(pos)
            kind = struct.unpack_from("<I", mm, pos)[0]
            meta[name], pos = value(kind, pos + 4)
        tensors = []
        for _ in range(n_tensors):
            name, pos = text(pos)
            dims = struct.unpack_from("<I", mm, pos)[0]
            pos += 4 + 8 * dims + 4                # dims, then ggml type
            tensors.append((struct.unpack_from("<Q", mm, pos)[0], name))
            pos += 8
    align = int(meta.get("general.alignment") or 32)
    data = -(-pos // align) * align
    tensors.sort()
    experts: dict[int, int] = {}
    other = 0
    for i, (offset, name) in enumerate(tensors):
        end = tensors[i + 1][0] if i + 1 < len(tensors) else size - data
        if end < offset:
            return None
        m = EXPS_RE.match(name)
        if m:
            experts[int(m.group(1))] = experts.get(int(m.group(1)), 0) + \
                end - offset
        else:
            other += end - offset
    if data > size:
        return None
    arch = str(meta.get("general.architecture") or "")
    blocks = max(int(meta.get(f"{arch}.block_count") or 0),
                 max(experts) + 1 if experts else 0)
    return {"arch": arch, "blocks": blocks,
            # expert bytes per block, blk.0 first; [] for a dense model
            "experts": [experts.get(i, 0) for i in range(blocks)]
            if experts else [],
            # every other tensor: attention, norms, shared experts, the
            # embeddings and the output head
            "other": other,
            "expert_count": int(meta.get(f"{arch}.expert_count") or 0),
            "expert_used": int(meta.get(f"{arch}.expert_used_count") or 0)}


def kv_bytes(conf: dict, ctx: int, kv_bits: int = 16) -> int:
    """Key and value, every layer, at the chosen context length."""
    per_token = 2 * conf.get("kv_layers", conf["layers"]) * conf["kv_heads"] \
        * conf["head_dim"] * (kv_bits / 8)
    return int(per_token * ctx)


# --------------------------------------------------------------------------- #
# the fit calculation
# --------------------------------------------------------------------------- #
OVERHEAD = 600 * 1024 * 1024   # CUDA context, compute buffers, a little slack


def assess(weights: int, conf: dict, hw: dict, ctx: int = 8192,
           kv_bits: int = 16) -> dict:
    """What happens if this file is run on this machine at this context."""
    vram = hw.get("vram") or 0
    ram = hw.get("ram") or 0
    kv = kv_bytes(conf, ctx, kv_bits)
    needed = weights + kv + OVERHEAD

    layers = max(conf["layers"], 1)
    per_layer = weights / layers
    if not vram:
        gpu_layers = 0
    else:
        room = vram - kv - OVERHEAD
        gpu_layers = int(max(0, min(layers, room // per_layer))) if per_layer \
            else 0

    if not vram:
        verdict = "cpu"
    elif needed <= vram * 0.95:
        verdict = "fits"
    elif gpu_layers >= layers * 0.75:
        verdict = "tight"
    else:
        verdict = "spills"

    # Speed: a token reads the whole model once, so bandwidth over bytes is the
    # ceiling. 0.8 is the usual real-world fraction of it.
    bw = hw.get("bandwidth") or 0
    speed = None
    # Below ~50 MB it is not a real model file (a stub, or a truncated
    # download), and bandwidth over size would report a fantasy number.
    # MoE models read only part of the file per token, so the formula does
    # not apply — no number beats a wrong one.
    if bw and weights > 50 * 1024 * 1024 and not conf.get("moe"):
        on_gpu = min(gpu_layers / layers, 1.0) if layers else 0
        gpu_part = weights * on_gpu
        cpu_part = weights * (1 - on_gpu)
        seconds = (gpu_part / (bw * 1e9)) + (cpu_part / (SYSTEM_BANDWIDTH * 1e9))
        if seconds > 0:
            speed = round(0.8 / seconds, 1)
            if speed > 2000:
                speed = None

    fits_ram = bool(ram and needed <= ram * 0.9)
    return {"weights": weights, "kv": kv, "needed": needed,
            "gpu_layers": gpu_layers, "layers": layers, "verdict": verdict,
            "speed": speed, "speed_is_estimate": True,
            "exact_kv": conf.get("exact", False), "fits_ram": fits_ram,
            "hybrid": conf.get("hybrid", False),
            "moe": conf.get("moe", False),
            "ctx": ctx}


def moe_plan(layout: dict | None, conf: dict, hw: dict, ctx: int,
             kv_bits: int = 16) -> dict | None:
    """Where a mixture-of-experts model goes when it is bigger than the card:
    every layer on the GPU — attention, norms, shared experts, KV cache —
    and the expert weights that do not fit held in system RAM, the first
    blocks' first, which is what llama.cpp's --n-cpu-moe N does.

    That beats splitting whole layers (plain -ngl) by a wide margin: a token
    reads only expert_used of expert_count experts in each layer, so the
    part left in slow RAM is a small fraction of what is read, while
    attention — read in full every token — stays on the fast card.

    Sizes come from the file's own tensor table (gguf_layout). None when
    this is not a MoE file, or when even the non-expert weights and the KV
    cache do not fit — the plain layer split is all that is left then."""
    if not layout or not layout.get("experts") or not hw.get("vram"):
        return None
    kv = kv_bytes(conf, ctx, kv_bits)
    room = hw["vram"] - kv - OVERHEAD - layout["other"]
    if room < 0:
        return None
    per = layout["experts"]
    kept, n_cpu = 0, len(per)
    # --n-cpu-moe keeps blocks 0..N-1 in RAM, so the card is filled from
    # the last block backwards.
    for i in range(len(per) - 1, -1, -1):
        if kept + per[i] > room:
            break
        kept += per[i]
        n_cpu = i
    return {"n_cpu_moe": n_cpu, "layers": layout["blocks"],
            "experts_gpu": kept, "experts_cpu": sum(per) - kept,
            "other": layout["other"], "kv": kv,
            "expert_used": layout.get("expert_used", 0),
            "expert_count": layout.get("expert_count", 0)}


# Auto context steps. 8k is the floor: below it a thinking model can spend
# the whole window deliberating and never reach the answer. 32k is where Auto
# stops — the native window of most models here, and what Qwen3 asks for a
# thinking answer; longer windows are picked by hand.
AUTO_CTX_STEPS = (8192, 16384, 32768)


def auto_ctx(weights: int, conf: dict, hw: dict, kv_bits: int = 16,
             layout: dict | None = None) -> int:
    """The context Auto gives a model: the longest step that costs nothing.

    A step is taken only while the model keeps every GPU layer it had at the
    floor, keeps fitting if it fitted there, and stays inside the window it
    was trained on (config.json). So a model with VRAM to spare gets a long
    context, and one that already spills is not pushed further off the card.

    A mixture-of-experts model placed with moe_plan is judged differently:
    its layers never leave the card, and a step of context only moves about
    one more block of experts to RAM, of which a token reads a small
    fraction. So it takes the longest step at which the attention and the
    cache still fit on the card whole.
    """
    cap = int(conf.get("max_ctx") or 0)
    steps = [s for s in AUTO_CTX_STEPS if not cap or s <= cap]
    if not steps:
        return max(512, cap)
    floor = assess(weights, conf, hw, steps[0], kv_bits)
    plan = moe_plan(layout, conf, hw, steps[0], kv_bits)
    if plan and plan["n_cpu_moe"] and floor["gpu_layers"] < floor["layers"]:
        best = steps[0]
        for step in steps[1:]:
            if not moe_plan(layout, conf, hw, step, kv_bits):
                break
            if floor["fits_ram"] and not assess(weights, conf, hw, step,
                                                kv_bits)["fits_ram"]:
                break
            best = step
        return best
    best = steps[0]
    for step in steps[1:]:
        a = assess(weights, conf, hw, step, kv_bits)
        if a["gpu_layers"] < floor["gpu_layers"]:
            break
        if floor["verdict"] == "fits" and a["verdict"] != "fits":
            break
        if floor["fits_ram"] and not a["fits_ram"]:
            break
        best = step
    return best


# Everything memory-shaped is shown in GiB, because that is the unit a GPU
# reports its VRAM in. HuggingFace lists file sizes in decimal GB, so a file
# the repo calls 4.9 GB shows here as 4.6 GB — same bytes, honest unit.
def verdict_text(a: dict, hw: dict) -> str:
    gb = 1024 ** 3
    experts_ride = a.get("moe") and a["verdict"] in ("tight", "spills") \
        and not a.get("moe_layer_split")
    if experts_ride:
        # A MoE model bigger than the card does not lose whole layers: its
        # experts go to RAM instead, so a layer count would mislead.
        base = (f"Bigger than the card: {a['needed']/gb:.1f} GB against "
                f"{hw.get('vram', 0)/gb:.1f} GB.")
    elif a["verdict"] == "fits":
        base = (f"Fits: {a['weights']/gb:.1f} GB of weights plus "
                f"{a['kv']/gb:.1f} GB of KV cache at {a['ctx']:,} tokens, "
                f"inside {hw.get('vram', 0)/gb:.1f} GB.")
    elif a["verdict"] == "tight":
        if a["gpu_layers"] >= a["layers"]:
            base = (f"Tight: every layer fits, but {a['needed']/gb:.1f} GB "
                    f"against {hw.get('vram', 0)/gb:.1f} GB leaves nothing "
                    "spare. Anything else using the GPU will push it over.")
        else:
            base = (f"Tight: needs {a['needed']/gb:.1f} GB against "
                    f"{hw.get('vram', 0)/gb:.1f} GB, so "
                    f"{a['gpu_layers']} of {a['layers']} layers sit on the GPU "
                    "and the rest on the CPU. A shorter context buys them back.")
    elif a["verdict"] == "spills":
        base = f"Spills: only {a['gpu_layers']} of {a['layers']} layers fit."
        if not a.get("moe"):
            base += (" Most of each token is read from system RAM, which is "
                     "roughly an order of magnitude slower.")
    else:
        base = "No GPU detected, so this runs on the CPU."
    if a.get("hybrid"):
        base += (" Hybrid attention: only the full-attention layers hold a KV "
                 "cache, which is why it is smaller than the layer count "
                 "suggests.")
    if a.get("moe"):
        if a.get("n_cpu_moe"):
            used, count = a.get("expert_used"), a.get("expert_count")
            share = (f"only {used} of {count} experts"
                     if used and count else "only a few experts")
            base += (f" Mixture of experts: every layer runs on the GPU and "
                     f"the experts of {a['n_cpu_moe']} of {a['layers']} "
                     "blocks wait in system RAM (--n-cpu-moe). A token reads "
                     f"{share} in each block, so that costs far less than "
                     "moving whole layers off the card.")
        elif a.get("moe_layer_split") == "fallback":
            base += (" Mixture of experts, but llama-server did not start "
                     "with its experts held in system RAM (a build older "
                     "than --n-cpu-moe, or not enough room), so whole layers "
                     "are split instead. Updating llama.cpp on the Engine "
                     "page usually brings the faster placement back.")
        elif a.get("moe_layer_split"):
            base += (" Mixture of experts, but even its attention and cache "
                     "do not fit on the card, so whole layers are split "
                     "instead.")
        elif experts_ride:
            base += (" Mixture of experts: it loads with every layer on the "
                     "GPU and the experts that do not fit in system RAM "
                     "(--n-cpu-moe). A token reads only a few experts per "
                     "block, so that costs far less than moving whole layers "
                     "off the card.")
        base += (" No speed estimate — the bytes read per token cannot be "
                 "measured from the file size.")
    if a.get("auto_ctx"):
        base += (f" Auto context gives it {a['ctx']:,} tokens — the longest "
                 + ("window at which its attention and cache still fit on "
                    "the card." if a.get("n_cpu_moe") else
                    "window that costs no GPU layer."))
    if not a["exact_kv"]:
        base += (" KV cache is estimated — the model's config.json could not "
                 "be read.")
    return base
