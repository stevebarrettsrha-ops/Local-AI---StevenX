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

Where a number is a guess it says so. Nothing here invents a benchmark.
"""

from __future__ import annotations

import json
import math
import platform
import re
import shutil
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
        if layers and kv_heads and head_dim:
            out = {"layers": layers, "kv_layers": kv_layers,
                   "kv_heads": kv_heads, "head_dim": head_dim,
                   "exact": True,
                   "hybrid": kv_layers != layers}
            _CONFIG_CACHE[repo] = out
            return out
    except Exception:
        pass
    return {"layers": 32, "kv_layers": 32, "kv_heads": 8, "head_dim": 128,
            "exact": False, "hybrid": False}


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
    if bw and weights > 50 * 1024 * 1024:
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
            "ctx": ctx}


# Everything memory-shaped is shown in GiB, because that is the unit a GPU
# reports its VRAM in. HuggingFace lists file sizes in decimal GB, so a file
# the repo calls 4.9 GB shows here as 4.6 GB — same bytes, honest unit.
def verdict_text(a: dict, hw: dict) -> str:
    gb = 1024 ** 3
    if a["verdict"] == "fits":
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
        base = (f"Spills: only {a['gpu_layers']} of {a['layers']} layers fit. "
                "Most of each token is read from system RAM, which is roughly "
                "an order of magnitude slower.")
    else:
        base = "No GPU detected, so this runs on the CPU."
    if a.get("hybrid"):
        base += (" Hybrid attention: only the full-attention layers hold a KV "
                 "cache, which is why it is smaller than the layer count "
                 "suggests.")
    if not a["exact_kv"]:
        base += (" KV cache is estimated — the model's config.json could not "
                 "be read.")
    return base
