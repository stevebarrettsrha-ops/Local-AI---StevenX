"""
server.py - Llama Studio backend.

Run:  python server.py        (opens http://127.0.0.1:7806)
"""

from __future__ import annotations

import html as html_mod
import json
import os
import re
import threading
import time
import urllib.parse
import uuid
import webbrowser
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

import engine
import fit

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CHATS_PATH = DATA_DIR / "chats.json"
CONFIG_PATH = DATA_DIR / "config.json"
WEB_DIR = APP_DIR / "web"
PORT = int(os.environ.get("LLAMA_STUDIO_PORT", "7806"))

app = Flask(__name__, static_folder=None)
server = engine.Server()
chats_lock = threading.Lock()

DEFAULTS = {
    "hf_token": "", "hf_endpoint": "https://huggingface.co",
    "ctx": 8192, "temperature": 0.7, "top_p": 0.95, "max_tokens": 1024,
    "system": "", "kv_bits": 16, "auto_start": True, "plan_for": "",
    "last_model": "", "setup_complete": False, "workspace": "",
}


def load_config() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


cfg = load_config()


# --------------------------------------------------------------------------- #
# chats
# --------------------------------------------------------------------------- #
def read_chats() -> list[dict]:
    with chats_lock:
        if not CHATS_PATH.exists():
            return []
        try:
            return json.loads(CHATS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []


def write_chats(items: list[dict]) -> None:
    with chats_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CHATS_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def get_chat(chat_id: str) -> dict | None:
    return next((c for c in read_chats() if c["id"] == chat_id), None)


def put_chat(chat: dict) -> None:
    items = read_chats()
    for i, c in enumerate(items):
        if c["id"] == chat["id"]:
            items[i] = chat
            break
    else:
        items.insert(0, chat)
    write_chats(items)


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/web/<path:name>")
def web_asset(name: str):
    return send_from_directory(WEB_DIR, name)


# --------------------------------------------------------------------------- #
# status / hardware
# --------------------------------------------------------------------------- #
@app.get("/api/status")
def api_status():
    hw = fit.hardware(cfg)
    binary = engine.server_binary()
    return jsonify({
        "hardware": hw,
        "llama_installed": bool(binary),
        "llama_path": str(binary) if binary else "",
        "running": server.alive(), "ready": server.ready(),
        "model": Path(server.model).name if server.model else "",
        "models": engine.local_models(),
        "models_dir": str(engine.MODELS_DIR),
        "engines": engines_list(),
        "workspace": str(workspace_root() or ""),
        "config": {k: cfg.get(k) for k in
                   ("ctx", "temperature", "top_p", "max_tokens", "system",
                    "kv_bits", "auto_start", "last_model", "plan_for")},
        "catalogue": fit.CATALOGUE,
        "presets": fit.presets(),
        "setup_complete": bool(cfg.get("setup_complete")),
        "tail": server.tail(12),
    })


@app.post("/api/config")
def api_config():
    b = request.get_json(silent=True) or {}
    for key in ("ctx", "temperature", "top_p", "max_tokens", "system",
                "kv_bits", "auto_start", "hf_token", "hf_endpoint",
                "plan_for", "setup_complete"):
        if key in b:
            cfg[key] = b[key]
    save_config(cfg)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# engines (switchable model families in their own folders under ./models)
# --------------------------------------------------------------------------- #
ENGINE_ORDER = ["qwen", "coder", "gemma", "glm"]   # qwen first: the primary
SETUP_PLAN = [("qwen3-8b", "qwen"),       # (catalogue id, folder)
              ("gemma-4-e4b-unc", "gemma")]


def engines_list() -> list[dict]:
    """One entry per engine family folder that holds a finished model."""
    groups: dict[str, list[dict]] = {}
    for m in engine.local_models():
        if m["folder"] and not m["partial"]:
            groups.setdefault(m["folder"], []).append(m)
    loaded_name = Path(server.model).name if server.model else ""
    out = []
    for fam in sorted(groups, key=lambda f: (ENGINE_ORDER.index(f)
                                             if f in ENGINE_ORDER else 99, f)):
        models = groups[fam]
        pick = next((m for m in models if m["name"] == loaded_name), None) \
            or next((m for m in models
                     if m["name"] == cfg.get("last_model")), None) \
            or models[0]
        out.append({"family": fam, "model": pick["name"],
                    "loaded": server.alive() and pick["name"] == loaded_name})
    return out


def load_model_by_name(name: str, ctx: int | None = None,
                       gpu_layers: int | None = None) -> dict:
    """Start llama-server on a local file, -ngl from the fit calculation."""
    model = next((m for m in engine.local_models() if m["name"] == name), None)
    if not model:
        raise RuntimeError("That model is not on disk.")
    if model["partial"]:
        raise RuntimeError("That download has not finished.")
    ctx = int(ctx or cfg.get("ctx") or 8192)
    hw = fit.hardware(cfg)
    entry = fit.catalogue_match(name)
    conf = fit.model_config(cfg, (entry or {}).get("config_repo", "")) \
        if entry else {"layers": 32, "kv_layers": 32, "kv_heads": 8,
                       "head_dim": 128, "exact": False}
    a = fit.assess(model["size"], conf, hw, ctx, int(cfg.get("kv_bits", 16)))
    gl = a["gpu_layers"] if gpu_layers is None else int(gpu_layers)
    extra = []
    if int(cfg.get("kv_bits", 16)) == 8:
        extra += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    server.start(model["path"], gl, ctx, extra)
    cfg["last_model"] = name
    cfg["ctx"] = ctx
    save_config(cfg)
    if not server.wait_ready(600):
        raise RuntimeError("llama-server did not come up.")
    return {"model": name, "gpu_layers": gl, "ctx": ctx, "fit": a,
            "why": fit.verdict_text(a, hw)}


def pick_quant(files: list[dict], conf: dict, hw: dict, ctx: int,
               kv_bits: int) -> dict | None:
    """The build first-run setup downloads: Q4_K_M when it runs here (the
    sane default quant), else the largest that fits VRAM, else the largest
    that at least fits RAM, else the smallest. Judged, never guessed."""
    cand = [f for f in files
            if not f["split"] and not f["companion"] and f["size"]]
    if not cand:
        return None
    scored = [(f, fit.assess(f["size"], conf, hw, ctx, kv_bits))
              for f in cand]
    q4 = next(((f, a) for f, a in scored if f["quant"] == "Q4_K_M"), None)
    if q4 and (q4[1]["verdict"] == "fits" or
               (not hw.get("vram") and q4[1]["fits_ram"])):
        return q4[0]
    fits = [f for f, a in scored if a["verdict"] == "fits"]
    if fits:
        return fits[-1]
    ram_ok = [f for f, a in scored if a["fits_ram"]]
    return ram_ok[-1] if ram_ok else cand[0]


@app.post("/api/engine/switch")
def api_engine_switch():
    fam = ((request.get_json(silent=True) or {}).get("family") or "").strip()
    eng = next((e for e in engines_list() if e["family"] == fam), None)
    if not eng:
        return jsonify({"error": f"No downloaded model for '{fam}' yet."}), 404
    if eng["loaded"]:
        return jsonify({"ok": True, "model": eng["model"], "already": True})
    try:
        out = load_model_by_name(eng["model"])
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc), "tail": server.tail(25)}), 500
    return jsonify({"ok": True, **out})


@app.post("/api/setup/firstrun")
def api_setup_firstrun():
    """Install llama.cpp if needed, download Qwen and Gemma into their own
    folders, then load Qwen as the primary engine."""
    running = next((t for t in engine.task_list()
                    if t.kind == "setup" and t.state == "running"), None)
    if running:
        return jsonify({"ok": True, "task": running.view()})

    def run(task):
        # First run judges against the machine actually here, never a
        # planned card.
        hw = fit.hardware({})
        if not engine.server_binary():
            task.set(detail="Installing llama.cpp…")
            engine.install_sync(task, hw)
        primary_file = ""
        spans = [(5, 50), (50, 95)]
        for (model_id, folder), (p0, p1) in zip(SETUP_PLAN, spans):
            if task.cancel:
                task.set(state="cancelled", detail="Cancelled — partial "
                         "downloads resume next time.")
                return
            entry = next(c for c in fit.CATALOGUE if c["id"] == model_id)
            task.set(pct=p0, detail=f"Choosing a {entry['name']} build…")
            files = engine.hf_files(cfg, entry["repo"])
            conf = fit.model_config(cfg, entry.get("config_repo")
                                    or entry["repo"])
            pick = pick_quant(files, conf, hw, int(cfg.get("ctx") or 8192),
                              int(cfg.get("kv_bits", 16)))
            if not pick:
                raise RuntimeError(f"No usable GGUF in {entry['repo']}.")
            task.log(f"{entry['name']}: {pick['name']} "
                     f"({pick['size']/1e9:.1f} GB) → models/{folder}/")
            dest = engine.download_sync(cfg, entry["repo"], pick["path"],
                                        folder, task, p0, p1,
                                        label=entry["name"])
            if task.cancel:
                return
            if model_id == SETUP_PLAN[0][0]:
                primary_file = dest.name
        task.set(pct=96, detail="Loading Qwen as the primary engine…")
        try:
            load_model_by_name(primary_file)
            task.log(f"Loaded {primary_file}.")
        except Exception as exc:  # noqa: BLE001
            task.log(f"Downloaded, but could not load the primary engine "
                     f"yet: {exc}")
        cfg["setup_complete"] = True
        save_config(cfg)
        task.set(pct=100, detail="Ready — Qwen is the primary engine; Gemma "
                                 "is one click away on the switcher.")

    task = engine.spawn("setup", "First-run setup", run)
    return jsonify({"ok": True, "task": task.view()})


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
@app.get("/api/fit")
def api_fit():
    """Every GGUF in a repo, each judged against this machine."""
    model_id = request.args.get("model", "")
    entry = next((m for m in fit.CATALOGUE if m["id"] == model_id), None)
    repo = request.args.get("repo") or (entry["repo"] if entry else "")
    if not repo:
        return jsonify({"error": "No model chosen."}), 400
    ctx = int(request.args.get("ctx") or cfg.get("ctx") or 8192)
    # Some entries exist for reference rather than for running: a serving
    # format this engine cannot load. Say why, and point at what does run.
    if entry and entry.get("runtime") and entry["runtime"] != "llama.cpp":
        hw = fit.hardware(cfg)
        need = entry.get("vram_needed") or 0
        alt = entry.get("gguf") or ""
        why = (f"{entry['name']} is a {entry['runtime']} format. llama.cpp "
               "loads GGUF only, so this build cannot run here.")
        if need:
            why += (f" It also wants about {need/1024**3:.0f} GB of VRAM "
                    "resident with no CPU offload, against your "
                    f"{hw.get('vram', 0)/1024**3:.0f} GB.")
        return jsonify({"repo": repo, "ctx": ctx, "hardware": hw,
                        "unsupported": True, "runtime": entry["runtime"],
                        "why": why, "alternative": alt, "files": [],
                        "recommended": ""})
    try:
        files = engine.hf_files(cfg, repo)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    hw = fit.hardware(cfg)
    conf = fit.model_config(cfg, (entry or {}).get("config_repo") or repo)
    out = []
    for f in files:
        a = fit.assess(f["size"], conf, hw, ctx, int(cfg.get("kv_bits", 16)))
        out.append({**f, "fit": a, "why": fit.verdict_text(a, hw)})
    best = next((f for f in out if f["fit"]["verdict"] == "fits"
                 and not f["split"] and not f.get("companion")), None)
    return jsonify({"repo": repo, "ctx": ctx, "hardware": hw, "config": conf,
                    "files": out, "recommended": best["name"] if best else ""})


@app.get("/api/fit/local")
def api_fit_local():
    """The same judgement for a file already on disk."""
    hw = fit.hardware(cfg)
    ctx = int(request.args.get("ctx") or cfg.get("ctx") or 8192)
    out = []
    for m in engine.local_models():
        entry = fit.catalogue_match(m["name"])
        conf = fit.model_config(cfg, (entry or {}).get("config_repo", "")) \
            if entry else {"layers": 32, "kv_heads": 8, "head_dim": 128,
                           "exact": False}
        a = fit.assess(m["size"], conf, hw, ctx, int(cfg.get("kv_bits", 16)))
        out.append({**m, "fit": a, "why": fit.verdict_text(a, hw)})
    return jsonify({"models": out, "hardware": hw, "ctx": ctx})


# --------------------------------------------------------------------------- #
# engine control
# --------------------------------------------------------------------------- #
@app.post("/api/engine/install")
def api_engine_install():
    try:
        return jsonify({"ok": True,
                        "task": engine.install(fit.hardware(cfg)).view()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/engine/load")
def api_engine_load():
    b = request.get_json(silent=True) or {}
    name = (b.get("model") or "").strip()
    try:
        out = load_model_by_name(
            name, b.get("ctx"),
            b.get("gpu_layers") if b.get("gpu_layers") is not None else None)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "not on disk" in msg:
            return jsonify({"error": msg}), 404
        if "did not come up" in msg:
            return jsonify({"error": msg, "tail": server.tail(25)}), 500
        return jsonify({"error": msg}), 400
    return jsonify({"ok": True, **out})


@app.post("/api/engine/stop")
def api_engine_stop():
    server.stop()
    return jsonify({"ok": True})


@app.get("/api/engine/log")
def api_engine_log():
    return jsonify({"lines": server.tail(60), "running": server.alive(),
                    "ready": server.ready()})


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
@app.get("/api/models")
def api_models():
    return jsonify({"local": engine.local_models(),
                    "dir": str(engine.MODELS_DIR)})


@app.post("/api/models/download")
def api_models_download():
    b = request.get_json(silent=True) or {}
    try:
        # Catalogue models with an engine family land in that family's own
        # folder, so the switcher finds them; anything else stays flat.
        entry = next((c for c in fit.CATALOGUE
                      if c["repo"] == b.get("repo")), None)
        folder = (entry or {}).get("family", "")
        task = engine.download(cfg, b.get("repo", ""), b.get("path", ""),
                               folder)
        return jsonify({"ok": True, "task": task.view()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.delete("/api/models")
def api_models_delete():
    b = request.get_json(silent=True) or {}
    try:
        name = b.get("name", "")
        if server.model and Path(server.model).name == name:
            server.stop()
        engine.delete_model(name)
        return jsonify({"ok": True})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.get("/api/tasks")
def api_tasks():
    task_id = request.args.get("id", "")
    since = int(request.args.get("since", 0))
    if task_id:
        task = engine.TASKS.get(task_id)
        if not task:
            return jsonify({"error": "No such task."}), 404
        return jsonify(task.view(since))
    return jsonify([t.view(t.view()["cursor"]) for t in engine.task_list()[:20]])


@app.post("/api/tasks/<task_id>/cancel")
def api_task_cancel(task_id: str):
    task = engine.TASKS.get(task_id)
    if task:
        task.cancel = True
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# web search (opt-in, per message; DuckDuckGo's keyless HTML endpoint)
# --------------------------------------------------------------------------- #
SEARCH_URL = os.environ.get("LLAMA_STUDIO_SEARCH_URL",
                            "https://html.duckduckgo.com/html/")


def _ddg_target(href: str) -> str:
    """DuckDuckGo wraps result links in a redirect; unwrap to the real URL."""
    m = re.search(r"[?&]uddg=([^&]+)", href)
    return urllib.parse.unquote(m.group(1)) if m else href


@app.get("/api/search")
def api_search():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "Nothing to search."}), 400
    try:
        r = requests.get(SEARCH_URL, params={"q": q}, timeout=12,
                         headers={"User-Agent": "Mozilla/5.0 (LlamaStudio)"})
        r.raise_for_status()
        page = r.text
        links = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            page, re.S)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>',
                              page, re.S)

        def clean(s: str) -> str:
            return html_mod.unescape(re.sub(r"<[^>]+>", "", s)).strip()

        results = []
        for i, (href, title) in enumerate(links[:5]):
            results.append({"title": clean(title),
                            "url": _ddg_target(html_mod.unescape(href)),
                            "snippet": clean(snippets[i])
                            if i < len(snippets) else ""})
        if not results:
            return jsonify({"error": "The search engine returned nothing "
                                     "readable."}), 502
        return jsonify({"query": q, "results": results})
    except requests.RequestException as exc:
        return jsonify({"error": f"Search failed: {exc}"}), 502


# --------------------------------------------------------------------------- #
# workspace (a folder on this computer the Code page can read and edit)
# --------------------------------------------------------------------------- #
WS_SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", "env",
           "dist", "build", "target", ".idea", ".vscode", ".next",
           "site-packages"}
WS_MAX_FILES = 400
WS_MAX_READ = 300_000


def workspace_root() -> Path | None:
    raw = (cfg.get("workspace") or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_dir() else None


def ws_resolve(rel: str) -> Path:
    """A path inside the open folder — never outside it."""
    root = workspace_root()
    if not root:
        raise RuntimeError("No folder is open.")
    target = (root / rel).resolve()
    root = root.resolve()
    if target != root and not str(target).startswith(str(root) + os.sep):
        raise RuntimeError("That path is outside the open folder.")
    return target


@app.post("/api/workspace")
def api_workspace_open():
    b = request.get_json(silent=True) or {}
    raw = (b.get("path") or "").strip()
    if not raw:
        cfg["workspace"] = ""
        save_config(cfg)
        return jsonify({"ok": True, "path": ""})
    p = Path(raw).expanduser()
    if not p.is_dir():
        return jsonify({"error": f"No such folder: {raw}"}), 400
    cfg["workspace"] = str(p)
    save_config(cfg)
    return jsonify({"ok": True, "path": str(p)})


@app.get("/api/workspace/tree")
def api_workspace_tree():
    root = workspace_root()
    if not root:
        return jsonify({"path": "", "files": []})
    files, truncated = [], False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames)
                       if d not in WS_SKIP and not d.startswith(".")]
        rel_dir = Path(dirpath).relative_to(root)
        for name in sorted(filenames):
            if name.startswith(".") or name.endswith(".bak"):
                continue
            if len(files) >= WS_MAX_FILES:
                truncated = True
                break
            f = Path(dirpath) / name
            rel = str(rel_dir / name).replace("\\", "/").lstrip("./")
            try:
                files.append({"path": rel, "size": f.stat().st_size})
            except OSError:
                continue
        if truncated:
            break
    return jsonify({"path": str(root), "files": files,
                    "truncated": truncated})


@app.get("/api/workspace/file")
def api_workspace_read():
    try:
        target = ws_resolve(request.args.get("path", ""))
        if not target.is_file():
            raise RuntimeError("That file does not exist.")
        if target.stat().st_size > WS_MAX_READ:
            raise RuntimeError("That file is too large to attach "
                               f"({WS_MAX_READ // 1000} KB cap).")
        data = target.read_bytes()
        if b"\0" in data[:8192]:
            raise RuntimeError("That looks like a binary file.")
        return jsonify({"path": request.args.get("path", ""),
                        "content": data.decode("utf-8", errors="replace"),
                        "size": len(data)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/workspace/file")
def api_workspace_write():
    """Write model output back into the folder. The previous version is
    kept beside it as .bak — an edit is undoable, never silent loss."""
    b = request.get_json(silent=True) or {}
    try:
        target = ws_resolve(b.get("path") or "")
        content = b.get("content")
        if content is None:
            raise RuntimeError("Nothing to write.")
        backup = ""
        if target.exists():
            bak = target.with_name(target.name + ".bak")
            bak.write_bytes(target.read_bytes())
            backup = bak.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return jsonify({"ok": True, "path": b.get("path"), "backup": backup})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #
@app.get("/api/chats")
def api_chats():
    return jsonify([{k: v for k, v in c.items() if k != "messages"}
                    | {"count": len(c.get("messages", []))}
                    for c in read_chats()])


@app.get("/api/chats/<chat_id>")
def api_chat(chat_id: str):
    chat = get_chat(chat_id)
    return jsonify(chat) if chat else (jsonify({"error": "Not found."}), 404)


@app.delete("/api/chats/<chat_id>")
def api_chat_delete(chat_id: str):
    write_chats([c for c in read_chats() if c["id"] != chat_id])
    return jsonify({"ok": True})


@app.post("/api/chat")
def api_chat_send():
    """Streams the reply back as it is generated, then stores the exchange."""
    b = request.get_json(silent=True) or {}
    text = (b.get("message") or "").strip()
    if not text:
        return jsonify({"error": "Nothing to send."}), 400
    if not server.ready():
        return jsonify({"error": "No model is loaded. Pick one on the Models "
                                 "page."}), 503

    chat_id = b.get("chat") or ""
    chat = get_chat(chat_id) if chat_id else None
    if not chat:
        chat = {"id": uuid.uuid4().hex[:12],
                "title": " ".join(text.split()[:7])[:60] or "New chat",
                "created": time.time(), "model": Path(server.model).name,
                # which workspace this conversation belongs to: the plain
                # chat page or the coding page. Old chats have no mode and
                # are treated as plain chat.
                "mode": "code" if b.get("mode") == "code" else "chat",
                "messages": []}
    chat["messages"].append({"role": "user", "content": text,
                             "at": time.time()})
    # Stored before a single token comes back: if the person presses Stop or
    # closes the tab mid-reply, their own message is not lost with it.
    put_chat(chat)
    history = [{"role": m["role"], "content": m["content"]}
               for m in chat["messages"]]
    params = {"temperature": b.get("temperature", cfg.get("temperature")),
              "top_p": b.get("top_p", cfg.get("top_p")),
              "max_tokens": b.get("max_tokens", cfg.get("max_tokens")),
              "system": b.get("system", cfg.get("system"))}

    def stream():
        started = time.time()
        pieces: list[str] = []
        finished = False
        try:
            yield "data: " + json.dumps({"chat": chat["id"],
                                         "title": chat["title"]}) + "\n\n"
            try:
                for delta in server.chat_stream(history, params):
                    pieces.append(delta)
                    yield "data: " + json.dumps({"delta": delta}) + "\n\n"
            except Exception as exc:  # noqa: BLE001
                yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"
            finished = True
            elapsed = max(time.time() - started, 0.001)
            tps = round((len("".join(pieces)) / 4) / elapsed, 1)
            yield "data: " + json.dumps({"done": True, "tps": tps,
                                         "seconds": round(elapsed, 1)}) + "\n\n"
        finally:
            # Runs on a clean finish AND on GeneratorExit when the client goes
            # away, so a half-written reply is kept rather than thrown out.
            reply = "".join(pieces)
            if reply:
                elapsed = max(time.time() - started, 0.001)
                chat["messages"].append({
                    "role": "assistant", "content": reply, "at": time.time(),
                    # ~4 characters a token: rough, but measured, not predicted.
                    "tps": round((len(reply) / 4) / elapsed, 1),
                    "seconds": round(elapsed, 1),
                    "stopped": not finished})
                chat["model"] = Path(server.model).name if server.model \
                    else chat.get("model")
                put_chat(chat)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------- #
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if cfg.get("auto_start") and cfg.get("last_model"):
        model = next((m for m in engine.local_models()
                      if m["name"] == cfg["last_model"] and not m["partial"]),
                     None)
        if model and engine.server_binary():
            hw = fit.hardware(cfg)
            conf = {"layers": 32, "kv_heads": 8, "head_dim": 128,
                    "exact": False}
            a = fit.assess(model["size"], conf, hw, int(cfg.get("ctx", 8192)),
                           int(cfg.get("kv_bits", 16)))
            try:
                server.start(model["path"], a["gpu_layers"],
                             int(cfg.get("ctx", 8192)))
                print(f"  loading {model['name']}…")
            except Exception as exc:  # noqa: BLE001
                print(f"  could not reload {model['name']}: {exc}")
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  Llama Studio  →  {url}\n")
    if os.environ.get("LLAMA_STUDIO_NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
