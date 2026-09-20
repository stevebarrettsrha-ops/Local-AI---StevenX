"""
server.py - Llama Studio backend.

Run:  python server.py        (opens http://127.0.0.1:7806)
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
import webbrowser
from pathlib import Path

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
    "last_model": "", "setup_complete": False,
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
        "config": {k: cfg.get(k) for k in
                   ("ctx", "temperature", "top_p", "max_tokens", "system",
                    "kv_bits", "auto_start", "last_model")},
        "catalogue": fit.CATALOGUE,
        "setup_complete": bool(cfg.get("setup_complete")),
        "tail": server.tail(12),
    })


@app.post("/api/config")
def api_config():
    b = request.get_json(silent=True) or {}
    for key in ("ctx", "temperature", "top_p", "max_tokens", "system",
                "kv_bits", "auto_start", "hf_token", "hf_endpoint",
                "plan_for"):
        if key in b:
            cfg[key] = b[key]
    save_config(cfg)
    return jsonify({"ok": True})


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
        entry = next((c for c in fit.CATALOGUE
                      if c["name"].lower().split()[0] in m["name"].lower()), None)
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
    model = next((m for m in engine.local_models() if m["name"] == name), None)
    if not model:
        return jsonify({"error": "That model is not on disk."}), 404
    if model["partial"]:
        return jsonify({"error": "That download has not finished."}), 400

    ctx = int(b.get("ctx") or cfg.get("ctx") or 8192)
    hw = fit.hardware(cfg)
    entry = next((c for c in fit.CATALOGUE
                  if c["name"].lower().split()[0] in name.lower()), None)
    conf = fit.model_config(cfg, (entry or {}).get("config_repo", "")) if entry \
        else {"layers": 32, "kv_heads": 8, "head_dim": 128, "exact": False}
    a = fit.assess(model["size"], conf, hw, ctx, int(cfg.get("kv_bits", 16)))
    gpu_layers = int(b.get("gpu_layers")) if b.get("gpu_layers") is not None \
        else a["gpu_layers"]
    extra = []
    if int(cfg.get("kv_bits", 16)) == 8:
        extra += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    try:
        server.start(model["path"], gpu_layers, ctx, extra)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    cfg["last_model"] = name
    cfg["ctx"] = ctx
    save_config(cfg)
    ok = server.wait_ready(600)
    if not ok:
        return jsonify({"error": "llama-server did not come up.",
                        "tail": server.tail(25)}), 500
    return jsonify({"ok": True, "model": name, "gpu_layers": gpu_layers,
                    "ctx": ctx, "fit": a, "why": fit.verdict_text(a, hw)})


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
        task = engine.download(cfg, b.get("repo", ""), b.get("path", ""))
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
