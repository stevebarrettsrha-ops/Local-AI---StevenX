"""
server.py - Llama Studio backend.

Run:  python server.py        (opens http://127.0.0.1:7806)
"""

from __future__ import annotations

import csv
import html as html_mod
import io
import json
import os
import re
import subprocess
import threading
import time
import urllib.parse
import uuid
import webbrowser
from pathlib import Path

import requests
from flask import (Flask, Response, jsonify, request, send_file,
                   send_from_directory)

import engine
import fit

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CHATS_PATH = DATA_DIR / "chats.json"
MEMORY_PATH = DATA_DIR / "memory.md"
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
    "run_command": "", "recall": True,
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
        "api_url": server.url + "/v1",
        "model": Path(server.model).name if server.model else "",
        "models": engine.local_models(),
        "models_dir": str(engine.MODELS_DIR),
        "engines": engines_list(),
        "workspace": str(workspace_root() or ""),
        "embed": {"installed": bool(embed_model_file()),
                  "ready": embedder.ready(),
                  "indexed": len(emb_index())},
        "voice": {"installed": bool(engine.whisper_binary()
                                    and voice_model_file()),
                  "ready": whisper.ready()},
        "image": {"installed": bool(engine.sd_binary()
                                    and image_model_file())},
        "config": {k: cfg.get(k) for k in
                   ("ctx", "temperature", "top_p", "max_tokens", "system",
                    "kv_bits", "auto_start", "last_model", "plan_for",
                    "run_command")},
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
                "plan_for", "setup_complete", "recall"):
        if key in b:
            cfg[key] = b[key]
    save_config(cfg)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# memory (persistent notes + recall of relevant past conversations)
# --------------------------------------------------------------------------- #
MEMORY_CAP = 16_000        # characters kept in the memory file
MEMORY_INJECT = 4_000      # characters of it sent with each request


def read_memory() -> str:
    try:
        return MEMORY_PATH.read_text(encoding="utf-8") \
            if MEMORY_PATH.exists() else ""
    except OSError:
        return ""


def write_memory(text: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(text[:MEMORY_CAP], encoding="utf-8")


def remember_lines(reply: str) -> int:
    """Lines the model marked `remember:` become permanent memory. The
    file is user-editable and capped, so a chatty model cannot flood it."""
    new = [ln.strip()[9:].strip() for ln in reply.splitlines()
           if ln.strip().lower().startswith("remember:")]
    new = [n for n in new if n]
    if not new:
        return 0
    mem = read_memory()
    have = {ln.lstrip("- ").strip().lower() for ln in mem.splitlines()}
    added = [n for n in new if n.lower() not in have]
    if added:
        write_memory((mem.rstrip() + "\n" if mem.strip() else "") +
                     "\n".join("- " + n for n in added) + "\n")
    return len(added)


# ---- semantic recall: a small embedding model beside the chat model ---- #
EMBED_REPO = "nomic-ai/nomic-embed-text-v1.5-GGUF"
EMBED_DIR = "embed"
EMBED_SIM_MIN = 0.5
EMB_PATH = DATA_DIR / "embeddings.json"
embedder = engine.EmbedServer(8082)
_emb_lock = threading.Lock()
_emb_index: dict | None = None
_emb_busy = False


def embed_model_file() -> Path | None:
    d = engine.MODELS_DIR / EMBED_DIR
    if d.is_dir():
        for f in sorted(d.glob("*.gguf")):
            return f
    return None


def emb_index() -> dict:
    global _emb_index
    with _emb_lock:
        if _emb_index is None:
            try:
                _emb_index = json.loads(
                    EMB_PATH.read_text(encoding="utf-8")) \
                    if EMB_PATH.exists() else {}
            except Exception:
                _emb_index = {}
        return _emb_index


def emb_save() -> None:
    with _emb_lock:
        EMB_PATH.write_text(json.dumps(_emb_index), encoding="utf-8")


def ensure_embedder(wait: int = 15) -> bool:
    model = embed_model_file()
    if not model or not engine.server_binary():
        return False
    if embedder.ready():
        return True
    if not embedder.alive():
        try:
            embedder.start(str(model), 0, 2048, ["--embeddings"])
        except Exception:
            return False
    return embedder.wait_ready(wait)


def _cos(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def _exchange_rows():
    for c in read_chats():
        msgs = c.get("messages", [])
        for i, m in enumerate(msgs):
            if m.get("role") != "user":
                continue
            reply = msgs[i + 1]["content"] if i + 1 < len(msgs) \
                and msgs[i + 1].get("role") == "assistant" else ""
            yield c, i, m.get("content", ""), reply


def index_exchanges() -> None:
    """Embed every stored exchange the index does not have yet. The
    nomic prefixes (search_document / search_query) are the model's own
    convention for asymmetric retrieval."""
    global _emb_busy
    if _emb_busy or not ensure_embedder():
        return
    _emb_busy = True
    try:
        idx = emb_index()
        todo = []
        for c, i, um, am in _exchange_rows():
            key = f'{c["id"]}:{i}'
            # An exchange whose reply has not arrived yet is skipped, not
            # indexed half-finished: once a key exists it is never
            # revisited, so indexing early would freeze an empty answer.
            if key in idx or not am.strip():
                continue
            doc = "search_document: " + um[:300].replace("\n", " ") + \
                " " + am[:500].replace("\n", " ")
            todo.append((key, c, um, am, doc))
        for start in range(0, len(todo), 8):
            batch = todo[start:start + 8]
            try:
                vecs = embedder.embed([t[4] for t in batch])
            except Exception:
                return
            for (key, c, um, am, _), v in zip(batch, vecs):
                idx[key] = {"chat": c["id"], "title": c.get("title", ""),
                            "q": um[:200].replace("\n", " "),
                            "a": am[:400].replace("\n", " "), "v": v}
        if todo:
            emb_save()
    finally:
        _emb_busy = False


def semantic_recall(query: str, exclude_id: str,
                    limit: int = 2) -> list[str] | None:
    """None means the embedding model is not available — the caller
    falls back to keyword matching rather than losing recall."""
    if not ensure_embedder(wait=3):
        return None
    index_exchanges()
    try:
        qv = embedder.embed(["search_query: " + query[:1000]])[0]
    except Exception:
        return None
    scored = sorted(((_cos(qv, e["v"]), e) for e in emb_index().values()
                     if e.get("chat") != exclude_id),
                    key=lambda t: -t[0])
    out = []
    for sim, e in scored[:limit]:
        if sim < EMBED_SIM_MIN:
            break
        out.append('From the conversation "%s" — asked: "%s" — answered: '
                   '"%s"' % (e["title"], e["q"], e["a"]))
    return out


@app.post("/api/embed/install")
def api_embed_install():
    if embed_model_file():
        threading.Thread(target=index_exchanges, daemon=True).start()
        return jsonify({"ok": True, "already": True})

    def run(task):
        files = engine.hf_files(cfg, EMBED_REPO)
        cand = [f for f in files if not f["split"] and not f["companion"]]
        pick = next((f for f in cand if f["quant"] == "Q8_0"), None) \
            or (min(cand, key=lambda f: f["size"]) if cand else None)
        if not pick:
            raise RuntimeError(f"No GGUF found in {EMBED_REPO}.")
        engine.download_sync(cfg, EMBED_REPO, pick["path"], EMBED_DIR,
                             task, 0, 90, label="recall model")
        if task.cancel:
            return
        task.set(pct=92, detail="Starting the embedding server…")
        if not ensure_embedder(30):
            raise RuntimeError("The embedding server did not come up.")
        task.set(pct=95, detail="Indexing past conversations…")
        index_exchanges()
        task.set(pct=100,
                 detail=f"Ready — {len(emb_index())} exchanges indexed.")

    return jsonify({"ok": True,
                    "task": engine.spawn("download", "recall model",
                                         run).view()})


# ---- Office documents: xlsx / docx / pptx built from tagged blocks ---- #
# The model writes a fenced block tagged xlsx, docx or pptx in a simple
# format (told to it via DOC_HOWTO); these builders turn that text into
# the real file. Libraries are imported lazily so the app still runs if
# they were not installed.
DOC_HOWTO = (
    "\n\nWhen the user asks for a spreadsheet, a Word document or a "
    "PowerPoint deck, put the content in ONE fenced code block tagged "
    "xlsx, docx or pptx. xlsx: lines of '# Sheet: <name>' each followed "
    "by CSV rows, first row the headers; after a sheet's rows you may add "
    "'# Chart: bar <title>' (or line, or pie) to embed a chart of that "
    "sheet — first column is the labels, the other columns the numbers. "
    "Use bar to compare categories, line for change over time, pie only "
    "for a few shares of a whole. docx: Markdown-style # headings, "
    "paragraphs, - bullets, **bold**, and Markdown tables (| cell | "
    "rows), which become real Word tables. pptx: '# <slide title>' "
    "followed by - bullet lines, one group per slide. The app offers the "
    "real file for download from that block.")


def build_xlsx(text: str) -> tuple[bytes, str]:
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, LineChart, PieChart, Reference
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    wb.remove(wb.active)
    # each sheet: (name, rows, chart) where chart is (type, title) or None
    sheets, name, rows, chart = [], "Sheet1", [], None
    for line in text.splitlines():
        m = re.match(r"\s*#+\s*Chart\s*:\s*(bar|line|pie)\b\s*(.*)$",
                     line, re.I)
        if m:
            chart = (m.group(1).lower(), m.group(2).strip())
            continue
        m = re.match(r"\s*#+\s*Sheet\s*:\s*(.+)$", line, re.I)
        if m:
            if rows:
                sheets.append((name, rows, chart))
            name, rows, chart = m.group(1).strip()[:31], [], None
        elif line.strip():
            rows.append(next(csv.reader([line])))
    if rows:
        sheets.append((name, rows, chart))
    if not sheets:
        raise RuntimeError("No rows found in the block.")
    for sname, srows, schart in sheets:
        ws = wb.create_sheet(title=sname or "Sheet")
        for r, row in enumerate(srows, 1):
            for c, val in enumerate(row, 1):
                v = val.strip()
                if v and re.fullmatch(r"-?\d+(\.\d+)?", v):
                    v = float(v) if "." in v else int(v)
                cell = ws.cell(row=r, column=c, value=v)
                if r == 1:
                    cell.font = Font(bold=True)
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col if c.value
                         is not None), default=8)
            ws.column_dimensions[col[0].column_letter].width = \
                min(40, width + 2)
        # An embedded chart of this sheet's data: labels from the first
        # column, values from the rest, series named by the header row.
        # Colors stay Excel's own theme — a fixed-order palette, never
        # hand-picked here.
        ncols = max((len(r) for r in srows), default=0)
        if schart and len(srows) >= 2 and ncols >= 2:
            ctype, ctitle = schart
            last_col = 2 if ctype == "pie" else ncols
            ch = {"bar": BarChart, "line": LineChart,
                  "pie": PieChart}[ctype]()
            ch.title = ctitle or sname
            data = Reference(ws, min_col=2, max_col=last_col,
                             min_row=1, max_row=len(srows))
            ch.add_data(data, titles_from_data=True)
            ch.set_categories(Reference(ws, min_col=1, min_row=2,
                                        max_row=len(srows)))
            ch.height, ch.width = 8, 15
            ws.add_chart(ch, f"{get_column_letter(ncols + 2)}2")
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue(), ("application/vnd.openxmlformats-officedocument"
                            ".spreadsheetml.sheet")


def _docx_runs(paragraph, text: str) -> None:
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            paragraph.add_run(part).bold = (i % 2 == 1)


def build_docx(text: str) -> tuple[bytes, str]:
    from docx import Document
    doc = Document()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        s = line.strip()
        if not s:
            i += 1
            continue
        # A run of |-delimited lines is a Markdown table: header row bold,
        # the |---| separator dropped, rendered as a real Word table.
        if s.startswith("|") and s.endswith("|"):
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in
                         lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    tbl.append(cells)
                i += 1
            if tbl:
                cols = max(len(r) for r in tbl)
                t = doc.add_table(rows=len(tbl), cols=cols)
                t.style = "Table Grid"
                for r, rowvals in enumerate(tbl):
                    for c in range(cols):
                        para = t.cell(r, c).paragraphs[0]
                        txt = rowvals[c] if c < len(rowvals) else ""
                        if r == 0:
                            para.add_run(
                                re.sub(r"\*\*", "", txt)).bold = True
                        else:
                            _docx_runs(para, txt)
            continue
        m = re.match(r"(#{1,4})\s+(.*)", line)
        if m:
            doc.add_heading(re.sub(r"\*\*", "", m.group(2)),
                            level=min(len(m.group(1)), 4))
        elif re.match(r"\s*[-*]\s+", line):
            _docx_runs(doc.add_paragraph(style="List Bullet"),
                       re.sub(r"^\s*[-*]\s+", "", line))
        elif re.match(r"\s*\d+[.)]\s+", line):
            _docx_runs(doc.add_paragraph(style="List Number"),
                       re.sub(r"^\s*\d+[.)]\s+", "", line))
        else:
            _docx_runs(doc.add_paragraph(), s)
        i += 1
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue(), ("application/vnd.openxmlformats-officedocument"
                            ".wordprocessingml.document")


def build_pptx(text: str) -> tuple[bytes, str]:
    from pptx import Presentation
    prs = Presentation()
    slides, cur = [], None
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        m = re.match(r"#+\s+(.*)", s)
        if m:
            cur = {"title": re.sub(r"\*\*", "", m.group(1)), "bullets": []}
            slides.append(cur)
        else:
            if cur is None:
                cur = {"title": "", "bullets": []}
                slides.append(cur)
            cur["bullets"].append(
                re.sub(r"\*\*", "", re.sub(r"^[-*]\s+", "", s)))
    if not slides:
        raise RuntimeError("No slides found in the block.")
    layout = prs.slide_layouts[1]          # title and content
    for sl in slides:
        slide = prs.slides.add_slide(layout)
        slide.shapes.title.text = sl["title"][:90]
        frame = slide.placeholders[1].text_frame
        for i, btxt in enumerate(sl["bullets"][:12]):
            para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            para.text = btxt[:200]
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue(), ("application/vnd.openxmlformats-officedocument"
                            ".presentationml.presentation")


DOC_BUILDERS = {"xlsx": build_xlsx, "docx": build_docx, "pptx": build_pptx}


@app.post("/api/doc")
def api_doc():
    b = request.get_json(silent=True) or {}
    kind = (b.get("kind") or "").lower()
    if kind not in DOC_BUILDERS:
        return jsonify({"error": "Unknown document kind."}), 400
    title = re.sub(r"[^A-Za-z0-9 _-]+", "",
                   b.get("title") or "document").strip()[:48] or "document"
    try:
        data, mime = DOC_BUILDERS[kind](b.get("content") or "")
    except ImportError as exc:
        return jsonify({"error": f"The .{kind} exporter needs a Python "
                        f"package that is not installed ({exc}). Run: "
                        "pip install -r requirements.txt"}), 500
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    return send_file(io.BytesIO(data), mimetype=mime, as_attachment=True,
                     download_name=f"{title}.{kind}")


# ---- images: local text-to-image via stable-diffusion.cpp ---- #
IMAGE_MODEL_REPO = "second-state/stable-diffusion-v1-5-GGUF"
IMAGE_MODEL_FILE = "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf"
IMAGE_DIR = "image"
IMAGES_OUT = DATA_DIR / "images"


def image_model_file() -> Path | None:
    d = engine.MODELS_DIR / IMAGE_DIR
    if d.is_dir():
        for f in sorted(d.glob("*.gguf")):
            return f
    return None


@app.post("/api/image/install")
def api_image_install():
    if engine.sd_binary() and image_model_file():
        return jsonify({"ok": True, "already": True})

    def run(task):
        if not engine.sd_binary():
            engine.sd_install_sync(task, fit.hardware({}))
        if not image_model_file():
            engine.download_sync(cfg, IMAGE_MODEL_REPO, IMAGE_MODEL_FILE,
                                 IMAGE_DIR, task, 30, 100,
                                 label="image model")

    return jsonify({"ok": True,
                    "task": engine.spawn("download", "image engine",
                                         run).view()})


@app.post("/api/image/generate")
def api_image_generate():
    b = request.get_json(silent=True) or {}
    prompt = (b.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Describe the image first."}), 400
    binary, model = engine.sd_binary(), image_model_file()
    if not binary or not model:
        return jsonify({"error": "The image engine is not installed yet — "
                                 "download it on the Images page."}), 400
    w = max(256, min(768, int(b.get("width") or 512))) // 64 * 64
    h = max(256, min(768, int(b.get("height") or 512))) // 64 * 64
    steps = max(1, min(50, int(b.get("steps") or 20)))
    seed = int(b.get("seed") or time.time()) % 2_000_000_000

    def run(task):
        IMAGES_OUT.mkdir(parents=True, exist_ok=True)
        name = f"img-{int(time.time())}-{seed}.png"
        out = IMAGES_OUT / name
        cmd = [str(binary), "-m", str(model), "-p", prompt,
               "-o", str(out), "--steps", str(steps),
               "-W", str(w), "-H", str(h), "-s", str(seed)]
        task.log(" ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, cwd=str(engine.SD_DIR))
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            task.log(line)
            task.set(detail=line[:120])
            m = re.search(r"(\d+)\s*/\s*(\d+)", line)
            if m and int(m.group(2)) == steps:
                task.set(pct=int(m.group(1)) / steps * 100)
            if task.cancel:
                proc.kill()
                task.set(state="cancelled", detail="Cancelled")
                return
        proc.wait()
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"sd exited with code {proc.returncode} — "
                               "see the task log.")
        (IMAGES_OUT / (name + ".json")).write_text(
            json.dumps({"prompt": prompt, "seed": seed, "w": w, "h": h,
                        "steps": steps}), encoding="utf-8")
        task.meta["file"] = name
        task.set(pct=100, detail=f"Saved {name}")

    return jsonify({"ok": True,
                    "task": engine.spawn("image", prompt[:48],
                                         run).view()})


@app.get("/api/images")
def api_images_list():
    out = []
    if IMAGES_OUT.is_dir():
        for f in sorted(IMAGES_OUT.glob("*.png"),
                        key=lambda p: -p.stat().st_mtime)[:60]:
            meta = {}
            side = IMAGES_OUT / (f.name + ".json")
            if side.exists():
                try:
                    meta = json.loads(side.read_text(encoding="utf-8"))
                except Exception:
                    pass
            out.append({"name": f.name, "url": f"/images/{f.name}",
                        "prompt": meta.get("prompt", ""),
                        "created": f.stat().st_mtime})
    return jsonify({"images": out})


@app.get("/images/<name>")
def api_image_file(name: str):
    if "/" in name or "\\" in name or ".." in name:
        return jsonify({"error": "Bad name."}), 400
    return send_from_directory(IMAGES_OUT, name)


@app.delete("/api/images")
def api_image_delete():
    name = ((request.get_json(silent=True) or {}).get("name") or "")
    if "/" in name or "\\" in name or ".." in name or not name:
        return jsonify({"error": "Bad name."}), 400
    (IMAGES_OUT / name).unlink(missing_ok=True)
    (IMAGES_OUT / (name + ".json")).unlink(missing_ok=True)
    return jsonify({"ok": True})


# ---- voice: local speech-to-text via whisper.cpp ---- #
WHISPER_MODEL_REPO = "ggerganov/whisper.cpp"
WHISPER_MODEL_FILE = "ggml-base.bin"
VOICE_DIR = "voice"
whisper = engine.WhisperServer(8083)


def voice_model_file() -> Path | None:
    d = engine.MODELS_DIR / VOICE_DIR
    if d.is_dir():
        for f in sorted(d.glob("*.bin")):
            return f
    return None


def ensure_whisper(wait: int = 15) -> bool:
    model = voice_model_file()
    if not model or not engine.whisper_binary():
        return False
    if whisper.ready():
        return True
    if not whisper.alive():
        try:
            whisper.start_whisper(str(model))
        except Exception:
            return False
    return whisper.wait_ready(wait)


@app.post("/api/voice/install")
def api_voice_install():
    if engine.whisper_binary() and voice_model_file():
        ensure_whisper(20)
        return jsonify({"ok": True, "already": True})

    def run(task):
        if not engine.whisper_binary():
            engine.whisper_install_sync(task)
        if not voice_model_file():
            engine.download_sync(cfg, WHISPER_MODEL_REPO,
                                 WHISPER_MODEL_FILE, VOICE_DIR, task,
                                 45, 95, label="voice model")
            if task.cancel:
                return
        task.set(pct=96, detail="Starting the speech server…")
        if not ensure_whisper(30):
            raise RuntimeError("The whisper server did not come up.")
        task.set(pct=100, detail="Ready — speak with the mic button.")

    return jsonify({"ok": True,
                    "task": engine.spawn("download", "voice (whisper)",
                                         run).view()})


@app.post("/api/voice/transcribe")
def api_voice_transcribe():
    audio = request.get_data(cache=False)
    if not audio:
        return jsonify({"error": "No audio received."}), 400
    if len(audio) > 25_000_000:
        return jsonify({"error": "Recording too long."}), 400
    if not ensure_whisper(10):
        return jsonify({"error": "Voice input is not set up — download "
                                 "the Whisper model under Parameters."}), 503
    try:
        return jsonify({"text": whisper.transcribe(audio)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 502


_STOP = {"the", "and", "that", "this", "with", "from", "have", "what",
         "your", "about", "into", "does", "how", "can", "for", "you",
         "are", "was", "were", "will", "would", "could", "should", "did",
         "when", "where", "which", "there", "here", "then", "than",
         "please", "make", "just", "like", "also", "some", "them",
         "they", "their", "file", "files", "code"}


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", (text or "").lower())
            if w not in _STOP}


def recall_snippets(query: str, exclude_id: str, limit: int = 2) -> list[str]:
    """Excerpts from OTHER conversations that share enough distinct words
    with the question to plausibly matter. Keyword overlap, not magic —
    cheap, local, and inspectable."""
    q = _terms(query)
    if len(q) < 2:
        return []
    scored = []
    for c in read_chats():
        if c["id"] == exclude_id:
            continue
        msgs = c.get("messages", [])
        for i, m in enumerate(msgs):
            if m.get("role") != "user":
                continue
            reply = msgs[i + 1]["content"] if i + 1 < len(msgs) \
                and msgs[i + 1].get("role") == "assistant" else ""
            score = len(q & _terms(m.get("content", "") + " " + reply))
            if score >= 2:
                scored.append((score, c.get("title", ""),
                               m.get("content", ""), reply))
    scored.sort(key=lambda t: -t[0])
    out = []
    for _, title, um, am in scored[:limit]:
        out.append('From the conversation "%s" — asked: "%s" — answered: '
                   '"%s"' % (title, um[:200].replace("\n", " "),
                             am[:400].replace("\n", " ")))
    return out


def context_extra(query: str, chat_id: str) -> str:
    """What rides along with the system prompt: remembered notes, and
    (when enabled) excerpts from earlier conversations."""
    parts = []
    mem = read_memory().strip()
    if mem:
        parts.append("Things to remember from earlier (persistent memory; "
                     "the user can edit these):\n" + mem[:MEMORY_INJECT])
    if cfg.get("recall", True):
        snips = semantic_recall(query, chat_id)
        mode = "meaning"
        if snips is None:
            snips = recall_snippets(query, chat_id)
            mode = "keywords"
        if snips:
            parts.append("Relevant excerpts from earlier conversations "
                         "(matched by " + mode + "):\n" + "\n".join(snips))
    if parts:
        parts.append("To permanently remember a new important fact, put it "
                     "on its own line starting with remember: in your "
                     "reply.")
    return ("\n\n" + "\n\n".join(parts)) if parts else ""


@app.get("/api/memory")
def api_memory_get():
    return jsonify({"text": read_memory(),
                    "recall": bool(cfg.get("recall", True))})


@app.post("/api/memory")
def api_memory_set():
    b = request.get_json(silent=True) or {}
    write_memory(str(b.get("text") or ""))
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
        # embedding and image models are helpers, never chat engines
        if m["folder"] and m["folder"] not in (EMBED_DIR, IMAGE_DIR) \
                and not m["partial"]:
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
        # The tiny recall model rides along; skipping it never fails setup.
        if not embed_model_file():
            task.set(pct=95, detail="Fetching the small recall model…")
            try:
                files = engine.hf_files(cfg, EMBED_REPO)
                cand = [f for f in files
                        if not f["split"] and not f["companion"]]
                pick = next((f for f in cand if f["quant"] == "Q8_0"),
                            None) or min(cand, key=lambda f: f["size"])
                engine.download_sync(cfg, EMBED_REPO, pick["path"],
                                     EMBED_DIR, task, 95, 97,
                                     label="recall model")
            except Exception as exc:  # noqa: BLE001
                task.log(f"Recall model skipped: {exc}")
        task.set(pct=97, detail="Loading Qwen as the primary engine…")
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
        if m.get("folder") in (EMBED_DIR, IMAGE_DIR):
            continue
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


@app.post("/api/workspace/run")
def api_workspace_run():
    """Run the user's own command inside the open folder and hand back
    exit code plus output. The command is always the one the person typed
    — model output never chooses what runs."""
    root = workspace_root()
    if not root:
        return jsonify({"error": "No folder is open."}), 400
    b = request.get_json(silent=True) or {}
    cmd = (b.get("command") or cfg.get("run_command") or "").strip()
    if not cmd:
        return jsonify({"error": "No run command set."}), 400
    cfg["run_command"] = cmd
    save_config(cfg)
    started = time.time()
    try:
        p = subprocess.run(cmd, shell=True, cwd=str(root),
                           capture_output=True, text=True, timeout=60)
        out = ((p.stdout or "") + (p.stderr or "")).strip()
        return jsonify({"ok": True, "exit": p.returncode,
                        "output": out[-8000:],
                        "seconds": round(time.time() - started, 1)})
    except subprocess.TimeoutExpired:
        return jsonify({"ok": True, "exit": -1,
                        "output": "Timed out after 60 seconds.",
                        "seconds": 60.0})
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
              "system": b.get("system", cfg.get("system")),
              # persistent memory, past-conversation excerpts, and the
              # Office-document block convention
              "system_extra": context_extra(text, chat["id"]) + DOC_HOWTO}

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
                # Any `remember:` lines in the reply become permanent
                # memory for every future conversation, and the fresh
                # exchange joins the semantic index in the background.
                remember_lines(reply)
                threading.Thread(target=index_exchanges,
                                 daemon=True).start()

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
        embedder.stop()
        whisper.stop()


if __name__ == "__main__":
    main()
