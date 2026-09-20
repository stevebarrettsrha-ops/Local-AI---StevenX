"""
engine.py - llama.cpp itself: getting a build, running llama-server, talking to
it, and pulling GGUF files.

No Ollama, no Python bindings. The app downloads an official llama.cpp release
binary for this machine, starts `llama-server` with flags worked out from the
fit calculation, and speaks to its OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import uuid
import zipfile
from pathlib import Path

import requests

import fit

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
BIN_DIR = APP_DIR / "llama.cpp"
MODELS_DIR = APP_DIR / "models"

RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
HF_BASE = "https://huggingface.co"


# --------------------------------------------------------------------------- #
# tasks (downloads, installs)
# --------------------------------------------------------------------------- #
class Task:
    def __init__(self, kind: str, title: str, meta: dict | None = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.title, self.meta = kind, title, meta or {}
        self.state, self.pct, self.detail = "running", 0.0, ""
        self.lines: list[str] = []
        self.created = time.time()
        self.cancel = False
        self._lock = threading.Lock()

    def log(self, msg: str) -> None:
        with self._lock:
            self.lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            if len(self.lines) > 800:
                del self.lines[:400]

    def set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def view(self, since: int = 0) -> dict:
        with self._lock:
            return {"id": self.id, "kind": self.kind, "title": self.title,
                    "meta": self.meta, "state": self.state,
                    "pct": round(self.pct, 1), "detail": self.detail,
                    "created": self.created, "cursor": len(self.lines),
                    "lines": self.lines[since:]}


TASKS: dict[str, Task] = {}
_tasks_lock = threading.Lock()


def spawn(kind: str, title: str, fn, meta: dict | None = None) -> Task:
    task = Task(kind, title, meta)
    with _tasks_lock:
        TASKS[task.id] = task
        finished = sorted((t for t in TASKS.values() if t.state != "running"),
                          key=lambda t: t.created)
        for old in finished[:-30]:
            TASKS.pop(old.id, None)

    def wrapper():
        try:
            fn(task)
            if task.state == "running":
                task.set(state="done", pct=100)
        except Exception as exc:  # noqa: BLE001
            task.log(f"FAILED: {exc}")
            task.set(state="error", detail=str(exc))

    threading.Thread(target=wrapper, daemon=True).start()
    return task


def task_list() -> list[Task]:
    with _tasks_lock:
        return sorted(TASKS.values(), key=lambda t: t.created, reverse=True)


# --------------------------------------------------------------------------- #
# picking a build
# --------------------------------------------------------------------------- #
def asset_pattern(hw: dict) -> tuple[str, str]:
    """(substring to look for in the release asset names, why)."""
    system = platform.system()
    if system == "Windows":
        if hw.get("vendor") == "nvidia":
            return "bin-win-cuda", "CUDA build for your NVIDIA card"
        if hw.get("vendor") == "amd":
            return "bin-win-hip", "HIP build for your AMD card"
        return "bin-win-cpu", "CPU build"
    if system == "Darwin":
        return "bin-macos-arm64", "Apple Silicon build with Metal"
    if hw.get("vendor") == "nvidia":
        return "bin-ubuntu-cuda", "CUDA build for your NVIDIA card"
    return "bin-ubuntu-x64", "CPU build"


def server_binary() -> Path | None:
    name = "llama-server.exe" if platform.system() == "Windows" \
        else "llama-server"
    for base in (BIN_DIR, BIN_DIR / "build" / "bin"):
        if not base.is_dir():
            continue
        direct = base / name
        if direct.exists():
            return direct
        for found in base.rglob(name):
            return found
    found = shutil.which(name)
    return Path(found) if found else None


def install(hw: dict) -> Task:
    """Download an official release build and unpack it into ./llama.cpp."""
    return spawn("install", "llama.cpp", lambda t: install_sync(t, hw),
                 {"pattern": asset_pattern(hw)[0]})


def install_sync(task: Task, hw: dict) -> None:
    """The install itself, runnable inside another task (first-run setup)."""
    pattern, why = asset_pattern(hw)
    task.set(detail="Asking GitHub for the latest release…")
    r = requests.get(RELEASES, timeout=30,
                     headers={"Accept": "application/vnd.github+json"})
    r.raise_for_status()
    release = r.json()
    assets = release.get("assets", [])
    match = next((a for a in assets
                  if pattern in a["name"] and a["name"].endswith(".zip")), None)
    if not match:
        names = ", ".join(a["name"] for a in assets[:8])
        raise RuntimeError(
            f"No asset matching '{pattern}' in {release.get('tag_name')}. "
            f"Available: {names}")
    task.log(f"{release.get('tag_name')} — {match['name']} ({why})")
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    archive = BIN_DIR / match["name"]
    _stream(match["browser_download_url"], archive, {},
            lambda got, total, speed, eta: task.set(
                pct=(got / total * 100) if total else 0,
                detail=f"{got/1e6:.0f} of {total/1e6:.0f} MB"),
            lambda: task.cancel)
    if task.cancel:
        task.set(state="cancelled", detail="Cancelled")
        return
    task.set(detail="Unpacking…")
    with zipfile.ZipFile(archive) as z:
        z.extractall(BIN_DIR)
    archive.unlink(missing_ok=True)
    binary = server_binary()
    if not binary:
        raise RuntimeError("Unpacked, but no llama-server binary was found "
                           "inside the archive.")
    if platform.system() != "Windows":
        for f in BIN_DIR.rglob("llama-*"):
            try:
                f.chmod(0o755)
            except OSError:
                pass
    task.set(detail=f"Ready — {binary}")


# --------------------------------------------------------------------------- #
# models on disk / on HuggingFace
# --------------------------------------------------------------------------- #
def local_models() -> list[dict]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for f in sorted(MODELS_DIR.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".gguf", ".part"):
            rel = f.relative_to(MODELS_DIR)
            out.append({"name": f.name, "path": str(f),
                        # the engine family folder it lives in, if any
                        "folder": rel.parts[0] if len(rel.parts) > 1 else "",
                        "size": f.stat().st_size,
                        "quant": fit.quant_of(f.name),
                        "partial": f.suffix.lower() == ".part"})
    return out


def hf_headers(cfg: dict) -> dict:
    token = (cfg.get("hf_token") or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_files(cfg: dict, repo: str) -> list[dict]:
    """Every GGUF in a repo, with its real size and quant label."""
    endpoint = (cfg.get("hf_endpoint") or HF_BASE).rstrip("/")
    r = requests.get(f"{endpoint}/api/models/{repo}/tree/main?recursive=1",
                     headers=hf_headers(cfg), timeout=30)
    if r.status_code == 401:
        raise RuntimeError("That repo needs a HuggingFace token — add one on "
                           "the Models page.")
    if r.status_code == 403:
        raise RuntimeError("Your token cannot read that repo. Some models ask "
                           "you to accept a licence first.")
    if r.status_code == 404:
        raise RuntimeError(f"No such repo: {repo}")
    r.raise_for_status()
    files = []
    have = {m["name"] for m in local_models()}
    for e in r.json():
        if e.get("type") != "file" or not e["path"].lower().endswith(".gguf"):
            continue
        size = (e.get("lfs") or {}).get("size") or e.get("size") or 0
        name = Path(e["path"]).name
        low = name.lower()
        # mmproj is the vision projector, not a model: it is loaded alongside
        # one with --mmproj. Listing it as a candidate (and recommending it,
        # since it is small enough to "fit") is nonsense.
        companion = low.startswith("mmproj") or "mmproj" in low
        files.append({"path": e["path"], "name": name, "size": size,
                      "quant": fit.quant_of(name), "installed": name in have,
                      "split": bool("-of-" in name),
                      "companion": companion,
                      "role": "vision projector" if companion else "model"})
    def rank(f):
        q = f["quant"]
        return (1 if f["companion"] else 0,
                fit.QUANT_ORDER.index(q) if q in fit.QUANT_ORDER else 99,
                f["size"])
    files.sort(key=rank)
    if not files:
        hint = gguf_sibling(cfg, repo)
        msg = (f"{repo} has no GGUF files. FP8, NVFP4, INT8 and MLX repos are "
               "for vLLM, TensorRT and Apple MLX; llama.cpp loads GGUF only.")
        raise RuntimeError(msg + (f" Try {hint} instead." if hint else
                                  " Look for a sibling repo ending in -GGUF."))
    return files


def gguf_sibling(cfg: dict, repo: str) -> str:
    """Guess the GGUF build of a safetensors repo, and only return it if it
    is really there — a suggestion that 404s is worse than none."""
    base = repo.rsplit("-", 1)[0] if repo.rsplit("-", 1)[-1].upper() in (
        "FP8", "NVFP4", "INT8", "MLX", "AWQ", "GPTQ") else repo
    endpoint = (cfg.get("hf_endpoint") or HF_BASE).rstrip("/")
    for candidate in (base + "-GGUF", repo + "-GGUF"):
        try:
            r = requests.get(
                f"{endpoint}/api/models/{candidate}/tree/main",
                headers=hf_headers(cfg), timeout=15)
            if r.status_code == 200:
                return candidate
        except Exception:
            continue
    return ""


def model_dest(path: str, folder: str = "") -> Path:
    """Where a downloaded file lands: ./models, or an engine family's own
    subfolder (models/qwen, models/gemma) when one is named."""
    folder = re.sub(r"[^A-Za-z0-9._-]", "", folder or "")
    base = MODELS_DIR / folder if folder else MODELS_DIR
    return base / Path(path).name


def download(cfg: dict, repo: str, path: str, folder: str = "") -> Task:
    dest = model_dest(path, folder)
    if dest.exists():
        raise RuntimeError(f"{dest.name} is already here.")
    if any(t.meta.get("dest") == str(dest) for t in task_list()
           if t.state == "running"):
        raise RuntimeError(f"{dest.name} is already downloading.")

    def run(task: Task) -> None:
        download_sync(cfg, repo, path, folder, task)

    return spawn("download", dest.name, run,
                 {"dest": str(dest), "repo": repo, "name": dest.name})


def download_sync(cfg: dict, repo: str, path: str, folder: str, task: Task,
                  pct_from: float = 0, pct_to: float = 100,
                  label: str = "") -> Path:
    """One file, fetched inside an existing task, its progress mapped onto
    [pct_from, pct_to] so multi-step tasks keep one honest bar."""
    endpoint = (cfg.get("hf_endpoint") or HF_BASE).rstrip("/")
    url = f"{endpoint}/{repo}/resolve/main/{urllib.parse.quote(path)}"
    dest = model_dest(path, folder)
    prefix = (label + ": ") if label else ""
    if dest.exists():
        task.log(f"{dest.name} is already here — skipping.")
        task.set(pct=pct_to)
        return dest
    task.log(f"{repo}/{path} → {dest.relative_to(MODELS_DIR).parent}/")
    span = pct_to - pct_from
    _stream(url, dest, hf_headers(cfg),
            lambda got, total, speed, eta: task.set(
                pct=pct_from + ((got / total * span) if total else 0),
                detail=f"{prefix}{got/1e9:.1f} of {total/1e9:.1f} GB · "
                       f"{speed/1e6:.0f} MB/s · "
                       f"{int(eta//60)}m {int(eta%60)}s left"),
            lambda: task.cancel)
    if task.cancel:
        task.set(state="cancelled",
                 detail="Cancelled — what downloaded is kept and will be "
                        "resumed.")
        return dest
    task.set(pct=pct_to,
             detail=f"{prefix}saved — {dest.stat().st_size/1e9:.1f} GB")
    return dest


def delete_model(name: str) -> None:
    if "/" in name or "\\" in name or ".." in name:
        raise RuntimeError("That path is not allowed.")
    # Look the file up rather than joining paths: models now live in per-
    # engine subfolders, and local_models() only ever lists files inside
    # MODELS_DIR, so anything it returns is safe to remove.
    model = next((m for m in local_models() if m["name"] == name), None)
    if not model:
        raise RuntimeError("That file is already gone.")
    Path(model["path"]).unlink()


def _stream(url: str, dest: Path, headers: dict, on_progress, should_cancel):
    """Resumable download: .part file, Range on retry, atomic replace."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(headers)
    if have:
        headers["Range"] = f"bytes={have}-"
    with requests.get(url, headers=headers, stream=True, timeout=60,
                      allow_redirects=True) as r:
        if r.status_code == 416:
            part.replace(dest)
            return
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if (have and r.status_code == 206) else "wb"
        if mode == "wb":
            have = 0
        got, last, started = have, 0.0, time.time()
        with open(part, mode) as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if should_cancel():
                    return
                if not chunk:
                    continue
                fh.write(chunk)
                got += len(chunk)
                now = time.time()
                if on_progress and now - last > 0.5:
                    last = now
                    speed = (got - have) / max(now - started, .1)
                    eta = (total - got) / speed if speed > 0 and total else 0
                    on_progress(got, total, speed, eta)
    part.replace(dest)


# --------------------------------------------------------------------------- #
# llama-server
# --------------------------------------------------------------------------- #
class Server:
    """One llama-server process, and the chat calls that go to it."""

    def __init__(self, port: int = 8080) -> None:
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.model: str = ""
        self.flags: list[str] = []
        self.lines: list[str] = []
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ready(self) -> bool:
        try:
            r = requests.get(f"{self.url}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def start(self, model_path: str, gpu_layers: int, ctx: int,
              extra: list[str] | None = None) -> None:
        binary = server_binary()
        if not binary:
            raise RuntimeError("llama.cpp is not installed yet — install it on "
                               "the Engine page.")
        if not Path(model_path).exists():
            raise RuntimeError(f"No such model file: {model_path}")
        self.stop()
        cmd = [str(binary), "-m", model_path, "--host", "127.0.0.1",
               "--port", str(self.port), "-c", str(ctx),
               "-ngl", str(max(0, gpu_layers)), "--no-webui"]
        if extra:
            cmd += extra
        with self._lock:
            self.lines = [" ".join(cmd)]
        env = dict(os.environ)
        if platform.system() != "Windows":
            lib = str(server_binary().parent)
            env["LD_LIBRARY_PATH"] = lib + ":" + env.get("LD_LIBRARY_PATH", "")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if platform.system() == "Windows" else 0
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, env=env, creationflags=flags)
        self.model, self.flags = model_path, cmd
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            with self._lock:
                self.lines.append(line.rstrip())
                if len(self.lines) > 500:
                    del self.lines[:250]

    def wait_ready(self, timeout: int = 600) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Death first: if OUR process has exited, a healthy /health can
            # only be an impostor already squatting on the port (say, an
            # orphaned llama-server from a crashed session) — that is a
            # failure to report, not a success to claim.
            if self.proc and self.proc.poll() is not None:
                return False
            if self.ready():
                return True
            time.sleep(1)
        return False

    def tail(self, n: int = 40) -> list[str]:
        with self._lock:
            return self.lines[-n:]

    def stop(self) -> None:
        if self.alive():
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None
        self.model = ""

    # -------------------------------------------------------------- chat --- #
    def chat_stream(self, messages: list[dict], params: dict):
        """Yields text deltas from llama-server's OpenAI-compatible endpoint."""
        body = {"messages": messages, "stream": True,
                "temperature": float(params.get("temperature", 0.7)),
                "top_p": float(params.get("top_p", 0.95)),
                "max_tokens": int(params.get("max_tokens", 1024))}
        if params.get("system"):
            body["messages"] = [{"role": "system",
                                 "content": params["system"]}] + messages
        with requests.post(f"{self.url}/v1/chat/completions", json=body,
                           stream=True, timeout=600) as r:
            if r.status_code >= 400:
                raise RuntimeError(f"llama-server said: {r.text[:300]}")
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield delta

    def props(self) -> dict:
        try:
            return requests.get(f"{self.url}/props", timeout=5).json()
        except Exception:
            return {}
