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
import socket
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

# Overridable for corporate mirrors and offline test rigs.
GH_API = os.environ.get("LLAMA_STUDIO_GITHUB_API",
                        "https://api.github.com").rstrip("/")
RELEASES = f"{GH_API}/repos/ggml-org/llama.cpp/releases"
WHISPER_RELEASES = f"{GH_API}/repos/ggml-org/whisper.cpp/releases"
WHISPER_DIR = APP_DIR / "whisper.cpp"
SD_RELEASES = f"{GH_API}/repos/leejet/stable-diffusion.cpp/releases"
SD_DIR = APP_DIR / "stable-diffusion.cpp"
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
def asset_choices(hw: dict) -> list[tuple[list[str], str]]:
    """Substring sets to look for in release asset names, best first.
    Later entries are graceful fallbacks — a CPU build that runs beats a
    CUDA build that was never found."""
    system = platform.system()
    if system == "Windows":
        cpu = [(["win", "cpu"], "CPU build"),
               (["win", "avx2"], "CPU build")]
        if hw.get("vendor") == "nvidia":
            return [(["win", "cuda"],
                     "CUDA build for your NVIDIA card")] + cpu
        if hw.get("vendor") == "amd":
            return [(["win", "hip"], "HIP build for your AMD card"),
                    (["win", "vulkan"], "Vulkan build for your AMD card")] \
                + cpu
        return cpu
    if system == "Darwin":
        return [(["macos", "arm64"], "Apple Silicon build with Metal")]
    cpu = [(["ubuntu", "x64"], "CPU build"),
           (["linux", "x64"], "CPU build")]
    if hw.get("vendor") == "nvidia":
        return [(["ubuntu", "cuda"], "CUDA build for your NVIDIA card"),
                (["linux", "cuda"], "CUDA build for your NVIDIA card")] + cpu
    return cpu


def _pick_asset(assets: list[dict],
                choices: list[tuple[list[str], str]]) -> tuple[dict, str] | None:
    """The best zip in this release for this machine. The cudart zips are
    runtime DLLs, not builds — never the main pick."""
    for tokens, why in choices:
        for a in assets:
            low = (a.get("name") or "").lower()
            if (low.endswith(".zip") and "cudart" not in low
                    and all(t in low for t in tokens)):
                return a, why
    return None


def _resolve_release(choices: list[tuple[list[str], str]],
                     task: Task, base: str = "") -> tuple[dict, dict, str]:
    """(release, asset, why) for the newest release that really carries a
    build for this machine.

    llama.cpp's 'latest' release is sometimes only a pointer: a tiny
    *tag*.txt asset naming the tag of the release the binaries actually
    live in. Follow that pointer, and failing that walk the recent
    releases, rather than declaring there is nothing to install."""
    base = base or RELEASES
    headers = {"Accept": "application/vnd.github+json"}
    tried: list[str] = []
    r = requests.get(base + "/latest", timeout=30, headers=headers)
    r.raise_for_status()
    release = r.json()
    found = _pick_asset(release.get("assets", []), choices)
    if found:
        return release, found[0], found[1]
    tried.append(release.get("tag_name") or "latest")
    pointer = next((a for a in release.get("assets", [])
                    if (a.get("name") or "").lower().endswith(".txt")
                    and "tag" in (a.get("name") or "").lower()), None)
    if pointer:
        try:
            text = requests.get(pointer["browser_download_url"],
                                timeout=30).text
            tag = text.strip().splitlines()[0].strip() if text.strip() else ""
        except Exception:
            tag = ""
        if tag:
            task.log(f"{release.get('tag_name')} only points at the real "
                     f"build — following it to {tag}.")
            rr = requests.get(f"{base}/tags/{tag}", timeout=30,
                              headers=headers)
            if rr.status_code == 200:
                rel = rr.json()
                found = _pick_asset(rel.get("assets", []), choices)
                if found:
                    return rel, found[0], found[1]
                tried.append(tag)
    rr = requests.get(base + "?per_page=15", timeout=30, headers=headers)
    if rr.status_code == 200:
        for rel in rr.json():
            found = _pick_asset(rel.get("assets", []), choices)
            if found:
                return rel, found[0], found[1]
            tried.append(rel.get("tag_name") or "?")
    sample = ", ".join(a.get("name", "")
                       for a in release.get("assets", [])[:8])
    raise RuntimeError(
        "No usable llama.cpp build in recent releases (looked at " +
        ", ".join(tried[:8]) + "). The latest release offers: " +
        (sample or "nothing") + ".")


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


def whisper_binary() -> Path | None:
    name = "whisper-server.exe" if platform.system() == "Windows" \
        else "whisper-server"
    if WHISPER_DIR.is_dir():
        direct = WHISPER_DIR / name
        if direct.exists():
            return direct
        for found in WHISPER_DIR.rglob(name):
            return found
    found = shutil.which(name)
    return Path(found) if found else None


def whisper_install_sync(task: Task) -> None:
    """whisper.cpp's server binary, from the official releases. Prebuilt
    zips exist for Windows; elsewhere the person supplies the binary."""
    if platform.system() != "Windows":
        raise RuntimeError(
            "whisper.cpp publishes prebuilt binaries for Windows only. On "
            "this OS, put a whisper-server binary into ./whisper.cpp "
            "yourself (package manager or your own build) and retry.")
    choices = [(["whisper-bin-x64"], "CPU build"),
               (["bin", "x64"], "CPU build")]
    task.set(detail="Asking GitHub for the latest whisper.cpp release…")
    release, match, why = _resolve_release(choices, task,
                                           base=WHISPER_RELEASES)
    task.log(f"{release.get('tag_name')} — {match['name']} ({why})")
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    if not _fetch_zip(task, match, 0, 40, dest_dir=WHISPER_DIR):
        task.set(state="cancelled", detail="Cancelled")
        return
    binary = whisper_binary()
    if not binary:
        raise RuntimeError("Unpacked, but no whisper-server binary was "
                           "found inside the archive.")
    if platform.system() != "Windows":
        for f in WHISPER_DIR.rglob("whisper-*"):
            try:
                f.chmod(0o755)
            except OSError:
                pass


def sd_binary() -> Path | None:
    name = "sd.exe" if platform.system() == "Windows" else "sd"
    if SD_DIR.is_dir():
        direct = SD_DIR / name
        if direct.exists():
            return direct
        for found in SD_DIR.rglob(name):
            return found
    return None


def sd_install_sync(task: Task, hw: dict) -> None:
    """stable-diffusion.cpp's CLI from the official releases (assets are
    sd-master-<hash>-bin-win-<backend>-x64.zip, with a cudart companion
    for the CUDA build)."""
    if platform.system() != "Windows":
        raise RuntimeError(
            "stable-diffusion.cpp publishes prebuilt binaries for Windows "
            "only. On this OS, put an sd binary into "
            "./stable-diffusion.cpp yourself and retry.")
    choices = [(["win", "avx2"], "CPU build")]
    if hw.get("vendor") == "nvidia":
        choices = [(["win", "cuda12"],
                    "CUDA build for your NVIDIA card")] + choices
    task.set(detail="Asking GitHub for the latest stable-diffusion.cpp "
                    "release…")
    release, match, why = _resolve_release(choices, task, base=SD_RELEASES)
    task.log(f"{release.get('tag_name')} — {match['name']} ({why})")
    SD_DIR.mkdir(parents=True, exist_ok=True)
    if not _fetch_zip(task, match, 0, 22, dest_dir=SD_DIR):
        task.set(state="cancelled", detail="Cancelled")
        return
    if "cuda" in match["name"].lower():
        runtime = next((a for a in release.get("assets", [])
                        if "cudart" in (a.get("name") or "").lower()
                        and (a.get("name") or "").lower().endswith(".zip")),
                       None)
        if runtime:
            task.set(pct=22, detail="Fetching the CUDA runtime DLLs…")
            if not _fetch_zip(task, runtime, 22, 30, dest_dir=SD_DIR):
                task.set(state="cancelled", detail="Cancelled")
                return
    if not sd_binary():
        raise RuntimeError("Unpacked, but no sd binary was found inside "
                           "the archive.")


def install(hw: dict) -> Task:
    """Download an official release build and unpack it into ./llama.cpp."""
    return spawn("install", "llama.cpp", lambda t: install_sync(t, hw))


def _fetch_zip(task: Task, asset: dict, pct_from: float,
               pct_to: float, dest_dir: Path | None = None) -> bool:
    """Download one release zip into BIN_DIR and unpack it there. False
    means the task was cancelled mid-way."""
    dest_dir = dest_dir or BIN_DIR
    archive = dest_dir / asset["name"]
    span = pct_to - pct_from
    _stream(asset["browser_download_url"], archive, {},
            lambda got, total, speed, eta: task.set(
                pct=pct_from + ((got / total * span) if total else 0),
                detail=f"{asset['name']}: {got/1e6:.0f} of "
                       f"{total/1e6:.0f} MB"),
            lambda: task.cancel)
    if task.cancel:
        archive.unlink(missing_ok=True)
        return False
    task.set(detail=f"Unpacking {asset['name']}…")
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest_dir)
    archive.unlink(missing_ok=True)
    return True


def install_sync(task: Task, hw: dict) -> None:
    """The install itself, runnable inside another task (first-run setup)."""
    task.set(detail="Asking GitHub for the latest release…")
    release, match, why = _resolve_release(asset_choices(hw), task)
    task.log(f"{release.get('tag_name')} — {match['name']} ({why})")
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    if not _fetch_zip(task, match, 0, 80):
        task.set(state="cancelled", detail="Cancelled")
        return
    # A CUDA build on Windows needs the CUDA runtime DLLs too, shipped as
    # a separate cudart zip beside it — without them llama-server.exe
    # only starts on machines that happen to have the CUDA toolkit.
    if "cuda" in match["name"].lower() and platform.system() == "Windows":
        runtime = next((a for a in release.get("assets", [])
                        if "cudart" in (a.get("name") or "").lower()
                        and (a.get("name") or "").lower().endswith(".zip")),
                       None)
        if runtime:
            task.set(pct=80, detail="Fetching the CUDA runtime DLLs…")
            if not _fetch_zip(task, runtime, 80, 95):
                task.set(state="cancelled", detail="Cancelled")
                return
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
        self.ctx: int = 0
        self._n_ctx: int = 0
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
        # --jinja: the model's own chat template, not llama.cpp's built-in
        # approximation of it. Current builds default to it; an older build
        # installed months ago does not, and without it a thinking model's
        # template (and its enable_thinking switch) is never applied.
        cmd = [str(binary), "-m", model_path, "--host", "127.0.0.1",
               "--port", str(self.port), "-c", str(ctx),
               "-ngl", str(max(0, gpu_layers)), "--no-webui", "--jinja"]
        if extra:
            cmd += extra
        self.ctx = int(ctx)
        self._n_ctx = 0
        self._spawn(cmd, model_path)

    def _spawn(self, cmd: list[str], model_path: str) -> None:
        with self._lock:
            self.lines = [" ".join(cmd)]
        env = dict(os.environ)
        if platform.system() != "Windows":
            lib = str(Path(cmd[0]).parent)
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
        self.ctx = 0
        self._n_ctx = 0

    # ------------------------------------------------------------ context --- #
    def n_ctx(self) -> int:
        """The context window the running server actually has, asked of the
        server rather than assumed from what we passed on the command line —
        llama.cpp can round it down to the model's own trained maximum.
        Asked once per loaded model, not once per message."""
        if self._n_ctx:
            return self._n_ctx
        p = self.props()
        gen = p.get("default_generation_settings") or {}
        for value in (gen.get("n_ctx"), p.get("n_ctx")):
            try:
                if int(value) > 0:
                    self._n_ctx = int(value)
                    return self._n_ctx
            except (TypeError, ValueError):
                continue
        return self.ctx

    def count_tokens(self, text: str) -> int:
        """Real token count from the model's own tokenizer. Falls back to the
        four-characters-a-token rule of thumb only when /tokenize is not
        there, and says so by returning the estimate rather than raising."""
        if not text:
            return 0
        try:
            r = requests.post(f"{self.url}/tokenize", json={"content": text},
                              timeout=20)
            if r.status_code < 400:
                toks = r.json().get("tokens")
                if isinstance(toks, list):
                    return len(toks)
        except Exception:
            pass
        return max(1, len(text) // 4)

    # -------------------------------------------------------------- chat --- #
    def chat_stream(self, messages: list[dict], params: dict,
                    handle: dict | None = None):
        """Yields events from llama-server's OpenAI-compatible endpoint:
        {"prompt": {"processed", "total", "cache"}} while it reads the
        prompt, {"think": text} for each piece of a thinking model's
        reasoning, {"delta": text} for each piece of the answer, then one
        {"stop": reason, "timings": {...}} — "length" when the reply hit the
        token limit, "stop" when the model finished of its own accord.
        Without that last event a cut-off reply is indistinguishable from a
        finished one, which is exactly how a truncated answer used to pass
        for a complete one.

        The reasoning arrives in its own field (reasoning_content) and used to
        be dropped here unseen: a thinking model looked frozen for a minute,
        and when the window ran out mid-thought there was no answer at all.

        `handle`, when given, is how the caller watches and ends the reply
        from another thread: the open response lands in it as "response",
        llama.cpp's latest measured timings as "timings", and setting
        "cancel" (see abort) hangs up, which is what makes llama-server stop
        generating."""
        body = {"messages": messages, "stream": True,
                "temperature": float(params.get("temperature", 0.7)),
                "top_p": float(params.get("top_p", 0.95)),
                "max_tokens": int(params.get("max_tokens", 1024)),
                # Reading a long prompt, and every token that is not text,
                # used to send nothing at all: the page could not tell a
                # working model from a stuck one. llama.cpp counts both for
                # us when asked; an older build ignores the two keys.
                "return_progress": True, "timings_per_token": True}
        # The rest of a model's published sampling rides along only when
        # there is one; otherwise llama-server's own defaults stand.
        for key in ("top_k", "min_p", "repeat_penalty"):
            if params.get(key) is not None:
                body[key] = params[key]
        if params.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = params["chat_template_kwargs"]
        sys_full = ((params.get("system") or "") +
                    (params.get("system_extra") or "")).strip()
        if sys_full:
            body["messages"] = [{"role": "system",
                                 "content": sys_full}] + messages
        reason, timings = "", {}
        live = handle if handle is not None else {}
        if live.get("cancel"):
            return
        with requests.post(f"{self.url}/v1/chat/completions", json=body,
                           stream=True, timeout=600) as r:
            live["response"] = r
            if r.status_code >= 400:
                raise RuntimeError(f"llama-server said: {r.text[:300]}")
            for raw in r.iter_lines(decode_unicode=True):
                if live.get("cancel"):
                    return
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                if isinstance(chunk.get("timings"), dict):
                    timings = live["timings"] = chunk["timings"]
                progress = chunk.get("prompt_progress")
                if isinstance(progress, dict):
                    yield {"prompt": {
                        key: int(progress.get(key) or 0)
                        for key in ("processed", "total", "cache")}}
                for choice in chunk.get("choices", []):
                    if choice.get("finish_reason"):
                        reason = str(choice["finish_reason"])
                    d = choice.get("delta") or {}
                    think = d.get("reasoning_content")
                    if think:
                        yield {"think": think}
                    delta = d.get("content")
                    if delta:
                        yield {"delta": delta}
        yield {"stop": reason or "stop", "timings": timings}

    @staticmethod
    def abort(handle: dict) -> None:
        """Hang up on a reply from another thread. llama-server stops a
        task only when its client goes away, and a reader blocked on a
        model that is producing nothing visible would otherwise hold the
        connection — and the model — until the token limit, long after
        Stop was pressed. Shutting the socket for reading wakes that
        reader; leaving the `with` block then closes the connection."""
        handle["cancel"] = True
        r = handle.get("response")
        if r is None:
            return
        try:
            r.raw.shutdown()  # urllib3 2.3+
            return
        except Exception:
            pass
        try:  # older urllib3: the same thing by hand
            r.raw._connection.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass

    def activity(self) -> dict:
        """What the model is doing right now, from llama-server's /slots:
        whether it is working on a reply, and how many tokens of it are
        generated. That count moves even when none of the tokens is text,
        which the stream itself never shows — a model emitting only
        padding looks exactly like a frozen page otherwise. {} when the
        server does not answer in time (it replies between batches, so a
        long prompt batch holds it up) or runs with --no-slots."""
        try:
            r = requests.get(f"{self.url}/slots", timeout=1)
            slots = r.json() if r.status_code < 400 else None
        except Exception:
            return {}
        if not isinstance(slots, list):
            return {}
        busy = [s for s in slots
                if isinstance(s, dict) and s.get("is_processing")]
        if not busy:
            return {"processing": False, "decoded": 0}
        # the newest task is the reply being streamed now
        slot = max(busy, key=lambda s: int(s.get("id_task") or 0))
        nxt = slot.get("next_token")
        if isinstance(nxt, list):
            nxt = nxt[0] if nxt else {}
        try:
            decoded = int((nxt or {}).get("n_decoded") or 0)
        except (TypeError, ValueError):
            decoded = 0
        return {"processing": True, "decoded": decoded}

    def props(self) -> dict:
        try:
            return requests.get(f"{self.url}/props", timeout=5).json()
        except Exception:
            return {}


class WhisperServer(Server):
    """whisper.cpp's server: local speech to text, no audio leaves the
    machine. Started from the whisper.cpp binary, not llama.cpp's."""

    def start_whisper(self, model_path: str) -> None:
        binary = whisper_binary()
        if not binary:
            raise RuntimeError("whisper.cpp is not installed yet.")
        if not Path(model_path).exists():
            raise RuntimeError(f"No such model file: {model_path}")
        self.stop()
        self._spawn([str(binary), "-m", model_path,
                     "--host", "127.0.0.1", "--port", str(self.port)],
                    model_path)

    def ready(self) -> bool:
        # whisper-server has no /health; its root page answering is
        # the ready signal.
        try:
            return requests.get(self.url + "/",
                                timeout=2).status_code < 500
        except Exception:
            return False

    def transcribe(self, wav: bytes) -> str:
        r = requests.post(self.url + "/inference",
                          files={"file": ("audio.wav", wav, "audio/wav")},
                          data={"temperature": "0.0",
                                "response_format": "json"},
                          timeout=120)
        r.raise_for_status()
        return (r.json().get("text") or "").strip()


class EmbedServer(Server):
    """A second llama-server running a small embedding model — used to
    match new questions to past conversations by meaning. Same official
    release binary, just started with --embeddings on its own port."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = requests.post(f"{self.url}/v1/embeddings",
                          json={"input": texts}, timeout=60)
        r.raise_for_status()
        data = r.json().get("data", [])
        return [d["embedding"]
                for d in sorted(data, key=lambda x: x["index"])]
