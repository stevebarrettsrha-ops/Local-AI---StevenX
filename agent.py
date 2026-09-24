"""
agent.py - the coding agent: the model works in the open folder with tools,
step after step, until the task is done.

A reply in the chat can only describe code. The agent writes it: it lists and
reads the project, writes and edits files, keeps a plan, runs the project and
reads the errors, and carries on — for as many steps as the task needs — so
"build me a snake game" ends as a folder holding a game that runs.

It runs here, beside the web server, not in the page: a long run survives a
reload or a trip to another page, and the page only watches it (events) and
answers its questions (approvals). Everything the model can touch goes
through the host the server hands in — the same guarded workspace functions
the Code page uses — so the agent can reach nothing the page could not.

Commands are the one thing a file tool cannot do by itself, and the one the
person decides on: every run_command waits for their approval, once or for
the rest of the run.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

MAX_STEPS = 80            # per run; a paused run carries on when told to
READ_CAP = 30_000         # characters of a file returned by one read_file
OUTPUT_CAP = 6_000        # characters of command output returned
LIST_CAP = 400            # entries in one list_files
SEARCH_CAP = 80           # matches in one search
REPLY_FLOOR = 2_048       # the least room a step gets to answer in
KEEP_RECENT = 6           # newest transcript messages never compacted
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env",
             "dist", "build", "target", ".idea", ".vscode", ".next",
             "site-packages"}

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List the files under a folder of the project (noise "
                       "like .git and node_modules is skipped), with sizes.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string",
                     "description": "Folder relative to the project root; "
                                    "'.' for the root."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file of the project. Long files come "
                       "back in parts: give start_line to read further.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer",
                           "description": "First line to return (1-based)."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Find text in the project's files. Returns "
                       "file:line: text for each match.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string",
                        "description": "Text to find (case-insensitive)."},
            "path": {"type": "string",
                     "description": "Folder to search; '.' for all."}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create a file, or replace one completely. Folders "
                       "are created as needed. Always the whole file — no "
                       "placeholders. For a very large file, write the first "
                       "part here and add the rest with append_file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "append_file",
        "description": "Add text to the end of a file (created if missing). "
                       "For writing a large file in parts.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Change part of an existing file: old_text must "
                       "appear exactly once (copy it from read_file, with "
                       "enough surrounding lines to be unique) and is "
                       "replaced by new_text.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "replace_all": {"type": "boolean",
                            "description": "Replace every occurrence."}},
            "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command in the project folder and get "
                       "its exit code and output. The person approves each "
                       "command. It must finish on its own: no servers, "
                       "watchers or interactive programs.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer",
                        "description": "Seconds before it is stopped "
                                       "(default 60, at most 300)."}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "update_plan",
        "description": "Set the checklist for the task and mark progress. "
                       "Send the whole list each time.",
        "parameters": {"type": "object", "properties": {
            "steps": {"type": "array", "items": {
                "type": "object", "properties": {
                    "step": {"type": "string"},
                    "status": {"type": "string",
                               "enum": ["pending", "in_progress", "done"]}},
                "required": ["step", "status"]}}},
            "required": ["steps"]}}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

SYSTEM = """You are a coding agent. You build and change software by calling \
tools in the project folder {root}. The person watches every step.

How to work:
1. Look before you change: list_files, read_file and search the parts you \
need. Never edit code you have not read.
2. For anything bigger than a small fix, call update_plan with a short \
checklist first, and update it as you go.
3. Write complete, working code. write_file creates or replaces a whole file; \
edit_file changes part of an existing one. No placeholders, no "rest of the \
code here", no TODOs standing in for logic.
4. Check your work. Run the project or its tests with run_command when that \
is possible, read the errors, fix them, and run again. Commands must finish \
on their own — no dev servers, watchers or prompts.
5. When the task is done and checked, stop calling tools and reply with a \
short summary: what you built, the files, and how to run it.

Web apps and games: a folder with index.html plus sibling .js and .css files \
(relative paths, no build step, no CDN needed) runs as it is, and the person \
can open it with the Preview button. Keep game state in plain JavaScript; \
guard localStorage in try/catch.

This machine: {system_info}
Project folder now:
{listing}"""

CUT_OFF = ("Your last message was cut off at the length limit before it "
           "finished, so nothing in it was run. Keep each step smaller: "
           "write a big file in parts (write_file, then append_file), and "
           "keep explanations short.")


def _flag(value) -> bool:
    """A boolean argument, also when a model sends it as text."""
    return value is True or str(value).strip().lower() in ("true", "1",
                                                           "yes")


def _tool_msg(call_id: str, text: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


# --------------------------------------------------------------------------- #
# tool calls the server's parser missed
# --------------------------------------------------------------------------- #
HERMES_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
XML_CALL = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>"
                      r"\s*</tool_call>", re.S)
XML_PARAM = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)


def calls_in_text(text: str) -> tuple[list[dict], str]:
    """Tool calls a model wrote into its text instead of the tool channel —
    what an older llama-server, or a template it has no parser for, leaves
    behind — in Hermes JSON (<tool_call>{...}</tool_call>) or Qwen3-Coder
    XML (<function=...><parameter=...>). Returns the calls and the text
    with them taken out."""
    calls = []
    for m in XML_CALL.finditer(text):
        args = {k: v for k, v in XML_PARAM.findall(m.group(2))}
        calls.append({"name": m.group(1), "arguments": json.dumps(args)})
    rest = XML_CALL.sub("", text)
    for m in HERMES_CALL.finditer(rest):
        try:
            obj = json.loads(m.group(1))
        except ValueError:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        calls.append({"name": str(obj.get("name", "")),
                      "arguments": args if isinstance(args, str)
                      else json.dumps(args)})
    rest = HERMES_CALL.sub("", rest)
    return ([{"id": "call_" + uuid.uuid4().hex[:10], **c} for c in calls],
            rest.strip() if calls else text)


# --------------------------------------------------------------------------- #
# a session: one task in one folder, with every step kept
# --------------------------------------------------------------------------- #
class Session:
    """The transcript the model sees, the steps the page shows, the plan,
    and the state of the run. Steps are what a reload renders; events are
    the same changes, numbered, for a page watching live."""

    def __init__(self, sid: str, title: str, root: str) -> None:
        self.id, self.title, self.root = sid, title, root
        self.created = time.time()
        self.transcript: list[dict] = []
        self.steps: list[dict] = []
        self.plan: list[dict] = []
        self.state = "idle"        # running, waiting, done, paused,
        self.note = ""             # stopped, error
        self.step = 0
        self.max_steps = MAX_STEPS
        self.auto_run = False
        self.backed_up: set[str] = set()
        self.last_html = ""
        self.seq = 0
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.stop = False
        self.pending: dict | None = None
        self.response = None        # the open stream, closed by Stop
        self.cost_cache: dict = {}  # token counts by text, across steps
        self.forgotten = False      # deleted: never saved again
        self.save_lock = threading.Lock()

    # ---- changes, applied here and sent to anyone watching ---- #
    def _emit(self, ev: dict) -> None:
        with self.cond:
            self.seq += 1
            ev["seq"] = self.seq
            self.events.append(ev)
            if len(self.events) > 4000:
                del self.events[:2000]
            self.cond.notify_all()

    def add(self, step: dict) -> int:
        self.steps.append(step)
        self._emit({"type": "add", "index": len(self.steps) - 1,
                    "step": step})
        return len(self.steps) - 1

    def patch(self, index: int, **fields) -> None:
        self.steps[index].update(fields)
        self._emit({"type": "patch", "index": index, "fields": fields})

    def delta(self, index: int, field: str, text: str) -> None:
        self.steps[index][field] = self.steps[index].get(field, "") + text
        self._emit({"type": "delta", "index": index, "field": field,
                    "text": text})

    def set_state(self, state: str, note: str = "") -> None:
        self.state, self.note = state, note
        self._emit({"type": "state", "state": state, "note": note,
                    "step": self.step, "max_steps": self.max_steps})

    def set_plan(self, items: list[dict]) -> None:
        self.plan = items
        self._emit({"type": "plan", "items": items})

    def events_since(self, seq: int, wait: float = 15.0) -> list[dict]:
        """Events after seq, waiting up to `wait` seconds for one."""
        with self.cond:
            if self.seq <= seq and self.state in ("running", "waiting"):
                self.cond.wait(wait)
            return [e for e in self.events if e["seq"] > seq]

    def view(self) -> dict:
        return {"id": self.id, "title": self.title, "root": self.root,
                "created": self.created, "steps": self.steps,
                "plan": self.plan, "state": self.state, "note": self.note,
                "step": self.step, "max_steps": self.max_steps,
                "auto_run": self.auto_run, "last_html": self.last_html,
                "seq": self.seq,
                "pending": ({"id": self.pending["id"],
                             "command": self.pending["command"]}
                            if self.pending else None)}

    def to_json(self) -> dict:
        return {**self.view(), "transcript": self.transcript,
                "backed_up": sorted(self.backed_up)}

    @classmethod
    def from_json(cls, d: dict) -> "Session":
        s = cls(d["id"], d.get("title", "Agent"), d.get("root", ""))
        s.created = d.get("created", s.created)
        s.transcript = d.get("transcript", [])
        s.steps = d.get("steps", [])
        s.plan = d.get("plan", [])
        s.step = int(d.get("step") or 0)
        s.last_html = d.get("last_html", "")
        s.backed_up = set(d.get("backed_up", []))
        s.state = d.get("state", "done")
        s.note = d.get("note", "")
        # A run cannot outlive the app: one that was going when it closed
        # is paused, and carries on when told to.
        if s.state in ("running", "waiting"):
            s.state, s.note = "paused", "The app closed during this run."
            for st in s.steps:
                if st.get("kind") == "tool" and st.get("status") in (
                        "running", "waiting"):
                    st["status"] = "stopped"
        return s


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
class Agent:
    """Runs sessions against a host. The host is the server's side of
    things — see server.AgentHost — so everything here is testable against
    a fake one."""

    def __init__(self, host, save) -> None:
        self.host, self.save = host, save
        self.lock = threading.Lock()
        self.running: Session | None = None

    def start(self, session: Session, instruction: str,
              auto_run: bool = False) -> None:
        with self.lock:
            if self.running and self.running.state in ("running",
                                                       "waiting"):
                raise RuntimeError("The agent is already working on "
                                   f"“{self.running.title}”. Stop it first.")
            session.stop = False
            session.auto_run = auto_run
            session.step = 0
            session.set_state("running")
            self.running = session
        threading.Thread(target=self._run, args=(session, instruction),
                         daemon=True).start()

    def stop(self, session: Session) -> None:
        session.stop = True
        if session.pending:
            session.pending["allow"] = False
            session.pending["event"].set()
        resp = session.response
        if resp is not None:
            try:
                resp.close()        # unblocks a read still waiting on it
            except Exception:  # noqa: BLE001
                pass

    def decide(self, session: Session, call_id: str, allow: bool,
               for_run: bool = False) -> None:
        p = session.pending
        if not p or p["id"] != call_id:
            raise RuntimeError("Nothing is waiting for that answer.")
        if allow and for_run:
            session.auto_run = True
        p["allow"] = bool(allow)
        p["event"].set()

    # ---- the loop ---- #
    def _run(self, s: Session, instruction: str) -> None:
        try:
            s.add({"kind": "user", "text": instruction, "at": time.time()})
            s.transcript.append({"role": "user", "content": instruction})
            self._loop(s)
        except Exception as exc:  # noqa: BLE001
            s.add({"kind": "note", "text": f"The run failed: {exc}"})
            s.set_state("error", str(exc))
        finally:
            s.pending = None
            s.response = None
            try:
                self.save(s)
            except Exception:  # noqa: BLE001
                pass

    def _loop(self, s: Session) -> None:
        while not s.stop:
            if s.step >= s.max_steps:
                s.add({"kind": "note", "text":
                       f"Paused after {s.max_steps} steps. Send “keep "
                       "going” to carry on, or a new instruction."})
                s.set_state("paused")
                return
            s.step += 1
            s.set_state("running")
            text, calls, reason = self._model_step(s)
            if s.stop:
                break
            if not calls:
                if reason == "length":
                    s.transcript.append({"role": "user", "content": CUT_OFF})
                    continue
                s.set_state("done")
                return
            for call in calls:
                if s.stop:
                    s.transcript.append(_tool_msg(call["id"],
                                                  "Stopped by the person."))
                    continue
                s.transcript.append(_tool_msg(call["id"],
                                              self._call(s, call, reason)))
            self.save(s)
        s.add({"kind": "note", "text": "Stopped."})
        s.set_state("stopped")

    def _model_step(self, s: Session) -> tuple[str, list[dict], str]:
        messages, max_tokens, trimmed = fit(self.host, s)
        idx = s.add({"kind": "assistant", "text": "", "reasoning": "",
                     "trimmed": trimmed})
        params = {**self.host.sampling(), "max_tokens": max_tokens,
                  "tools": TOOLS}
        calls: dict[int, dict] = {}
        reason, shown = "", {}

        def hold(resp):
            s.response = resp
        stream = self.host.stream(messages, params, hold)
        try:
            for ev in stream:
                if s.stop:
                    break
                if "think" in ev:
                    s.delta(idx, "reasoning", ev["think"])
                elif "delta" in ev:
                    s.delta(idx, "text", ev["delta"])
                elif "tool" in ev:
                    t = ev["tool"]
                    c = calls.setdefault(t.get("index", 0),
                                         {"id": "", "name": "",
                                          "arguments": ""})
                    c["id"] = t.get("id") or c["id"]
                    c["name"] = t.get("name") or c["name"]
                    c["arguments"] += t.get("args") or ""
                    # a big write streams for a while: say so, now and then
                    n = len(c["arguments"])
                    if n - shown.get(t.get("index", 0), -1) >= 2000:
                        shown[t.get("index", 0)] = n
                        s.patch(idx, writing={"name": c["name"],
                                              "chars": n})
                elif "stop" in ev:
                    reason = ev["stop"]
        except Exception as exc:  # noqa: BLE001
            if not s.stop:
                raise RuntimeError(f"the model server: {exc}") from exc
        finally:
            s.response = None
            stream.close()          # ends the request if Stop broke off
        text = s.steps[idx]["text"]
        found = [{"id": c["id"] or "call_" + uuid.uuid4().hex[:10],
                  "name": c["name"], "arguments": c["arguments"]}
                 for _, c in sorted(calls.items()) if c["name"]]
        if not found and "<tool_call>" in text and not s.stop:
            found, text = calls_in_text(text)
            s.patch(idx, text=text)
        if s.stop:
            # A call cut off by Stop never runs, and must not stay in the
            # history unanswered: the next instruction would send the model
            # a call with no result after it.
            found = []
        if s.steps[idx].get("writing"):
            s.patch(idx, writing=None)
        # Reasoning is shown, never sent back (the thinking models ask it).
        msg = {"role": "assistant", "content": text}
        if found:
            msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"],
                                               "arguments": c["arguments"]}}
                                 for c in found]
        s.transcript.append(msg)
        return text, found, reason

    # ---- one tool call ---- #
    def _call(self, s: Session, call: dict, reason: str) -> str:
        name = call["name"]
        try:
            args = json.loads(call["arguments"] or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except ValueError as exc:
            step = s.add({"kind": "tool", "id": call["id"], "name": name,
                          "args": {"raw": call["arguments"][:2000]},
                          "status": "error"})
            why = (CUT_OFF if reason == "length" else
                   f"The arguments were not valid JSON ({exc}). Send the "
                   "call again with valid JSON.")
            s.patch(step, result=why, ok=False)
            return why
        step = s.add({"kind": "tool", "id": call["id"], "name": name,
                      "args": args, "status": "running"})
        if name not in TOOL_NAMES:
            out, ok = (f"There is no tool called {name}. The tools are: "
                       + ", ".join(sorted(TOOL_NAMES)) + "."), False
        else:
            try:
                out, ok = getattr(self, "t_" + name)(s, args, step)
            except Exception as exc:  # noqa: BLE001
                out, ok = f"Error: {exc}", False
        status = s.steps[step].get("status")
        s.patch(step, result=out if len(out) <= 20_000 else
                out[:20_000] + "\n…", ok=ok,
                status="denied" if status == "denied" else
                ("done" if ok else "error"))
        return out

    # ---- the tools ---- #
    def t_list_files(self, s, a, step):
        base = self.host.resolve(a.get("path") or ".")
        if not base.is_dir():
            return f"No such folder: {a.get('path')}", False
        root = Path(self.host.root()).resolve()
        rows, more = [], False
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in SKIP_DIRS
                                 and not d.startswith("."))
            for n in sorted(filenames):
                if n.endswith(".bak"):
                    continue
                if len(rows) >= LIST_CAP:
                    more = True
                    break
                f = Path(dirpath) / n
                try:
                    size = f.stat().st_size
                except OSError:
                    continue
                rows.append(f"{f.relative_to(root).as_posix()}  "
                            f"({size:,} bytes)")
            if more:
                break
        if not rows:
            return "The folder is empty.", True
        return "\n".join(rows) + ("\n… (more files not listed)"
                                  if more else ""), True

    def t_read_file(self, s, a, step):
        f = self.host.resolve(a["path"])
        if not f.is_file():
            return f"No such file: {a['path']}", False
        data = f.read_bytes()
        if b"\0" in data[:8192]:
            return f"{a['path']} is a binary file.", False
        lines = data.decode("utf-8", errors="replace").splitlines()
        start = max(1, int(a.get("start_line") or 1))
        # About a quarter of the window at ~4 characters a token: one read
        # of a long file must not crowd out the rest of the task.
        cap = min(READ_CAP, max(4000, self.host.n_ctx()))
        out, used, end = [], 0, start - 1
        for ln in lines[start - 1:]:
            if used + len(ln) + 1 > cap and out:
                break
            out.append(ln)
            used += len(ln) + 1
            end += 1
        head = f"{a['path']} — lines {start}-{end} of {len(lines)}"
        if end < len(lines):
            head += f" (read on with start_line={end + 1})"
        return head + "\n" + "\n".join(out), True

    def t_search(self, s, a, step):
        pat = str(a.get("pattern") or "")
        if not pat:
            return "Give a pattern to search for.", False
        base = self.host.resolve(a.get("path") or ".")
        root = Path(self.host.root()).resolve()
        low, hits = pat.lower(), []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                           and not d.startswith(".")]
            for n in sorted(filenames):
                f = Path(dirpath) / n
                try:
                    if f.stat().st_size > 1_000_000:
                        continue
                    data = f.read_bytes()
                except OSError:
                    continue
                if b"\0" in data[:8192]:
                    continue
                for i, ln in enumerate(data.decode("utf-8", "replace")
                                       .splitlines(), 1):
                    if low in ln.lower():
                        hits.append(f"{f.relative_to(root).as_posix()}:"
                                    f"{i}: {ln.strip()[:200]}")
                        if len(hits) >= SEARCH_CAP:
                            return "\n".join(hits) + "\n… (more)", True
        return "\n".join(hits) if hits else f"No match for {pat!r}.", True

    def _write(self, s, rel: str, text: str) -> dict:
        # The first change to a file this session keeps the person's own
        # version as .bak; later changes leave that .bak alone.
        out = self.host.write(rel, text, keep_backup=rel in s.backed_up)
        s.backed_up.add(rel)
        if rel.lower().endswith((".html", ".htm")):
            s.last_html = rel
        return out

    def t_write_file(self, s, a, step):
        rel, text = a["path"], str(a.get("content") or "")
        out = self._write(s, rel, text)
        return (f"{'Replaced' if out.get('existed') else 'Created'} {rel} "
                f"({text.count(chr(10)) + 1} lines).", True)

    def t_append_file(self, s, a, step):
        rel, text = a["path"], str(a.get("content") or "")
        f = self.host.resolve(rel)
        old = f.read_text(encoding="utf-8", errors="replace") \
            if f.is_file() else ""
        self._write(s, rel, old + text)
        return (f"Appended {text.count(chr(10)) + 1} lines to {rel} (now "
                f"{(old + text).count(chr(10)) + 1} lines).", True)

    def t_edit_file(self, s, a, step):
        rel = a["path"]
        f = self.host.resolve(rel)
        if not f.is_file():
            return f"No such file: {rel}. Create it with write_file.", False
        text = f.read_text(encoding="utf-8", errors="replace")
        old, new = str(a.get("old_text") or ""), str(a.get("new_text", ""))
        if not old:
            return "old_text is empty.", False
        count = text.count(old)
        if count == 0 and "\r\n" in text:
            text = text.replace("\r\n", "\n")
            count = text.count(old)
        if count == 0:
            return (f"old_text was not found in {rel}. Read the file again "
                    "and copy the exact text, spaces included."), False
        every = _flag(a.get("replace_all"))
        if count > 1 and not every:
            return (f"old_text appears {count} times in {rel}. Include more "
                    "of the surrounding lines so it is unique, or set "
                    "replace_all."), False
        self._write(s, rel, text.replace(old, new) if every
                    else text.replace(old, new, 1))
        return (f"Edited {rel}: replaced {count if every else 1} "
                "occurrence(s)."), True

    def t_run_command(self, s, a, step):
        cmd = str(a.get("command") or "").strip()
        if not cmd:
            return "Give a command to run.", False
        timeout = max(5, min(300, int(a.get("timeout") or 60)))
        if not s.auto_run:
            ev = threading.Event()
            s.pending = {"id": s.steps[step]["id"], "command": cmd,
                         "event": ev, "allow": None}
            s.patch(step, status="waiting")
            s.set_state("waiting")
            self.save(s)
            while not ev.wait(0.5):
                if s.stop:
                    break
            allow = s.pending and s.pending["allow"]
            s.pending = None
            s.set_state("running")
            if not allow:
                s.patch(step, status="denied")
                return ("The person declined to run this command. Do not "
                        "run it again; carry on another way, or ask them in "
                        "your reply."), False
            s.patch(step, status="running")
        r = self.host.run(cmd, timeout)
        out = r.get("output") or "(no output)"
        if len(out) > OUTPUT_CAP:
            out = "…" + out[-OUTPUT_CAP:]
        return f"exit code {r.get('exit')}\n{out}", r.get("exit") == 0

    def t_update_plan(self, s, a, step):
        items, given = [], a.get("steps") or []
        if isinstance(given, str):      # XML-style calls send it as text
            try:
                given = json.loads(given)
            except ValueError:
                given = []
        for it in given if isinstance(given, list) else []:
            if isinstance(it, dict) and str(it.get("step") or "").strip():
                st = it.get("status")
                items.append({"step": str(it["step"])[:200],
                              "status": st if st in (
                                  "pending", "in_progress", "done")
                              else "pending"})
        s.set_plan(items[:30])
        return "Plan updated.", True


# --------------------------------------------------------------------------- #
# fitting a long run into the window
# --------------------------------------------------------------------------- #
def system_prompt(host) -> str:
    return SYSTEM.format(root=host.root(), system_info=host.system_info(),
                         listing=host.listing())


def _cost(host, msg: dict, cache: dict) -> int:
    text = (msg.get("content") or "") + "".join(
        c["function"]["name"] + c["function"]["arguments"]
        for c in msg.get("tool_calls") or [])
    key = hash(text)
    if key not in cache:
        cache[key] = host.count(text) + 8
    return cache[key]


def _slim(msg: dict) -> dict:
    """An old message cut to what the rest of the run needs from it: that a
    tool ran, not every byte it printed; that a file was written, not its
    whole content again. The model can read a file again when it must."""
    if msg["role"] == "tool":
        body = msg["content"]
        if len(body) <= 400:
            return msg
        return {**msg, "content": body[:200] + f"\n… [{len(body):,} "
                "characters of this result dropped to save room; run the "
                "tool again if you need them]"}
    if msg["role"] == "assistant" and msg.get("tool_calls"):
        calls = []
        for c in msg["tool_calls"]:
            fn = c["function"]
            try:
                args = json.loads(fn["arguments"])
            except ValueError:
                args = None
            if isinstance(args, dict) and len(str(args.get("content", ""))
                                              ) > 400:
                args = {**args, "content": f"[{len(args['content']):,} "
                        "characters written — read the file for them]"}
                fn = {**fn, "arguments": json.dumps(args)}
            calls.append({**c, "function": fn})
        return {**msg, "tool_calls": calls}
    return msg


def _alternate(msgs: list[dict]) -> list[dict]:
    """Consecutive user messages joined: a task whose steps were all dropped
    leaves its instruction beside the next one, and strict chat templates
    (Gemma, Mistral) refuse two user turns in a row."""
    out: list[dict] = []
    for m in msgs:
        if out and m["role"] == "user" and out[-1]["role"] == "user":
            out[-1] = {**out[-1], "content": out[-1]["content"] + "\n\n" +
                       m["content"]}
        else:
            out.append(m)
    return out


def fit(host, s: Session) -> tuple[list[dict], int, bool]:
    """The run so far, inside the context window. Old tool output and old
    file contents are cut to a line first; if that is not enough, the
    oldest steps go, whole — an assistant message leaves with the tool
    results that answer it, and the person's instructions always stay.
    Only when even the newest steps are too big are they cut too."""
    n_ctx = host.n_ctx()
    system = system_prompt(host)
    cache = s.cost_cache
    limit = n_ctx - REPLY_FLOOR - 256
    msgs = [dict(m) for m in s.transcript]
    costs = [_cost(host, m, cache) for m in msgs]
    total = host.count(system) + 256 + sum(costs)
    trimmed = False

    def slim(upto: int) -> None:
        nonlocal total, trimmed
        for i in range(1, upto):
            if total <= limit:
                return
            small = _slim(msgs[i])
            if small is not msgs[i]:
                msgs[i] = small
                total -= costs[i]
                costs[i] = _cost(host, small, cache)
                total += costs[i]
                trimmed = True

    slim(max(1, len(msgs) - KEEP_RECENT))
    while total > limit:
        j = next((i for i in range(1, len(msgs) - KEEP_RECENT)
                  if msgs[i]["role"] == "assistant"), None)
        if j is None:
            break
        k = j + 1
        while k < len(msgs) and msgs[k]["role"] == "tool":
            k += 1
        if k > len(msgs) - KEEP_RECENT:
            break
        total -= sum(costs[j:k])
        del msgs[j:k]
        del costs[j:k]
        trimmed = True
    slim(max(1, len(msgs) - 1))       # the newest steps, as a last resort
    if trimmed and msgs:
        msgs[0] = {**msgs[0], "content": msgs[0]["content"] + (
            "\n\n(Earlier steps of this task were shortened to fit the "
            "context window. Read files again rather than relying on "
            "memory of them.)")}
        total += 40
    return ([{"role": "system", "content": system}] + _alternate(msgs),
            max(512, n_ctx - total - 64), trimmed)
