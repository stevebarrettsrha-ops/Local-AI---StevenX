"""Tests for the coding agent: runs driven by a scripted model against a real
temporary folder, the approval gate, fitting long runs into the window, the
endpoints, and the live-preview server."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import agent
import server


def call(name: str, index: int = 0, cid: str = "", **args) -> dict:
    return {"tool": {"index": index, "id": cid or f"c_{name}_{index}",
                     "name": name, "args": json.dumps(args)}}


def say(text: str, reason: str = "stop") -> list[dict]:
    return [{"delta": text}, {"stop": reason}]


class FakeHost:
    """The server's side, with a scripted model: each step of the script is
    the list of events one model turn streams."""

    def __init__(self, root: Path, script: list, ctx: int = 32768) -> None:
        self._root, self.script, self.ctx = root, list(script), ctx
        self.runs: list[str] = []
        self.seen: list[list[dict]] = []

    def root(self):
        return str(self._root)

    def resolve(self, rel):
        p = (self._root / rel).resolve()
        if not p.is_relative_to(self._root.resolve()):
            raise RuntimeError("That path is outside the open folder.")
        return p

    def write(self, rel, content, keep_backup=False):
        p = self.resolve(rel)
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"existed": existed}

    def run(self, cmd, timeout):
        self.runs.append(cmd)
        return {"exit": 0, "output": f"ran {cmd}"}

    def stream(self, messages, params, hook):
        self.seen.append(messages)
        self.params = params
        events = self.script.pop(0) if self.script else say("Done.")
        yield from events

    def count(self, text):
        return len(text) // 4

    def n_ctx(self):
        return self.ctx

    def sampling(self):
        return {"temperature": 0.7}

    def system_info(self):
        return "Test OS"

    def listing(self):
        return "(empty — a new project)"


def wait_for(cond, timeout: float = 10.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


class AgentRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.saved: list = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_agent(self, script, instruction="Build a snake game",
                  auto_run=False, ctx=32768, session=None):
        host = FakeHost(self.root, script, ctx)
        ag = agent.Agent(host, lambda s: self.saved.append(s.state))
        s = session or agent.Session("abc123def456", "t", str(self.root))
        ag.start(s, instruction, auto_run=auto_run)
        return ag, s, host

    def test_a_task_is_built_run_and_summarised(self) -> None:
        html = "<!doctype html><canvas id=c></canvas><script src=game.js>" \
               "</script>"
        ag, s, host = self.run_agent([
            [call("update_plan", steps=[{"step": "Write the page",
                                         "status": "in_progress"}]),
             {"stop": "tool_calls"}],
            [call("write_file", path="index.html", content=html),
             call("write_file", 1, path="game.js", content="let x = 1;\n"),
             {"stop": "tool_calls"}],
            [call("run_command", command="node --check game.js"),
             {"stop": "tool_calls"}],
            say("Built a snake game: open index.html."),
        ], auto_run=True)
        wait_for(lambda: s.state == "done")
        self.assertEqual((self.root / "index.html").read_text(), html)
        self.assertTrue((self.root / "game.js").is_file())
        self.assertEqual(host.runs, ["node --check game.js"])
        self.assertEqual(s.plan[0]["step"], "Write the page")
        self.assertEqual(s.last_html, "index.html")
        kinds = [st["kind"] for st in s.steps]
        self.assertEqual(kinds[0], "user")
        self.assertEqual(kinds.count("tool"), 4)
        self.assertEqual(s.steps[-1]["text"],
                         "Built a snake game: open index.html.")
        # every tool call is answered, in order, in the transcript
        roles = [m["role"] for m in s.transcript]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant",
                                 "tool", "tool", "assistant", "tool",
                                 "assistant"])
        # the tools went to the model, and its own system prompt first
        self.assertEqual(host.seen[0][0]["role"], "system")
        self.assertIn("run_command",
                      [t["function"]["name"] for t in host.params["tools"]])

    def test_commands_wait_for_the_person(self) -> None:
        ag, s, host = self.run_agent([
            [call("run_command", cid="r1", command="npm test"),
             {"stop": "tool_calls"}],
            [call("run_command", cid="r2", command="rm -rf /"),
             {"stop": "tool_calls"}],
            say("Stopped there."),
        ])
        wait_for(lambda: s.state == "waiting")
        self.assertEqual(host.runs, [])
        self.assertEqual(s.view()["pending"]["command"], "npm test")
        ag.decide(s, "r1", True)
        wait_for(lambda: s.pending and s.pending["id"] == "r2")
        self.assertEqual(host.runs, ["npm test"])
        ag.decide(s, "r2", False)
        wait_for(lambda: s.state == "done")
        self.assertEqual(host.runs, ["npm test"])
        denied = [st for st in s.steps if st.get("id") == "r2"][0]
        self.assertEqual(denied["status"], "denied")
        self.assertIn("declined", s.transcript[-2]["content"])

    def test_allow_for_the_run_stops_asking(self) -> None:
        ag, s, host = self.run_agent([
            [call("run_command", cid="a", command="python -V"),
             {"stop": "tool_calls"}],
            [call("run_command", cid="b", command="python app.py"),
             {"stop": "tool_calls"}],
            say("Both ran."),
        ])
        wait_for(lambda: s.state == "waiting")
        ag.decide(s, "a", True, for_run=True)
        wait_for(lambda: s.state == "done")
        self.assertEqual(host.runs, ["python -V", "python app.py"])

    def test_stop_while_waiting_ends_the_run(self) -> None:
        ag, s, host = self.run_agent([
            [call("run_command", command="make"), {"stop": "tool_calls"}]])
        wait_for(lambda: s.state == "waiting")
        ag.stop(s)
        wait_for(lambda: s.state == "stopped")
        self.assertEqual(host.runs, [])

    def test_stop_mid_call_leaves_a_clean_history(self) -> None:
        gate = threading.Event()

        def slow():
            yield {"tool": {"index": 0, "id": "w", "name": "write_file",
                            "args": '{"path": "a.js", "cont'}}
            gate.wait(5)
            yield {"tool": {"index": 0, "args": 'ent": "x"}'}}
            yield {"stop": "tool_calls"}
        ag, s, host = self.run_agent([slow()])
        wait_for(lambda: len(s.steps) >= 2)
        ag.stop(s)
        gate.set()
        wait_for(lambda: s.state == "stopped")
        self.assertFalse((self.root / "a.js").exists())
        self.assertNotIn("tool_calls", s.transcript[-1])
        # carrying on sends a well-formed history
        ag.start(s, "carry on")
        wait_for(lambda: s.state == "done")
        roles = [m["role"] for m in host.seen[-1]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_edits_must_be_exact_and_unique(self) -> None:
        (self.root / "a.py").write_text("x = 1\ny = 1\nx = 1\n")
        ag, s, host = self.run_agent([
            [call("edit_file", path="a.py", old_text="x = 1",
                  new_text="x = 2"), {"stop": "tool_calls"}],
            [call("edit_file", path="a.py", old_text="nope",
                  new_text="z"), {"stop": "tool_calls"}],
            [call("edit_file", path="a.py", old_text="y = 1",
                  new_text="y = 3"), {"stop": "tool_calls"}],
            say("Edited."),
        ])
        wait_for(lambda: s.state == "done")
        results = [st["result"] for st in s.steps if st["kind"] == "tool"]
        self.assertIn("appears 2 times", results[0])
        self.assertIn("was not found", results[1])
        self.assertEqual((self.root / "a.py").read_text(),
                         "x = 1\ny = 3\nx = 1\n")

    def test_files_are_read_listed_and_searched(self) -> None:
        (self.root / "src").mkdir()
        (self.root / "src" / "main.py").write_text(
            "\n".join(f"line {i}" for i in range(1, 6)) + "\n")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "junk.js").write_text("x")
        ag, s, host = self.run_agent([
            [call("list_files", path="."), call("read_file", 1,
                                                path="src/main.py",
                                                start_line=3),
             call("search", 2, pattern="LINE 4"), {"stop": "tool_calls"}],
            say("Read it."),
        ])
        wait_for(lambda: s.state == "done")
        listed, read, found = [st["result"] for st in s.steps
                               if st["kind"] == "tool"]
        self.assertIn("src/main.py", listed)
        self.assertNotIn("node_modules", listed)
        self.assertTrue(read.startswith("src/main.py — lines 3-5 of 5"))
        self.assertIn("src/main.py:4: line 4", found)

    def test_nothing_leaves_the_folder(self) -> None:
        ag, s, host = self.run_agent([
            [call("write_file", path="../escape.txt", content="x"),
             {"stop": "tool_calls"}],
            say("Could not."),
        ])
        wait_for(lambda: s.state == "done")
        self.assertFalse((self.root.parent / "escape.txt").exists())
        self.assertFalse(s.steps[2]["ok"])

    def test_a_cut_off_call_is_explained_not_run(self) -> None:
        ag, s, host = self.run_agent([
            [{"tool": {"index": 0, "id": "w", "name": "write_file",
                       "args": '{"path": "big.js", "content": "funct'}},
             {"stop": "length"}],
            say("Wrote it in parts."),
        ])
        wait_for(lambda: s.state == "done")
        self.assertFalse((self.root / "big.js").exists())
        self.assertIn("cut off", s.transcript[2]["content"])

    def test_calls_written_as_text_are_still_run(self) -> None:
        xml = ("<tool_call>\n<function=write_file>\n<parameter=path>\n"
               "a.txt\n</parameter>\n<parameter=content>\nhello\n"
               "</parameter>\n</function>\n</tool_call>")
        hermes = ('<tool_call>{"name": "write_file", "arguments": '
                  '{"path": "b.txt", "content": "hi"}}</tool_call>')
        ag, s, host = self.run_agent([say("Writing.\n" + xml, "stop"),
                                      say(hermes, "stop"), say("Done.")])
        wait_for(lambda: s.state == "done")
        self.assertEqual((self.root / "a.txt").read_text(), "hello")
        self.assertEqual((self.root / "b.txt").read_text(), "hi")
        self.assertEqual(s.steps[1]["text"], "Writing.")

    def test_a_long_task_pauses_and_carries_on(self) -> None:
        loop = [[call("list_files"), {"stop": "tool_calls"}]] * 3
        host = FakeHost(self.root, loop + [say("Finished.")])
        ag = agent.Agent(host, lambda s: None)
        s = agent.Session("abc123def456", "t", str(self.root))
        s.max_steps = 2
        ag.start(s, "Go")
        wait_for(lambda: s.state == "paused")
        self.assertIn("Paused after 2 steps", s.steps[-1]["text"])
        ag.start(s, "keep going")
        wait_for(lambda: s.state == "done")
        self.assertEqual(s.steps[-1]["text"], "Finished.")

    def test_one_run_at_a_time(self) -> None:
        ag, s, host = self.run_agent([
            [call("run_command", command="x"), {"stop": "tool_calls"}]])
        wait_for(lambda: s.state == "waiting")
        other = agent.Session("fedcba654321", "u", str(self.root))
        with self.assertRaisesRegex(RuntimeError, "already working"):
            ag.start(other, "second task")
        ag.stop(s)

    def test_a_saved_run_comes_back_paused(self) -> None:
        s = agent.Session("abc123def456", "t", str(self.root))
        s.state = "running"
        s.steps = [{"kind": "tool", "status": "waiting"}]
        back = agent.Session.from_json(json.loads(json.dumps(s.to_json())))
        self.assertEqual(back.state, "paused")
        self.assertEqual(back.steps[0]["status"], "stopped")


class StreamToolTests(unittest.TestCase):
    def test_tool_call_pieces_are_passed_on_as_they_stream(self) -> None:
        import io
        import requests
        import engine
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "write_file", "arguments": ""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": '{"path": "a.js",'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": ' "content": "é"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}]
        body = "".join("data: " + json.dumps(c, ensure_ascii=False) + "\n\n"
                       for c in chunks) + "data: [DONE]\n\n"
        resp = requests.models.Response()
        resp.status_code = 200
        resp.headers["Content-Type"] = "text/event-stream"
        resp.raw = io.BytesIO(body.encode("utf-8"))
        cm = mock.MagicMock()
        cm.__enter__.return_value = resp
        held = []
        with mock.patch.object(engine.requests, "post",
                               return_value=cm) as post:
            evs = list(engine.Server().chat_stream(
                [], {"tools": agent.TOOLS}, held.append))
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["tools"], agent.TOOLS)
        self.assertEqual(sent["tool_choice"], "auto")
        self.assertNotIn("parallel_tool_calls", sent)
        self.assertEqual(held, [resp])
        args = "".join(e["tool"]["args"] for e in evs if "tool" in e)
        self.assertEqual(json.loads(args), {"path": "a.js", "content": "é"})
        self.assertEqual(evs[0]["tool"]["name"], "write_file")
        self.assertEqual(evs[-1]["stop"], "tool_calls")


class FitTests(unittest.TestCase):
    def host(self, ctx):
        return FakeHost(Path("."), [], ctx)

    def session(self, n_steps: int, big: int = 8000) -> agent.Session:
        s = agent.Session("abc123def456", "t", ".")
        s.transcript.append({"role": "user", "content": "Build a game"})
        for i in range(n_steps):
            args = json.dumps({"path": f"f{i}.js", "content": "x" * big})
            s.transcript.append({"role": "assistant", "content": "",
                                 "tool_calls": [{"id": f"c{i}",
                                                 "type": "function",
                                                 "function": {
                                                     "name": "write_file",
                                                     "arguments": args}}]})
            s.transcript.append({"role": "tool", "tool_call_id": f"c{i}",
                                 "content": "y" * big})
        s.transcript.append({"role": "user", "content": "now add sound"})
        return s

    def test_a_short_run_goes_as_it_is(self) -> None:
        s = self.session(2, big=100)
        msgs, room, trimmed = agent.fit(self.host(32768), s)
        self.assertFalse(trimmed)
        self.assertEqual(len(msgs), len(s.transcript) + 1)

    def test_a_long_run_is_cut_to_fit_and_stays_well_formed(self) -> None:
        s = self.session(30)
        msgs, room, trimmed = agent.fit(self.host(16384), s)
        self.assertTrue(trimmed)
        total = sum(len(json.dumps(m)) for m in msgs) // 4
        self.assertLess(total, 16384)
        self.assertGreaterEqual(room, 512)
        # every tool result still follows the call it answers
        open_calls = set()
        for m in msgs:
            if m["role"] == "assistant":
                open_calls = {c["id"] for c in m.get("tool_calls") or []}
            elif m["role"] == "tool":
                self.assertIn(m["tool_call_id"], open_calls)
        # the task and the newest instruction are both still there
        self.assertIn("Build a game", msgs[1]["content"])
        self.assertEqual(msgs[-1]["content"], "now add sound")
        roles = [m["role"] for m in msgs]
        self.assertFalse(any(a == b == "user"
                             for a, b in zip(roles, roles[1:])))
        # the full transcript itself is untouched
        self.assertEqual(len(s.transcript), 62)


class EndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.ws = base / "project"
        self.ws.mkdir()
        self.patches = [
            mock.patch.object(server, "AGENTS_DIR", base / "agents"),
            mock.patch.object(server, "CHATS_PATH", base / "chats.json"),
            mock.patch.dict(server.cfg, {"workspace": str(self.ws)}),
            mock.patch.object(server, "save_config"),
            mock.patch.object(server.server, "ready", return_value=True),
            mock.patch.object(server.server, "n_ctx", return_value=32768),
            mock.patch.object(server.server, "count_tokens",
                              side_effect=lambda t: len(t) // 4),
            mock.patch.object(server, "machine_summary",
                              return_value="Test OS"),
            mock.patch.dict(server.agent_sessions, clear=True),
        ]
        for p in self.patches:
            p.start()
        server.AGENT.running = None
        self.client = server.app.test_client()

    def tearDown(self) -> None:
        if server.AGENT.running:
            server.AGENT.stop(server.AGENT.running)
            wait_for(lambda: server.AGENT.running.state not in (
                "running", "waiting"))
        server.AGENT.running = None
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def start(self, script, **body):
        def fake(messages, params, hook=None):
            yield from (script.pop(0) if script else say("Done."))
        self.fake = mock.patch.object(server.server, "chat_stream",
                                      side_effect=fake)
        self.fake.start()
        self.addCleanup(self.fake.stop)
        return self.client.post("/api/agent", json={
            "instruction": "Build a page", **body})

    def test_a_task_runs_in_the_open_folder_and_is_listed(self) -> None:
        r = self.start([[call("write_file", path="index.html",
                              content="<h1>hi</h1>"),
                         {"stop": "tool_calls"}], say("Made index.html.")])
        self.assertEqual(r.status_code, 200)
        sid = r.get_json()["session"]["id"]
        sess = server.agent_session(sid)
        wait_for(lambda: sess.state == "done")
        self.assertEqual((self.ws / "index.html").read_text(), "<h1>hi</h1>")
        self.assertEqual(server.get_chat(sid)["mode"], "agent")
        view = self.client.get(f"/api/agent/{sid}").get_json()
        self.assertEqual(view["state"], "done")
        raw = self.client.get(f"/api/agent/{sid}/events?since=0")\
            .get_data(as_text=True)
        types = [json.loads(c[5:])["type"] for c in raw.split("\n\n")
                 if c.startswith("data:")]
        self.assertIn("add", types)
        self.assertEqual(types[-1], "end")
        # saved: a restart finds it again
        self.assertTrue((server.AGENTS_DIR / f"{sid}.json").is_file())

    def test_approval_through_the_api(self) -> None:
        with mock.patch.object(server, "ws_run",
                               return_value={"exit": 0, "output": "ok",
                                             "seconds": 0.1}) as run:
            r = self.start([[call("run_command", cid="k", command="npm t"),
                             {"stop": "tool_calls"}], say("Ran.")])
            sid = r.get_json()["session"]["id"]
            sess = server.agent_session(sid)
            wait_for(lambda: sess.state == "waiting")
            self.assertEqual(self.client.post(
                f"/api/agent/{sid}/approve",
                json={"id": "k", "allow": True}).status_code, 200)
            wait_for(lambda: sess.state == "done")
        run.assert_called_once_with("npm t", 60)

    def test_it_needs_a_folder_and_a_model(self) -> None:
        with mock.patch.dict(server.cfg, {"workspace": ""}):
            self.assertEqual(self.client.post("/api/agent", json={
                "instruction": "x"}).status_code, 400)
        with mock.patch.object(server.server, "ready", return_value=False):
            self.assertEqual(self.client.post("/api/agent", json={
                "instruction": "x"}).status_code, 503)

    def test_deleting_the_task_deletes_its_steps(self) -> None:
        r = self.start([say("Nothing to do.")])
        sid = r.get_json()["session"]["id"]
        wait_for(lambda: server.agent_session(sid).state == "done")
        self.client.delete(f"/api/chats/{sid}")
        self.assertIsNone(server.agent_session(sid))
        self.assertFalse((server.AGENTS_DIR / f"{sid}.json").exists())

    def test_a_new_folder_can_be_created(self) -> None:
        new = Path(self.tmp.name) / "games" / "snake"
        r = self.client.post("/api/workspace", json={"path": str(new),
                                                     "create": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(new.is_dir())


class PreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        (self.ws / "game").mkdir()
        (self.ws / "game" / "index.html").write_text("<h1>game</h1>")
        (self.ws / "game" / "main.js").write_text("let x;")
        (self.ws / ".env").write_text("SECRET=1")
        self.patch = mock.patch.dict(server.cfg, {"workspace": str(self.ws)})
        self.patch.start()
        self.client = server.preview_app.test_client()
        self.base = f"/{server.PREVIEW_TOKEN}/"

    def tearDown(self) -> None:
        self.patch.stop()
        self.tmp.cleanup()

    def test_pages_and_scripts_are_served_as_a_site(self) -> None:
        r = self.client.get(self.base)
        self.assertEqual(r.status_code, 302)
        self.assertTrue(r.headers["Location"].endswith("/game/index.html"))
        with self.client.get(self.base + "game/main.js") as r:
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.mimetype.startswith("text/javascript"))
            self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")

    def test_it_gives_away_nothing_else(self) -> None:
        self.assertEqual(self.client.get("/wrong-token/game/index.html")
                         .status_code, 404)
        self.assertEqual(self.client.get(self.base + ".env").status_code,
                         404)
        self.assertEqual(self.client.get(self.base + "../x").status_code,
                         404)
        self.assertEqual(self.client.get(
            self.base + "game/index.html",
            headers={"Host": "evil.example:7807"}).status_code, 403)
        self.assertEqual(self.client.post(self.base + "game/index.html")
                         .status_code, 403)

    def test_the_api_refuses_a_preview_page(self) -> None:
        c = server.app.test_client()
        r = c.post("/api/workspace/run", json={"command": "echo x"},
                   headers={"Origin": f"http://127.0.0.1:"
                                      f"{server.PREVIEW_PORT}"})
        self.assertEqual(r.status_code, 403)


if __name__ == "__main__":
    unittest.main()
