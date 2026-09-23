"""Regression tests for the second full audit: request origin, workspace
safety, chat persistence, stream decoding, process and download handling,
build selection, model identification and the KV cache estimate."""

from __future__ import annotations

import http.server
import io
import json
import os
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import requests

import engine
import fit
import server

GB = 1024 ** 3


class RequestOriginTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = server.app.test_client()

    def test_only_this_machine_may_drive_the_api(self) -> None:
        get = lambda host: self.client.get(  # noqa: E731
            "/api/chats", headers={"Host": host}).status_code
        self.assertEqual(get("127.0.0.1:7806"), 200)
        self.assertEqual(get("localhost:7806"), 200)
        # a rebound domain resolving to 127.0.0.1 still names itself
        self.assertEqual(get("attacker.example:7806"), 403)

    def test_cross_origin_writes_are_refused(self) -> None:
        for origin in ("https://attacker.example", "null"):
            r = self.client.post("/api/memory", json={"text": "x"},
                                 headers={"Origin": origin})
            self.assertEqual(r.status_code, 403, origin)

    def test_memory_is_replaced_only_by_an_explicit_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(server, "MEMORY_PATH",
                                  Path(tmp) / "memory.md"):
            self.assertEqual(self.client.post(
                "/api/memory", data="", content_type="text/plain")
                .status_code, 400)
            self.assertEqual(self.client.post(
                "/api/memory", json={}).status_code, 400)
            self.assertEqual(self.client.post(
                "/api/memory", json={"text": "- likes tea"}).status_code, 200)

    def test_file_names_cannot_leave_their_folder(self) -> None:
        for bad in ("", "..", "a/b", "a\\b", "C:x", "C:\\x", "x..y"):
            self.assertFalse(server.plain_name(bad), bad)
        self.assertTrue(server.plain_name("img-1-2.png"))


class WorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [mock.patch.dict(server.cfg,
                                        {"workspace": str(self.root)}),
                        mock.patch.object(server, "save_config")]
        for p in self.patches:
            p.start()
        self.client = server.app.test_client()

    def tearDown(self) -> None:
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def write(self, path: str, content: str, **extra):
        return self.client.post("/api/workspace/file",
                                json={"path": path, "content": content,
                                      **extra})

    def test_absolute_unc_and_drive_paths_are_refused_unresolved(self) -> None:
        for bad in ("/etc/passwd", "\\\\host\\share\\x", "C:\\x", "C:x",
                    "../outside.txt"):
            with self.assertRaises(RuntimeError, msg=bad):
                server.ws_resolve(bad)
        self.assertEqual(server.ws_resolve("src/a.py"),
                         (self.root / "src" / "a.py").resolve())

    def test_git_internals_are_never_written(self) -> None:
        r = self.write(".git/hooks/pre-commit", "echo pwned")
        self.assertEqual(r.status_code, 400)
        self.assertFalse((self.root / ".git").exists())

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_a_bak_link_is_replaced_not_written_through(self) -> None:
        outside = Path(self.tmp.name + "-outside.txt")
        outside.write_text("untouched")
        try:
            (self.root / "app.py").write_text("print(1)\n")
            (self.root / "app.py.bak").symlink_to(outside)
            self.assertEqual(self.write("app.py", "print(2)\n").status_code,
                             200)
            self.assertEqual(outside.read_text(), "untouched")
            self.assertFalse((self.root / "app.py.bak").is_symlink())
            self.assertEqual((self.root / "app.py.bak").read_text(),
                             "print(1)\n")
        finally:
            outside.unlink(missing_ok=True)

    def test_run_and_fix_keeps_the_persons_own_version(self) -> None:
        (self.root / "app.py").write_text("mine\n")
        self.write("app.py", "round one\n")
        self.write("app.py", "round two\n", keep_backup=True)
        self.assertEqual((self.root / "app.py.bak").read_text(), "mine\n")
        self.write("app.py", "later apply\n")
        self.assertEqual((self.root / "app.py.bak").read_text(),
                         "round two\n")

    def test_line_endings_are_the_files_own(self) -> None:
        (self.root / "win.bat").write_bytes(b"echo a\r\necho b\r\n")
        self.write("win.bat", "echo c\necho d\n")
        self.write("run.sh", "#!/bin/sh\necho hi\n")
        self.assertEqual((self.root / "win.bat").read_bytes(),
                         b"echo c\r\necho d\r\n")
        self.assertEqual((self.root / "run.sh").read_bytes(),
                         b"#!/bin/sh\necho hi\n")

    def test_run_needs_its_command_named(self) -> None:
        server.cfg["run_command"] = "echo saved"
        self.assertEqual(self.client.post("/api/workspace/run", json={})
                         .status_code, 400)
        self.assertEqual(self.client.post(
            "/api/workspace/run", data="x", content_type="text/plain")
            .status_code, 400)
        r = self.client.post("/api/workspace/run",
                             json={"command": "echo hello"}).get_json()
        self.assertEqual((r["exit"], r["output"]), (0, "hello"))

    @unittest.skipIf(os.name == "nt", "process groups differ on Windows")
    def test_a_timeout_stops_everything_the_command_started(self) -> None:
        marker = f"sleep {int(time.time()) % 1000 + 1000}"
        with mock.patch.object(server, "RUN_TIMEOUT", 1):
            r = self.client.post("/api/workspace/run", json={
                "command": f"{marker} & echo started; {marker}"}).get_json()
        self.assertEqual(r["exit"], -1)
        self.assertIn("started", r["output"])
        time.sleep(0.3)
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True,
                            text=True).stdout
        left = [ln for ln in ps.splitlines() if ln.strip() == marker]
        self.assertEqual(left, [])


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(server, "MEMORY_PATH",
                                       Path(self.tmp.name) / "memory.md")
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()
        self.tmp.cleanup()

    def test_code_is_not_a_note_and_additions_are_reported(self) -> None:
        reply = ("remember: prefers tabs\n```python\n"
                 "print('remember: not this')\nremember: nor this\n```\n")
        self.assertEqual(server.remember_lines(reply), ["prefers tabs"])
        self.assertEqual(server.remember_lines(reply), [])   # no duplicate

    def test_a_full_file_takes_nothing_it_would_drop(self) -> None:
        server.write_memory("x" * (server.MEMORY_CAP - 5))
        self.assertEqual(server.remember_lines("remember: too long a note"),
                         [])

    def test_the_newest_notes_are_the_ones_sent(self) -> None:
        mem = "\n".join(f"- note {i:04d}" for i in range(1000))
        out = server.memory_excerpt(mem)
        self.assertIn("note 0999", out)
        self.assertNotIn("note 0000", out)
        self.assertLessEqual(len(out), server.MEMORY_INJECT + 40)


class ChatPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.path = base / "chats.json"
        self.patches = [
            mock.patch.object(server, "CHATS_PATH", self.path),
            mock.patch.object(server, "MEMORY_PATH", base / "memory.md"),
            mock.patch.object(server, "context_extra", return_value=""),
            mock.patch.object(server, "index_exchanges"),
            mock.patch.object(server.server, "ready", return_value=True),
            mock.patch.object(server.server, "n_ctx", return_value=4096),
            mock.patch.object(server.server, "count_tokens",
                              side_effect=lambda t: len(t) // 4),
        ]
        for p in self.patches:
            p.start()
        self.sent = []
        self.client = server.app.test_client()

    def tearDown(self) -> None:
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def chat(self, body, events):
        def fake(history, params):
            self.sent.append((history, params))
            for ev in events:
                if isinstance(ev, Exception):
                    raise ev
                yield ev
        with mock.patch.object(server.server, "chat_stream",
                               side_effect=fake):
            raw = self.client.post("/api/chat", json=body)\
                .get_data(as_text=True)
        return [json.loads(c[5:]) for c in raw.split("\n\n")
                if c.startswith("data:")]

    def test_a_reply_lands_after_its_question_not_over_newer_ones(self) -> None:
        self.chat({"message": "q0"}, [{"delta": "a0"}, {"stop": "stop"}])
        cid = server.read_chats()[0]["id"]
        # A second reply finishes while this one is still streaming.
        stale = server.get_chat(cid)
        stale["messages"].append({"role": "user", "content": "qA",
                                  "at": 1.0})
        server.put_chat(stale)
        stale["messages"].append({"role": "user", "content": "qB",
                                  "at": 2.0})
        server.put_chat(stale)
        server.merge_reply(cid, 2.0, False, "aB", "", "stop", True, 1, 1, "")
        server.merge_reply(cid, 1.0, False, "aA", "", "stop", True, 1, 1, "")
        self.assertEqual([m["content"] for m in server.get_chat(cid)
                          ["messages"]], ["q0", "a0", "qA", "aA", "qB", "aB"])

    def test_a_chat_deleted_mid_reply_stays_deleted(self) -> None:
        self.chat({"message": "q"}, [{"delta": "a"}, {"stop": "stop"}])
        cid = server.read_chats()[0]["id"]
        self.client.delete(f"/api/chats/{cid}")
        server.merge_reply(cid, 0, False, "late", "", "stop", True, 1, 1, "")
        self.assertIsNone(server.get_chat(cid))

    def test_a_failure_mid_reply_is_not_a_finished_reply(self) -> None:
        out = self.chat({"message": "q"},
                        [{"delta": "half an ans"}, RuntimeError("OOM")])
        done = next(e for e in out if e.get("done"))
        self.assertEqual((done["stop"], done["error"]), ("error", "OOM"))
        self.assertTrue(done["unfinished"])
        stored = server.read_chats()[0]["messages"][-1]
        self.assertEqual((stored["stop"], stored["error"]), ("error", "OOM"))
        self.assertTrue(stored["unfinished"])

    def test_a_failed_turn_keeps_the_roles_alternating(self) -> None:
        self.chat({"message": "q1"}, [RuntimeError("down")])
        cid = server.read_chats()[0]["id"]
        self.chat({"message": "q2", "chat": cid},
                  [{"delta": "ok"}, {"stop": "stop"}])
        roles = [m["role"] for m in self.sent[-1][0]]
        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_continuation_keeps_question_and_partial_reply(self) -> None:
        # a reply that filled the whole window
        self.chat({"message": "Write the parser."},
                  [{"delta": "x" * 16000}, {"stop": "length"}])
        cid = server.read_chats()[0]["id"]
        self.chat({"chat": cid, "continue": True},
                  [{"delta": "done"}, {"stop": "stop"}])
        history, params = self.sent[-1]
        self.assertEqual([m["role"] for m in history],
                         ["user", "assistant", "user"])
        self.assertEqual(history[0]["content"], "Write the parser.")
        self.assertTrue(history[1]["content"].startswith("…x"))
        self.assertEqual(history[2]["content"], server.CONTINUE_NUDGE)
        self.assertGreaterEqual(params["max_tokens"], 4096 // 4 - 64)
        self.assertTrue(server.read_chats()[0]["messages"][-1]["content"]
                        .endswith("xdone"))

    def test_code_page_rules_ride_with_its_continuations(self) -> None:
        self.chat({"message": "q", "mode": "code", "system": "CODE RULES"},
                  [{"delta": "```python\nx = 1\n"}, {"stop": "length"}])
        cid = server.read_chats()[0]["id"]
        self.chat({"chat": cid, "continue": True},
                  [{"delta": "```\n"}, {"stop": "stop"}])
        self.assertTrue(self.sent[-1][1]["system"].startswith("CODE RULES"))

    def test_fence_lines_not_backticks_in_prose(self) -> None:
        self.assertFalse(server.looks_unfinished(
            "Wrap code in ``` fences, like so.", "stop"))
        self.assertFalse(server.looks_unfinished(
            "````md\n```py\nx\n```\n````\n", "stop"))
        self.assertTrue(server.looks_unfinished("```py\nx = 1\n", "stop"))

    def test_an_unreadable_history_is_set_aside_not_wiped(self) -> None:
        self.path.write_text("[{broken json")
        self.assertEqual(server.read_chats(), [])
        kept = list(self.path.parent.glob("chats.unreadable-*.json"))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].read_text(), "[{broken json")


class ConfigSanityTests(unittest.TestCase):
    def test_hand_edited_values_fall_back_one_by_one(self) -> None:
        c = server.sane_config({**server.DEFAULTS, "ctx": "auto",
                                "kv_bits": None, "max_tokens": "lots",
                                "temperature": float("nan"), "top_p": 0.9,
                                "recall": "yes", "sampling": "greedy",
                                "model_configs": [], "system": 5})
        self.assertEqual((c["ctx"], c["kv_bits"], c["max_tokens"]),
                         (0, server.DEFAULTS["kv_bits"], 0))
        self.assertEqual(c["temperature"], server.DEFAULTS["temperature"])
        self.assertEqual(c["top_p"], 0.9)
        self.assertEqual((c["recall"], c["sampling"], c["model_configs"],
                          c["system"]), (True, "model", {}, ""))

    def test_a_broken_config_file_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"hf_token": "hf_secret", oops')
            with mock.patch.object(server, "CONFIG_PATH", path), \
                    mock.patch.object(server, "DATA_DIR", Path(tmp)):
                server.load_config()
            kept = list(Path(tmp).glob("config.unreadable-*.json"))
            self.assertEqual(len(kept), 1)
            self.assertIn("hf_secret", kept[0].read_text())

    def test_hf_address_must_be_encrypted_or_local(self) -> None:
        c = server.app.test_client()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(server, "CONFIG_PATH",
                                  Path(tmp) / "config.json"), \
                mock.patch.dict(server.cfg, {}):
            for bad in ("http://mirror.example", "ftp://x", "javascript:1"):
                self.assertEqual(c.post("/api/config", json={
                    "hf_endpoint": bad}).status_code, 400, bad)
            for good in ("https://hf-mirror.com", "http://127.0.0.1:9911"):
                self.assertEqual(c.post("/api/config", json={
                    "hf_endpoint": good}).status_code, 200, good)


class QuantChoiceTests(unittest.TestCase):
    def test_best_here_is_the_largest_that_fits_not_the_smallest(self) -> None:
        files = [{"name": n, "quant": q, "size": int(sz * GB),
                  "split": False, "companion": False}
                 for n, q, sz in (("a-Q2_K", "Q2_K", 3.0),
                                  ("a-Q5_K_M", "Q5_K_M", 5.3),
                                  ("a-Q6_K", "Q6_K", 6.3),
                                  ("a-IQ2_M", "IQ2_M", 2.8))]
        fits = {"verdict": "fits", "fits_ram": True}
        spills = {"verdict": "spills", "fits_ram": True}
        scored = [(f, fits if f["size"] < 6 * GB else spills) for f in files]
        self.assertEqual(server.choose_quant(scored, {"vram": 8 * GB})
                         ["name"], "a-Q5_K_M")


class OfficeTests(unittest.TestCase):
    def test_sheet_names_excel_refuses_and_ids_with_zeros(self) -> None:
        import openpyxl
        data, _ = server.build_xlsx(
            "# Sheet: Budget 2024/25: [v2]?\nZIP,Amount\n00123,1.5\n0,-3\n")
        wb = openpyxl.load_workbook(io.BytesIO(data))
        ws = wb.worksheets[0]
        self.assertNotRegex(ws.title, r"[\\/*?:\[\]]")
        self.assertEqual([c.value for c in ws[2]], ["00123", 1.5])
        self.assertEqual([c.value for c in ws[3]], [0, -3])


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
def _sse_response(body: bytes) -> requests.Response:
    r = requests.models.Response()
    r.status_code = 200
    r.headers["Content-Type"] = "text/event-stream"   # no charset
    r.encoding = requests.utils.get_encoding_from_headers(r.headers)
    r.raw = io.BytesIO(body)
    return r


class StreamTests(unittest.TestCase):
    def stream(self, body: bytes) -> list:
        resp = _sse_response(body)
        cm = mock.MagicMock()
        cm.__enter__.return_value = resp
        with mock.patch.object(engine.requests, "post", return_value=cm):
            return list(engine.Server().chat_stream([], {}))

    def test_replies_are_utf8_whatever_the_header_says(self) -> None:
        text = "Café — “quoted” 日本 🙂"
        body = ('data: ' + json.dumps({"choices": [{"delta": {
            "content": text}}]}, ensure_ascii=False) + "\n\n"
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n").encode("utf-8")
        self.assertEqual(self.stream(body)[0], {"delta": text})

    def test_an_error_event_mid_reply_is_raised(self) -> None:
        body = (b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                b'data: {"error":{"message":"context full"}}\n\n')
        with self.assertRaisesRegex(RuntimeError, "context full"):
            self.stream(body)


class ProcessTests(unittest.TestCase):
    def test_a_server_we_did_not_start_is_not_ready(self) -> None:
        srv = engine.Server()
        with mock.patch.object(engine.requests, "get") as get:
            get.return_value.status_code = 200
            self.assertFalse(srv.ready())
            get.assert_not_called()

    def test_stop_during_a_load_ends_the_wait_at_once(self) -> None:
        srv = engine.Server()
        srv.proc = mock.Mock()
        srv.proc.poll.return_value = None
        with mock.patch.object(srv, "ready", return_value=False):
            threading.Timer(0.2, lambda: setattr(srv, "proc", None)).start()
            t = time.time()
            self.assertFalse(srv.wait_ready(30))
        self.assertLess(time.time() - t, 3)

    def test_an_occupied_port_is_named_not_misread(self) -> None:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        try:
            srv = engine.Server(sock.getsockname()[1])
            with tempfile.NamedTemporaryFile(suffix=".gguf") as f, \
                    mock.patch.object(engine, "server_binary",
                                      return_value=Path("/bin/true")), \
                    mock.patch.object(srv, "_spawn") as spawn:
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    srv.start(f.name, 10, 8192)
            spawn.assert_not_called()
        finally:
            sock.close()


RELEASE = """cudart-llama-bin-win-cuda-12.4-x64.zip
cudart-llama-bin-win-cuda-13.3-x64.zip cudart-llama-bin-win-cuda-13.4-arm64.zip
llama-b1-bin-macos-arm64.tar.gz llama-b1-bin-macos-x64.tar.gz
llama-b1-bin-ubuntu-arm64.tar.gz llama-b1-bin-ubuntu-vulkan-x64.tar.gz
llama-b1-bin-ubuntu-openvino-2026.3.1-x64.tar.gz llama-b1-bin-ubuntu-x64.tar.gz
llama-b1-bin-win-cpu-arm64.zip llama-b1-bin-win-cpu-x64.zip
llama-b1-bin-win-cuda-12.4-x64.zip llama-b1-bin-win-cuda-13.3-x64.zip
llama-b1-bin-win-cuda-13.4-arm64.zip llama-b1-bin-win-vulkan-x64.zip
llama-b1-ui.tar.gz""".split()


class BuildChoiceTests(unittest.TestCase):
    def pick(self, system, machine, vendor):
        with mock.patch.object(engine.platform, "system",
                               return_value=system), \
                mock.patch.object(engine.platform, "machine",
                                  return_value=machine):
            got = engine._pick_asset([{"name": n} for n in RELEASE],
                                     engine.asset_choices({"vendor": vendor}))
        return got[0]["name"] if got else None

    def test_each_machine_gets_a_build_it_can_run(self) -> None:
        self.assertEqual(self.pick("Windows", "AMD64", "nvidia"),
                         "llama-b1-bin-win-cuda-12.4-x64.zip")
        # it used to be the arm64 build, listed first
        self.assertEqual(self.pick("Windows", "AMD64", ""),
                         "llama-b1-bin-win-cpu-x64.zip")
        self.assertEqual(self.pick("Windows", "AMD64", "amd"),
                         "llama-b1-bin-win-vulkan-x64.zip")
        # .tar.gz builds were skipped entirely
        self.assertEqual(self.pick("Linux", "x86_64", ""),
                         "llama-b1-bin-ubuntu-x64.tar.gz")
        self.assertEqual(self.pick("Darwin", "x86_64", ""),
                         "llama-b1-bin-macos-x64.tar.gz")

    def test_install_swaps_the_whole_build_after_stopping_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "llama.cpp"
            bin_dir.mkdir()
            (bin_dir / "stale.dll").write_text("old")
            order = []

            def fetch(task, asset, a, b, dest_dir=None):
                (dest_dir / "llama-server").write_text("new")
                return True

            asset = {"name": "llama-b1-bin-ubuntu-x64.tar.gz"}
            with mock.patch.object(engine, "BIN_DIR", bin_dir), \
                    mock.patch.object(engine, "_resolve_release",
                                      return_value=({"assets": []}, asset,
                                                    "CPU build")), \
                    mock.patch.object(engine, "_fetch_zip",
                                      side_effect=fetch), \
                    mock.patch.object(engine.platform, "system",
                                      return_value="Linux"):
                engine.install_sync(engine.Task("install", "t"), {},
                                    before_swap=lambda: order.append(
                                        (bin_dir / "stale.dll").exists()))
            self.assertEqual(order, [True])      # stopped before the swap
            self.assertFalse((bin_dir / "stale.dll").exists())
            self.assertEqual((bin_dir / "llama-server").read_text(), "new")
            self.assertFalse(bin_dir.with_name("llama.cpp.new").exists())
            self.assertFalse(bin_dir.with_name("llama.cpp.old").exists())


class _Files(http.server.BaseHTTPRequestHandler):
    """Serves one file with ETag and Range support, as a CDN would."""
    body = b""
    etag = '"v1"'
    truncate = 0

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()

    def do_GET(self):
        body, rng = self.body, self.headers.get("Range")
        if rng and self.headers.get("If-Range") in (None, self.etag):
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= len(body):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
            part = body[start:]
        else:
            self.send_response(200)
            part = body
        self.send_header("ETag", self.etag)
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part[:len(part) - self.truncate])


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                     _Files)
        threading.Thread(target=self.httpd.serve_forever,
                         daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/f.gguf"
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "f.gguf"
        self.part = Path(str(self.dest) + ".part")
        _Files.truncate, _Files.etag = 0, '"v1"'
        self.env = mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1",
                                                "no_proxy": "127.0.0.1"})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def get(self):
        engine._stream(self.url, self.dest, {}, None, lambda: False)

    def test_a_resume_of_the_same_file_appends(self) -> None:
        _Files.body = b"0123456789"
        self.part.write_bytes(b"0123")
        Path(str(self.part) + ".etag").write_text('"v1"')
        self.get()
        self.assertEqual(self.dest.read_bytes(), b"0123456789")

    def test_a_file_replaced_since_is_fetched_afresh(self) -> None:
        _Files.body, _Files.etag = b"NEW-NEW-NEW-", '"v2"'
        self.part.write_bytes(b"OLD-")
        Path(str(self.part) + ".etag").write_text('"v1"')
        self.get()
        self.assertEqual(self.dest.read_bytes(), b"NEW-NEW-NEW-")

    def test_a_short_download_stays_a_part(self) -> None:
        _Files.body, _Files.truncate = b"0123456789", 3
        with self.assertRaises(Exception):
            self.get()
        self.assertFalse(self.dest.exists())

    def test_a_stale_complete_looking_part_is_discarded(self) -> None:
        _Files.body = b"short"
        self.part.write_bytes(b"much longer old file")
        with self.assertRaisesRegex(RuntimeError, "no longer matches"):
            self.get()
        self.assertFalse(self.dest.exists() or self.part.exists())

    def test_one_writer_per_file(self) -> None:
        engine._ACTIVE_PARTS.add(str(self.part.resolve()).lower())
        try:
            with self.assertRaisesRegex(RuntimeError, "already downloading"):
                self.get()
        finally:
            engine._ACTIVE_PARTS.clear()


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
class IdentifyTests(unittest.TestCase):
    def test_a_family_word_alone_is_not_a_match(self) -> None:
        name = lambda f: (fit.catalogue_match(f) or {}).get("name")  # noqa
        self.assertEqual(name("Qwen3-8B-Q4_K_M.gguf"), "Qwen3 8B")
        self.assertIsNone(name("Qwen3-32B-Q4_K_M.gguf"))
        self.assertIsNone(name("gemma-3-27b-it-Q4_K_M.gguf"))
        self.assertIsNone(name("Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"))
        self.assertEqual(name("phi-4-Q4_K_M.gguf"), "Phi-4 14B")
        self.assertEqual(name("Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"),
                         "Qwen3 Coder 30B A3B")
        self.assertEqual(name("Qwen3.8-27B-Uncensored-Q4_K_M.gguf"),
                         "Qwen3.8 27B Uncensored — GGUF")

    def test_bandwidth_is_matched_as_a_whole_name(self) -> None:
        self.assertEqual(fit.bandwidth_of("NVIDIA GeForce RTX 4060"), 272)
        self.assertEqual(fit.bandwidth_of("NVIDIA GeForce RTX 4060 Ti"), 288)
        self.assertEqual(fit.bandwidth_of("NVIDIA RTX A1000"), 0)
        self.assertEqual(fit.bandwidth_of("NVIDIA L40S"), 0)
        self.assertEqual(fit.bandwidth_of("RTX 4090 Laptop GPU"), 0)

    def test_every_gpu_is_read_and_na_costs_only_its_field(self) -> None:
        out = mock.Mock(returncode=0, stdout=(
            "NVIDIA GeForce GT 1030, 2048, 1900\n"
            "NVIDIA GeForce RTX 4060, 8188, [N/A]\n"))
        with mock.patch.object(fit.shutil, "which", return_value="x"), \
                mock.patch.object(fit.subprocess, "run", return_value=out):
            g = fit.gpu_info()
        self.assertEqual((g["name"], g["vram"], g["vram_free"]),
                         ("NVIDIA GeForce RTX 4060", 8188 * 1024 ** 2,
                          8188 * 1024 ** 2))


class KvCacheTests(unittest.TestCase):
    def test_q8_is_34_bytes_per_32_values(self) -> None:
        conf = {"layers": 36, "kv_layers": 36, "kv_heads": 8,
                "head_dim": 128}
        self.assertEqual(fit.kv_bytes(conf, 1024, 8),
                         int(2 * 36 * 8 * 128 * 34 / 32 * 1024))

    def test_latent_attention_caches_the_latent(self) -> None:
        c = fit._config_from({"num_hidden_layers": 47,
                              "num_attention_heads": 20,
                              "num_key_value_heads": 20, "head_dim": 256,
                              "kv_lora_rank": 512, "qk_rope_head_dim": 64},
                             {})
        self.assertEqual((c["kv_heads"], c["head_dim"]), (1, 576))

    def test_sliding_layers_hold_only_their_window(self) -> None:
        types = (["sliding_attention"] * 5 + ["full_attention"]) * 8
        c = fit._config_from({"num_hidden_layers": 48, "layer_types": types,
                              "sliding_window": 1024,
                              "num_attention_heads": 16,
                              "num_key_value_heads": 8, "head_dim": 256},
                             {})
        self.assertEqual((c["kv_layers"], c["swa_layers"], c["hybrid"]),
                         (8, 40, False))
        per = 2 * 8 * 256 * 2
        self.assertEqual(fit.kv_bytes(c, 32768, 16),
                         int(8 * per * 32768 + 40 * per * (1024 + 512)))


def _gguf(path: Path, meta: list, tensors: list) -> None:
    def s(t):
        b = t.encode()
        return struct.pack("<Q", len(b)) + b
    out = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(tensors),
                                           len(meta))
    for k, v in meta:
        out += s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str)
                       else struct.pack("<II", 4, v))
    off = 0
    for n, size in tensors:
        out += s(n) + struct.pack("<IQQIQ", 2, size, 1, 0, off)
        off += size
    out += b"\0" * (-(-len(out) // 32) * 32 - len(out))
    path.write_bytes(bytes(out) + bytes(off))


class FileArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        fit._LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def file(self, name: str, blocks: int) -> dict:
        path = Path(self.tmp.name) / name
        _gguf(path, [("general.architecture", "qwen3"),
                     ("qwen3.block_count", blocks),
                     ("qwen3.attention.head_count", 32),
                     ("qwen3.attention.head_count_kv", 8),
                     ("qwen3.attention.key_length", 128),
                     ("qwen3.context_length", 40960)],
              [(f"blk.{i}.attn_q.weight", 64) for i in range(blocks)])
        return {"name": name, "path": str(path),
                "size": path.stat().st_size, "partial": False}

    def test_an_unknown_file_describes_itself(self) -> None:
        conf = fit.file_config(fit.gguf_layout(self.file("x.gguf", 40)
                                               ["path"]))
        self.assertEqual((conf["layers"], conf["kv_heads"], conf["head_dim"],
                          conf["max_ctx"], conf["exact"]),
                         (40, 8, 128, 40960, True))

    def test_a_config_for_another_model_is_refused(self) -> None:
        model = self.file("Qwen3-8B-Q4_K_M.gguf", 64)   # really 64 blocks
        eight_b = {**fit.GENERIC_CONF, "layers": 36, "kv_layers": 36,
                   "exact": True, "nextn": 0}
        with mock.patch.object(server.fit, "model_config",
                               return_value=eight_b), \
                mock.patch.object(server, "saved_model_config",
                                  return_value=None):
            conf, entry = server.local_conf(model)
        self.assertIsNone(entry)            # its sampling is not applied
        self.assertEqual(conf["layers"], 64)
        self.assertTrue(conf.get("from_file"))

    def test_corrupt_headers_are_refused_not_copied(self) -> None:
        path = Path(self.tmp.name) / "bad.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1) +
                         struct.pack("<Q", 1 << 40) + b"x" * 64)
        self.assertIsNone(fit.gguf_layout(str(path)))


if __name__ == "__main__":
    unittest.main()
