"""Regression tests for what reaches the model: sampling, context, reasoning,
and the system prompt."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import engine
import fit
import server

GB = 1024 ** 3


def _response(status: int = 200, body=None):
    r = mock.Mock()
    r.status_code = status
    r.json.return_value = body
    r.raise_for_status.side_effect = (
        None if status < 400 else RuntimeError(str(status)))
    return r


class SamplingProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        fit._SAMPLING_CACHE.clear()

    def test_reads_qwen3_thinking_settings_from_generation_config(self) -> None:
        body = {"do_sample": True, "temperature": 0.6, "top_k": 20,
                "top_p": 0.95, "eos_token_id": [151645, 151643]}
        with mock.patch.object(fit.requests, "get",
                               return_value=_response(200, body)) as get:
            prof = fit.model_sampling({}, "Qwen/Qwen3-8B")
        self.assertIn("Qwen/Qwen3-8B/resolve/main/generation_config.json",
                      get.call_args.args[0])
        self.assertEqual(prof, {"temperature": 0.6, "top_p": 0.95,
                                "top_k": 20, "min_p": 0.0,
                                "repeat_penalty": 1.0,
                                "source": "Qwen/Qwen3-8B"})

    def test_repetition_penalty_maps_to_llama_cpp_name(self) -> None:
        body = {"do_sample": True, "temperature": 0.7, "top_p": 0.8,
                "top_k": 20, "repetition_penalty": 1.05}
        with mock.patch.object(fit.requests, "get",
                               return_value=_response(200, body)):
            prof = fit.model_sampling({}, "Qwen/Qwen3-Coder-30B-A3B-Instruct")
        self.assertEqual(prof["repeat_penalty"], 1.05)
        self.assertEqual(prof["temperature"], 0.7)

    def test_missing_keys_take_the_reference_defaults(self) -> None:
        # Gemma ships top_k / top_p and leaves temperature at transformers'
        # sampling default of 1.0.
        body = {"do_sample": True, "top_k": 64, "top_p": 0.95}
        with mock.patch.object(fit.requests, "get",
                               return_value=_response(200, body)):
            prof = fit.model_sampling({}, "google/gemma-3-4b-it")
        self.assertEqual(prof["temperature"], 1.0)
        self.assertEqual(prof["top_k"], 64)
        self.assertEqual(prof["min_p"], 0.0)

    def test_no_profile_when_absent_greedy_or_insane(self) -> None:
        for status, body in ((404, None),
                             (200, {"do_sample": False, "temperature": 0.6}),
                             (200, {"bos_token_id": 1}),
                             (200, {"temperature": 9.0, "top_p": 0.9})):
            fit._SAMPLING_CACHE.clear()
            with mock.patch.object(fit.requests, "get",
                                   return_value=_response(status, body)):
                self.assertEqual(fit.model_sampling({}, "x/y"), {}, body)

    def test_network_failure_is_not_cached(self) -> None:
        with mock.patch.object(fit.requests, "get",
                               side_effect=OSError("offline")):
            self.assertIsNone(fit.model_sampling({}, "x/y"))
        self.assertNotIn("x/y", fit._SAMPLING_CACHE)

    def test_offline_load_keeps_a_saved_profile_and_retries_a_missing_one(
            self) -> None:
        old = dict(server.cfg)
        prof = {"model": "Qwen3-8B-Q4_K_M.gguf", "temperature": 0.6,
                "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                "repeat_penalty": 1.0, "source": "Qwen/Qwen3-8B"}
        try:
            with tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(server, "CONFIG_PATH",
                                      Path(tmp) / "config.json"), \
                    mock.patch.object(server.fit, "model_sampling",
                                      return_value=None):
                server.cfg["last_model_sampling"] = dict(prof)
                kept = server.remember_sampling("Qwen3-8B-Q4_K_M.gguf")
                self.assertEqual(kept["temperature"], 0.6)
                self.assertEqual(server.cfg["last_model_sampling"], prof)
                server.remember_sampling("Qwen3-14B-Q4_K_M.gguf")
                # nothing recorded for the new model, so autostart retries
                self.assertEqual(server.cfg["last_model_sampling"], {})
        finally:
            server.cfg.clear()
            server.cfg.update(old)


class AutoContextTests(unittest.TestCase):
    QWEN3_8B = {"layers": 36, "kv_layers": 36, "kv_heads": 8,
                "head_dim": 128, "exact": True, "moe": False,
                "hybrid": False, "max_ctx": 40960}
    Q4 = 5_027_783_488          # Qwen3-8B-Q4_K_M.gguf

    def hw(self, vram_mib: int = 8188) -> dict:
        return {"vram": vram_mib * 1024 ** 2, "ram": 32 * GB,
                "bandwidth": 272}

    def test_idle_8gb_card_gives_a_thinking_model_32k(self) -> None:
        self.assertEqual(fit.auto_ctx(self.Q4, self.QWEN3_8B, self.hw(), 8),
                         32768)

    def test_busy_card_gets_less_but_never_under_the_floor(self) -> None:
        self.assertEqual(fit.auto_ctx(self.Q4, self.QWEN3_8B,
                                      self.hw(7000), 8), 16384)
        self.assertEqual(fit.auto_ctx(self.Q4, self.QWEN3_8B,
                                      self.hw(5000), 8), 8192)

    def test_trained_window_caps_auto(self) -> None:
        conf = {**self.QWEN3_8B, "max_ctx": 16384}
        self.assertEqual(fit.auto_ctx(self.Q4, conf, self.hw(), 8), 16384)

    def test_spilling_model_is_not_pushed_further_off_the_card(self) -> None:
        coder = {"layers": 48, "kv_layers": 48, "kv_heads": 4,
                 "head_dim": 128, "exact": True, "moe": True,
                 "hybrid": False, "max_ctx": 262144}
        weights = int(17.3 * GB)
        ctx = fit.auto_ctx(weights, coder, self.hw(), 8)
        floor = fit.assess(weights, coder, self.hw(), 8192, 8)
        self.assertEqual(fit.assess(weights, coder, self.hw(), ctx,
                                    8)["gpu_layers"], floor["gpu_layers"])

    def test_load_measures_free_vram_after_stopping_the_old_model(self) -> None:
        model = {"name": "Qwen3-8B-Q4_K_M.gguf", "path": "/tmp/q.gguf",
                 "size": self.Q4, "partial": False}
        hw = {**self.hw(), "vram_free": 7000 * 1024 ** 2, "name": "RTX 4060"}
        order = []
        old = dict(server.cfg)
        try:
            server.cfg.update({"ctx": 0, "kv_bits": 8})
            with tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(server, "CONFIG_PATH",
                                      Path(tmp) / "config.json"), \
                    mock.patch.object(server.engine, "local_models",
                                      return_value=[model]), \
                    mock.patch.object(server.fit, "model_config",
                                      return_value=self.QWEN3_8B), \
                    mock.patch.object(server.fit, "model_sampling",
                                      return_value={}), \
                    mock.patch.object(server.server, "stop",
                                      side_effect=lambda: order.append(
                                          "stop")), \
                    mock.patch.object(server.fit, "hardware",
                                      side_effect=lambda c: order.append(
                                          "measure") or hw), \
                    mock.patch.object(server.server, "start") as start, \
                    mock.patch.object(server.server, "wait_ready",
                                      return_value=True):
                result = server.load_model_by_name(model["name"])
                with self.assertRaisesRegex(RuntimeError, "Context must"):
                    server.load_model_by_name(model["name"], ctx=100)
            self.assertEqual(order[:2], ["stop", "measure"])
            # a rejected size never stops the running model
            self.assertEqual(order.count("stop"), 1)
            self.assertEqual(result["ctx"], 16384)
            self.assertTrue(result["auto_ctx"])
            self.assertEqual(start.call_args.args[2], 16384)
            # Auto stays the setting; only the resolved window is recorded.
            self.assertEqual(server.cfg["ctx"], 0)
            self.assertEqual(server.cfg["last_ctx"], 16384)
        finally:
            server.cfg.clear()
            server.cfg.update(old)


class ChatStreamTests(unittest.TestCase):
    def test_reasoning_answer_and_timings_are_all_surfaced(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"Let me '
            'think."}}]}',
            'data: {"choices":[{"delta":{"content":"def f(): pass"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"timings":{"predicted_per_second":41.7}}',
            "data: [DONE]",
        ]
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = lines
        resp.__enter__.return_value = resp
        srv = engine.Server()
        with mock.patch.object(engine.requests, "post",
                               return_value=resp) as post:
            events = list(srv.chat_stream(
                [{"role": "user", "content": "hi"}],
                {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                 "min_p": 0.0, "repeat_penalty": 1.0, "max_tokens": 100,
                 "chat_template_kwargs": {"enable_thinking": False}}))
        self.assertEqual(events, [
            {"think": "Let me think."}, {"delta": "def f(): pass"},
            {"stop": "stop", "timings": {"predicted_per_second": 41.7}}])
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["top_k"], 20)
        self.assertEqual(sent["min_p"], 0.0)
        self.assertEqual(sent["chat_template_kwargs"],
                         {"enable_thinking": False})

    def test_llama_defaults_stand_when_no_profile(self) -> None:
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = ["data: [DONE]"]
        resp.__enter__.return_value = resp
        with mock.patch.object(engine.requests, "post",
                               return_value=resp) as post:
            list(engine.Server().chat_stream([], {"temperature": 0.7}))
        sent = post.call_args.kwargs["json"]
        for key in ("top_k", "min_p", "repeat_penalty",
                    "chat_template_kwargs"):
            self.assertNotIn(key, sent)

    def test_prompt_progress_and_live_timings_are_asked_for_and_surfaced(
            self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"role":"assistant","content":null}}],'
            '"prompt_progress":{"total":1007,"cache":0,"processed":512,'
            '"time_ms":900}}',
            'data: {"choices":[{"delta":{"content":"Hi"}}],'
            '"timings":{"predicted_n":1,"predicted_per_second":20.5}}',
            "data: [DONE]",
        ]
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = lines
        resp.__enter__.return_value = resp
        handle: dict = {}
        with mock.patch.object(engine.requests, "post",
                               return_value=resp) as post:
            events = list(engine.Server().chat_stream(
                [], {"max_tokens": 10}, handle=handle))
        self.assertEqual(events[0], {"prompt": {
            "processed": 512, "total": 1007, "cache": 0}})
        self.assertIn({"delta": "Hi"}, events)
        self.assertIs(handle["response"], resp)
        self.assertEqual(handle["timings"]["predicted_n"], 1)
        sent = post.call_args.kwargs["json"]
        self.assertTrue(sent["return_progress"])
        self.assertTrue(sent["timings_per_token"])

    def test_cancel_ends_the_reply_at_the_next_chunk(self) -> None:
        handle: dict = {}

        def lines():
            yield 'data: {"choices":[{"delta":{"content":"a"}}]}'
            handle["cancel"] = True
            yield 'data: {"choices":[{"delta":{"content":"b"}}]}'
        resp = mock.MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = lines()
        resp.__enter__.return_value = resp
        with mock.patch.object(engine.requests, "post", return_value=resp):
            events = list(engine.Server().chat_stream([], {},
                                                      handle=handle))
        self.assertEqual(events, [{"delta": "a"}])
        resp.__exit__.assert_called()

    def test_abort_shuts_the_socket_under_a_blocked_reader(self) -> None:
        resp = mock.MagicMock()
        handle = {"response": resp}
        engine.Server.abort(handle)
        self.assertTrue(handle["cancel"])
        resp.raw.shutdown.assert_called_once()
        # older urllib3 without shutdown(): the socket by hand
        resp = mock.MagicMock()
        resp.raw.shutdown.side_effect = AttributeError
        engine.Server.abort({"response": resp})
        resp.raw._connection.sock.shutdown.assert_called_once()
        engine.Server.abort({})  # nothing open yet: just marks it

    def test_activity_reads_tokens_generated_from_slots(self) -> None:
        slots = [
            {"id": 0, "is_processing": False},
            {"id": 1, "is_processing": True, "id_task": 7,
             "next_token": [{"n_decoded": 812}]},
            {"id": 2, "is_processing": True, "id_task": 3,
             "next_token": {"n_decoded": 5}},
        ]
        srv = engine.Server()
        with mock.patch.object(engine.requests, "get",
                               return_value=_response(200, slots)):
            self.assertEqual(srv.activity(),
                             {"processing": True, "decoded": 812})
        with mock.patch.object(engine.requests, "get",
                               return_value=_response(200, [
                                   {"id": 0, "is_processing": False}])):
            self.assertEqual(srv.activity(),
                             {"processing": False, "decoded": 0})
        # --no-slots, or a server busy with a long prompt batch
        with mock.patch.object(engine.requests, "get",
                               return_value=_response(501, {"error": {}})):
            self.assertEqual(srv.activity(), {})
        with mock.patch.object(engine.requests, "get",
                               side_effect=engine.requests.Timeout):
            self.assertEqual(srv.activity(), {})

    def test_server_is_started_with_the_models_own_template(self) -> None:
        srv = engine.Server()
        with tempfile.NamedTemporaryFile(suffix=".gguf") as f, \
                mock.patch.object(engine, "server_binary",
                                  return_value=Path("/bin/llama-server")), \
                mock.patch.object(srv, "_spawn") as spawn:
            srv.start(f.name, 36, 32768)
        self.assertIn("--jinja", spawn.call_args.args[0])


class SamplingChoiceTests(unittest.TestCase):
    PROFILE = {"model": "q.gguf", "temperature": 0.6, "top_p": 0.95,
               "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
               "source": "Qwen/Qwen3-8B"}

    def setUp(self) -> None:
        self.old = dict(server.cfg)
        self.old_model = server.server.model
        server.server.model = "/models/qwen/q.gguf"
        server.cfg.update({"temperature": 0.9, "top_p": 0.5,
                           "last_model_sampling": dict(self.PROFILE)})

    def tearDown(self) -> None:
        server.server.model = self.old_model
        server.cfg.clear()
        server.cfg.update(self.old)

    def test_model_settings_by_default(self) -> None:
        server.cfg["sampling"] = "model"
        p = server.sampling_params({})
        self.assertEqual((p["temperature"], p["top_p"], p["top_k"],
                          p["min_p"]), (0.6, 0.95, 20, 0.0))

    def test_manual_uses_the_sliders_only(self) -> None:
        server.cfg["sampling"] = "manual"
        self.assertEqual(server.sampling_params({}),
                         {"temperature": 0.9, "top_p": 0.5})

    def test_profile_of_another_model_is_ignored(self) -> None:
        server.cfg["sampling"] = "model"
        server.server.model = "/models/gemma/other.gguf"
        self.assertEqual(server.sampling_params({})["temperature"], 0.9)

    def test_an_explicit_request_value_wins(self) -> None:
        server.cfg["sampling"] = "model"
        self.assertEqual(server.sampling_params(
            {"temperature": 0.3})["temperature"], 0.3)


class PromptDietTests(unittest.TestCase):
    def test_coding_question_carries_no_office_convention(self) -> None:
        history = [{"role": "user",
                    "content": "Write a Flask endpoint that returns JSON."}]
        self.assertEqual(server.doc_howto(history), "")

    def test_office_request_and_its_follow_up_do(self) -> None:
        ask = [{"role": "user", "content": "Make me a spreadsheet of sales"}]
        self.assertIn("xlsx", server.doc_howto(ask))
        follow = [ask[0],
                  {"role": "assistant", "content": "```xlsx\n# Sheet: A\n```"},
                  {"role": "user", "content": "add a total row"}]
        self.assertTrue(server.wants_documents(follow))

    def test_attached_source_files_do_not_trigger_it(self) -> None:
        msg = ("Fix the off-by-one in the carousel.\n\nProject files:\n\n"
               "```js path=slides.js\nconst deck = [];\n```")
        self.assertFalse(server.wants_documents(
            [{"role": "user", "content": msg}]))


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.old = dict(server.cfg)
        self.path = Path(self.tmp.name) / "config.json"
        self.patch = mock.patch.object(server, "CONFIG_PATH", self.path)
        self.patch.start()
        self.client = server.app.test_client()

    def tearDown(self) -> None:
        server.cfg.clear()
        server.cfg.update(self.old)
        self.patch.stop()
        self.tmp.cleanup()

    def test_auto_context_and_sampling_mode_are_accepted(self) -> None:
        r = self.client.post("/api/config",
                             json={"ctx": 0, "sampling": "manual"})
        self.assertEqual(r.status_code, 200)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual((saved["ctx"], saved["sampling"]), (0, "manual"))
        self.assertEqual(self.client.post(
            "/api/config", json={"sampling": "greedy"}).status_code, 400)
        self.assertEqual(self.client.post(
            "/api/config", json={"ctx": 100}).status_code, 400)

    def test_old_fixed_8k_default_moves_to_auto_once(self) -> None:
        self.path.write_text(json.dumps({"ctx": 8192,
                                         "reply_limit_migrated": True}))
        self.assertEqual(server.load_config()["ctx"], 0)
        self.path.write_text(json.dumps({"ctx": 8192,
                                         "reply_limit_migrated": True,
                                         "ctx_auto_migrated": True}))
        self.assertEqual(server.load_config()["ctx"], 8192)
        self.path.write_text(json.dumps({"ctx": 16384,
                                         "reply_limit_migrated": True}))
        self.assertEqual(server.load_config()["ctx"], 16384)


class ChatReasoningTests(unittest.TestCase):
    """The full /api/chat path with llama-server stubbed out."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.old = dict(server.cfg)
        self.patches = [
            mock.patch.object(server, "CHATS_PATH", base / "chats.json"),
            mock.patch.object(server, "MEMORY_PATH", base / "memory.md"),
            mock.patch.object(server, "CONFIG_PATH", base / "config.json"),
            mock.patch.object(server, "context_extra", return_value=""),
            mock.patch.object(server, "index_exchanges"),
            mock.patch.object(server.server, "ready", return_value=True),
            mock.patch.object(server.server, "n_ctx", return_value=32768),
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
        server.cfg.clear()
        server.cfg.update(self.old)
        self.tmp.cleanup()

    def run_chat(self, body: dict, events: list[dict]) -> list[dict]:
        def fake(history, params, **_):
            self.sent.append((history, params))
            yield from events
        with mock.patch.object(server.server, "chat_stream",
                               side_effect=fake):
            r = self.client.post("/api/chat", json=body)
            raw = r.get_data(as_text=True)
        return [json.loads(chunk[5:]) for chunk in raw.split("\n\n")
                if chunk.startswith("data:")]

    def test_reasoning_is_streamed_stored_and_never_sent_back(self) -> None:
        out = self.run_chat({"message": "sort a list"}, [
            {"think": "They want sorted()."},
            {"delta": "Use `sorted(xs)`."},
            {"stop": "stop", "timings": {"predicted_per_second": 38.2}}])
        self.assertIn({"think": "They want sorted()."}, out)
        done = next(e for e in out if e.get("done"))
        self.assertEqual(done["tps"], 38.2)
        self.assertFalse(done["thought_only"])
        chat = server.read_chats()[0]
        reply = chat["messages"][-1]
        self.assertEqual(reply["reasoning"], "They want sorted().")
        self.assertEqual(reply["content"], "Use `sorted(xs)`.")
        # next turn: history is content only
        self.run_chat({"message": "and reversed?", "chat": chat["id"]},
                      [{"delta": "sorted(xs, reverse=True)"},
                       {"stop": "stop"}])
        history = self.sent[-1][0]
        self.assertNotIn("They want sorted().", json.dumps(history))

    def test_window_spent_thinking_is_kept_and_carried_into_an_answer(
            self) -> None:
        out = self.run_chat({"message": "write a parser"}, [
            {"think": "Hmm, many cases... " * 5},
            {"stop": "length", "timings": {}}])
        done = next(e for e in out if e.get("done"))
        self.assertTrue(done["thought_only"])
        self.assertTrue(done["unfinished"])
        chat = server.read_chats()[0]
        self.assertEqual(chat["messages"][-1]["content"], "")
        self.run_chat({"chat": chat["id"], "continue": True},
                      [{"delta": "def parse(s): ..."}, {"stop": "stop"}])
        history, params = self.sent[-1]
        self.assertEqual(history[-1]["content"], server.ANSWER_NUDGE)
        self.assertEqual(params["chat_template_kwargs"],
                         {"enable_thinking": False})
        merged = server.read_chats()[0]["messages"][-1]
        self.assertEqual(merged["content"], "def parse(s): ...")
        self.assertIn("many cases", merged["reasoning"])

    def test_plain_continuation_runs_without_thinking(self) -> None:
        self.run_chat({"message": "long file"},
                      [{"delta": "```python\nx = 1\n"},
                       {"stop": "length"}])
        chat = server.read_chats()[0]
        self.run_chat({"chat": chat["id"], "continue": True},
                      [{"delta": "```"}, {"stop": "stop"}])
        history, params = self.sent[-1]
        self.assertEqual(history[-1]["content"], server.CONTINUE_NUDGE)
        self.assertEqual(params["chat_template_kwargs"],
                         {"enable_thinking": False})
        self.assertNotIn("chat_template_kwargs", self.sent[0][1])

    def run_live(self, body: dict, fake, slot=None) -> list[dict]:
        with mock.patch.object(server, "BEAT", 0.05), \
                mock.patch.object(server, "QUIET", 0.05), \
                mock.patch.object(server.server, "activity",
                                  return_value=slot or {}), \
                mock.patch.object(server.server, "chat_stream",
                                  side_effect=fake):
            raw = self.client.post("/api/chat", json=body).get_data(
                as_text=True)
        return [json.loads(chunk[5:]) for chunk in raw.split("\n\n")
                if chunk.startswith("data:")]

    def test_a_silent_model_still_reports_what_it_is_doing(self) -> None:
        def fake(history, params, handle=None):
            yield {"prompt": {"processed": 512, "total": 1007, "cache": 0}}
            time.sleep(0.2)
            handle["timings"] = {"predicted_n": 40,
                                 "predicted_per_second": 21.0}
            yield {"delta": "Hello"}
            time.sleep(0.2)
            yield {"stop": "stop", "timings": {"predicted_n": 41}}
        out = self.run_live({"message": "hi"}, fake)
        stages = [e["alive"] for e in out if "alive" in e]
        self.assertTrue(stages)
        self.assertEqual(stages[0]["stage"], "prompt")
        self.assertEqual(stages[0]["prompt"], [512, 1007])
        self.assertEqual(stages[-1]["stage"], "write")
        self.assertEqual(stages[-1]["tokens"], 40)
        self.assertEqual(stages[-1]["tps"], 21.0)
        done = next(e for e in out if e.get("done"))
        self.assertEqual(done["tokens"], 41)
        self.assertFalse(done["empty"])

    def test_tokens_that_are_not_text_are_said_to_be_so(self) -> None:
        def fake(history, params, handle=None):
            time.sleep(0.25)
            yield {"stop": "length", "timings": {"predicted_n": 900}}
        out = self.run_live({"message": "hi"}, fake,
                            slot={"processing": True, "decoded": 900})
        stages = [e["alive"] for e in out if "alive" in e]
        self.assertEqual(stages[-1]["stage"], "hidden")
        self.assertEqual(stages[-1]["tokens"], 900)
        done = next(e for e in out if e.get("done"))
        self.assertTrue(done["empty"])
        self.assertEqual(done["tokens"], 900)
        # nothing written is nothing to carry on from
        self.assertFalse(done["unfinished"])

    def test_the_model_is_hung_up_on_when_the_reply_ends(self) -> None:
        with mock.patch.object(server.server, "abort") as abort:
            self.run_chat({"message": "hi"},
                          [{"delta": "ok"}, {"stop": "stop"}])
        abort.assert_called_once()
        self.assertIsInstance(abort.call_args.args[0], dict)

    def test_page_leaving_mid_reply_keeps_the_text_and_hangs_up(
            self) -> None:
        def fake(history, params, handle=None):
            yield {"delta": "partial "}
            while not handle.get("cancel"):
                time.sleep(0.01)
        with mock.patch.object(server, "BEAT", 0.05), \
                mock.patch.object(server.server, "activity",
                                  return_value={}), \
                mock.patch.object(server.server, "chat_stream",
                                  side_effect=fake):
            r = self.client.post("/api/chat", json={"message": "hi"},
                                 buffered=False)
            chunks = iter(r.response)
            seen = ""
            while "partial" not in seen:
                seen += next(chunks).decode()
            r.close()  # what werkzeug does when the page goes away
        reply = server.read_chats()[0]["messages"][-1]
        self.assertEqual(reply["content"], "partial ")
        self.assertEqual(reply["stop"], "aborted")

    def test_an_unanswered_question_is_folded_into_the_next(self) -> None:
        self.assertEqual(server.turns([
            {"role": "user", "content": "first", "at": 1},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "answer", "reasoning": "x"},
            {"role": "user", "content": "next"}]), [
            {"role": "user", "content": "first\n\nagain"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"}])


if __name__ == "__main__":
    unittest.main()
