"""Regression tests for the local API and hardware-fit safety rails."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import fit
import server


class ConfigApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.old_config = dict(server.cfg)
        self.config_path = Path(self.tmp.name) / "config.json"
        self.path_patch = mock.patch.object(server, "CONFIG_PATH", self.config_path)
        self.path_patch.start()
        self.client = server.app.test_client()

    def tearDown(self) -> None:
        server.cfg.clear()
        server.cfg.update(self.old_config)
        self.path_patch.stop()
        self.tmp.cleanup()

    def test_accepts_and_persists_valid_values(self) -> None:
        response = self.client.post("/api/config", json={
            "ctx": 8192, "temperature": 0.2, "top_p": 0.9,
            "kv_bits": 8, "max_tokens": 0, "recall": False,
        })
        self.assertEqual(response.status_code, 200)
        saved = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["ctx"], 8192)
        self.assertEqual(saved["kv_bits"], 8)
        self.assertFalse(saved["recall"])

    def test_rejects_out_of_range_values_without_mutating_config(self) -> None:
        old = dict(server.cfg)
        response = self.client.post("/api/config", json={"ctx": 1})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(server.cfg, old)

    def test_api_responses_have_hardening_headers(self) -> None:
        response = self.client.get("/api/chats")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertEqual(response.headers["Cache-Control"], "no-store")


class PersistenceTests(unittest.TestCase):
    def test_concurrent_chat_updates_do_not_drop_distinct_chats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chats.json"
            with mock.patch.object(server, "CHATS_PATH", path):
                threads = [threading.Thread(
                    target=server.put_chat,
                    args=({"id": str(i), "messages": []},),
                ) for i in range(20)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                self.assertEqual(len(server.read_chats()), 20)
                self.assertFalse(list(path.parent.glob("*.tmp")))


class FitTests(unittest.TestCase):
    def test_4060_default_profile_offloads_an_8b_q4_model_fully(self) -> None:
        hw = {"vram": 8 * 1024**3, "ram": 32 * 1024**3,
              "bandwidth": 272}
        conf = {"layers": 36, "kv_layers": 36, "kv_heads": 8,
                "head_dim": 128, "exact": True, "moe": False,
                "hybrid": False}
        result = fit.assess(int(4.6 * 1024**3), conf, hw, 8192, 8)
        self.assertEqual(result["gpu_layers"], 36)
        self.assertEqual(result["verdict"], "fits")
        self.assertTrue(result["fits_ram"])

    def test_speed_guard_suppresses_truncated_files(self) -> None:
        hw = {"vram": 8 * 1024**3, "ram": 32 * 1024**3,
              "bandwidth": 272}
        conf = {"layers": 32, "kv_heads": 8, "head_dim": 128,
                "exact": False}
        self.assertIsNone(fit.assess(1024, conf, hw)["speed"])


class EngineProfileTests(unittest.TestCase):
    def test_q8_profile_is_applied_to_llama_server(self) -> None:
        self.assertEqual(server.model_runtime_args(8), [
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        ])
        self.assertEqual(server.model_runtime_args(16), [])

    def test_loading_ignores_plan_card_and_caps_manual_gpu_layers(self) -> None:
        model = {"name": "custom.gguf", "path": "/tmp/custom.gguf",
                 "size": 4 * 1024**3, "partial": False}
        actual = {"vram": 8 * 1024**3, "ram": 32 * 1024**3,
                  "bandwidth": 272, "name": "RTX 4060"}
        old = dict(server.cfg)
        try:
            server.cfg.update({"plan_for": "RTX 5090", "kv_bits": 8})
            with tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(server, "CONFIG_PATH",
                                      Path(tmp) / "config.json"), \
                    mock.patch.object(server.engine, "local_models",
                                      return_value=[model]), \
                    mock.patch.object(server.fit, "hardware",
                                      return_value=actual) as hardware, \
                    mock.patch.object(server.server, "start") as start, \
                    mock.patch.object(server.server, "wait_ready",
                                      return_value=True):
                result = server.load_model_by_name(
                    "custom.gguf", ctx=8192, gpu_layers=999)
            hardware.assert_called_once_with({})
            self.assertLess(result["gpu_layers"], 999)
            self.assertEqual(start.call_args.args[3], [
                "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            ])
            self.assertEqual(server.cfg["last_model_config"]["model"],
                             "custom.gguf")
            self.assertEqual(server.cfg["last_model_config"]["layers"], 32)
        finally:
            server.cfg.clear()
            server.cfg.update(old)


if __name__ == "__main__":
    unittest.main()
