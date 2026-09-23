"""Regression tests for mixture-of-experts placement (--n-cpu-moe) and the
output-layer -ngl fix."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fit
import server

GB = 1024 ** 3


def write_gguf(path: Path, meta: list, tensors: list, align: int = 32) -> None:
    """A minimal GGUF v3: metadata, tensor table, zero-filled data."""
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    out = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for key, kind, value in meta:
        out += s(key) + struct.pack("<I", kind)
        if kind == 8:
            out += s(value)
        elif kind == 4:
            out += struct.pack("<I", value)
        elif kind == 9:
            inner, items = value
            out += struct.pack("<IQ", inner, len(items))
            for item in items:
                out += s(item) if inner == 8 else struct.pack("<i", item)
    offset = 0
    for name, size in tensors:
        out += s(name) + struct.pack("<I", 2) + struct.pack("<QQ", size, 1)
        out += struct.pack("<I", 0) + struct.pack("<Q", offset)
        offset += -(-size // align) * align
    out += b"\0" * (-(-len(out) // align) * align - len(out))
    path.write_bytes(bytes(out) + bytes(offset))


def moe_file(path: Path) -> None:
    tensors = [("token_embd.weight", 4096)]
    for i in range(4):
        tensors += [(f"blk.{i}.attn_q.weight", 1024),
                    (f"blk.{i}.ffn_gate_exps.weight", 2048 * (i + 1)),
                    (f"blk.{i}.ffn_up_exps.weight", 2048),
                    (f"blk.{i}.ffn_down_exps.weight", 2048),
                    (f"blk.{i}.ffn_gate_shexp.weight", 512)]
    tensors.append(("output.weight", 4096))
    write_gguf(path, [
        ("general.architecture", 8, "qwen3moe"),
        ("general.alignment", 4, 32),
        ("qwen3moe.block_count", 4, 4),
        ("qwen3moe.expert_count", 4, 8),
        ("qwen3moe.expert_used_count", 4, 2),
        # the vocabulary: the part of a real header that is big to skip
        ("tokenizer.ggml.tokens", 9, (8, [f"tok{i}" for i in range(1000)])),
        ("tokenizer.ggml.token_type", 9, (5, [1] * 1000)),
    ], tensors)


class LayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        fit._LAYOUT_CACHE.clear()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_expert_bytes_per_block_come_from_the_tensor_table(self) -> None:
        path = self.dir / "moe.gguf"
        moe_file(path)
        lay = fit.gguf_layout(str(path))
        self.assertEqual(lay["arch"], "qwen3moe")
        self.assertEqual(lay["blocks"], 4)
        self.assertEqual(lay["experts"],
                         [2048 * (i + 1) + 4096 for i in range(4)])
        # attention, shared experts, embeddings and output head
        self.assertEqual(lay["other"], 4096 + 4 * (1024 + 512) + 4096)
        self.assertEqual((lay["expert_used"], lay["expert_count"]), (2, 8))

    def test_dense_file_has_no_experts(self) -> None:
        path = self.dir / "dense.gguf"
        write_gguf(path, [("general.architecture", 8, "qwen3"),
                          ("qwen3.block_count", 4, 2)],
                   [("blk.0.ffn_up.weight", 1024),
                    ("blk.1.ffn_up.weight", 1024)])
        lay = fit.gguf_layout(str(path))
        self.assertEqual(lay["experts"], [])
        self.assertIsNone(fit.moe_plan(lay, {}, {"vram": 8 * GB}, 8192))

    def test_anything_unreadable_is_none(self) -> None:
        junk = self.dir / "junk.gguf"
        junk.write_bytes(b"not a gguf at all")
        cut = self.dir / "cut.gguf"
        moe_file(cut)
        cut.write_bytes(cut.read_bytes()[:200])
        for p in (junk, cut, self.dir / "missing.gguf"):
            self.assertIsNone(fit.gguf_layout(str(p)), p.name)

    def test_regex_matches_what_llama_cpp_moves(self) -> None:
        moved = ["blk.0.ffn_up_exps.weight", "blk.3.ffn_down_exps.weight",
                 "blk.7.ffn_gate_exps.weight", "blk.2.ffn_gate_up_exps.weight",
                 "blk.5.ffn_up_chexps.weight", "blk.1.ffn_up_exps.bias"]
        kept = ["blk.0.ffn_up_shexp.weight", "blk.0.attn_q.weight",
                "blk.0.ffn_up.weight", "token_embd.weight"]
        for name in moved:
            self.assertTrue(fit.EXPS_RE.match(name), name)
        for name in kept:
            self.assertFalse(fit.EXPS_RE.match(name), name)
        self.assertEqual(fit.EXPS_RE.match("blk.10.ffn_up_exps").group(1),
                         "10")


# Qwen3 Coder 30B A3B at Q4_K_M, in round numbers: ~1 GiB of attention,
# embeddings and head, ~0.34 GiB of experts in each of 48 blocks.
CODER = {"layers": 48, "kv_layers": 48, "kv_heads": 4, "head_dim": 128,
         "exact": True, "moe": True, "hybrid": False, "max_ctx": 262144}
CODER_LAYOUT = {"arch": "qwen3moe", "blocks": 48,
                "experts": [int(0.34 * GB)] * 48, "other": 1 * GB,
                "expert_count": 128, "expert_used": 8}
CODER_SIZE = 1 * GB + 48 * int(0.34 * GB)
CARD = {"vram": 8 * GB, "ram": 32 * GB, "bandwidth": 272}


class PlanTests(unittest.TestCase):
    def test_card_is_filled_from_the_last_block_backwards(self) -> None:
        plan = fit.moe_plan(CODER_LAYOUT, CODER, CARD, 8192, 8)
        room = 8 * GB - fit.kv_bytes(CODER, 8192, 8) - fit.OVERHEAD - GB
        on_gpu = 48 - plan["n_cpu_moe"]
        self.assertEqual(on_gpu, int(room // int(0.34 * GB)))
        self.assertLessEqual(plan["experts_gpu"], room)
        self.assertEqual(plan["layers"], 48)

    def test_none_when_even_attention_does_not_fit(self) -> None:
        self.assertIsNone(fit.moe_plan(CODER_LAYOUT, CODER,
                                       {"vram": 1 * GB}, 8192, 8))

    def test_auto_context_keeps_attention_whole_not_expert_blocks(
            self) -> None:
        # The layer rule would hold a spilling model at 8k; with its experts
        # parked in RAM a longer window costs only a block of experts.
        self.assertEqual(fit.auto_ctx(CODER_SIZE, CODER, CARD, 8), 8192)
        self.assertEqual(fit.auto_ctx(CODER_SIZE, CODER, CARD, 8,
                                      CODER_LAYOUT), 32768)

    def test_verdict_describes_the_placement(self) -> None:
        a = {**fit.assess(CODER_SIZE, CODER, CARD, 8192, 8),
             "n_cpu_moe": 33, "expert_used": 8, "expert_count": 128}
        why = fit.verdict_text(a, CARD)
        self.assertIn("--n-cpu-moe", why)
        self.assertIn("33 of 48", why)
        self.assertIn("8 of 128 experts", why)
        self.assertNotIn("only 16 of 48 layers", why)


class LaunchTests(unittest.TestCase):
    MODEL = {"name": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
             "path": "/tmp/coder.gguf", "size": CODER_SIZE,
             "partial": False}

    def launch(self, ready, layout=CODER_LAYOUT, conf=CODER, model=None,
               gpu_layers=None):
        with mock.patch.object(server.fit, "gguf_layout",
                               return_value=layout), \
                mock.patch.object(server.server, "start") as start, \
                mock.patch.object(server.server, "stop"), \
                mock.patch.object(server.server, "wait_ready",
                                  side_effect=ready):
            out = server.launch(model or self.MODEL, conf, CARD, 8192, 8,
                                gpu_layers)
        return out, [c.args for c in start.call_args_list]

    def test_moe_goes_up_with_experts_in_ram(self) -> None:
        out, starts = self.launch([True])
        self.assertEqual(len(starts), 1)
        path, ngl, ctx, flags = starts[0]
        self.assertEqual(ngl, 49)            # every block plus the head
        n = fit.moe_plan(CODER_LAYOUT, CODER, CARD, 8192, 8)["n_cpu_moe"]
        self.assertEqual(flags[-2:], ["--n-cpu-moe", str(n)])
        self.assertIn("--cache-type-k", flags)
        self.assertEqual(out["n_cpu_moe"], n)
        self.assertFalse(out["moe_fallback"])
        self.assertIn("--n-cpu-moe", out["why"])

    def test_placement_that_does_not_start_falls_back_to_layers(self) -> None:
        out, starts = self.launch([False, True])
        self.assertEqual(len(starts), 2)
        self.assertNotIn("--n-cpu-moe", starts[1][3])
        self.assertEqual(starts[1][1], out["gpu_layers"])
        self.assertLess(out["gpu_layers"], 48)
        self.assertTrue(out["moe_fallback"])
        self.assertEqual(out["n_cpu_moe"], 0)

    def test_fallback_repicks_an_auto_window_by_the_layer_rule(self) -> None:
        with mock.patch.object(server.fit, "gguf_layout",
                               return_value=CODER_LAYOUT), \
                mock.patch.object(server.server, "start") as start, \
                mock.patch.object(server.server, "stop"), \
                mock.patch.object(server.server, "wait_ready",
                                  side_effect=[False, True]):
            out = server.launch(self.MODEL, CODER, CARD, 32768, 8,
                                auto=True)
        ctxs = [c.args[2] for c in start.call_args_list]
        self.assertEqual(ctxs, [32768, 8192])
        self.assertEqual(out["ctx"], 8192)

    def test_both_failing_is_an_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not come up"):
            self.launch([False, False])

    def test_a_manual_split_is_honoured_as_given(self) -> None:
        out, starts = self.launch([True], gpu_layers=10)
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0][1], 10)
        self.assertNotIn("--n-cpu-moe", starts[0][3])

    def test_dense_model_that_fits_takes_its_output_head_too(self) -> None:
        conf = {"layers": 36, "kv_layers": 36, "kv_heads": 8,
                "head_dim": 128, "exact": True, "moe": False}
        model = {"name": "Qwen3-8B-Q4_K_M.gguf", "path": "/tmp/q.gguf",
                 "size": int(4.7 * GB), "partial": False}
        out, starts = self.launch([True], layout=None, conf=conf,
                                  model=model)
        self.assertEqual(starts[0][1], 37)   # 36 blocks + the output head
        self.assertEqual(out["gpu_layers"], 36)

    def test_partial_split_is_passed_as_is(self) -> None:
        self.assertEqual(server.ngl_flag(20, 36), 20)
        self.assertEqual(server.ngl_flag(36, 36), 37)


if __name__ == "__main__":
    unittest.main()
