"""An app folder on a drive shared by Windows and Linux.

Set up on Windows, then opened on Linux: ./llama.cpp holds the Windows
build. Linux could not see it, so it read "not installed", and Install then
swapped the folder out whole — deleting the Windows build, which Windows
then deleted back. A build for the other system is never run (it would be
[Errno 8] Exec format error) and never installed over.
"""

from __future__ import annotations

import errno
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import engine
import server

MZ = b"MZ\x90\x00" + b"\x00" * 60
ELF = b"\x7fELF\x02\x01\x01" + b"\x00" * 57
MACHO = b"\xcf\xfa\xed\xfe" + b"\x00" * 60


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class ForeignBinaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_first_bytes_name_the_system(self) -> None:
        exe = _write(self.root / "a.exe", MZ)
        elf = _write(self.root / "a", ELF)
        mac = _write(self.root / "m", MACHO)
        with mock.patch.object(engine.platform, "system",
                               return_value="Linux"):
            self.assertTrue(engine.foreign_binary(exe))
            self.assertFalse(engine.foreign_binary(elf))
            self.assertTrue(engine.foreign_binary(mac))
        with mock.patch.object(engine.platform, "system",
                               return_value="Windows"):
            self.assertFalse(engine.foreign_binary(exe))
            self.assertTrue(engine.foreign_binary(elf))
        with mock.patch.object(engine.platform, "system",
                               return_value="Darwin"):
            self.assertFalse(engine.foreign_binary(mac))
            self.assertTrue(engine.foreign_binary(elf))

    def test_unreadable_or_unknown_is_never_called_foreign(self) -> None:
        script = _write(self.root / "s", b"#!/bin/sh\n")
        with mock.patch.object(engine.platform, "system",
                               return_value="Linux"):
            self.assertFalse(engine.foreign_binary(self.root / "missing"))
            self.assertFalse(engine.foreign_binary(script))
            self.assertFalse(engine.foreign_binary(self.root))


class SharedFolderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bin = self.root / "llama.cpp"
        self.patches = [
            mock.patch.object(engine, "BIN_DIR", self.bin),
            mock.patch.object(engine.platform, "system",
                              return_value="Linux"),
            mock.patch.object(engine.shutil, "which", return_value=None),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_windows_build_is_not_this_systems_and_is_said_so(self) -> None:
        _write(self.bin / "llama-server.exe", MZ)
        self.assertIsNone(engine.server_binary())
        self.assertEqual(engine.install_dir(self.bin, "llama-server"),
                         self.root / "llama.cpp-linux")
        note = engine.foreign_build(self.bin, "llama-server")
        self.assertIn("another operating system", note)
        self.assertIn("llama.cpp-linux", note)

    def test_a_foreign_file_under_the_native_name_is_passed_over(self) -> None:
        _write(self.bin / "llama-server", MZ)
        self.assertIsNone(engine.server_binary())
        good = _write(self.root / "llama.cpp-linux" / "llama-server", ELF)
        self.assertEqual(engine.server_binary(), good)

    def test_a_native_folder_stays_where_it_is(self) -> None:
        good = _write(self.bin / "build" / "bin" / "llama-server", ELF)
        _write(self.bin / "llama-server.exe", MZ)   # both, side by side
        self.assertEqual(engine.install_dir(self.bin, "llama-server"),
                         self.bin)
        self.assertEqual(engine.server_binary(), good)
        self.assertEqual(engine.foreign_build(self.bin, "llama-server"), "")

    def test_install_leaves_the_windows_build_alone(self) -> None:
        win = _write(self.bin / "llama-server.exe", MZ)
        _write(self.bin / "ggml-cuda.dll", MZ)
        asset = {"name": "llama-b1-bin-ubuntu-x64.zip",
                 "browser_download_url": "http://example.invalid/x.zip"}

        def fetch(task, match, p0, p1, dest_dir=None):
            _write(Path(dest_dir) / "build" / "bin" / "llama-server", ELF)
            return True

        task = engine.Task("install", "llama.cpp")
        with mock.patch.object(engine, "_resolve_release",
                               return_value=({"tag_name": "b1"}, asset,
                                             "CPU build")), \
                mock.patch.object(engine, "_fetch_zip", side_effect=fetch):
            engine.install_sync(task, {})
        self.assertEqual(win.read_bytes(), MZ)
        self.assertTrue((self.bin / "ggml-cuda.dll").exists())
        found = engine.server_binary()
        self.assertEqual(found, self.root / "llama.cpp-linux" / "build" /
                         "bin" / "llama-server")
        self.assertFalse((self.root / "llama.cpp-linux.new").exists())
        self.assertIn("another operating", "\n".join(task.lines))

    def test_status_says_why_it_is_not_installed(self) -> None:
        _write(self.bin / "llama-server.exe", MZ)
        st = server.app.test_client().get("/api/status").get_json()
        self.assertFalse(st["llama_installed"])
        self.assertIn("another operating system", st["llama_note"])

    def test_exec_format_error_is_a_sentence(self) -> None:
        srv = engine.Server(port=engine.free_port())
        err = OSError(errno.ENOEXEC, "Exec format error")
        with mock.patch.object(engine.subprocess, "Popen", side_effect=err):
            with self.assertRaises(RuntimeError) as cm:
                srv._spawn([str(self.bin / "llama-server"), "-m", "x"], "x")
        self.assertIn("another operating system", str(cm.exception))
        self.assertNotIn("Errno", str(cm.exception))
        self.assertIn("another operating system", "\n".join(srv.tail()))


if __name__ == "__main__":
    unittest.main()
