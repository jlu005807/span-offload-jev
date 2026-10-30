from hashlib import sha256
import subprocess

import pytest

from deploy.jev import prepare_ollama


def test_partial_transfer_retries_with_existing_partial_file(tmp_path, monkeypatch):
    target = tmp_path / "archive"
    part = tmp_path / "archive.part"
    calls = []
    def run(args, check):
        calls.append(args)
        assert args[args.index("-C") + 1] == "-"
        if len(calls) == 1:
            part.write_bytes(b"ab")
            if check:
                raise subprocess.CalledProcessError(18, args)
            return subprocess.CompletedProcess(args, 18)
        assert part.read_bytes() == b"ab"
        part.write_bytes(b"abcd")
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(prepare_ollama.subprocess, "run", run)
    monkeypatch.setattr(prepare_ollama, "pause_before_retry", lambda: None, raising=False)
    prepare_ollama.download("https://example.invalid/archive", target, sha256(b"abcd").hexdigest())
    assert target.read_bytes() == b"abcd" and len(calls) == 2


def test_checksum_failure_does_not_publish_download(tmp_path, monkeypatch):
    target = tmp_path / "archive"
    def run(args, check):
        (tmp_path / "archive.part").write_bytes(b"wrong")
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(prepare_ollama.subprocess, "run", run)
    with pytest.raises(ValueError, match="checksum"):
        prepare_ollama.download("https://example.invalid/archive", target, sha256(b"right").hexdigest())
    assert not target.exists()
