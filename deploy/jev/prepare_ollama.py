"""Install pinned user-local Ollama and checksum-verified official embedding blobs."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

VERSION = "0.11.11"
MODEL = "qwen3-embedding:0.6b"
ROOT = Path.home() / "ozy"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def pause_before_retry():
    time.sleep(2)


def download(url, path, expected=None):
    if path.exists() and (expected is None or digest(path) == expected):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    command = ["curl", "-fL", "--retry", "3", "--connect-timeout", "20", "--max-time", "5400",
               "-C", "-", "-o", str(part), url]
    for attempt in range(4):
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            break
        if attempt == 3:
            raise subprocess.CalledProcessError(result.returncode, command)
        print("Retrying download from retained partial file", attempt + 1, flush=True)
        pause_before_retry()
    if expected is not None and digest(part) != expected:
        raise ValueError("Downloaded checksum mismatch")
    part.replace(path)


def main():
    root = ROOT / "tools"
    root.mkdir(parents=True, exist_ok=True)
    # Pinned official release metadata; avoid unauthenticated GitHub API rate limits
    # on shared compute-node egress. The archive fingerprint is saved after download.
    asset = {"size": 1908755362, "browser_download_url":
             f"https://github.com/ollama/ollama/releases/download/v{VERSION}/ollama-linux-amd64.tgz"}
    archive = root / f"ollama-{VERSION}-linux-amd64.tgz"
    checksum = (asset.get("digest") or "").removeprefix("sha256:") or None
    download(asset["browser_download_url"], archive, checksum)
    if archive.stat().st_size != asset["size"]:
        raise ValueError("Ollama archive size mismatch")
    install = root / f"ollama-{VERSION}"
    if not (install / "install-complete.json").exists():
        install.mkdir(exist_ok=True)
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(install)], check=True)
        if not (install / "bin/ollama").is_file():
            raise ValueError("Ollama executable missing")
        (install / "install-complete.json").write_text(json.dumps({"version": VERSION, "sha256": digest(archive)}))
    manifest = get_json("https://registry.ollama.ai/v2/library/qwen3-embedding/manifests/0.6b")
    models = ROOT / "ollama-models"
    for blob in [manifest["config"], *manifest["layers"]]:
        name = blob["digest"]
        path = models / "blobs" / name.replace(":", "-")
        download("https://registry.ollama.ai/v2/library/qwen3-embedding/blobs/" + name,
                 path, name.split(":")[1])
        if path.stat().st_size != blob["size"]:
            raise ValueError("Embedding blob size mismatch")
    path = models / "manifests/registry.ollama.ai/library/qwen3-embedding/0.6b"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, separators=(",", ":")))
    receipt = {"version": VERSION, "binary": str(install / "bin/ollama"), "models": str(models),
               "model": MODEL, "manifest": manifest, "archive_sha256": digest(archive)}
    (root / "ollama-embedding-ready.json").write_text(json.dumps(receipt, indent=2))
    print("OLLAMA_PREPARED", VERSION, MODEL, flush=True)


if __name__ == "__main__":
    main()
