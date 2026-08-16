"""Fetch the MSU/ORNL Power System Attack Dataset archives, verify SHA256, extract.

Hashes were recorded in configs/data.yaml on first download (2026-08-13). Any
mismatch on re-download fails loud — it means the upstream file changed and the
recorded provenance no longer holds.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import py7zr
import requests

from qgridbench.utils.run_tracking import REPO_ROOT, get_logger, load_yaml

log = get_logger(__name__)

VARIANTS = ("binary", "triple", "multiclass")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    log.info("downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def ensure_raw_data(config_path: Path | None = None) -> dict[str, Path]:
    """Download (if missing), verify, and extract all variants. Returns variant->dir."""
    cfg = load_yaml(config_path or REPO_ROOT / "configs" / "data.yaml")
    raw_dir = REPO_ROOT / cfg["paths"]["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    base = cfg["source"]["base_url"]

    out: dict[str, Path] = {}
    for name, spec in cfg["source"]["files"].items():
        archive = raw_dir / spec["archive"]
        if not archive.exists():
            fetch(f"{base}/{spec['archive']}", archive)
        digest = sha256_file(archive)
        if digest != spec["sha256"]:
            raise RuntimeError(
                f"SHA256 mismatch for {archive.name}: got {digest}, "
                f"expected {spec['sha256']} (recorded {cfg['source']['retrieved']}). "
                "Upstream file changed — re-verify against the dataset README before use."
            )
        log.info("verified %s (%s)", archive.name, digest[:12])
        if name in VARIANTS:
            extract_dir = raw_dir / name
            if not extract_dir.exists() or not any(extract_dir.glob("*.csv")):
                extract_dir.mkdir(exist_ok=True)
                log.info("extracting %s -> %s", archive.name, extract_dir)
                with py7zr.SevenZipFile(archive) as z:
                    z.extractall(extract_dir)
            out[name] = extract_dir
    return out


if __name__ == "__main__":
    dirs = ensure_raw_data()
    for variant, d in dirs.items():
        n = len(list(d.rglob("*.csv")))
        log.info("%s: %d csv files in %s", variant, n, d)
    if not dirs:
        sys.exit("no variants extracted")
