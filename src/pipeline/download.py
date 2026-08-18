"""
src/pipeline/download.py — Q1: Download raw datasets
=====================================================

Strategy (in order of preference):
  1. Skip if already downloaded/extracted
  2. Try huggingface_hub (respects HF_TOKEN env variable)
  3. Try direct URL download (requests with browser User-Agent)
  4. If all fail → print clear manual download instructions and exit

MIND-small manual download:
  Visit https://msnews.github.io  →  agree to terms  →  download
  Place MINDsmall_train.zip and MINDsmall_dev.zip inside data/raw/mind/

EB-NeRD demo manual download:
  Visit https://recsys.eb.dk/dataset/  →  download ebnerd_demo.zip
  Place it inside data/raw/ebnerd/
"""

import logging
import os
import sys
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

# ── MIND-small URLs ───────────────────────────────────────────────────────────
# Official Azure blob (may require Microsoft portal visit to unlock)
MIND_URLS = {
    "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
    "dev":   "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
}
# Fallback: HuggingFace (requires HF_TOKEN or `huggingface-cli login`)
MIND_HF_REPO = "yjw1029/MIND"
MIND_HF_FILES = {
    "train": "MINDsmall_train.zip",
    "dev":   "MINDsmall_dev.zip",
}

# ── EB-NeRD demo URL ──────────────────────────────────────────────────────────
EBNERD_URLS = {
    "demo": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip archive if not already extracted."""
    marker = dest_dir / ".extracted"
    if marker.exists():
        log.info(f"  [skip] {dest_dir.name} already extracted")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    marker.touch()
    log.info(f"  Extracted to {dest_dir}")


def _try_requests_download(url: str, dest: Path) -> bool:
    """Try downloading with requests + browser User-Agent. Returns True on success."""
    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; research-pipeline/1.0)"
        }
        log.info(f"  Trying requests download: {url}")
        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            if r.status_code != 200:
                log.warning(f"  requests got HTTP {r.status_code}")
                return False
            total = int(r.headers.get("content-length", 0))
            dest.parent.mkdir(parents=True, exist_ok=True)
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (5 * 1024 * 1024) < 8192:
                        pct = downloaded * 100 // total
                        print(f"\r    {pct}%  ({downloaded/1e6:.1f} MB)", end="", flush=True)
            print()
            return True
    except Exception as e:
        log.warning(f"  requests download failed: {e}")
        return False


def _try_hf_download(hf_repo: str, filename: str, dest: Path) -> bool:
    """Try downloading via huggingface_hub. Returns True on success."""
    try:
        from huggingface_hub import hf_hub_download
        log.info(f"  Trying huggingface_hub: {hf_repo}/{filename}")
        local = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            repo_type="dataset",
        )
        import shutil
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, dest)
        return True
    except Exception as e:
        log.warning(f"  huggingface_hub download failed: {e}")
        return False


def _manual_instructions_mind(raw_dir: Path):
    raw_str = str(raw_dir)
    msg = (
        "\n"
        "=" * 68 + "\n"
        "  MIND-small: Manual Download Required\n"
        "=" * 68 + "\n"
        "\n"
        "  The MIND dataset requires accepting Microsoft Research terms.\n"
        "\n"
        "  Steps:\n"
        "  1. Visit  https://msnews.github.io\n"
        "  2. Click 'Get Dataset' and agree to the license terms\n"
        "  3. Download:  MINDsmall_train.zip\n"
        "               MINDsmall_dev.zip\n"
        "  4. Place both zip files in:\n"
        f"       {raw_str}\n"
        "  5. Re-run:  python build_pipeline.py\n"
        "\n"
        "  Alternatively, if you have a HuggingFace account:\n"
        "    pip install huggingface_hub\n"
        "    huggingface-cli login   (paste your HF token)\n"
        "    python build_pipeline.py\n"
        "\n"
        "=" * 68 + "\n"
    )
    print(msg)


def _manual_instructions_ebnerd(raw_dir: Path):
    raw_str = str(raw_dir)
    msg = (
        "\n"
        "=" * 68 + "\n"
        "  EB-NeRD demo: Manual Download Required\n"
        "=" * 68 + "\n"
        "\n"
        "  Steps:\n"
        "  1. Visit  https://recsys.eb.dk/dataset/\n"
        "  2. Download:  ebnerd_demo.zip\n"
        "  3. Place it in:\n"
        f"       {raw_str}\n"
        "  4. Re-run:  python build_pipeline.py\n"
        "\n"
        "=" * 68 + "\n"
    )
    print(msg)


# ── Public API ────────────────────────────────────────────────────────────────

def download_mind(raw_dir: Path) -> None:
    """Download and extract MIND-small train and dev splits."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for split in ("train", "dev"):
        zip_path = raw_dir / f"MINDsmall_{split}.zip"
        extract_dir = raw_dir / split

        # Skip if already extracted
        if (extract_dir / ".extracted").exists():
            log.info(f"  [skip] MIND {split} already extracted")
            continue

        # Skip download if zip already present
        if zip_path.exists():
            log.info(f"  [skip] {zip_path.name} already downloaded")
        else:
            url = MIND_URLS[split]
            # Try 1: direct requests download
            ok = _try_requests_download(url, zip_path)
            # Try 2: huggingface_hub
            if not ok:
                ok = _try_hf_download(MIND_HF_REPO, MIND_HF_FILES[split], zip_path)
            if not ok:
                all_ok = False
                continue

        _extract_zip(zip_path, extract_dir)

    if not all_ok:
        _manual_instructions_mind(raw_dir)
        # Don't sys.exit — let EB-NeRD continue; pipeline will handle missing data gracefully
        log.warning("MIND download incomplete. Please follow manual instructions above.")
    else:
        log.info("MIND download complete.")


def download_ebnerd(raw_dir: Path) -> None:
    """Download and extract EB-NeRD demo bundle."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ebnerd_demo.zip"
    extract_dir = raw_dir / "demo"

    if (extract_dir / ".extracted").exists():
        log.info("  [skip] EB-NeRD demo already extracted")
        return

    if zip_path.exists():
        log.info(f"  [skip] {zip_path.name} already downloaded")
    else:
        ok = _try_requests_download(EBNERD_URLS["demo"], zip_path)
        if not ok:
            _manual_instructions_ebnerd(raw_dir)
            log.warning("EB-NeRD download incomplete. Please follow manual instructions above.")
            return

    _extract_zip(zip_path, extract_dir)
    log.info("EB-NeRD download complete.")
