"""Fetch the official MSRA HRNet ImageNet checkpoint.

Target file:
    data/external/pretrained/hrnetv2_w32_imagenet.pth

The official ckpt is hosted on OneDrive / Google Drive. We try, in order:

  1. If the file already exists at the target path, verify size + (optional)
     SHA256 and exit.
  2. OneDrive shared-file direct download via the public
     ``api.onedrive.com/v1.0/shares/u!<base64-link>/root/content`` endpoint.
     This works for files shared via a 1drv.ms link without interactive auth.
  3. Google Drive via ``gdown`` (if installed). Handles the >100 MB virus-scan
     confirmation page that plain urllib cannot.
  4. Fall back to manual-download instructions pointing at the official HRNet
     README, plus the recorded size/SHA256 (when known).

We intentionally do NOT pull from timm's HuggingFace mirror
(``timm/hrnet_w32.ms_in1k``) — those weights are renamed to timm's key scheme
and will fail our ``_remap_key`` (see src/models/hrnet_pretrained.py).

See ``docs/project_decisions.md`` section 1 (2026-05-04 revision).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Source links from the official HRNet-Image-Classification README, HRNet-W32-C row.
ONEDRIVE_SHORT_URL = "https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w"
GDRIVE_FILE_ID = "1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38"

# SHA256 of the MSRA HRNet-W32 ImageNet checkpoint stored at DEFAULT_DEST.
EXPECTED_SHA256: str | None = "19bc083708bb8d873211e50d85d56344c10290c6e8b564c813fdde09645c4c1c"
# Soft sanity check: the official w32 pth is much larger than an HTML error
# page from a CDN/drive host, which is typically only a few KB.
MIN_EXPECTED_BYTES = 50 * 1024 * 1024

DEFAULT_DEST = Path("data/external/pretrained/hrnetv2_w32_imagenet.pth")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _onedrive_direct_url(short_url: str) -> str:
    """Convert a 1drv.ms shared link to the public binary-content endpoint.

    Works for files shared without sign-in; no auth header required.
    """
    encoded = base64.urlsafe_b64encode(short_url.encode("utf-8")).decode("ascii").rstrip("=")
    return f"https://api.onedrive.com/v1.0/shares/u!{encoded}/root/content"


def _try_urllib_download(url: str, dest: Path) -> bool:
    print(f"[download] trying {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (download_hrnet_imagenet)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"[download] failed: {e}")
        return False
    return _looks_like_real_ckpt(dest)


def _try_gdown(file_id: str, dest: Path) -> bool:
    try:
        import gdown  # type: ignore
    except ImportError:
        print("[download] gdown not installed; skipping Google Drive attempt "
              "(`pip install gdown` to enable)")
        return False
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[download] trying gdown for {url}")
    try:
        gdown.download(url, str(dest), quiet=False)
    except Exception as e:  # gdown raises a wide range of exceptions
        print(f"[download] gdown failed: {e}")
        return False
    return _looks_like_real_ckpt(dest)


def _looks_like_real_ckpt(dest: Path) -> bool:
    if not dest.exists():
        return False
    size = dest.stat().st_size
    if size < MIN_EXPECTED_BYTES:
        print(f"[download] file at {dest} is only {size} bytes (<{MIN_EXPECTED_BYTES}); "
              "looks like an error page, discarding")
        dest.unlink(missing_ok=True)
        return False
    return True


def _print_manual_instructions(dest: Path) -> None:
    print(
        "\n[download] Could not fetch the HRNet ImageNet checkpoint automatically.\n"
        "Manual steps:\n"
        "  1. Open the HRNet-Image-Classification README:\n"
        "       https://github.com/HRNet/HRNet-Image-Classification\n"
        "  2. Download the HRNet-W32-C row's checkpoint from the OneDrive or\n"
        "     Google Drive link (file id 1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38).\n"
        f"  3. Save the file to: {dest}\n"
        "  4. Re-run this script to verify.\n"
    )
    if EXPECTED_SHA256:
        print(f"Expected SHA256: {EXPECTED_SHA256}\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = p.parse_args()

    dest: Path = args.dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[download] already present: {dest} ({dest.stat().st_size} bytes)")
        if EXPECTED_SHA256:
            got = _sha256(dest)
            if got != EXPECTED_SHA256:
                print(f"[download] SHA256 mismatch: got {got}, expected {EXPECTED_SHA256}")
                return 2
            print("[download] SHA256 OK")
        return 0

    if _try_urllib_download(_onedrive_direct_url(ONEDRIVE_SHORT_URL), dest):
        print(f"[download] saved to {dest} via OneDrive")
    elif _try_gdown(GDRIVE_FILE_ID, dest):
        print(f"[download] saved to {dest} via Google Drive (gdown)")
    else:
        _print_manual_instructions(dest)
        return 1

    if EXPECTED_SHA256:
        got = _sha256(dest)
        if got != EXPECTED_SHA256:
            print(f"[download] SHA256 mismatch: got {got}, expected {EXPECTED_SHA256}")
            dest.unlink(missing_ok=True)
            return 2
        print("[download] SHA256 OK")
    print(f"[download] {dest.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
