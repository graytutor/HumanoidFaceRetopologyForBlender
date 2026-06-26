#!/usr/bin/env python3
"""Build an installable HFR release zip.

The release zip hides DevOption by setting HFR_SHOW_DEV_OPTIONS = False.
Run from the repository root:

    python tools/build_release.py
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import zipfile

EXCLUDE_DIRS = {"__pycache__"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "addon" / "HumanoidFaceRetopology"
DIST = ROOT / "dist"
VERSION = "v1_0_1"
PACKAGE_NAME = "HumanoidFaceRetopology"


def copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    DIST.mkdir(exist_ok=True)
    build_root = DIST / PACKAGE_NAME
    copytree_clean(SRC, build_root)

    init_py = build_root / "__init__.py"
    text = init_py.read_text(encoding="utf-8")
    text = re.sub(r"HFR_SHOW_DEV_OPTIONS\s*=\s*(True|False)", "HFR_SHOW_DEV_OPTIONS = False", text)
    init_py.write_text(text, encoding="utf-8")

    zip_path = DIST / f"HumanoidFaceRetopology_{VERSION}_release.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in build_root.rglob("*"):
            if any(part in EXCLUDE_DIRS for part in path.parts):
                continue
            if path.suffix in EXCLUDE_SUFFIXES:
                continue
            zf.write(path, path.relative_to(DIST))

    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()
