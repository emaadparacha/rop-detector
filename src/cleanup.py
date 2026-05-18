"""
Clean up generated artifacts from the ROP project.

This script lives in src/ but operates on the parent project root.

Removes:
  - data/                       raw images (downloaded by prepare_data.py)
  - dataset/                    re-organized training set
  - samples/                    held-out demo images
  - models/                     trained weights and metrics (folder + contents)
  - zip information.xlsx        downloaded metadata
  - rop_dataset.zip             downloaded zip
  - _dataset_extract/           temporary extraction folder
  - outputs_smoke_heatmap.png   ad-hoc test output
  - heatmap.png                 ad-hoc test output
  - **/__pycache__              Python bytecode caches
  - **/.DS_Store                macOS junk
  - *.pyc files anywhere in the project tree

After cleanup the repo is in a "fresh clone" state. To rebuild everything
just run:

    python src/prepare_data.py
    python src/train.py

Usage (from the project root):
    python src/cleanup.py                prompts for confirmation
    python src/cleanup.py --yes          skips the prompt
    python src/cleanup.py --dry-run      show what would be removed
    python src/cleanup.py --include-venv also remove .venv/, venv/, env/, ENV/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# This file is src/cleanup.py, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Top-level files and folders to delete.
TOP_LEVEL_TARGETS: list[str] = [
    "data",
    "dataset",
    "samples",
    "models",
    "_dataset_extract",
    "zip information.xlsx",
    "rop_dataset.zip",
    "outputs_smoke_heatmap.png",
    "heatmap.png",
    "cleanup.py",  # remove the old root-level copy if it still exists
]

# Additional folders only removed when --include-venv is set.
VENV_DIRS: list[str] = [".venv", "venv", "env", "ENV"]

# Patterns to delete recursively from the entire project tree.
RECURSIVE_DIR_NAMES: set[str] = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}

RECURSIVE_FILE_GLOBS: list[str] = [
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "Thumbs.db",
]


def _human_size(path: Path) -> str:
    """Return a rough human readable size for a file or folder."""
    try:
        if path.is_file():
            size = path.stat().st_size
        else:
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return "?"
    size_f = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if size_f < 1024:
            return f"{size_f:.1f} {unit}"
        size_f /= 1024
    return f"{size_f:.1f} TB"


def _collect_targets(include_venv: bool) -> list[Path]:
    targets: list[Path] = []

    for name in TOP_LEVEL_TARGETS:
        p = PROJECT_ROOT / name
        if p.exists():
            targets.append(p)

    if include_venv:
        for name in VENV_DIRS:
            p = PROJECT_ROOT / name
            if p.exists():
                targets.append(p)

    for path in PROJECT_ROOT.rglob("*"):
        try:
            rel_parts = path.relative_to(PROJECT_ROOT).parts
        except ValueError:
            continue
        # Skip anything inside a venv unless include_venv is requested.
        if not include_venv and any(part in VENV_DIRS for part in rel_parts):
            continue
        if path.is_dir() and path.name in RECURSIVE_DIR_NAMES:
            targets.append(path)
        elif path.is_file():
            for pattern in RECURSIVE_FILE_GLOBS:
                if path.match(pattern):
                    targets.append(path)
                    break

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    deduped: list[Path] = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def _delete(path: Path) -> tuple[bool, str]:
    """Delete a file or directory. Returns (success, error_message)."""
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            return False, "not found"
        return True, ""
    except OSError as exc:
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean up ROP project artifacts.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )
    parser.add_argument(
        "--include-venv",
        action="store_true",
        help="Also delete .venv/, venv/, env/, ENV/ (default: keep them).",
    )
    args = parser.parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    targets = _collect_targets(include_venv=args.include_venv)

    if not targets:
        print("Nothing to clean. The project is already in a fresh state.")
        return 0

    print(f"\nFound {len(targets)} item(s) to remove:")
    total_size = 0
    for p in targets:
        try:
            if p.is_file():
                total_size += p.stat().st_size
            else:
                total_size += sum(q.stat().st_size for q in p.rglob("*") if q.is_file())
        except OSError:
            pass
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = p
        kind = "dir " if p.is_dir() else "file"
        print(f"  {kind}  {rel}   ({_human_size(p)})")

    print(f"\nTotal: roughly {total_size / (1024 * 1024):.1f} MB")

    if args.dry_run:
        print("\nDry run only. Nothing was deleted.")
        return 0

    if not args.yes:
        try:
            answer = input("\nDelete all of the above? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Aborted. Nothing was deleted.")
            return 1

    print("\nDeleting ...")
    failed: list[tuple[Path, str]] = []
    for p in targets:
        ok, err = _delete(p)
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = p
        if ok:
            print(f"  removed  {rel}")
        else:
            print(f"  FAILED   {rel}   ({err})")
            failed.append((p, err))

    print()
    if failed:
        print(f"Cleanup finished with {len(failed)} failure(s).")
        return 2

    print("Cleanup complete. To rebuild:")
    print("    python src/prepare_data.py")
    print("    python src/train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
