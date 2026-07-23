#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path


ARCHIVE_ROOT = "0shotProt-code"
ROOT_FILES = (
    ".python-version",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "pyrightconfig.json",
    "uv.lock",
)
SOURCE_DIRECTORIES = (
    "assets/prosst_structure_tokens",
    "bash",
    "datasets",
    "scripts",
    "src/prospero",
    "tests",
)
IGNORED_NAMES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build_anonymous_archive.py",
}
IGNORED_SUFFIXES = {
    ".pyc",
    ".pyo",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".csv",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_TEXT = (
    "/home/",
    "git@",
    "lamsade",
    "marcosbolanos",
    "mbolanos",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and audit the anonymous 0shotProt code archive."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/0shotProt-code.zip"),
        help="Destination ZIP path.",
    )
    return parser.parse_args()


def ignored(path: Path) -> bool:
    return any(part in IGNORED_NAMES for part in path.parts) or (
        path.suffix in IGNORED_SUFFIXES
    )


def copy_directory(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if ignored(relative) or not path.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def audit_tree(root: Path) -> None:
    leaked = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        relative_text = relative.as_posix().lower()
        for forbidden in FORBIDDEN_TEXT:
            if forbidden in relative_text:
                leaked.append(f"{relative}: forbidden path text {forbidden!r}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for forbidden in FORBIDDEN_TEXT:
            if forbidden in text:
                leaked.append(f"{relative}: forbidden text {forbidden!r}")
    if leaked:
        details = "\n".join(leaked)
        raise RuntimeError(f"Anonymous archive audit failed:\n{details}")


def write_manifest(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text(
        "\n".join(entries) + "\n", encoding="utf-8"
    )


def write_zip(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(f"{output.suffix}.tmp")
    with zipfile.ZipFile(
        temporary_output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(ARCHIVE_ROOT) / path.relative_to(root)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 7, 1, 0, 0, 0))
            mode = 0o755 if path.suffix == ".sh" else 0o644
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    temporary_output.replace(output)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output = args.output
    if not output.is_absolute():
        output = repo_root / output

    with tempfile.TemporaryDirectory(prefix="0shotprot-archive-") as tmp:
        staging = Path(tmp) / ARCHIVE_ROOT
        staging.mkdir()

        for relative in ROOT_FILES:
            source = repo_root / relative
            if not source.is_file():
                raise FileNotFoundError(f"Required archive file is missing: {source}")
            shutil.copy2(source, staging / relative)

        for relative in SOURCE_DIRECTORIES:
            source = repo_root / relative
            if not source.is_dir():
                raise FileNotFoundError(
                    f"Required archive directory is missing: {source}"
                )
            copy_directory(source, staging / relative)

        token_files = list(
            (staging / "assets/prosst_structure_tokens").rglob("*.fasta")
        )
        if len(token_files) != 8:
            raise RuntimeError(
                f"Expected 8 benchmark structure-token files, found {len(token_files)}."
            )

        audit_tree(staging)
        write_manifest(staging)
        write_zip(staging, output)

    size_mib = output.stat().st_size / (1024 * 1024)
    print(f"Built {output} ({size_mib:.1f} MiB)")


if __name__ == "__main__":
    main()
