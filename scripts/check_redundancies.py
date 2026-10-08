"""Report exact duplicate files, duplicate Python declarations, and stale Markdown links.

This is a review aid only: it never deletes or rewrites files.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".cache", ".ray",
    "build", "dist", "node_modules",
}
SKIP_SUFFIXES = {".pyc", ".pt", ".pth"}
MAX_HASH_SIZE = 64 * 1024 * 1024
MARKDOWN_LINK = re.compile(r"(?<!!)(?:\[[^\]]*\])\(([^)]+)\)")


def files_to_scan(include_outputs: bool) -> list[Path]:
    files: list[Path] = []
    for current, dirs, names in os.walk(ROOT):
        current_path = Path(current)
        relative = current_path.relative_to(ROOT)
        dirs[:] = [
            name for name in dirs
            if name not in SKIP_DIRS
            and (
                include_outputs
                or (relative / name).as_posix() not in {"results/experiments", "results/summary"}
            )
        ]
        for name in names:
            path = current_path / name
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            try:
                if path.stat().st_size <= MAX_HASH_SIZE:
                    files.append(path)
            except OSError:
                continue
    return files


def duplicate_files(paths: list[Path]) -> list[list[Path]]:
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            continue
        by_hash[digest.hexdigest()].append(path)
    return [group for group in by_hash.values() if len(group) > 1]


def duplicate_python_names(paths: list[Path]) -> list[tuple[Path, str, int, int]]:
    findings: list[tuple[Path, str, int, int]] = []
    for path in paths:
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as error:
            print(f"Could not inspect Python file {path.relative_to(ROOT)}: {error}")
            continue
        declared: dict[str, list[int]] = defaultdict(list)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                declared[node.name].append(node.lineno)
        for name, lines in declared.items():
            if len(lines) > 1:
                findings.append((path, name, lines[0], lines[-1]))
    return findings


def broken_markdown_links(paths: list[Path]) -> list[tuple[Path, int, str]]:
    findings: list[tuple[Path, int, str]] = []
    for markdown in paths:
        if markdown.suffix.lower() not in {".md", ".markdown"}:
            continue
        try:
            lines = markdown.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line_number, line in enumerate(lines, start=1):
            for match in MARKDOWN_LINK.finditer(line):
                raw_target = match.group(1).strip()
                if raw_target.startswith("<") and ">" in raw_target:
                    target = raw_target[1:raw_target.index(">")]
                else:
                    target = raw_target.split(maxsplit=1)[0]
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith("#"):
                    continue
                relative_target = unquote(parsed.path)
                if not relative_target:
                    continue
                target_path = (
                    ROOT / relative_target.lstrip("/")
                    if relative_target.startswith("/")
                    else markdown.parent / relative_target
                )
                if not target_path.exists():
                    findings.append((markdown, line_number, target))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true",
        help="exit with status 1 when any possible redundancy or stale link is found",
    )
    parser.add_argument(
        "--include-outputs", action="store_true",
        help="also scan results/experiments (excluded by default because they contain generated logs)",
    )
    args = parser.parse_args()

    paths = files_to_scan(args.include_outputs)
    duplicates = duplicate_files(paths)
    duplicate_names = duplicate_python_names(paths)
    stale_links = broken_markdown_links(paths)

    print(f"Scanned {len(paths)} files under {ROOT}")
    print("Possible duplicate files:")
    if duplicates:
        for group in duplicates:
            print("  " + " == ".join(str(path.relative_to(ROOT)) for path in group))
    else:
        print("  none")

    print("Repeated top-level Python declarations:")
    if duplicate_names:
        for path, name, first, last in duplicate_names:
            print(f"  {path.relative_to(ROOT)}: {name} (lines {first} and {last})")
    else:
        print("  none")

    print("Broken local Markdown links:")
    if stale_links:
        for path, line, target in stale_links:
            print(f"  {path.relative_to(ROOT)}:{line}: {target}")
    else:
        print("  none")

    finding_count = len(duplicates) + len(duplicate_names) + len(stale_links)
    print(f"Review {finding_count} finding(s); nothing was changed.")
    return 1 if args.strict and finding_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
