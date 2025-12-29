from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_URL = "https://github.com/reactjs/react.dev.git"
DOCS_SUBDIR_CANDIDATES = [
    Path("src/content"),
]
DEFAULT_INCLUDE_SUBDIRS = [
    "learn",
    "reference",
    "community",
    "errors",
    "warnings",
]
DEFAULT_ROOT_FILES = [
    "index.md",
    "versions.md",
]
ALLOWED_EXTS = {".md", ".mdx"}


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True, capture_output=True)


def _is_empty_dir(path: Path) -> bool:
    return not any(path.iterdir())


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch React docs markdown into a local domain data folder.")
    parser.add_argument("--out", default="domains/react_docs/data", help="Output directory for markdown files")
    parser.add_argument("--ref", default="main", help="Git ref/branch/tag (default: main)")
    parser.add_argument(
        "--commit",
        default=None,
        help="Exact commit SHA to checkout (more reproducible than --ref).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Subdirectory under src/content to include (repeatable). Default: learn, reference, community, errors, warnings.",
    )
    parser.add_argument("--include-blog", action="store_true", help="Also include src/content/blog")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output dir")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not _is_empty_dir(out_dir) and not args.overwrite:
        print(f"ERROR: output dir is not empty: {out_dir} (use --overwrite)", file=sys.stderr)
        return 2

    include_subdirs = [s.strip() for s in (args.include or []) if s.strip()]
    if not include_subdirs:
        include_subdirs = list(DEFAULT_INCLUDE_SUBDIRS)
    if args.include_blog and "blog" not in include_subdirs:
        include_subdirs.append("blog")

    with tempfile.TemporaryDirectory(prefix="react-docs-") as td:
        clone_dir = Path(td) / "react-dev"
        try:
            if args.commit:
                clone_dir.mkdir(parents=True, exist_ok=True)
                _run(["git", "init"], cwd=clone_dir)
                _run(["git", "remote", "add", "origin", REPO_URL], cwd=clone_dir)
                try:
                    _run(["git", "fetch", "--depth", "1", "origin", args.commit], cwd=clone_dir)
                except subprocess.CalledProcessError:
                    _run(["git", "fetch", "origin", args.commit], cwd=clone_dir)
                _run(["git", "checkout", "FETCH_HEAD"], cwd=clone_dir)
            else:
                _run(["git", "clone", "--depth", "1", "--branch", args.ref, REPO_URL, str(clone_dir)])
        except subprocess.CalledProcessError as exc:
            print(exc.stderr or exc.stdout, file=sys.stderr)
            return 2

        docs_root: Path | None = None
        for cand in DOCS_SUBDIR_CANDIDATES:
            p = clone_dir / cand
            if p.exists():
                docs_root = p
                break
        if docs_root is None:
            print("ERROR: could not locate docs folder in cloned repo.", file=sys.stderr)
            return 2

        try:
            commit = _run(["git", "rev-parse", "HEAD"], cwd=clone_dir).stdout.strip()
        except subprocess.CalledProcessError:
            commit = "unknown"

        files: list[Path] = []
        for root_file in DEFAULT_ROOT_FILES:
            p = docs_root / root_file
            if p.exists() and p.suffix.lower() in ALLOWED_EXTS:
                files.append(p)

        for subdir in include_subdirs:
            p = docs_root / subdir
            if not p.exists():
                print(f"WARN: missing docs subdir: {p}", file=sys.stderr)
                continue
            for src in sorted(p.rglob("*")):
                if not src.is_file():
                    continue
                if src.suffix.lower() in ALLOWED_EXTS:
                    files.append(src)

        files = sorted({p.resolve() for p in files})
        if not files:
            print(f"ERROR: no docs files found under {docs_root} (allowed: {sorted(ALLOWED_EXTS)})", file=sys.stderr)
            return 2

        for src in files:
            rel = src.relative_to(docs_root)
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

        (out_dir / "FETCH_META.txt").write_text(
            (
                f"source_repo={REPO_URL}\n"
                f"ref={(args.ref if not args.commit else '')}\n"
                f"commit={commit}\n"
                f"source_path={docs_root.relative_to(clone_dir).as_posix()}\n"
                f"included_subdirs={','.join(include_subdirs)}\n"
            ),
            encoding="utf-8",
        )

        print(f"Copied {len(files)} files to {out_dir}")
        print(f"Source commit: {commit}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

