from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_URL = "https://github.com/fastapi/fastapi.git"
DOCS_SUBDIR_CANDIDATES = [
    Path("docs/en/docs"),
    Path("docs/docs"),  # fallback for older layouts
]


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True, capture_output=True)


def _is_empty_dir(path: Path) -> bool:
    return not any(path.iterdir())


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch FastAPI docs markdown into a local domain data folder.")
    parser.add_argument("--out", default="domains/fastapi_docs/data", help="Output directory for markdown files")
    parser.add_argument("--ref", default="master", help="Git ref/branch/tag (default: master)")
    parser.add_argument(
        "--commit",
        default=None,
        help="Exact commit SHA to checkout (more reproducible than --ref).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output dir")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not _is_empty_dir(out_dir) and not args.overwrite:
        print(f"ERROR: output dir is not empty: {out_dir} (use --overwrite)", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="fastapi-docs-") as td:
        clone_dir = Path(td) / "fastapi"
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

        md_files = sorted(docs_root.rglob("*.md"))
        if not md_files:
            print(f"ERROR: no markdown files found under {docs_root}", file=sys.stderr)
            return 2

        for src in md_files:
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
            ),
            encoding="utf-8",
        )

        print(f"Copied {len(md_files)} files to {out_dir}")
        print(f"Source commit: {commit}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
