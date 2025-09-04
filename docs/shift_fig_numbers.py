#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys
from typing import List, Tuple

# Fixed configuration
ROOT_DIR = pathlib.Path("./content/user_guide")
EXTENSIONS = [".md", ".rst", ".txt", ".adoc"]  # file types to process

# Regex patterns for figure references
FIGURE_RX = re.compile(r"\bFigure\s+(\d+)\b")
FIGSLUG_RX = re.compile(r"\bfig-(\d+)\b", re.IGNORECASE)

# Optional: skip code fences ```...```
CODE_FENCE_RX = re.compile(r"^```")


def find_numbers_in_text(text: str) -> List[int]:
    """Extract all figure numbers from text (both 'Figure X' and 'fig-X')."""
    nums = [int(m.group(1)) for m in FIGURE_RX.finditer(text)]
    nums += [int(m.group(1)) for m in FIGSLUG_RX.finditer(text)]
    return nums


def list_files(root: pathlib.Path, exts: List[str]) -> List[pathlib.Path]:
    """Return all files under root with the given extensions."""
    exts_lower = tuple(e.lower() for e in exts)
    return [
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_lower
    ]


def compute_global_max(files: List[pathlib.Path]) -> int:
    """Find the maximum figure number across all files."""
    gmax = 0
    for f in files:
        try:
            txt = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                txt = f.read_text(encoding="utf-8-sig")
            except Exception:
                continue
        nums = find_numbers_in_text(txt)
        if nums:
            gmax = max(gmax, max(nums))
    return gmax


def renumber_text_desc(
    text: str, start_at: int, count: int, skip_code_blocks: bool
) -> Tuple[str, int]:
    """
    Replace figure references in descending order to avoid collisions:
      - 'Figure x' -> 'Figure x+count'
      - 'fig-x'    -> 'fig-(x+count)'
    Only numbers >= start_at are shifted.
    Returns (new_text, num_replacements).
    """
    nums = set(find_numbers_in_text(text))
    if not nums:
        return text, 0
    local_candidates = [n for n in nums if n >= start_at]
    if not local_candidates:
        return text, 0
    local_max = max(local_candidates)
    replacements = 0

    def replace_in_segment(seg: str) -> str:
        nonlocal replacements
        s = seg
        for x in range(local_max, start_at - 1, -1):
            s, n1 = re.subn(rf"\bFigure\s+{x}\b", f"Figure {x+count}", s)
            s, n2 = re.subn(rf"\bfig-{x}\b", f"fig-{x+count}", s, flags=re.IGNORECASE)
            replacements += n1 + n2
        return s

    if not skip_code_blocks:
        return replace_in_segment(text), replacements

    # Skip fenced code blocks ```...```
    lines = text.splitlines(keepends=True)
    in_fence = False
    buf = []
    for ln in lines:
        if CODE_FENCE_RX.match(ln.strip()):
            in_fence = not in_fence
            buf.append(ln)
        else:
            if in_fence:
                buf.append(ln)
            else:
                buf.append(replace_in_segment(ln))
    return "".join(buf), replacements


def process_file(
    path: pathlib.Path, start_at: int, count: int, dry_run: bool, skip_code_blocks: bool
) -> int:
    """Process one file: shift figure numbers if needed."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    original = text
    new_text, nrep = renumber_text_desc(text, start_at, count, skip_code_blocks)
    if nrep > 0 and not dry_run:
        # Backup original
        path.with_suffix(path.suffix + ".bak").write_text(original, encoding="utf-8")
        path.write_text(new_text, encoding="utf-8")
    return nrep


def main():
    ap = argparse.ArgumentParser(
        description="Shift figure references ('Figure X' and 'fig-X') in ./content/user_guide."
    )
    ap.add_argument(
        "--insert-at",
        type=int,
        required=True,
        help="First figure number to shift (inclusive).",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=1,
        help="How many new figures you want to insert. Default = 1.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would change, do not modify files.",
    )
    ap.add_argument(
        "--skip-code-blocks",
        action="store_true",
        help="Do not renumber inside fenced code blocks ```...```.",
    )
    ap.add_argument(
        "--print-max",
        action="store_true",
        help="Print the maximum figure number found and exit.",
    )
    args = ap.parse_args()

    files = list_files(ROOT_DIR, EXTENSIONS)
    if not files:
        print(
            "No files found in ./content/user_guide with given extensions.",
            file=sys.stderr,
        )
        sys.exit(1)

    gmax = compute_global_max(files)
    if args.print_max:
        print(f"Global maximum figure number: {gmax}")
        sys.exit(0)

    if gmax < args.insert_at:
        print(
            f"Global maximum ({gmax}) < insert-at ({args.insert_at}). Nothing to shift."
        )
        sys.exit(0)

    total_repl = 0
    for f in files:
        nrep = process_file(
            f, args.insert_at, args.count, args.dry_run, args.skip_code_blocks
        )
        if nrep:
            print(f"[{'DRY' if args.dry_run else 'OK'}] {f}: {nrep} replacements")
            total_repl += nrep

    print(f"Total replacements: {total_repl}")
    if args.dry_run:
        print("Dry run: no files were modified. Remove --dry-run to apply changes.")


if __name__ == "__main__":
    main()
