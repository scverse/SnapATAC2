from __future__ import annotations

import argparse
import sys
from pathlib import Path

import snapatac2

from auto_api.extractor import extract_api_docs
from auto_api.markdown import render_markdown


PLACEHOLDER = "[PLACE HOLDER]"
REPO_ROOT = Path(__file__).resolve().parent.parent


def extract_bullet_list(target: str) -> str:
    """Extract just the bullet list from auto-api's `--no-body` output."""
    docs = extract_api_docs(target, include_private_submodules=False)
    rendered = render_markdown(docs, include_body=False)

    marker = "## List of functions"
    if marker not in rendered:
        return ""

    after = rendered.split(marker, 1)[1]
    lines = after.splitlines()

    bullets: list[str] = []
    seen_first = False
    for line in lines:
        if line.startswith("- "):
            bullets.append(line)
            seen_first = True
        elif seen_first and line.strip() == "":
            bullets.append(line)
        elif seen_first:
            break

    while bullets and bullets[-1].strip() == "":
        bullets.pop()

    return "\n".join(bullets) + "\n" if bullets else ""


def replace_placeholder(text: str, replacement: str) -> str:
    """Replace the first occurrence of `[PLACE HOLDER]` with `replacement`."""
    occurrences = text.count(PLACEHOLDER)
    if occurrences == 0:
        raise ValueError(f"{PLACEHOLDER!r} not found in input file")
    if occurrences > 1:
        print(
            f"Warning: {PLACEHOLDER!r} appears {occurrences} times; "
            f"replacing only the first occurrence.",
            file=sys.stderr,
        )
    return text.replace(PLACEHOLDER, replacement, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an API function list from auto-api and inject it into llms.txt.",
    )
    parser.add_argument("--target", default="snapatac2", help="Module/package to extract.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "scripts" / "llms_template.txt",
        help="Source file containing the [PLACE HOLDER] token (read-only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "llms.txt",
        help="New file to write the result to.",
    )
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_path = resolve(args.input)
    output_path = resolve(args.output)

    if input_path == output_path:
        print(
            f"Error: --input and --output resolve to the same path: {input_path}",
            file=sys.stderr,
        )
        return 1

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    docs = extract_api_docs(args.target, include_private_submodules=False)
    if any(doc.error for doc in docs):
        print(
            f"Error: auto-api reported unresolved APIs for target {args.target!r}:",
            file=sys.stderr,
        )
        for doc in docs:
            if doc.error:
                print(f"  - {doc.requested_name}: {doc.error}", file=sys.stderr)
        return 1

    bullet_list = extract_bullet_list(args.target)
    if not bullet_list:
        print("Error: auto-api produced an empty bullet list.", file=sys.stderr)
        return 1

    original = input_path.read_text(encoding="utf-8")

    if PLACEHOLDER not in original:
        print(
            f"Error: {PLACEHOLDER!r} not found in {input_path}; "
            f"add the token to the file before running this script.",
            file=sys.stderr,
        )
        return 1

    updated = replace_placeholder(original, bullet_list)
    if not updated.endswith("\n"):
        updated += "\n"

    output_path.write_text(updated, encoding="utf-8")

    function_count = sum(1 for line in bullet_list.splitlines() if line.startswith("- "))
    print(f"Wrote {function_count} functions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
