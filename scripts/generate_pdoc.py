"""Build the project's static HTML API reference with pdoc.

Run ``python scripts/generate_pdoc.py`` from the repository root. The pdoc theme,
module navigation, and search index are written to ``docs/api/``.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "src"
OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "api"
MODULES = (
    "mma3001",
)


def main() -> None:
    """Generate static pdoc HTML for the pipeline modules."""
    try:
        import pdoc
    except ImportError as error:
        raise SystemExit(
            "pdoc is not installed. Install documentation dependencies with "
            "`python -m pip install -r requirements-dev.txt`."
        ) from error

    sys.path.insert(0, str(SOURCE_DIRECTORY))
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Document the package once so pdoc discovers its submodules without
    # creating duplicate entries for explicitly repeated module names.
    pdoc.pdoc(*MODULES, output_directory=OUTPUT_DIRECTORY)
    for stale_page in OUTPUT_DIRECTORY.glob("*.html"):
        if stale_page.name not in {"index.html", "mma3001.html"}:
            # Remove pages from the previous flat-module layout only after the
            # new package pages have been generated successfully.
            stale_page.unlink()
    for page_path in OUTPUT_DIRECTORY.rglob("*.html"):
        page = page_path.read_text(encoding="utf-8")
        page_path.write_text(
            "\n".join(line.rstrip() for line in page.splitlines()) + "\n",
            encoding="utf-8",
        )
    print(f"Generated pdoc HTML in {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
