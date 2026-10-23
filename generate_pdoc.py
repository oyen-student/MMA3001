"""Build the project's static HTML API reference with pdoc.

Run ``python generate_pdoc.py`` from the repository root. The pdoc theme,
module navigation, and search index are written to ``docs/api/``.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "docs" / "api"
MODULES = (
    "main",
    "data_prep",
    "linear_regression",
    "neural_network",
    "hyperparameter_tuning",
    "plots",
    "src.data_prep",
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

    sys.path.insert(0, str(PROJECT_ROOT))
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Remove the old Pydoc-only filename; pdoc writes src/data_prep.html.
    legacy_page = OUTPUT_DIRECTORY / "src.data_prep.html"
    if legacy_page.exists():
        legacy_page.unlink()

    pdoc.pdoc(*MODULES, output_directory=OUTPUT_DIRECTORY)
    for page_path in OUTPUT_DIRECTORY.rglob("*.html"):
        page = page_path.read_text(encoding="utf-8")
        page_path.write_text(
            "\n".join(line.rstrip() for line in page.splitlines()) + "\n",
            encoding="utf-8",
        )
    print(f"Generated pdoc HTML in {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
