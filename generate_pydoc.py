"""Generate static HTML API documentation for the project modules.

Run ``python generate_pydoc.py`` from any working directory. The generated
pages are written to ``docs/api/`` using Python's standard-library ``pydoc``.
"""

from __future__ import annotations

import html
import importlib
import os
from pathlib import Path
import pydoc
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


def _write_index() -> None:
    """Write a small landing page linking to each generated module page."""
    links = "\n".join(
        f'<li><a href="{html.escape(name, quote=True)}.html">'
        f"{html.escape(name)}</a></li>"
        for name in MODULES
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MMA3001 API Reference</title>
</head>
<body>
  <h1>MMA3001 API Reference</h1>
  <p>Module documentation generated with Python's standard-library pydoc.</p>
  <ul>
{links}
  </ul>
  <p>See <a href="../../README.md">README.md</a> for setup and usage.</p>
</body>
</html>
"""
    (OUTPUT_DIRECTORY / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    """Generate HTML pages for the pipeline and compatibility modules."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PROJECT_ROOT))

    # Import before changing directory so project imports resolve from root.
    modules = [importlib.import_module(name) for name in MODULES]
    original_directory = Path.cwd()
    try:
        os.chdir(OUTPUT_DIRECTORY)
        for module in modules:
            pydoc.writedoc(module)
            page_path = OUTPUT_DIRECTORY / f"{module.__name__}.html"
            page = page_path.read_text(encoding="utf-8")
            page_path.write_text(
                "\n".join(line.rstrip() for line in page.splitlines()) + "\n",
                encoding="utf-8",
            )
    finally:
        os.chdir(original_directory)

    _write_index()
    print(f"Generated Pydoc HTML for {len(MODULES)} modules in {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
