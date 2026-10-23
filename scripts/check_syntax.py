"""Parse project Python files without writing bytecode into the workspace."""
import ast
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Exclude generated environments and build products so the scan covers project
# source, tests, and scripts without parsing copied third-party packages.
EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "venv", "__pycache__", "build", "dist", ".pytest_cache"
}


def main():
    """Parse every project Python file outside generated/vendor directories."""
    # Collect paths first, then parse without importing modules or generating
    # bytecode; syntax checking therefore does not require dataset access.
    files = []
    # Parse source and test files directly instead of importing them; this
    # avoids executing pipeline code or requiring the large local dataset.
    for current_dir, directory_names, file_names in os.walk(PROJECT_ROOT):
        directory_names[:] = [
            name for name in directory_names if name not in EXCLUDED_DIRECTORIES
        ]
        files.extend(
            Path(current_dir) / name for name in file_names if name.endswith(".py")
        )
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"Syntax OK: {len(files)} Python files")


if __name__ == "__main__":
    main()
