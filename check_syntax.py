"""Parse project Python files without writing bytecode into the workspace."""
import ast
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EXCLUDED_DIRECTORIES = {".git", ".venv", "venv", "__pycache__", "build", "dist"}


def main():
    """Parse every project Python file outside generated/vendor directories."""
    files = []
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
