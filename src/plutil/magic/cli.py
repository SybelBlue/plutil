"""Command-line tools for :mod:`plutil.magic`."""

import argparse
import ast
from collections.abc import Sequence
from pathlib import Path

from .type_gen import DEFAULT_TYPE_FILE_NAME, write_plmagic_types_file


def _uses_plmagic(path: Path) -> bool:
    """Return whether *path* contains a function decorated with ``plmagic``."""
    source = path.read_text(encoding="utf-8")
    # Large courses can contain legacy or otherwise unparsable Python files. Avoid
    # parsing files that cannot possibly use the decorator.
    if "plmagic" not in source:
        return False
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        location = f"{path}:{error.lineno or 1}:{error.offset or 1}"
        source_line = (error.text or "").strip()
        detail = f": {source_line}" if source_line else ""
        raise SyntaxError(f"Cannot inspect {location}: {error.msg}{detail}") from error
    plmagic_names = {"plmagic"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module not in {"plutil", "plutil.magic"}:
            continue
        plmagic_names.update(
            alias.asname or alias.name
            for alias in node.names
            if alias.name == "plmagic"
        )

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            # Accept both ``@plmagic`` and configurable decorator forms such as
            # ``@plmagic(...)`` without importing or executing the module.
            while isinstance(decorator, ast.Call):
                decorator = decorator.func
            if isinstance(decorator, ast.Name) and decorator.id in plmagic_names:
                return True
            if isinstance(decorator, ast.Attribute) and decorator.attr == "plmagic":
                return True
    return False


def generate_plmagic_type_files(directory: Path) -> list[Path]:
    """Generate type files for every Python source using ``@plmagic`` below a directory.

    A matching source is treated as a PrairieLearn question server, so its companion
    ``info.json`` and generated ``__plmagic_types__.py`` live in the same directory.
    Question directories are de-duplicated when they contain multiple matching files.
    """
    directory = directory.resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    question_directories = {
        source.parent
        for source in directory.rglob("*.py")
        if source.name != DEFAULT_TYPE_FILE_NAME and _uses_plmagic(source)
    }
    output_paths: list[Path] = []
    for question_directory in sorted(question_directories):
        info_json_path = question_directory / "info.json"
        if not info_json_path.is_file():
            raise FileNotFoundError(
                f"A Python file using @plmagic has no companion info.json in "
                f"{question_directory}"
            )
        output_path = question_directory / DEFAULT_TYPE_FILE_NAME
        if write_plmagic_types_file(info_json_path, output_path.name):
            output_paths.append(output_path)
    return output_paths


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``plmagic-types`` console command."""
    parser = argparse.ArgumentParser(
        description="Generate type files for Python files decorated with @plmagic."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="directory to search recursively (default: current directory)",
    )
    args = parser.parse_args(argv)

    try:
        output_paths = generate_plmagic_type_files(args.directory)
    except (OSError, SyntaxError, ValueError) as error:
        parser.error(str(error))

    for output_path in output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
