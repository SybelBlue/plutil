"""Command-line tools for :mod:`plutil.magic`."""

import argparse
import ast
from collections.abc import Sequence
from pathlib import Path

from .type_gen import write_plmagic_types_file


def _uses_plmagic(path: Path) -> bool:
    """Return whether *path* contains a function decorated with ``plmagic``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "plmagic":
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
        if source.name != "__plmagic_types__.py" and _uses_plmagic(source)
    }
    output_paths: list[Path] = []
    for question_directory in sorted(question_directories):
        info_json_path = question_directory / "info.json"
        if not info_json_path.is_file():
            raise FileNotFoundError(
                f"A Python file using @plmagic has no companion info.json in "
                f"{question_directory}"
            )
        output_path = question_directory / "__plmagic_types__.py"
        write_plmagic_types_file(info_json_path, output_path.name)
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
