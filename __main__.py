"""Package entry point for ``python -m plutil``."""

from collections.abc import Sequence

from plutil.magic.cli import main as plmagic_types_main


def main(argv: Sequence[str] | None = None) -> int:
    """Generate plmagic type files below the requested directory."""
    return plmagic_types_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
