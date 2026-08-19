#!/usr/bin/env python3
"""Check that plutil's dependencies match PrairieLearn's pinned dependencies."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_UPSTREAM = (
    "https://raw.githubusercontent.com/PrairieLearn/PrairieLearn/master/pyproject.toml"
)
HOST_DEPENDENCY = "prairielearn"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def read_pyproject(source: str) -> dict[str, object]:
    """Read a pyproject from a local path or an HTTP(S) URL."""
    if urlparse(source).scheme in {"http", "https"}:
        with urllib.request.urlopen(source, timeout=30) as response:
            return tomllib.loads(response.read().decode())
    with Path(source).open("rb") as file:
        return tomllib.load(file)


def dependencies(pyproject: dict[str, object]) -> list[str]:
    """Return the PEP 621 runtime dependency list."""
    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise TypeError("missing [project] table")
    requirements = project.get("dependencies")
    if not isinstance(requirements, list) or not all(
        isinstance(requirement, str) for requirement in requirements
    ):
        raise TypeError("project.dependencies must be a list of strings")
    return requirements


def check(local: list[str], upstream: list[str]) -> list[str]:
    """Return errors for dependencies that do not exactly match upstream."""
    upstream_by_name = {
        requirement_name(requirement): requirement
        for requirement in upstream
        if "==" in requirement
    }
    errors: list[str] = []
    checked = 0

    for requirement in local:
        name = requirement_name(requirement)
        if name == HOST_DEPENDENCY:
            continue
        checked += 1
        expected = upstream_by_name.get(name)
        if expected is None:
            errors.append(f"{requirement!r} is not a PrairieLearn dependency")
        elif requirement != expected:
            errors.append(f"{requirement!r} must exactly match {expected!r}")

    if checked == 0:
        errors.append("no dependencies other than the PrairieLearn host were declared")
    if checked >= len(upstream):
        errors.append("plutil dependencies are not a strict subset of PrairieLearn's")
    return errors


def requirement_name(requirement: str) -> str:
    """Return a normalized distribution name from a requirement string."""
    match = NAME_PATTERN.match(requirement)
    if match is None:
        return ""
    return re.sub(r"[-_.]+", "-", match.group().lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", nargs="?", default=DEFAULT_UPSTREAM)
    parser.add_argument("--local", default="pyproject.toml")
    args = parser.parse_args()

    try:
        local = dependencies(read_pyproject(args.local))
        upstream = dependencies(read_pyproject(args.upstream))
    except (OSError, TypeError, UnicodeError, tomllib.TOMLDecodeError) as error:
        print(f"Could not read dependencies: {error}", file=sys.stderr)
        return 2

    errors = check(local, upstream)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print("plutil dependencies match a strict, versioned subset of PrairieLearn's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
