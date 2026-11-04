"""Demonstrate scalar and bounded-batch trigonometric obfuscations.

Run with ``uv run python examples/trig_obfuscations.py``.
"""

from sympy.abc import x

from plutil import function_names, obfuscations, rand


def describe(index: int, candidate: obfuscations.TrigObfuscation) -> None:
    """Print one candidate and its grading-relevant metadata."""
    print(f"Candidate {index} (gas={candidate.gas})")
    print(f"  source:    {candidate.source}")
    print(f"  rewritten: {candidate.rewritten}")
    print(f"  functions: {sorted(function_names(candidate.rewritten))}")


def main() -> None:
    """Generate repeatable scalar and batch examples."""
    rand.base.seed(18)

    single = obfuscations.trig(1, angle=x, max_gas=7)
    describe(1, single)

    print("\nUp to three distinct candidates:")
    batch = obfuscations.trig(1, angle=x, max_gas=7, max_k=3)
    assert isinstance(batch, tuple)
    for number, candidate in enumerate(batch, start=1):
        describe(number, candidate)

    all_candidates = obfuscations.trig(1, angle=x, max_gas=7, max_k=None)
    print(f"\nAll eligible candidates: {len(all_candidates)}")


if __name__ == "__main__":
    main()
