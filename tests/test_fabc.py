import pytest
import sympy
import sympy.abc
from sympy.core.function import FunctionClass

import plutil
from plutil import fabc

UPPERCASE_GREEK_NAMES = {
    "Alpha",
    "Beta",
    "Chi",
    "Delta",
    "Epsilon",
    "Eta",
    "Gamma",
    "Iota",
    "Kappa",
    "Lambda",
    "Mu",
    "Nu",
    "Omega",
    "Omicron",
    "Phi",
    "Pi",
    "Psi",
    "Rho",
    "Sigma",
    "Tau",
    "Theta",
    "Upsilon",
    "Xi",
    "Zeta",
}


def test_fabc_is_available_from_package_namespace() -> None:
    assert plutil.fabc is fabc


def test_fabc_mirrors_sympy_abc_symbol_names() -> None:
    symbol_names = {
        name
        for name, value in vars(sympy.abc).items()
        if isinstance(value, sympy.Symbol)
    }

    assert set(fabc.__all__) == symbol_names | UPPERCASE_GREEK_NAMES


def test_fabc_names_are_undefined_functions() -> None:
    for name in fabc.__all__:
        function = getattr(fabc, name)

        assert isinstance(function, FunctionClass)
        assert function.__name__ == name


def test_fabc_function_can_be_applied_to_a_symbol() -> None:
    assert fabc.f(sympy.abc.x) == sympy.Function("f")(sympy.abc.x)


@pytest.mark.parametrize(
    ("name", "latex_name"),
    [
        ("alpha", r"\alpha"),
        ("beta", r"\beta"),
        ("gamma", r"\gamma"),
        ("delta", r"\delta"),
        ("epsilon", r"\epsilon"),
        ("zeta", r"\zeta"),
        ("eta", r"\eta"),
        ("theta", r"\theta"),
        ("iota", r"\iota"),
        ("kappa", r"\kappa"),
        ("lamda", r"\lambda"),
        ("mu", r"\mu"),
        ("nu", r"\nu"),
        ("xi", r"\xi"),
        ("omicron", "o"),
        ("pi", r"\pi"),
        ("rho", r"\rho"),
        ("sigma", r"\sigma"),
        ("tau", r"\tau"),
        ("upsilon", r"\upsilon"),
        ("phi", r"\phi"),
        ("chi", r"\chi"),
        ("psi", r"\psi"),
        ("omega", r"\omega"),
        ("Alpha", r"\mathrm{A}"),
        ("Beta", r"\mathrm{B}"),
        ("Gamma", r"\Gamma"),
        ("Delta", r"\Delta"),
        ("Epsilon", r"\mathrm{E}"),
        ("Zeta", r"\mathrm{Z}"),
        ("Eta", r"\mathrm{H}"),
        ("Theta", r"\Theta"),
        ("Iota", r"\mathrm{I}"),
        ("Kappa", r"\mathrm{K}"),
        ("Lambda", r"\Lambda"),
        ("Mu", r"\mathrm{M}"),
        ("Nu", r"\mathrm{N}"),
        ("Xi", r"\Xi"),
        ("Omicron", r"\mathrm{O}"),
        ("Pi", r"\Pi"),
        ("Rho", r"\mathrm{P}"),
        ("Sigma", r"\Sigma"),
        ("Tau", r"\mathrm{T}"),
        ("Upsilon", r"\Upsilon"),
        ("Phi", r"\Phi"),
        ("Chi", r"\mathrm{X}"),
        ("Psi", r"\Psi"),
        ("Omega", r"\Omega"),
    ],
)
def test_greek_function_latex_uses_greek_letter(name: str, latex_name: str) -> None:
    function = getattr(fabc, name)

    assert sympy.latex(function) == latex_name
    assert sympy.latex(function(sympy.abc.x)) == rf"{latex_name}{{\left(x \right)}}"
