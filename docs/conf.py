"""Sphinx configuration for the plutil documentation."""

project = "plutil"
author = "plutil contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

# Pygments can't lex Mustache tags inside the guide's HTML examples; it
# falls back to relaxed highlighting, which renders fine.
suppress_warnings = ["misc.highlighting_failure"]

# Generate GitHub-style heading anchors so guide pages can link to sections.
myst_heading_anchors = 4

# Docstrings use single backticks for code in places; render them as code
# rather than reST's default italic "title reference".
default_role = "code"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Render "Attributes:" as a field list so autodoc doesn't document them twice.
napoleon_use_ivar = True

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_title = "plutil"
html_theme_options = {
    "source_repository": "https://github.com/SybelBlue/plutil/",
    "source_branch": "main",
    "source_directory": "docs/",
}
