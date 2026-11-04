# plutil

Utilities for PrairieLearn question `server.py` files: typed answer lenses,
SymPy parsing and comparison, nested `data` access, partial credit,
deterministic formatting, randomized parameters, and calculus helpers.

New here? Start with {doc}`getting-started`.

## Guide

| Topic | What it covers |
| --- | --- |
| {doc}`guide/common` | Core symbolic types, expression equality and triviality checks, nested `data` access |
| {doc}`guide/grading` | Rich scores, multiple-choice/checkbox credit, and rule-based partial credit |
| {doc}`guide/choices` | Building `pl-multiple-choice` and `pl-checkbox` options from `server.py` |
| {doc}`guide/formatting` | Deterministic number formatting (`fmt`), parsing it back (`defmt`), exact decimal input |
| {doc}`guide/functions` | Domains, evaluation, translating/scaling, and rearranging equations |
| {doc}`guide/calculus` | Derivatives, tangent lines, integrals, missing-constant credit, stock feedback |
| {doc}`guide/rand` | Random integers, rationals, polynomials, partitions, and reusable factories |
| {doc}`guide/obfuscations` | Random equivalent expressions built from trigonometric identities |
| {doc}`guide/assumptions` | Checking SymPy assumptions on values |
| {doc}`guide/fabc` | `sympy.abc`-style undefined function symbols |

For every public name, see the {doc}`api`.

```{toctree}
:hidden:
:caption: Getting started

getting-started
```

```{toctree}
:hidden:
:caption: Guide

guide/common
guide/grading
guide/choices
guide/formatting
guide/functions
guide/calculus
guide/rand
guide/obfuscations
guide/assumptions
guide/fabc
```

```{toctree}
:hidden:
:caption: Reference

api
```
