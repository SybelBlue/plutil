"""Common SymPy undefined functions.

This module mirrors the symbol names exported by :mod:`sympy.abc`, adds
uppercase Greek names, and binds each name to an undefined function instead of
a symbol.  For example, ``f(x)`` constructs an application of the undefined
function ``f``.
"""

from __future__ import annotations

from typing import Final

import sympy

a = sympy.Function("a")
b = sympy.Function("b")
c = sympy.Function("c")
d = sympy.Function("d")
e = sympy.Function("e")
f = sympy.Function("f")
g = sympy.Function("g")
h = sympy.Function("h")
i = sympy.Function("i")
j = sympy.Function("j")
k = sympy.Function("k")
l = sympy.Function("l")
m = sympy.Function("m")
n = sympy.Function("n")
o = sympy.Function("o")
p = sympy.Function("p")
q = sympy.Function("q")
r = sympy.Function("r")
s = sympy.Function("s")
t = sympy.Function("t")
u = sympy.Function("u")
v = sympy.Function("v")
w = sympy.Function("w")
x = sympy.Function("x")
y = sympy.Function("y")
z = sympy.Function("z")

A = sympy.Function("A")
B = sympy.Function("B")
C = sympy.Function("C")
D = sympy.Function("D")
E = sympy.Function("E")
F = sympy.Function("F")
G = sympy.Function("G")
H = sympy.Function("H")
I = sympy.Function("I")
J = sympy.Function("J")
K = sympy.Function("K")
L = sympy.Function("L")
M = sympy.Function("M")
N = sympy.Function("N")
O = sympy.Function("O")
P = sympy.Function("P")
Q = sympy.Function("Q")
R = sympy.Function("R")
S = sympy.Function("S")
T = sympy.Function("T")
U = sympy.Function("U")
V = sympy.Function("V")
W = sympy.Function("W")
X = sympy.Function("X")
Y = sympy.Function("Y")
Z = sympy.Function("Z")

alpha = sympy.Function("alpha")
beta = sympy.Function("beta")
gamma = sympy.Function("gamma")
delta = sympy.Function("delta")
epsilon = sympy.Function("epsilon")
zeta = sympy.Function("zeta")
eta = sympy.Function("eta")
theta = sympy.Function("theta")
iota = sympy.Function("iota")
kappa = sympy.Function("kappa")
lamda = sympy.Function("lamda")
mu = sympy.Function("mu")
nu = sympy.Function("nu")
xi = sympy.Function("xi")
omicron = sympy.Function("omicron")
pi = sympy.Function("pi")
rho = sympy.Function("rho")
sigma = sympy.Function("sigma")
tau = sympy.Function("tau")
upsilon = sympy.Function("upsilon")
phi = sympy.Function("phi")
chi = sympy.Function("chi")
psi = sympy.Function("psi")
omega = sympy.Function("omega")

Alpha = sympy.Function("Alpha")
Beta = sympy.Function("Beta")
Gamma = sympy.Function("Gamma")
Delta = sympy.Function("Delta")
Epsilon = sympy.Function("Epsilon")
Zeta = sympy.Function("Zeta")
Eta = sympy.Function("Eta")
Theta = sympy.Function("Theta")
Iota = sympy.Function("Iota")
Kappa = sympy.Function("Kappa")
Lambda = sympy.Function("Lambda")
Mu = sympy.Function("Mu")
Nu = sympy.Function("Nu")
Xi = sympy.Function("Xi")
Omicron = sympy.Function("Omicron")
Pi = sympy.Function("Pi")
Rho = sympy.Function("Rho")
Sigma = sympy.Function("Sigma")
Tau = sympy.Function("Tau")
Upsilon = sympy.Function("Upsilon")
Phi = sympy.Function("Phi")
Chi = sympy.Function("Chi")
Psi = sympy.Function("Psi")
Omega = sympy.Function("Omega")

__all__: Final[tuple[str, ...]] = (
    "A",
    "Alpha",
    "B",
    "Beta",
    "C",
    "Chi",
    "D",
    "Delta",
    "E",
    "Epsilon",
    "Eta",
    "F",
    "G",
    "Gamma",
    "H",
    "I",
    "Iota",
    "J",
    "K",
    "Kappa",
    "L",
    "Lambda",
    "M",
    "Mu",
    "N",
    "Nu",
    "O",
    "Omega",
    "Omicron",
    "P",
    "Phi",
    "Pi",
    "Psi",
    "Q",
    "R",
    "Rho",
    "S",
    "Sigma",
    "T",
    "Tau",
    "Theta",
    "U",
    "Upsilon",
    "V",
    "W",
    "X",
    "Xi",
    "Y",
    "Z",
    "Zeta",
    "a",
    "alpha",
    "b",
    "beta",
    "c",
    "chi",
    "d",
    "delta",
    "e",
    "epsilon",
    "eta",
    "f",
    "g",
    "gamma",
    "h",
    "i",
    "iota",
    "j",
    "k",
    "kappa",
    "l",
    "lamda",
    "m",
    "mu",
    "n",
    "nu",
    "o",
    "omega",
    "omicron",
    "p",
    "phi",
    "pi",
    "psi",
    "q",
    "r",
    "rho",
    "s",
    "sigma",
    "t",
    "tau",
    "theta",
    "u",
    "upsilon",
    "v",
    "w",
    "x",
    "xi",
    "y",
    "z",
    "zeta",
)
