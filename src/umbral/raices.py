"""Cinco métodos para la ecuación escalar g(c) = 0 (sección 3.4).

Cada función devuelve un dict con la raíz, el historial de puntos evaluados y el
número de evaluaciones de g y de g'.
"""
from __future__ import annotations

import math
import time
from typing import Callable

from scipy.optimize import brentq


class _Contador:
    def __init__(self, f):
        self.f, self.puntos = f, []

    def __call__(self, x):
        self.puntos.append(float(x))
        return float(self.f(x))


def _resultado(nombre, raiz, cg, n_dg=0, ok=True):
    return {"metodo": nombre, "raiz": raiz, "puntos": cg.puntos,
            "eval_g": len(cg.puntos), "eval_dg": n_dg, "convergio": ok}


def biseccion(g, a, b, tol=1e-14, max_iter=200):
    cg = _Contador(g)
    ga, gb = cg(a), cg(b)
    if ga * gb > 0:
        raise ValueError("g(a) y g(b) deben tener signos opuestos.")
    c = (a + b) / 2
    for _ in range(max_iter):
        c = (a + b) / 2
        gc = cg(c)
        if gc == 0 or (b - a) / 2 < tol:
            break
        if ga * gc < 0:
            b = c
        else:
            a, ga = c, gc
    return _resultado("Bisección", c, cg)


def newton(g, dg, x0, tol=1e-14, max_iter=50, dominio=(0.0, 1.0)):
    cg, n_dg, x = _Contador(g), 0, x0
    for _ in range(max_iter):
        gx = cg(x)
        d = float(dg(x)); n_dg += 1
        x_new = x - gx / d
        if not (dominio[0] < x_new < dominio[1]) or not math.isfinite(x_new):
            return _resultado("Newton–Raphson", x_new, cg, n_dg, ok=False)
        if abs(x_new - x) <= tol:
            return _resultado("Newton–Raphson", x_new, cg, n_dg)
        x = x_new
    return _resultado("Newton–Raphson", x, cg, n_dg, ok=False)


def secante(g, x0, x1, tol=1e-14, max_iter=100, dominio=(0.0, 1.0)):
    cg = _Contador(g)
    g0, g1 = cg(x0), cg(x1)
    for _ in range(max_iter):
        if g1 == g0:
            break
        x2 = x1 - g1 * (x1 - x0) / (g1 - g0)
        if not (dominio[0] < x2 < dominio[1]) or not math.isfinite(x2):
            return _resultado("Secante", x2, cg, ok=False)
        if abs(x2 - x1) <= tol:
            return _resultado("Secante", x2, cg)
        x0, g0, x1 = x1, g1, x2
        g1 = cg(x1)
    return _resultado("Secante", x1, cg, ok=abs(g1) < 1e-12)


def brent(g, a, b, tol=1e-14):
    cg = _Contador(g)
    r = brentq(cg, a, b, xtol=tol, rtol=4 * 2.220446049250313e-16, full_output=False)
    return _resultado("Brent", r, cg)


def itp(g, a, b, tol=1e-14, k1=None, k2=2.0, n0=1, max_iter=500):
    """Interpolación, truncamiento y proyección (Oliveira & Takahashi, 2020)."""
    cg = _Contador(g)
    if k1 is None:
        k1 = 0.2 / (b - a)
    ya, yb = cg(a), cg(b)
    if ya * yb > 0:
        raise ValueError("g(a) y g(b) deben tener signos opuestos.")
    signo = 1.0 if yb > ya else -1.0       # trabajar con función creciente
    ya, yb = signo * ya, signo * yb
    n_half = math.ceil(math.log2((b - a) / (2 * tol)))
    n_max = n_half + n0
    j = 0
    while (b - a) > 2 * tol and j < max_iter:
        x_half = (a + b) / 2
        r = tol * 2 ** (n_max - j) - (b - a) / 2
        delta = k1 * (b - a) ** k2
        x_f = (yb * a - ya * b) / (yb - ya)                 # regula falsi
        s = math.copysign(1.0, x_half - x_f)
        x_t = x_f + s * delta if delta <= abs(x_half - x_f) else x_half
        x_itp = x_t if abs(x_t - x_half) <= r else x_half - s * r
        y = signo * cg(x_itp)
        if y > 0:
            b, yb = x_itp, y
        elif y < 0:
            a, ya = x_itp, y
        else:
            a = b = x_itp
        j += 1
    return _resultado("ITP", (a + b) / 2, cg)


def comparar(g, dg, raiz_ref, a=0.01, b=0.99, x0=0.5, x1=0.45, tol=1e-14, repeticiones=50):
    """Tabla 3: evaluaciones, evaluaciones hasta 1e-12 y tiempo medio por método."""
    corridas = {
        "Bisección": lambda: biseccion(g, a, b, tol),
        "Newton–Raphson": lambda: newton(g, dg, x0, tol),
        "Secante": lambda: secante(g, x0, x1, tol),
        "Brent": lambda: brent(g, a, b, tol),
        "ITP": lambda: itp(g, a, b, tol),
    }
    filas = []
    for nombre, f in corridas.items():
        r = f()
        t0 = time.perf_counter()
        for _ in range(repeticiones):
            f()
        t = (time.perf_counter() - t0) / repeticiones * 1e6
        hasta = next((i + 1 for i, p in enumerate(r["puntos"]) if abs(p - raiz_ref) < 1e-12), None)
        filas.append({"Método": nombre, "Raíz": r["raiz"], "Eval. g": r["eval_g"],
                      "Eval. g'": r["eval_dg"], "Eval. hasta 1e-12": hasta,
                      "Tiempo (µs)": round(t, 1), "Convergió": r["convergio"],
                      "Errores": [abs(p - raiz_ref) for p in r["puntos"]]})
    return filas
