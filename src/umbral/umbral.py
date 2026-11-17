"""Umbral propuesto c* = argmax_c min{C_J(c), B_J(c; P)} (sección 3).

Dos caminos:
* `desde_matriz`: solo con la matriz de confusión en c0 = 0,5 (modelo logit-binormal).
* `desde_puntajes`: con los puntajes fuera de muestra (observación 2: cuantil 1 - P).
"""
from __future__ import annotations

import numpy as np

from .binormal import ModeloBinormal
from . import metricas as M
from . import raices


def desde_matriz(VP, FN, FP, VN, P=None, kappa=1.0, metodo="newton"):
    n1, n0 = VP + FN, FP + VN
    if P is None:
        P = n1 / (n1 + n0)
    mod = ModeloBinormal.desde_conteos(VP, FN, FP, VN, kappa)
    g = lambda c: mod.g(c, P)
    dg = lambda c: mod.dg(c, P)
    r = raices.newton(g, dg, 0.5) if metodo == "newton" else {"convergio": False}
    if not r["convergio"]:
        r = raices.brent(g, 0.01, 0.99) if g(0.01) * g(0.99) < 0 else raices.brent(g, 1e-6, 1 - 1e-6)
    c_star = r["raiz"]
    umbrales = {"c0 (red)": 0.5, "c* (propuesto)": c_star,
                "c_CJ (Youden clásico)": mod.c_youden_clasico(), "c_VPP=0,5": mod.c_vpp(P)}
    tabla = {}
    for nombre, c in umbrales.items():
        if not np.isfinite(c):
            continue
        if nombre.startswith("c0"):
            cel = {"VP": VP, "FN": FN, "FP": FP, "VN": VN}
            met = M.desde_conteos(VP, FN, FP, VN, P)
        else:
            cel = mod.matriz_esperada(c, n1, n0)
            met = M.desde_tasas(float(mod.Se(c)), float(mod.Sp(c)), P)
        tabla[nombre] = {"c": c, "eta": float(np.log(c / (1 - c))), **cel, **met.como_dict()}
    return {"modelo": mod, "P": P, "c_star": c_star, "tabla": tabla}


def curvas_modelo(mod: ModeloBinormal, P, n=2001):
    c = np.linspace(1e-3, 1 - 1e-3, n)
    Se, Sp = mod.Se(c), mod.Sp(c)
    Q = P * Se + (1 - P) * (1 - Sp)
    CJ = Se + Sp - 1
    BJ = CJ * P * (1 - P) / (Q * (1 - Q))
    return {"c": c, "CJ": CJ, "BJ": BJ, "F": np.minimum(CJ, BJ)}


def desde_puntajes(y, p, P=None):
    """Umbral empírico: cuantil 1 - P de las salidas (observación 2) y curvas empíricas."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    if P is None:
        P = y.mean()
    c_star = float(np.quantile(p, 1 - P))
    cs = np.unique(np.concatenate([np.linspace(0.001, 0.999, 999), p]))
    n1, n0 = y.sum(), (1 - y).sum()
    Se = np.array([(p[y == 1] >= c).sum() / n1 for c in cs])
    Sp = np.array([(p[y == 0] < c).sum() / n0 for c in cs])
    Q = P * Se + (1 - P) * (1 - Sp)
    CJ = Se + Sp - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        BJ = np.where((Q > 0) & (Q < 1), CJ * P * (1 - P) / (Q * (1 - Q)), np.nan)
    F = np.fmin(CJ, BJ)
    c_maxF = float(cs[np.nanargmax(F)])
    def conteos(c):
        yh = (p >= c).astype(int)
        return {"VP": int(((yh == 1) & (y == 1)).sum()), "FN": int(((yh == 0) & (y == 1)).sum()),
                "FP": int(((yh == 1) & (y == 0)).sum()), "VN": int(((yh == 0) & (y == 0)).sum())}
    tabla = {}
    for nombre, c in {"c0 (0,5)": 0.5, "c* (cuantil 1-P)": c_star, "argmax F (malla)": c_maxF}.items():
        k = conteos(c)
        met = M.desde_conteos(k["VP"], k["FN"], k["FP"], k["VN"], P) if min(k["VP"] + k["FN"], k["FP"] + k["VN"]) > 0 else None
        tabla[nombre] = {"c": c, **k, **(met.como_dict() if met else {})}
    return {"P": P, "c_star": c_star, "tabla": tabla,
            "curvas": {"c": cs, "CJ": CJ, "BJ": BJ, "F": F}}
