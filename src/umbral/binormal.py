"""Modelo logit-binormal reconstruido a partir de una matriz de confusión (sección 3.3)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def logit(c):
    c = np.asarray(c, dtype=float)
    return np.log(c / (1 - c))


def sigmoide(t):
    return 1.0 / (1.0 + np.exp(-np.asarray(t, dtype=float)))


class ModeloBinormal:
    """eta | Y=0 ~ N(m0, tau^2), eta | Y=1 ~ N(m1, kappa^2 tau^2), con tau = d' (ec. 3.9)."""

    def __init__(self, Se0: float, Sp0: float, kappa: float = 1.0):
        if not (0 < Se0 < 1 and 0 < Sp0 < 1):
            raise ValueError("Se y Sp en c0 = 0,5 deben estar estrictamente entre 0 y 1.")
        self.Se0, self.Sp0, self.kappa = Se0, Sp0, kappa
        self.zSe, self.zSp = norm.ppf(Se0), norm.ppf(Sp0)
        self.tau = self.zSe + self.zSp            # d'
        if self.tau <= 0:
            raise ValueError("d' <= 0: el clasificador no separa las clases (C_J <= 0).")
        self.m0 = -self.tau * self.zSp
        self.m1 = kappa * self.tau * self.zSe

    @classmethod
    def desde_conteos(cls, VP, FN, FP, VN, kappa: float = 1.0):
        return cls(VP / (VP + FN), VN / (VN + FP), kappa)

    # --- características operativas (ec. 3.6) ---
    def _s0(self, c):
        return (logit(c) - self.m0) / self.tau

    def _s1(self, c):
        return (logit(c) - self.m1) / (self.kappa * self.tau)

    def Se(self, c):
        return norm.sf(self._s1(c))

    def Sp(self, c):
        return norm.cdf(self._s0(c))

    def Q(self, c, P):
        return P * self.Se(c) + (1 - P) * norm.sf(self._s0(c))

    def auc(self):
        return norm.cdf((self.m1 - self.m0) / (self.tau * np.sqrt(1 + self.kappa ** 2)))

    def intercepto(self, P):
        """b de la ec. (3.8) (solo para kappa = 1)."""
        return np.log(P / (1 - P)) - (self.m1 ** 2 - self.m0 ** 2) / (2 * self.tau ** 2)

    # --- ecuación del umbral (3.10) y su derivada (3.11) ---
    def g(self, c, P):
        return (1 - P) * norm.sf(self._s0(c)) - P * norm.cdf(self._s1(c))

    def dg(self, c, P):
        c = np.asarray(c, dtype=float)
        j = c * (1 - c)
        dSp = norm.pdf(self._s0(c)) / (self.tau * j)
        dSe = -norm.pdf(self._s1(c)) / (self.kappa * self.tau * j)
        return -(1 - P) * dSp + P * dSe

    # --- umbrales de referencia (sección 4.4) ---
    def c_youden_clasico(self):
        if self.kappa != 1:
            from scipy.optimize import minimize_scalar
            r = minimize_scalar(lambda t: -(self.Se(sigmoide(t)) + self.Sp(sigmoide(t))),
                                bounds=(-20, 20), method="bounded")
            return float(sigmoide(r.x))
        return float(sigmoide((self.m0 + self.m1) / 2))

    def c_vpp(self, P, objetivo=0.5):
        from scipy.optimize import brentq
        def h(c):
            se, fpr = self.Se(c), norm.sf(self._s0(c))
            return P * se / (P * se + (1 - P) * fpr) - objetivo
        try:
            return brentq(h, 1e-6, 1 - 1e-6)
        except ValueError:      # con kappa != 1 el VPP puede no alcanzar el objetivo
            return float("nan")

    def matriz_esperada(self, c, n1, n0):
        se, sp = float(self.Se(c)), float(self.Sp(c))
        return {"VP": n1 * se, "FN": n1 * (1 - se), "FP": n0 * (1 - sp), "VN": n0 * sp}
