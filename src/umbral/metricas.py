"""Métricas clásicas y bayesianas de una matriz de confusión (sección 2 del artículo)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass
class Metricas:
    P: float
    Q: float
    Se: float
    Sp: float
    VPP: float
    VPN: float
    CJ: float
    BJ: float
    F: float
    CA: float
    BG: float
    F1: float
    MCC: float

    def como_dict(self) -> dict:
        return asdict(self)


def desde_tasas(Se: float, Sp: float, P: float) -> Metricas:
    """Métricas a partir de Se, Sp y la prevalencia P (ecuaciones 2.1 a 2.6)."""
    vp, fn = P * Se, P * (1 - Se)
    fp, vn = (1 - P) * (1 - Sp), (1 - P) * Sp
    Q = vp + fp
    vpp = vp / Q if Q > 0 else float("nan")
    vpn = vn / (1 - Q) if Q < 1 else float("nan")
    cj = Se + Sp - 1
    bj = vpp + vpn - 1
    ca = vp + vn
    bg = math.sqrt(vpp * vpn) if vpp >= 0 and vpn >= 0 else float("nan")
    f1 = 2 * vp / (2 * vp + fp + fn) if (2 * vp + fp + fn) > 0 else float("nan")
    mcc = math.copysign(math.sqrt(abs(cj * bj)), cj) if cj * bj >= 0 else float("nan")
    return Metricas(P, Q, Se, Sp, vpp, vpn, cj, bj, min(cj, bj), ca, bg, f1, mcc)


def desde_conteos(VP: float, FN: float, FP: float, VN: float, P: float | None = None) -> Metricas:
    """Métricas a partir de los conteos; si no se da P, se usa la prevalencia empírica."""
    n1, n0 = VP + FN, FP + VN
    if P is None:
        P = n1 / (n1 + n0)
    return desde_tasas(VP / n1, VN / n0, P)


def prevalencias_equilibrio(Se: float, Sp: float) -> dict:
    """Proposición 3: prevalencias en que B_J = C_J, maximizador de B_J y su máximo (Yule)."""
    cj = Se + Sp - 1
    u, v = Sp * (1 - Sp), Se * (1 - Se)
    a, b = math.sqrt(Se * Sp), math.sqrt((1 - Se) * (1 - Sp))
    return {
        "P1": (1 - Sp) / (1 - cj),
        "P2": Sp / (Se + Sp),
        "Popt": math.sqrt(u) / (math.sqrt(u) + math.sqrt(v)),
        "max_BJ": (a - b) / (a + b),
        "max_BG": a / (a + b),
        "DOR": (a / b) ** 2,
    }
