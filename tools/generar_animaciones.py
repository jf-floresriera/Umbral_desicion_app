"""Genera las animaciones del README (assets/*.gif).

    python tools/generar_animaciones.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from umbral import raices, umbral  # noqa: E402
from umbral.binormal import ModeloBinormal  # noqa: E402

SUP, TXT, TXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SALIDA = RAIZ / "assets"
SALIDA.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": TXT2, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2, "axes.facecolor": SUP,
                     "figure.facecolor": SUP, "axes.spines.top": False, "axes.spines.right": False})

mod = ModeloBinormal.desde_conteos(9, 18, 10, 175)


def _guardar_estatica(fig, nombre):
    """Último cuadro como PNG de alta resolución y SVG vectorial (para descargar)."""
    fig.savefig(SALIDA / f"{nombre}.png", dpi=220, facecolor=SUP)
    fig.savefig(SALIDA / f"{nombre}.svg", facecolor=SUP)


def umbral_vs_prevalencia():
    Ps = np.concatenate([np.linspace(0.03, 0.45, 36), np.linspace(0.45, 0.03, 36)])
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=90)
    lCJ, = ax.plot([], [], color=S1, lw=2, label="$C_J(c)$  Youden clásico")
    lBJ, = ax.plot([], [], color=S2, lw=2, label="$B_J(c;P)$  Youden bayesiano")
    lF, = ax.plot([], [], color=S3, lw=6, alpha=0.35, solid_capstyle="round", label="$F=\\min\\{C_J,B_J\\}$")
    v0 = ax.axvline(0.5, color=TXT2, ls=":", lw=1)
    vs = ax.axvline(0.5, color=S3, lw=1.5)
    punto, = ax.plot([], [], "o", ms=9, color=S3, mec=SUP, mew=2)
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", color=TXT, fontsize=11)
    ax.text(0.505, 0.02, "0,5", color=TXT2, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("umbral c"); ax.set_ylabel("índice")
    ax.grid(color=GRID, lw=0.8); ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()

    def cuadro(i):
        P = Ps[i]
        cv = umbral.curvas_modelo(mod, P, n=600)
        r = umbral.desde_matriz(9, 18, 10, 175, P=P)
        cs, F = r["c_star"], r["tabla"]["c* (propuesto)"]["F"]
        lCJ.set_data(cv["c"], cv["CJ"]); lBJ.set_data(cv["c"], cv["BJ"]); lF.set_data(cv["c"], cv["F"])
        vs.set_xdata([cs, cs]); punto.set_data([cs], [F])
        txt.set_text(f"prevalencia P = {P:.2f}\nc* = {cs:.3f}   F(c*) = {F:.3f}")
        return lCJ, lBJ, lF, vs, punto, txt

    FuncAnimation(fig, cuadro, frames=len(Ps), blit=False).save(
        SALIDA / "umbral_prevalencia.gif", writer=PillowWriter(fps=12))
    # versión estática (caso del artículo, P = 27/212) para ver en pausa y descargar
    Ps[0] = 27 / 212
    cuadro(0)
    _guardar_estatica(fig, "umbral_prevalencia")
    plt.close(fig)


def convergencia():
    P = 27 / 212
    g, dg = (lambda c: mod.g(c, P)), (lambda c: mod.dg(c, P))
    ref = 0.4323951143849492
    res = [(raices.biseccion(g, .01, .99), S1), (raices.newton(g, dg, .5), S2),
           (raices.secante(g, .5, .45), S3), (raices.brent(g, .01, .99), S4), (raices.itp(g, .01, .99), S5)]
    errs = [(r["metodo"], np.maximum([abs(p - ref) for p in r["puntos"]], 1e-16), col) for r, col in res]
    nmax = max(len(e) for _, e, _ in errs)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=90)
    lineas = [ax.semilogy([], [], marker="o", ms=4, lw=2, color=col, label=nom)[0] for nom, _, col in errs]
    ax.set_xlim(0.5, nmax + 0.5); ax.set_ylim(1e-16, 1)
    ax.set_xlabel("evaluaciones de g"); ax.set_ylabel("|c(k) − c*|")
    ax.grid(color=GRID, lw=0.8, which="major"); ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper center",
                                                          bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    frames = list(range(1, nmax + 1)) + [nmax] * 12

    def cuadro(k):
        for l, (_, e, _) in zip(lineas, errs):
            m = min(k, len(e)); l.set_data(np.arange(1, m + 1), e[:m])
        return lineas

    FuncAnimation(fig, cuadro, frames=frames).save(SALIDA / "convergencia.gif", writer=PillowWriter(fps=6))
    cuadro(nmax)
    _guardar_estatica(fig, "convergencia")
    plt.close(fig)


if __name__ == "__main__":
    umbral_vs_prevalencia()
    convergencia()
    print("Animaciones en", SALIDA)
