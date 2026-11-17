"""Verifica que el paquete reproduce los números del artículo (P = 27/212)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from umbral import metricas as M, raices, umbral, sinteticos, validacion  # noqa: E402
from umbral.binormal import ModeloBinormal  # noqa: E402

VP, FN, FP, VN = 9, 18, 10, 175
P = 27 / 212


def test_metricas_c0():
    m = M.desde_conteos(VP, FN, FP, VN)
    assert round(m.CJ, 3) == 0.279 and round(m.BJ, 3) == 0.380
    assert round(m.CA, 3) == 0.868 and round(m.BG, 3) == 0.655


def test_proposicion_3():
    r = M.prevalencias_equilibrio(9 / 27, 175 / 185)
    assert round(r["P1"], 3) == 0.075 and round(r["P2"], 3) == 0.739
    assert round(r["Popt"], 3) == 0.324 and round(r["max_BJ"], 3) == 0.495
    assert round(r["max_BG"], 3) == 0.747


def test_modelo_binormal():
    mod = ModeloBinormal.desde_conteos(VP, FN, FP, VN)
    assert round(mod.tau, 3) == 1.176 and round(mod.m0, 3) == -1.890 and round(mod.m1, 3) == -0.507
    assert round(mod.auc(), 3) == 0.797 and round(mod.intercepto(P), 3) == -0.726
    assert math.isclose(float(mod.Se(0.5)), 9 / 27) and math.isclose(float(mod.Sp(0.5)), 175 / 185)


def test_umbral_propuesto_tabla_2():
    r = umbral.desde_matriz(VP, FN, FP, VN)
    assert abs(r["c_star"] - 0.4323951144) < 1e-10
    t = r["tabla"]["c* (propuesto)"]
    assert round(t["Se"], 3) == 0.421 and round(t["Sp"], 3) == 0.915
    assert round(t["F"], 3) == 0.336 and abs(t["CJ"] - t["BJ"]) < 1e-9
    assert round(t["FP"], 2) == round(t["FN"], 2) == 15.63
    assert round(r["tabla"]["c_CJ (Youden clásico)"]["c"], 4) == 0.2318
    assert round(r["tabla"]["c_VPP=0,5"]["c"], 4) == 0.5327


@pytest.mark.parametrize("kappa,c_esperado", [(0.75, 0.4380), (1.25, 0.4286), (1.5, 0.4259)])
def test_tabla_5_kappa(kappa, c_esperado):
    assert round(umbral.desde_matriz(VP, FN, FP, VN, kappa=kappa)["c_star"], 4) == c_esperado


def test_metodos_numericos():
    mod = ModeloBinormal.desde_conteos(VP, FN, FP, VN)
    g, dg = (lambda c: mod.g(c, P)), (lambda c: mod.dg(c, P))
    ref = 0.4323951143849492
    for r in [raices.biseccion(g, .01, .99), raices.newton(g, dg, .5), raices.secante(g, .5, .45),
              raices.brent(g, .01, .99), raices.itp(g, .01, .99)]:
        assert r["convergio"] and abs(r["raiz"] - ref) < 1e-12, r["metodo"]
    nr = raices.newton(g, dg, .5)
    errores = [abs(x - ref) for x in nr["puntos"][1:4]]
    assert [f"{e:.2e}" for e in errores] == ["1.05e-02", "2.30e-04", "1.11e-07"]


def test_mcc_cota():
    r = umbral.desde_matriz(VP, FN, FP, VN)
    t = r["tabla"]["c* (propuesto)"]
    assert abs(t["MCC"] - t["F"]) < 1e-9          # en c*, F = MCC


def test_sinteticos_y_cv_agrupada():
    from sklearn.linear_model import LogisticRegression
    df = sinteticos.generar()
    X, y = df[sinteticos.PREDICTORES].to_numpy(), df["enfermo"].to_numpy()
    p = validacion.predicciones_cv(lambda s: LogisticRegression(max_iter=1000), X, y,
                                   grupos=df["ubicacion"].to_numpy())
    assert not np.isnan(p).any()
    r = umbral.desde_puntajes(y, p)
    assert 0 < r["c_star"] < 1
