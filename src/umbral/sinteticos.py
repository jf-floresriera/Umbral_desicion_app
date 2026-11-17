"""Datos sintéticos con la estructura del experimento de campo.

Ubicaciones en un lote con coordenadas, observadas en varias campañas; cinco
predictores (NDVI, EVI, NDRE, GLI, altura) que dependen de un estado latente de
enfermedad con correlación espacial y avance temporal. Sirven para probar la app
sin usar los datos de campo, que no son públicos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PREDICTORES = ["NDVI", "EVI", "NDRE", "GLI", "altura"]


def generar(n_ubicaciones=103, campanas=(99, 113, 127, 141), prevalencia=0.13,
            separacion=1.2, alcance=15.0, lado=130.0, prob_obs=0.5, semilla=20260212):
    rng = np.random.default_rng(semilla)
    xy = rng.uniform(0, lado, size=(n_ubicaciones, 2))
    # campo latente con correlación espacial exponencial
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    L = np.linalg.cholesky(np.exp(-d / alcance) + 1e-8 * np.eye(n_ubicaciones))
    campo = L @ rng.standard_normal(n_ubicaciones)
    filas = []
    for k, dds in enumerate(campanas):
        obs = rng.random(n_ubicaciones) < prob_obs
        avance = 0.6 * k
        for i in np.flatnonzero(obs):
            filas.append((i, dds, campo[i] + avance + 0.5 * rng.standard_normal()))
    df = pd.DataFrame(filas, columns=["ubicacion", "dds", "latente"])
    corte = np.quantile(df["latente"], 1 - prevalencia)
    df["enfermo"] = (df["latente"] >= corte).astype(int)
    y = df["enfermo"].to_numpy()
    edad = (df["dds"] - df["dds"].min()) / 42.0
    ruido = rng.standard_normal((len(df), 5))
    efecto = separacion * y
    df["NDVI"] = 0.80 - 0.05 * edad - 0.06 * efecto + 0.04 * ruido[:, 0]
    df["EVI"] = 0.55 - 0.04 * edad - 0.05 * efecto + 0.05 * ruido[:, 1]
    df["NDRE"] = 0.35 - 0.03 * edad - 0.02 * efecto + 0.04 * ruido[:, 2]
    df["GLI"] = 0.25 - 0.02 * edad - 0.04 * efecto + 0.04 * ruido[:, 3]
    df["altura"] = 0.60 - 0.05 * edad - 0.08 * efecto + 0.10 * ruido[:, 4]
    df["x"], df["y"] = xy[df["ubicacion"], 0], xy[df["ubicacion"], 1]
    return df.drop(columns="latente")
