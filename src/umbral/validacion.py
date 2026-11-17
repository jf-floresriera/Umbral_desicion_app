"""Predicciones fuera de muestra con validación cruzada estratificada o agrupada."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold


def predicciones_cv(fabrica_modelo, X, y, grupos=None, k=5, semilla=20260212):
    """Devuelve p fuera de muestra. Si `grupos` no es None, ninguna ubicación aparece
    a la vez en entrenamiento y en prueba (StratifiedGroupKFold)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
    if grupos is None:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla)
        particiones = cv.split(X, y)
    else:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semilla)
        particiones = cv.split(X, y, groups=np.asarray(grupos))
    p = np.full(len(y), np.nan)
    for f, (tr, te) in enumerate(particiones):
        modelo = fabrica_modelo(semilla + f)
        modelo.fit(X[tr], y[tr])
        p[te] = modelo.predict_proba(X[te])[:, 1]
    return p


def matriz(y, p, c=0.5):
    y, yh = np.asarray(y).astype(int), (np.asarray(p) >= c).astype(int)
    return {"VP": int(((yh == 1) & (y == 1)).sum()), "FN": int(((yh == 0) & (y == 1)).sum()),
            "FP": int(((yh == 1) & (y == 0)).sum()), "VN": int(((yh == 0) & (y == 0)).sum())}
