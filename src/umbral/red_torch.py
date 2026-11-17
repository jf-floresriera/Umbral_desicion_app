"""Perceptrón multicapa 5-10-5-1 en PyTorch con los hiperparámetros de la sección 4.3.

Interfaz tipo scikit-learn (fit / predict_proba) para usarlo en `validacion.predicciones_cv`.
La estandarización se ajusta solo con los datos de entrenamiento de cada partición.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError("Instala PyTorch: pip install torch") from e


class _MLP(nn.Module):
    def __init__(self, d_in=5, h1=10, h2=5):
        super().__init__()
        self.capas = nn.Sequential(nn.Linear(d_in, h1), nn.ReLU(),
                                   nn.Linear(h1, h2), nn.ReLU(),
                                   nn.Linear(h2, 1))
        for m in self.capas:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # Glorot & Bengio (2010)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.capas(x).squeeze(-1)           # preactivación eta


class RedTorch:
    def __init__(self, h1=10, h2=5, lr=0.05, momento=0.9, lam=1e-4, epocas=500,
                 lote=200, tol=1e-4, paciencia=10, semilla=20260212, peso_positivos=None,
                 dispositivo="cpu"):
        self.h1, self.h2, self.lr, self.momento, self.lam = h1, h2, lr, momento, lam
        self.epocas, self.lote, self.tol, self.paciencia = epocas, lote, tol, paciencia
        self.semilla, self.peso_positivos = semilla, peso_positivos
        self.dispositivo = dispositivo
        self.historial = []

    def _device(self):
        """'cpu' (por defecto), 'cuda' o 'auto' (GPU NVIDIA si está disponible)."""
        if self.dispositivo == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dispositivo == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se pidió 'cuda' pero no hay GPU NVIDIA disponible; usa 'cpu' o 'auto'.")
        return torch.device(self.dispositivo)

    def fit(self, X, y):
        torch.manual_seed(self.semilla)
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        self.media_, self.de_ = X.mean(0), X.std(0) + 1e-12
        self.device_ = self._device()
        Xt = torch.from_numpy((X - self.media_) / self.de_).to(self.device_)
        yt = torch.from_numpy(y).to(self.device_)
        n = len(yt)
        self.red_ = _MLP(X.shape[1], self.h1, self.h2).to(self.device_)
        pesos = [p for nombre, p in self.red_.named_parameters() if "weight" in nombre]
        sesgos = [p for nombre, p in self.red_.named_parameters() if "bias" in nombre]
        # penalización (lam / 2n) ||W||^2  ->  weight_decay = lam / n (solo pesos), ec. (4.6)
        opt = torch.optim.SGD([{"params": pesos, "weight_decay": self.lam / n},
                               {"params": sesgos, "weight_decay": 0.0}],
                              lr=self.lr, momentum=self.momento, nesterov=True)
        pw = None if self.peso_positivos is None else torch.tensor(self.peso_positivos, device=self.device_)
        perdida = nn.BCEWithLogitsLoss(pos_weight=pw)
        lote = min(self.lote, n)
        g = torch.Generator().manual_seed(self.semilla)
        mejor, sin_mejora = np.inf, 0
        for _ in range(self.epocas):
            orden = torch.randperm(n, generator=g).to(self.device_)
            total = 0.0
            for i in range(0, n, lote):
                idx = orden[i:i + lote]
                opt.zero_grad()
                l = perdida(self.red_(Xt[idx]), yt[idx])
                l.backward()
                opt.step()
                total += l.item() * len(idx)
            total /= n
            self.historial.append(total)
            if total > mejor - self.tol:
                sin_mejora += 1
                if sin_mejora >= self.paciencia:
                    break
            else:
                sin_mejora = 0
            mejor = min(mejor, total)
        return self

    @torch.no_grad()
    def preactivacion(self, X):
        X = (np.asarray(X, dtype=np.float32) - self.media_) / self.de_
        return self.red_(torch.from_numpy(X).to(self.device_)).cpu().numpy()

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.preactivacion(X)))
        return np.column_stack([1 - p, p])
