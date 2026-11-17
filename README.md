# Umbral de decisión sensible a la prevalencia (índice de Youden bayesiano)

Aplicación y código de apoyo del artículo *«Umbral de decisión sensible a la
prevalencia para perceptrones multicapa basado en el índice de Youden bayesiano:
aplicación en sensoramiento remoto»*.

El umbral propuesto es

c\* = argmax_c min{ C_J(c), B_J(c; P) },

la única raíz de Q(c) = P: el umbral en que se igualan los falsos positivos y los
falsos negativos esperados.

## Contenido

```
src/umbral/
  metricas.py     Se, Sp, VPP, VPN, C_J, B_J, B_G, F1, MCC; proposición 3
  binormal.py     modelo logit-binormal desde la matriz de confusión (sección 3.3)
  raices.py       bisección, Newton–Raphson, secante, Brent e ITP (sección 3.4)
  umbral.py       c* desde la matriz (modelo) o desde los puntajes (cuantil 1 − P)
  red_torch.py    perceptrón 5-10-5-1 en PyTorch con los hiperparámetros de la sección 4.3
  validacion.py   validación cruzada estratificada o agrupada por ubicación
  sinteticos.py   datos sintéticos con la estructura del experimento de campo
app/app.py        app Shiny para Python
tests/            pruebas que reproducen los números del artículo
data/             datos locales (excluidos de git)
```

## Instalación y uso

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                      # reproduce las tablas 2, 3, 5 y 6 del artículo
shiny run --reload app/app.py  # abre la app en http://127.0.0.1:8000
```

## La app

1. **Desde la matriz de confusión**: c\*, modelo logit-binormal, tabla de los cuatro
   umbrales y proposición 3, a partir de VP, FN, FP y VN.
2. **Métodos numéricos**: comparación de los cinco métodos para la ecuación (3.10).
3. **Datos y red (PyTorch)**: entrena el perceptrón con datos sintéticos o con un CSV
   local, con validación cruzada estratificada y agrupada por ubicación, y calcula el
   umbral empírico.
4. **Cambio de prevalencia**: c\*(P) para varios valores de κ.

## Datos

Los datos de campo no son públicos. Ver `data/README.md`.
