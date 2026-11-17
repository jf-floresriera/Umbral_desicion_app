# Carpeta de datos (local)

Los datos de campo **no se publican** en este repositorio: todo lo que se ponga aquí,
salvo este archivo, está excluido por `.gitignore`.

Coloca tu archivo (por ejemplo `healthy_diseased.csv`) en esta carpeta o cárgalo
desde la pestaña 3 de la app. El CSV debe tener, por fila (observación sitio-campaña):

| Columna sugerida | Contenido |
| --- | --- |
| clase | severidad original 0–5 o la clase binaria |
| ubicacion | identificador de la ubicación (para la validación cruzada agrupada) |
| campana o dds | campaña o días después de la siembra |
| NDVI, EVI, NDRE, GLI, altura | los cinco predictores |

En la app se elige qué valor de la clase cuenta como positivo (enfermo); así se
puede verificar la codificación sin modificar el archivo.
