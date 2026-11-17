# Instala lo necesario para la app en R (una sola vez).
# torch se instala SOLO para CPU: no necesita tarjeta gráfica ni controladores de video.
paquetes <- c("shiny", "bslib", "bsicons", "plotly", "DT", "torch")
faltan <- setdiff(paquetes, rownames(installed.packages()))
if (length(faltan)) install.packages(faltan)

# Sin tarjeta gráfica NVIDIA, install_torch() baja automáticamente la versión CPU (~200 MB).
if (!torch::torch_is_installed()) torch::install_torch()

message("Listo. Ejecuta: shiny::runApp('app_r')")
