# Instala lo necesario para la app en R (una sola vez).
# torch se instala SOLO para CPU: no necesita tarjeta gráfica ni controladores de video.

# Acepta automáticamente la descarga de libtorch (evita la pregunta "Do you want to continue?")
Sys.setenv(TORCH_INSTALL = "1")

paquetes <- c("shiny", "bslib", "bsicons", "plotly", "DT", "torch")
faltan <- setdiff(paquetes, rownames(installed.packages()))
if (length(faltan)) install.packages(faltan, repos = "https://cloud.r-project.org")

# Sin tarjeta gráfica NVIDIA, install_torch() baja automáticamente la versión CPU.
if (!torch::torch_is_installed()) torch::install_torch()

if (torch::torch_is_installed()) {
  message("✔ torch listo: ", paste(dim(torch::torch_randn(2, 3)), collapse = "x"), " (prueba de tensor)")
  message("Ejecuta: shiny::runApp('app_r')")
} else {
  message("✖ torch no quedó instalado; ejecuta torch::install_torch() y responde Yes.")
}
