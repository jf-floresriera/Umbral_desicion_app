# App R Shiny — Umbral de decisión con el índice de Youden bayesiano
# Ejecutar:  shiny::runApp("app_r")   (desde la raíz del repositorio)

library(shiny)
library(bslib)
library(bsicons)
library(plotly)
library(DT)
source("R/umbral.R", encoding = "UTF-8")

# Paleta (validada para daltonismo) y superficie
COL <- list(CJ = "#2a78d6", BJ = "#eb6834", F = "#1baf7a", s4 = "#eda100", s5 = "#e87ba4",
            txt2 = "#52514e", sup = "#fcfcfb")
COL_MET <- c("Bisección" = "#2a78d6", "Newton–Raphson" = "#eb6834", "Secante" = "#1baf7a",
             "Brent" = "#eda100", "ITP" = "#e87ba4")

tema <- bs_theme(version = 5, bg = COL$sup, fg = "#0b0b0b", primary = COL$CJ,
                 secondary = COL$txt2, base_font = font_google("Inter"),
                 heading_font = font_google("Inter Tight"), "border-radius" = "0.75rem")

grafico_curvas <- function(cv, marcas, titulo) {
  p <- plot_ly(cv, x = ~c) |>
    add_lines(y = ~F, name = "F = min{C<sub>J</sub>, B<sub>J</sub>}",
              line = list(color = COL$F, width = 10), opacity = 0.35) |>
    add_lines(y = ~CJ, name = "C<sub>J</sub>(c) Youden clásico", line = list(color = COL$CJ, width = 2)) |>
    add_lines(y = ~BJ, name = "B<sub>J</sub>(c;P) Youden bayesiano", line = list(color = COL$BJ, width = 2))
  lineas <- lapply(seq_along(marcas), function(i) list(
    type = "line", x0 = marcas[[i]], x1 = marcas[[i]], y0 = 0, y1 = 1, yref = "paper",
    line = list(color = COL$txt2, dash = "dot", width = 1)))
  notas <- lapply(seq_along(marcas), function(i) list(
    x = marcas[[i]], y = 1, yref = "paper", text = names(marcas)[i], showarrow = FALSE,
    textangle = -90, xanchor = "right", yanchor = "top", font = list(size = 11, color = COL$txt2)))
  layout(p, title = list(text = titulo, font = list(size = 14)), hovermode = "x unified",
         xaxis = list(title = "umbral c", range = c(0, 1), gridcolor = "#e6e5e1"),
         yaxis = list(title = "índice", gridcolor = "#e6e5e1"),
         shapes = lineas, annotations = notas, paper_bgcolor = COL$sup, plot_bgcolor = COL$sup,
         legend = list(orientation = "h", y = -0.2)) |> config(displaylogo = FALSE)
}

tabla_dt <- function(df) {
  num <- vapply(df, is.numeric, TRUE)
  df[num] <- lapply(df[num], round, 4)
  datatable(df, rownames = FALSE, options = list(dom = "t", scrollX = TRUE, ordering = FALSE),
            class = "compact stripe hover")
}

# ------------------------------------------------------------------------ UI ----
ui <- page_navbar(
  title = span(bs_icon("bullseye"), " Umbral de Youden bayesiano"),
  theme = tema, fillable = FALSE,

  nav_panel("Desde la matriz", icon = bs_icon("grid-3x3"),
    layout_sidebar(
      sidebar = sidebar(width = 300, title = "Matriz en c₀ = 0,5",
        layout_column_wrap(width = 1/2,
          numericInput("vp", "VP", 9, min = 0), numericInput("fn", "FN", 18, min = 0),
          numericInput("fp", "FP", 10, min = 0), numericInput("vn", "VN", 175, min = 0)),
        checkboxInput("p_emp", "Usar la prevalencia empírica", TRUE),
        conditionalPanel("!input.p_emp",
          sliderInput("p_uso", "Prevalencia de uso P", 0.01, 0.5, 0.127, step = 0.001)),
        radioButtons("kappa", "κ (cociente de desviaciones)", c("0.75", "1", "1.25", "1.5"),
                     selected = "1", inline = TRUE)),
      layout_column_wrap(width = 1/4, fill = FALSE,
        value_box("Umbral propuesto c*", textOutput("vb_c"), showcase = bs_icon("crosshair"), theme = "primary"),
        value_box("F(c*) = C_J = B_J", textOutput("vb_f"), showcase = bs_icon("graph-up-arrow"), theme = "success"),
        value_box("Sensibilidad en c*", textOutput("vb_se"), showcase = bs_icon("activity")),
        value_box("AUC del modelo", textOutput("vb_auc"), showcase = bs_icon("bezier2"))),
      card(full_screen = TRUE, card_header("Función objetivo F(c) = min{C_J, B_J}"),
           plotlyOutput("graf_matriz", height = "380px")),
      card(card_header("Métricas en los cuatro umbrales (tabla 2)"), DTOutput("tabla_matriz")),
      card(card_header("Prevalencias de equilibrio y óptima (proposición 3)"), DTOutput("tabla_prop3"))
    )),

  nav_panel("Métodos numéricos", icon = bs_icon("calculator"),
    card(full_screen = TRUE, card_header("Convergencia de los cinco métodos — ecuación (3.10)"),
         plotlyOutput("graf_metodos", height = "380px")),
    card(card_header("Comparación (tabla 3)"), DTOutput("tabla_metodos"))),

  nav_panel("Datos y red (torch)", icon = bs_icon("cpu"),
    layout_sidebar(
      sidebar = sidebar(width = 320, title = "Datos",
        radioButtons("fuente", NULL, c("Sintéticos" = "sint", "Mi CSV" = "csv"), inline = TRUE),
        conditionalPanel("input.fuente == 'csv'",
          fileInput("archivo", "CSV (no se guarda ni se sube)", accept = ".csv"),
          selectInput("col_y", "Columna de la clase", NULL),
          selectInput("val_pos", "Valor positivo (enfermo)", NULL),
          selectizeInput("col_x", "Predictores", NULL, multiple = TRUE),
          selectInput("col_g", "Ubicación (CV agrupada)", NULL)),
        conditionalPanel("input.fuente == 'sint'",
          sliderInput("sint_prev", "Prevalencia sintética", 0.05, 0.5, 0.13, step = 0.01),
          sliderInput("sint_sep", "Separación entre clases", 0.2, 3, 1.2, step = 0.1)),
        radioButtons("tipo_cv", "Validación cruzada (5 particiones)",
                     c("Estratificada" = "estr", "Agrupada por ubicación" = "agr", "Comparar ambas" = "ambas"),
                     selected = "ambas"),
        numericInput("epocas", "Épocas máximas", 500, min = 10, max = 5000),
        checkboxInput("pesar", "Ponderar positivos en la pérdida", FALSE),
        selectInput("disp", "Dispositivo", c("CPU" = "cpu", "GPU NVIDIA si está disponible" = "auto")),
        input_task_button("entrenar", "Entrenar y evaluar", icon = bs_icon("play-fill"))),
      uiOutput("aviso_torch"),
      uiOutput("cajas_red"),
      card(full_screen = TRUE, card_header("Curvas empíricas con las salidas fuera de muestra"),
           plotlyOutput("graf_red", height = "380px")),
      card(card_header("Umbrales empíricos"), DTOutput("tabla_red"))
    )),

  nav_panel("Cambio de prevalencia", icon = bs_icon("sliders"),
    card(full_screen = TRUE, card_header("c*(P) para cuatro valores de κ (figura 6)"),
         plotlyOutput("graf_prev", height = "380px")),
    card(card_header("F en el umbral fijo y en el recalculado (tabla 6, κ = 1)"), DTOutput("tabla_prev"))),

  nav_spacer(),
  nav_item(input_dark_mode(id = "modo"))
)

# -------------------------------------------------------------------- server ----
server <- function(input, output, session) {

  res_m <- reactive({
    req(input$vp, input$fn, input$fp, input$vn)
    P <- if (isTRUE(input$p_emp)) NULL else input$p_uso
    umbral_desde_matriz(input$vp, input$fn, input$fp, input$vn, P = P, kappa = as.numeric(input$kappa))
  })
  fila_cs <- reactive(subset(res_m()$tabla, Umbral == "c* (propuesto)"))

  output$vb_c <- renderText(sprintf("%.4f", res_m()$c_star))
  output$vb_f <- renderText(sprintf("%.3f", fila_cs()$F))
  output$vb_se <- renderText(sprintf("%.1f %%", 100 * fila_cs()$Se))
  output$vb_auc <- renderText(sprintf("%.3f", res_m()$modelo$auc))

  output$graf_matriz <- renderPlotly({
    r <- res_m(); t <- r$tabla
    grafico_curvas(curvas_modelo(r$modelo, r$P), setNames(t$c, sub(" .*", "", t$Umbral)),
                   sprintf("P = %.3f", r$P))
  })
  output$tabla_matriz <- renderDT(tabla_dt(res_m()$tabla))
  output$tabla_prop3 <- renderDT({
    tabla_dt(prevalencias_equilibrio(input$vp / (input$vp + input$fn), input$vn / (input$vn + input$fp)))
  })

  comp <- reactive({ r <- res_m(); comparar_metodos(r$modelo, r$P, r$c_star) })
  output$tabla_metodos <- renderDT(tabla_dt(comp()$tabla))
  output$graf_metodos <- renderPlotly({
    e <- comp()$errores
    plot_ly(e, x = ~k, y = ~error, color = ~metodo, colors = COL_MET, type = "scatter",
            mode = "lines+markers", marker = list(size = 8), line = list(width = 2)) |>
      layout(yaxis = list(type = "log", title = "|c(k) − c*|", exponentformat = "e", gridcolor = "#e6e5e1"),
             xaxis = list(title = "evaluaciones de g", gridcolor = "#e6e5e1"),
             paper_bgcolor = COL$sup, plot_bgcolor = COL$sup, legend = list(orientation = "h", y = -0.2)) |>
      config(displaylogo = FALSE)
  })

  # --- datos y red ---
  torch_listo <- reactive(requireNamespace("torch", quietly = TRUE) && torch::torch_is_installed())
  output$aviso_torch <- renderUI({
    if (torch_listo()) return(NULL)
    card(class = "border-warning", card_body(bs_icon("exclamation-triangle"),
      " Falta torch. En R: install.packages('torch'); torch::install_torch()  (solo CPU, no necesita tarjeta gráfica)."))
  })

  datos <- reactive({ f <- input$archivo; if (is.null(f)) NULL else read.csv(f$datapath, check.names = FALSE) })
  observeEvent(datos(), {
    cols <- names(datos())
    updateSelectInput(session, "col_y", choices = cols)
    updateSelectizeInput(session, "col_x", choices = cols)
    updateSelectInput(session, "col_g", choices = c("(ninguna)", cols))
  })
  observeEvent(input$col_y, {
    req(datos(), input$col_y %in% names(datos()))
    updateSelectInput(session, "val_pos", choices = sort(unique(as.character(datos()[[input$col_y]]))))
  })

  entrenamiento <- eventReactive(input$entrenar, {
    validate(need(torch_listo(), "Instala torch para entrenar la red."))
    if (input$fuente == "sint") {
      d <- datos_sinteticos(prevalencia = input$sint_prev, separacion = input$sint_sep)
      X <- as.matrix(d[, PREDICTORES]); y <- d$enfermo; grupos <- d$ubicacion
    } else {
      validate(need(!is.null(datos()) && length(input$col_x) > 0, "Carga el CSV y elige clase y predictores."))
      d <- na.omit(datos()[, unique(c(input$col_y, input$col_x, if (input$col_g != "(ninguna)") input$col_g))])
      X <- as.matrix(d[, input$col_x]); y <- as.integer(as.character(d[[input$col_y]]) == input$val_pos)
      grupos <- if (input$col_g == "(ninguna)") NULL else d[[input$col_g]]
    }
    P <- mean(y); pw <- if (isTRUE(input$pesar)) (1 - P) / P else NULL
    esq <- switch(input$tipo_cv, estr = "Estratificada", agr = "Agrupada", ambas = c("Estratificada", "Agrupada"))
    if (is.null(grupos)) esq <- "Estratificada"
    out <- list()
    withProgress(message = "Entrenando la red", value = 0, {
      for (nm in esq) {
        incProgress(1 / length(esq), detail = paste("CV", tolower(nm)))
        p <- predicciones_cv(X, y, if (nm == "Agrupada") grupos else NULL, epocas = input$epocas, peso_pos = pw,
                              dispositivo = input$disp)
        out[[nm]] <- list(p = p, emp = umbral_desde_puntajes(y, p))
      }
    })
    list(y = y, P = P, esq = out)
  })

  output$cajas_red <- renderUI({
    r <- entrenamiento()
    cajas <- lapply(names(r$esq), function(nm) {
      t <- r$esq[[nm]]$emp$tabla; t0 <- t[1, ]
      value_box(paste("CV", tolower(nm)), sprintf("c* = %.3f", r$esq[[nm]]$emp$c_star),
                sprintf("En 0,5: VP=%d FN=%d FP=%d VN=%d", t0$VP, t0$FN, t0$FP, t0$VN),
                showcase = bs_icon(if (nm == "Agrupada") "geo-alt" else "shuffle"),
                theme = if (nm == "Agrupada") "success" else "primary")
    })
    vb_p <- value_box("Prevalencia de los datos", sprintf("%.3f", r$P), sprintf("n = %d", length(r$y)),
                      showcase = bs_icon("pie-chart"))
    do.call(layout_column_wrap, c(list(width = 1/2, fill = FALSE, vb_p), cajas))
  })
  output$graf_red <- renderPlotly({
    r <- entrenamiento(); nm <- tail(names(r$esq), 1); e <- r$esq[[nm]]$emp
    grafico_curvas(e$curvas, c(`c*` = e$c_star, `0,5` = 0.5), paste("CV", tolower(nm)))
  })
  output$tabla_red <- renderDT({
    r <- entrenamiento()
    tabla_dt(do.call(rbind, lapply(names(r$esq), function(nm) {
      t <- r$esq[[nm]]$emp$tabla; t$Umbral <- paste(nm, "·", t$Umbral); t })))
  })

  # --- cambio de prevalencia ---
  barrido <- reactive({
    Ps <- seq(0.01, 0.5, length.out = 50)
    do.call(rbind, lapply(c(0.75, 1, 1.25, 1.5), function(k) do.call(rbind, lapply(Ps, function(P) {
      r <- umbral_desde_matriz(input$vp, input$fn, input$fp, input$vn, P = P, kappa = k)
      data.frame(kappa = k, P = P, c_star = r$c_star, F_c0 = r$tabla$F[1],
                 F_cstar = r$tabla$F[r$tabla$Umbral == "c* (propuesto)"])
    }))))
  })
  output$graf_prev <- renderPlotly({
    plot_ly(barrido(), x = ~P, y = ~c_star, color = ~factor(paste("κ =", kappa)),
            colors = unname(unlist(COL[c("CJ", "BJ", "F", "s4")])), type = "scatter", mode = "lines",
            line = list(width = 2)) |>
      layout(xaxis = list(title = "prevalencia de uso P", gridcolor = "#e6e5e1"),
             yaxis = list(title = "c*(P)", gridcolor = "#e6e5e1"),
             shapes = list(list(type = "line", x0 = 0, x1 = 0.5, y0 = 0.5, y1 = 0.5,
                                line = list(color = COL$txt2, dash = "dot", width = 1))),
             paper_bgcolor = COL$sup, plot_bgcolor = COL$sup, legend = list(orientation = "h", y = -0.2)) |>
      config(displaylogo = FALSE)
  })
  output$tabla_prev <- renderDT({
    b <- subset(barrido(), kappa == 1); tabla_dt(b[seq(1, nrow(b), by = 7), ])
  })
}

shinyApp(ui, server)
