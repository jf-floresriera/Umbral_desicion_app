# App R Shiny — Umbral de decisión con el índice de Youden bayesiano
# Archivo único: basta copiar este app.R a cualquier carpeta y ejecutar shiny::runApp("carpeta")

library(shiny)
library(bslib)
library(bsicons)
library(plotly)
library(DT)
# ============================================================================
# NÚCLEO DE CÁLCULO (incluido aquí para que app.R funcione solo, en cualquier carpeta)
# ============================================================================
# Núcleo de cálculo en R (traducción del paquete Python src/umbral, validado contra el artículo).

# ---------------------------------------------------------------- métricas ----
metricas_tasas <- function(Se, Sp, P) {
  vp <- P * Se; fn <- P * (1 - Se); fp <- (1 - P) * (1 - Sp); vn <- (1 - P) * Sp
  Q <- vp + fp
  VPP <- vp / Q; VPN <- vn / (1 - Q)
  CJ <- Se + Sp - 1; BJ <- VPP + VPN - 1
  data.frame(P = P, Q = Q, Se = Se, Sp = Sp, VPP = VPP, VPN = VPN, CJ = CJ, BJ = BJ,
             F = pmin(CJ, BJ), MCC = sign(CJ) * sqrt(abs(CJ * BJ)), CA = vp + vn,
             BG = sqrt(VPP * VPN), F1 = 2 * vp / (2 * vp + fp + fn))
}

metricas_conteos <- function(VP, FN, FP, VN, P = NULL) {
  if (is.null(P)) P <- (VP + FN) / (VP + FN + FP + VN)
  metricas_tasas(VP / (VP + FN), VN / (VN + FP), P)
}

prevalencias_equilibrio <- function(Se, Sp) {
  CJ <- Se + Sp - 1; u <- Sp * (1 - Sp); v <- Se * (1 - Se)
  a <- sqrt(Se * Sp); b <- sqrt((1 - Se) * (1 - Sp))
  data.frame(P1 = (1 - Sp) / (1 - CJ), P2 = Sp / (Se + Sp),
             Popt = sqrt(u) / (sqrt(u) + sqrt(v)), max_BJ = (a - b) / (a + b),
             max_BG = a / (a + b), DOR = (a / b)^2)
}

# ------------------------------------------------------ modelo logit-binormal ----
logit <- function(c) log(c / (1 - c))
sigmoide <- function(t) 1 / (1 + exp(-t))

modelo_binormal <- function(VP, FN, FP, VN, kappa = 1) {
  Se0 <- VP / (VP + FN); Sp0 <- VN / (VN + FP)
  if (Se0 <= 0 || Se0 >= 1 || Sp0 <= 0 || Sp0 >= 1)
    stop("Se y Sp en c0 = 0,5 deben estar estrictamente entre 0 y 1.")
  zSe <- qnorm(Se0); zSp <- qnorm(Sp0); tau <- zSe + zSp
  if (tau <= 0) stop("d' <= 0: el clasificador no separa las clases.")
  m <- list(kappa = kappa, tau = tau, m0 = -tau * zSp, m1 = kappa * tau * zSe,
            n1 = VP + FN, n0 = FP + VN)
  m$s0 <- function(c) (logit(c) - m$m0) / m$tau
  m$s1 <- function(c) (logit(c) - m$m1) / (m$kappa * m$tau)
  m$Se <- function(c) pnorm(m$s1(c), lower.tail = FALSE)
  m$Sp <- function(c) pnorm(m$s0(c))
  m$g <- function(c, P) (1 - P) * pnorm(m$s0(c), lower.tail = FALSE) - P * pnorm(m$s1(c))
  m$dg <- function(c, P) {
    j <- c * (1 - c)
    -(1 - P) * dnorm(m$s0(c)) / (m$tau * j) - P * dnorm(m$s1(c)) / (m$kappa * m$tau * j)
  }
  m$auc <- pnorm((m$m1 - m$m0) / (m$tau * sqrt(1 + kappa^2)))
  m$c_cj <- if (kappa == 1) sigmoide((m$m0 + m$m1) / 2) else {
    sigmoide(optimize(function(t) -(m$Se(sigmoide(t)) + m$Sp(sigmoide(t))), c(-20, 20))$minimum)
  }
  m
}

c_vpp <- function(m, P, objetivo = 0.5) {
  h <- function(c) { se <- m$Se(c); fpr <- 1 - m$Sp(c); P * se / (P * se + (1 - P) * fpr) - objetivo }
  r <- try(uniroot(h, c(1e-6, 1 - 1e-6), tol = 1e-12)$root, silent = TRUE)
  if (inherits(r, "try-error")) NA_real_ else r
}

# ---------------------------------------------------------- métodos numéricos ----
.contador <- function(f) {
  env <- new.env(); env$x <- numeric(0)
  list(f = function(x) { env$x <- c(env$x, x); f(x) }, puntos = function() env$x)
}
.res <- function(metodo, raiz, cnt, n_dg = 0, ok = TRUE)
  list(metodo = metodo, raiz = raiz, puntos = cnt$puntos(), eval_g = length(cnt$puntos()),
       eval_dg = n_dg, convergio = ok)

biseccion <- function(g, a, b, tol = 1e-14, max_iter = 200) {
  cnt <- .contador(g); ga <- cnt$f(a); cnt$f(b); c <- (a + b) / 2
  for (i in seq_len(max_iter)) {
    c <- (a + b) / 2; gc <- cnt$f(c)
    if (gc == 0 || (b - a) / 2 < tol) break
    if (ga * gc < 0) b <- c else { a <- c; ga <- gc }
  }
  .res("Bisección", c, cnt)
}

newton <- function(g, dg, x0, tol = 1e-14, max_iter = 50) {
  cnt <- .contador(g); x <- x0; n_dg <- 0
  for (i in seq_len(max_iter)) {
    gx <- cnt$f(x); d <- dg(x); n_dg <- n_dg + 1
    xn <- x - gx / d
    if (!is.finite(xn) || xn <= 0 || xn >= 1) return(.res("Newton–Raphson", xn, cnt, n_dg, FALSE))
    if (abs(xn - x) <= tol) return(.res("Newton–Raphson", xn, cnt, n_dg))
    x <- xn
  }
  .res("Newton–Raphson", x, cnt, n_dg, FALSE)
}

secante <- function(g, x0, x1, tol = 1e-14, max_iter = 100) {
  cnt <- .contador(g); g0 <- cnt$f(x0); g1 <- cnt$f(x1)
  for (i in seq_len(max_iter)) {
    if (g1 == g0) break
    x2 <- x1 - g1 * (x1 - x0) / (g1 - g0)
    if (!is.finite(x2) || x2 <= 0 || x2 >= 1) return(.res("Secante", x2, cnt, ok = FALSE))
    if (abs(x2 - x1) <= tol) return(.res("Secante", x2, cnt))
    x0 <- x1; g0 <- g1; x1 <- x2; g1 <- cnt$f(x1)
  }
  .res("Secante", x1, cnt, ok = abs(g1) < 1e-12)
}

brent <- function(g, a, b, tol = 1e-14) {
  cnt <- .contador(g)
  r <- uniroot(cnt$f, c(a, b), tol = tol, maxiter = 1000)$root   # Brent–Dekker
  .res("Brent", r, cnt)
}

itp <- function(g, a, b, tol = 1e-14, k1 = NULL, k2 = 2, n0 = 1, max_iter = 500) {
  cnt <- .contador(g); if (is.null(k1)) k1 <- 0.2 / (b - a)
  ya <- cnt$f(a); yb <- cnt$f(b); s0 <- if (yb > ya) 1 else -1
  ya <- s0 * ya; yb <- s0 * yb
  n_max <- ceiling(log2((b - a) / (2 * tol))) + n0; j <- 0
  while ((b - a) > 2 * tol && j < max_iter) {
    xh <- (a + b) / 2; r <- tol * 2^(n_max - j) - (b - a) / 2; delta <- k1 * (b - a)^k2
    xf <- (yb * a - ya * b) / (yb - ya); s <- sign(xh - xf); if (s == 0) s <- 1
    xt <- if (delta <= abs(xh - xf)) xf + s * delta else xh
    x <- if (abs(xt - xh) <= r) xt else xh - s * r
    y <- s0 * cnt$f(x)
    if (y > 0) { b <- x; yb <- y } else if (y < 0) { a <- x; ya <- y } else { a <- x; b <- x }
    j <- j + 1
  }
  .res("ITP", (a + b) / 2, cnt)
}

comparar_metodos <- function(m, P, ref) {
  g <- function(c) m$g(c, P); dg <- function(c) m$dg(c, P)
  res <- list(biseccion(g, .01, .99), newton(g, dg, .5), secante(g, .5, .45),
              brent(g, .01, .99), itp(g, .01, .99))
  tiempo <- function(f) { t0 <- proc.time()[["elapsed"]]; for (i in 1:50) f(); (proc.time()[["elapsed"]] - t0) / 50 * 1e6 }
  fs <- list(function() biseccion(g, .01, .99), function() newton(g, dg, .5),
             function() secante(g, .5, .45), function() brent(g, .01, .99), function() itp(g, .01, .99))
  tabla <- do.call(rbind, lapply(seq_along(res), function(i) {
    r <- res[[i]]; e <- abs(r$puntos - ref)
    data.frame(Método = r$metodo, Raíz = sprintf("%.10f", r$raiz), `Eval. g` = r$eval_g,
               `Eval. g'` = r$eval_dg, `Eval. hasta 1e-12` = which(e < 1e-12)[1],
               `Tiempo (µs)` = round(tiempo(fs[[i]]), 1), Convergió = r$convergio, check.names = FALSE)
  }))
  errores <- do.call(rbind, lapply(res, function(r)
    data.frame(metodo = r$metodo, k = seq_along(r$puntos), error = pmax(abs(r$puntos - ref), 1e-16))))
  list(tabla = tabla, errores = errores)
}

# --------------------------------------------------------- umbral propuesto ----
umbral_desde_matriz <- function(VP, FN, FP, VN, P = NULL, kappa = 1) {
  if (is.null(P)) P <- (VP + FN) / (VP + FN + FP + VN)
  m <- modelo_binormal(VP, FN, FP, VN, kappa)
  r <- newton(function(c) m$g(c, P), function(c) m$dg(c, P), 0.5)
  if (!r$convergio) r <- brent(function(c) m$g(c, P), 1e-6, 1 - 1e-6)
  cs <- r$raiz
  umbrales <- c(`c0 (red)` = 0.5, `c* (propuesto)` = cs, `c_CJ (Youden clásico)` = m$c_cj,
                `c_VPP=0,5` = c_vpp(m, P))
  umbrales <- umbrales[is.finite(umbrales)]
  filas <- lapply(names(umbrales), function(nm) {
    c <- umbrales[[nm]]
    if (nm == "c0 (red)") {
      cel <- c(VP = VP, FN = FN, FP = FP, VN = VN); met <- metricas_conteos(VP, FN, FP, VN, P)
    } else {
      se <- m$Se(c); sp <- m$Sp(c)
      cel <- c(VP = m$n1 * se, FN = m$n1 * (1 - se), FP = m$n0 * (1 - sp), VN = m$n0 * sp)
      met <- metricas_tasas(se, sp, P)
    }
    cbind(data.frame(Umbral = nm, c = c), as.data.frame(t(cel)), met[, setdiff(names(met), "P")])
  })
  list(modelo = m, P = P, c_star = cs, tabla = do.call(rbind, filas))
}

curvas_modelo <- function(m, P, n = 600) {
  c <- seq(1e-3, 1 - 1e-3, length.out = n)
  Se <- m$Se(c); Sp <- m$Sp(c); Q <- P * Se + (1 - P) * (1 - Sp)
  CJ <- Se + Sp - 1; BJ <- CJ * P * (1 - P) / (Q * (1 - Q))
  data.frame(c = c, CJ = CJ, BJ = BJ, F = pmin(CJ, BJ))
}

umbral_desde_puntajes <- function(y, p, P = mean(y)) {
  c_star <- unname(quantile(p, 1 - P, type = 7))
  cs <- sort(unique(c(seq(0.001, 0.999, by = 0.001), p)))
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  Se <- vapply(cs, function(c) sum(p[y == 1] >= c) / n1, 0)
  Sp <- vapply(cs, function(c) sum(p[y == 0] < c) / n0, 0)
  Q <- P * Se + (1 - P) * (1 - Sp); CJ <- Se + Sp - 1
  BJ <- ifelse(Q > 0 & Q < 1, CJ * P * (1 - P) / (Q * (1 - Q)), NA)
  F <- pmin(CJ, BJ)
  conteos <- function(c) { yh <- as.integer(p >= c)
    c(VP = sum(yh == 1 & y == 1), FN = sum(yh == 0 & y == 1), FP = sum(yh == 1 & y == 0), VN = sum(yh == 0 & y == 0)) }
  umbrales <- c(`c0 (0,5)` = 0.5, `c* (cuantil 1-P)` = c_star, `argmax F (malla)` = cs[which.max(F)])
  tabla <- do.call(rbind, lapply(names(umbrales), function(nm) {
    k <- conteos(umbrales[[nm]])
    cbind(data.frame(Umbral = nm, c = umbrales[[nm]]), as.data.frame(t(k)),
          metricas_conteos(k[["VP"]], k[["FN"]], k[["FP"]], k[["VN"]], P)[, -1])
  }))
  list(P = P, c_star = c_star, tabla = tabla, curvas = data.frame(c = cs, CJ = CJ, BJ = BJ, F = F))
}

# ------------------------------------------------------- datos sintéticos ----
PREDICTORES <- c("NDVI", "EVI", "NDRE", "GLI", "altura")

datos_sinteticos <- function(n_ubic = 103, campanas = c(99, 113, 127, 141), prevalencia = 0.13,
                             separacion = 1.2, alcance = 15, lado = 130, prob_obs = 0.5, semilla = 20260212) {
  set.seed(semilla)
  xy <- matrix(runif(2 * n_ubic, 0, lado), ncol = 2)
  L <- t(chol(exp(-as.matrix(dist(xy)) / alcance) + diag(1e-8, n_ubic)))
  campo <- as.vector(L %*% rnorm(n_ubic))
  df <- do.call(rbind, lapply(seq_along(campanas), function(k) {
    i <- which(runif(n_ubic) < prob_obs)
    data.frame(ubicacion = i, dds = campanas[k], latente = campo[i] + 0.6 * (k - 1) + 0.5 * rnorm(length(i)))
  }))
  df$enfermo <- as.integer(df$latente >= quantile(df$latente, 1 - prevalencia))
  edad <- (df$dds - min(df$dds)) / 42; e <- separacion * df$enfermo; n <- nrow(df)
  df$NDVI <- 0.80 - 0.05 * edad - 0.06 * e + 0.04 * rnorm(n)
  df$EVI <- 0.55 - 0.04 * edad - 0.05 * e + 0.05 * rnorm(n)
  df$NDRE <- 0.35 - 0.03 * edad - 0.02 * e + 0.04 * rnorm(n)
  df$GLI <- 0.25 - 0.02 * edad - 0.04 * e + 0.04 * rnorm(n)
  df$altura <- 0.60 - 0.05 * edad - 0.08 * e + 0.10 * rnorm(n)
  df$x <- xy[df$ubicacion, 1]; df$y <- xy[df$ubicacion, 2]
  df$latente <- NULL
  df
}

# ------------------------------------------------- validación cruzada ----
particiones <- function(y, grupos = NULL, k = 5, semilla = 20260212) {
  set.seed(semilla)
  if (is.null(grupos)) {                       # estratificada
    f <- integer(length(y))
    for (cl in unique(y)) { i <- which(y == cl); f[i] <- sample(rep_len(1:k, length(i))) }
    return(f)
  }
  # agrupada y estratificada: cada ubicación entera va a una sola partición
  g <- unique(grupos); g <- g[sample(length(g))]
  pos <- tapply(y, grupos, sum)[as.character(g)]; tam <- tapply(y, grupos, length)[as.character(g)]
  orden <- order(-pos, -tam); carga_pos <- numeric(k); carga_n <- numeric(k); asign <- integer(length(g))
  for (j in orden) { f <- order(carga_pos, carga_n)[1]; asign[j] <- f
    carga_pos[f] <- carga_pos[f] + pos[j]; carga_n[f] <- carga_n[f] + tam[j] }
  asign[match(grupos, g)]
}

# ------------------------------------------------------- red en torch (R) ----
# Usa el mismo motor C++ (libtorch) que PyTorch. Funciona solo con CPU.
entrenar_red <- function(X, y, h1 = 10, h2 = 5, lr = 0.05, momento = 0.9, lam = 1e-4,
                         epocas = 500, lote = 200, tol = 1e-4, paciencia = 10,
                         semilla = 20260212, peso_pos = NULL, dispositivo = "cpu") {
  # dispositivo: "cpu" (por defecto) o "auto" (GPU NVIDIA si torch la detecta)
  dev <- if (dispositivo == "auto" && torch::cuda_is_available()) "cuda" else "cpu"
  torch::torch_manual_seed(semilla); set.seed(semilla)
  media <- colMeans(X); de <- apply(X, 2, sd) + 1e-12
  Xs <- sweep(sweep(X, 2, media), 2, de, "/")
  xt <- torch::torch_tensor(Xs, dtype = torch::torch_float(), device = dev)
  yt <- torch::torch_tensor(as.numeric(y), dtype = torch::torch_float(), device = dev)
  n <- nrow(X)
  Red <- torch::nn_module(
    initialize = function(d) {
      self$l1 <- torch::nn_linear(d, h1); self$l2 <- torch::nn_linear(h1, h2); self$l3 <- torch::nn_linear(h2, 1)
      for (l in list(self$l1, self$l2, self$l3)) {
        torch::nn_init_xavier_uniform_(l$weight); torch::nn_init_zeros_(l$bias)
      }
    },
    forward = function(x) {
      x <- torch::nnf_relu(self$l1(x)); x <- torch::nnf_relu(self$l2(x))
      torch::torch_squeeze(self$l3(x), 2)
    })
  red <- Red(ncol(X)); red$to(device = dev)
  pars <- red$parameters
  es_peso <- grepl("weight", names(pars))
  opt <- torch::optim_sgd(list(list(params = unname(pars[es_peso]), weight_decay = lam / n),
                               list(params = unname(pars[!es_peso]), weight_decay = 0)),
                          lr = lr, momentum = momento, nesterov = TRUE)
  pw <- if (is.null(peso_pos)) NULL else torch::torch_tensor(peso_pos, device = dev)
  b <- min(lote, n); mejor <- Inf; sin_mejora <- 0
  for (ep in seq_len(epocas)) {
    orden <- sample(n); total <- 0
    for (i in seq(1, n, by = b)) {
      idx <- orden[i:min(i + b - 1, n)]
      opt$zero_grad()
      l <- torch::nnf_binary_cross_entropy_with_logits(red(xt[idx, ]), yt[idx], pos_weight = pw)
      l$backward(); opt$step()
      total <- total + l$item() * length(idx)
    }
    total <- total / n
    if (total > mejor - tol) { sin_mejora <- sin_mejora + 1; if (sin_mejora >= paciencia) break } else sin_mejora <- 0
    mejor <- min(mejor, total)
  }
  list(red = red, media = media, de = de, dev = dev)
}

predecir_red <- function(modelo, X) {
  Xs <- sweep(sweep(X, 2, modelo$media), 2, modelo$de, "/")
  torch::with_no_grad({
    eta <- modelo$red(torch::torch_tensor(Xs, dtype = torch::torch_float(), device = modelo$dev))
  })
  1 / (1 + exp(-as.numeric(torch::as_array(eta$cpu()))))
}

predicciones_cv <- function(X, y, grupos = NULL, k = 5, semilla = 20260212, ...) {
  f <- particiones(y, grupos, k, semilla); p <- rep(NA_real_, length(y))
  for (j in seq_len(k)) {
    tr <- f != j; te <- f == j
    mod <- entrenar_red(X[tr, , drop = FALSE], y[tr], semilla = semilla + j, ...)
    p[te] <- predecir_red(mod, X[te, , drop = FALSE])
  }
  p
}

# ============================================================================
# APP
# ============================================================================


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
