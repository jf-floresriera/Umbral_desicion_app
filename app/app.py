"""App Shiny (Python) para evaluar el umbral propuesto c* con PyTorch.

Ejecutar desde la raíz del repositorio:
    shiny run --reload app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from umbral import metricas as M, raices, sinteticos, umbral, validacion  # noqa: E402
from umbral.binormal import ModeloBinormal  # noqa: E402

COLUMNAS_TABLA = ["c", "VP", "FN", "FP", "VN", "Q", "Se", "Sp", "VPP", "VPN",
                  "CJ", "BJ", "F", "MCC", "CA", "BG", "F1"]


def _tabla(d: dict) -> pd.DataFrame:
    df = pd.DataFrame(d).T
    cols = [c for c in COLUMNAS_TABLA if c in df.columns]
    return df[cols].astype(float).round(4).reset_index(names="Umbral")


def _grafico_curvas(curvas, marcas: dict, titulo: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curvas["c"], curvas["CJ"], label="$C_J(c)$")
    ax.plot(curvas["c"], curvas["BJ"], label="$B_J(c;P)$")
    ax.plot(curvas["c"], curvas["F"], lw=4, alpha=0.35, label="$F=\\min\\{C_J,B_J\\}$")
    for nombre, c in marcas.items():
        ax.axvline(c, ls="--", lw=1, color="gray")
        ax.text(c, ax.get_ylim()[1] * 0.95, nombre, rotation=90, va="top", fontsize=8)
    ax.set_xlabel("umbral c"); ax.set_ylabel("índice"); ax.set_title(titulo)
    ax.set_xlim(0, 1); ax.legend(loc="lower right"); fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- UI
pestaña_matriz = ui.nav_panel(
    "1 · Desde la matriz de confusión",
    ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Matriz en c₀ = 0,5"),
            ui.input_numeric("vp", "VP", 9, min=0), ui.input_numeric("fn", "FN", 18, min=0),
            ui.input_numeric("fp", "FP", 10, min=0), ui.input_numeric("vn", "VN", 175, min=0),
            ui.input_checkbox("p_emp", "Usar la prevalencia empírica", True),
            ui.panel_conditional("!input.p_emp",
                                 ui.input_slider("p_uso", "Prevalencia de uso P", 0.01, 0.5, 0.127, step=0.001)),
            ui.input_select("kappa", "κ (cociente de desviaciones)", ["0.75", "1", "1.25", "1.5"], selected="1"),
            width=300,
        ),
        ui.output_text_verbatim("res_umbral"),
        ui.output_plot("graf_matriz", height="380px"),
        ui.h5("Métricas en los cuatro umbrales (tabla 2)"),
        ui.output_data_frame("tabla_matriz"),
        ui.h5("Prevalencias de equilibrio y óptima (proposición 3)"),
        ui.output_data_frame("tabla_prop3"),
    ),
)

pestaña_metodos = ui.nav_panel(
    "2 · Métodos numéricos",
    ui.p("Se resuelve la ecuación (3.10) con la matriz y la prevalencia de la pestaña 1."),
    ui.output_data_frame("tabla_metodos"),
    ui.output_plot("graf_metodos", height="380px"),
)

pestaña_red = ui.nav_panel(
    "3 · Datos y red (PyTorch)",
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_radio_buttons("fuente", "Datos", {"sint": "Sintéticos", "csv": "Mi CSV"}, selected="sint"),
            ui.panel_conditional("input.fuente === 'csv'",
                                 ui.input_file("archivo", "CSV (no se guarda)", accept=[".csv"]),
                                 ui.input_select("col_y", "Columna de la clase", []),
                                 ui.input_select("val_pos", "Valor de la clase positiva (enfermo)", []),
                                 ui.input_selectize("col_x", "Predictores", [], multiple=True),
                                 ui.input_select("col_g", "Columna de ubicación (para CV agrupada)", [])),
            ui.panel_conditional("input.fuente === 'sint'",
                                 ui.input_slider("sint_prev", "Prevalencia sintética", 0.05, 0.5, 0.13, step=0.01),
                                 ui.input_slider("sint_sep", "Separación entre clases", 0.2, 3.0, 1.2, step=0.1)),
            ui.input_radio_buttons("tipo_cv", "Validación cruzada (5 particiones)",
                                   {"estr": "Estratificada", "agr": "Agrupada por ubicación", "ambas": "Comparar ambas"},
                                   selected="ambas"),
            ui.input_numeric("epocas", "Épocas máximas", 500, min=10, max=5000),
            ui.input_checkbox("pesar", "Ponderar positivos en la pérdida (pos_weight)", False),
            ui.input_action_button("entrenar", "Entrenar y evaluar", class_="btn-primary"),
            width=320,
        ),
        ui.output_text_verbatim("res_red"),
        ui.output_data_frame("tabla_red"),
        ui.output_plot("graf_red", height="380px"),
    ),
)

pestaña_prev = ui.nav_panel(
    "4 · Cambio de prevalencia",
    ui.p("c*(P) para la matriz de la pestaña 1 y los cuatro valores de κ (figura 6 y tabla 6)."),
    ui.output_plot("graf_prev", height="380px"),
    ui.output_data_frame("tabla_prev"),
)

app_ui = ui.page_navbar(pestaña_matriz, pestaña_metodos, pestaña_red, pestaña_prev,
                        title="Umbral de Youden bayesiano")


# ----------------------------------------------------------------------- server
def server(input, output, session):

    @reactive.calc
    def resultado_matriz():
        P = None if input.p_emp() else input.p_uso()
        return umbral.desde_matriz(input.vp(), input.fn(), input.fp(), input.vn(),
                                   P=P, kappa=float(input.kappa()))

    @render.text
    def res_umbral():
        r = resultado_matriz(); mod = r["modelo"]
        return (f"P = {r['P']:.4f}   c* = {r['c_star']:.10f}   logit c* = {np.log(r['c_star']/(1-r['c_star'])):.3f}\n"
                f"Modelo: m0 = {mod.m0:.3f}, m1 = {mod.m1:.3f}, τ = d' = {mod.tau:.3f}, AUC = {mod.auc():.3f}")

    @render.plot
    def graf_matriz():
        r = resultado_matriz()
        curvas = umbral.curvas_modelo(r["modelo"], r["P"])
        marcas = {k.split(" ")[0]: v["c"] for k, v in r["tabla"].items()}
        return _grafico_curvas(curvas, marcas, "Función objetivo (modelo logit-binormal)")

    @render.data_frame
    def tabla_matriz():
        return _tabla(resultado_matriz()["tabla"])

    @render.data_frame
    def tabla_prop3():
        n1, n0 = input.vp() + input.fn(), input.fp() + input.vn()
        d = M.prevalencias_equilibrio(input.vp() / n1, input.vn() / n0)
        return pd.DataFrame([{k: round(v, 4) for k, v in d.items()}])

    # --- métodos numéricos ---
    @reactive.calc
    def comparacion():
        r = resultado_matriz(); mod, P = r["modelo"], r["P"]
        return raices.comparar(lambda c: mod.g(c, P), lambda c: mod.dg(c, P), r["c_star"])

    @render.data_frame
    def tabla_metodos():
        return pd.DataFrame([{k: v for k, v in f.items() if k != "Errores"} for f in comparacion()])

    @render.plot
    def graf_metodos():
        fig, ax = plt.subplots(figsize=(7, 4))
        for f in comparacion():
            e = np.maximum(np.array(f["Errores"]), 1e-16)
            ax.semilogy(np.arange(1, len(e) + 1), e, marker="o", ms=3, label=f["Método"])
        ax.set_xlabel("evaluación de g"); ax.set_ylabel("|c(k) − c*|"); ax.legend(); fig.tight_layout()
        return fig

    # --- datos y red ---
    @reactive.calc
    def datos_csv():
        f = input.archivo()
        return None if not f else pd.read_csv(f[0]["datapath"])

    @reactive.effect
    def _actualizar_columnas():
        df = datos_csv()
        if df is None:
            return
        cols = list(df.columns)
        ui.update_select("col_y", choices=cols)
        ui.update_selectize("col_x", choices=cols)
        ui.update_select("col_g", choices=["(ninguna)"] + cols)

    @reactive.effect
    def _actualizar_valores():
        df = datos_csv()
        if df is None or not input.col_y():
            return
        ui.update_select("val_pos", choices=[str(v) for v in sorted(df[input.col_y()].dropna().unique())])

    @reactive.calc
    @reactive.event(input.entrenar)
    def entrenamiento():
        from umbral.red_torch import RedTorch
        if input.fuente() == "sint":
            df = sinteticos.generar(prevalencia=input.sint_prev(), separacion=input.sint_sep())
            X, y, grupos = df[sinteticos.PREDICTORES].to_numpy(), df["enfermo"].to_numpy(), df["ubicacion"].to_numpy()
        else:
            df = datos_csv()
            if df is None or not input.col_x():
                raise ValueError("Carga el CSV y elige la clase y los predictores.")
            d = df.dropna(subset=list(input.col_x()) + [input.col_y()])
            X = d[list(input.col_x())].to_numpy(dtype=float)
            y = (d[input.col_y()].astype(str) == input.val_pos()).astype(int).to_numpy()
            grupos = None if input.col_g() in ("", "(ninguna)") else d[input.col_g()].to_numpy()
        P = y.mean()
        pw = (1 - P) / P if input.pesar() else None
        fabrica = lambda s: RedTorch(epocas=input.epocas(), semilla=s, peso_positivos=pw)
        esquemas = {"estr": ["Estratificada"], "agr": ["Agrupada"], "ambas": ["Estratificada", "Agrupada"]}[input.tipo_cv()]
        if "Agrupada" in esquemas and grupos is None:
            esquemas = ["Estratificada"]
        salida = {}
        with ui.Progress(min=0, max=len(esquemas)) as prog:
            for i, nombre in enumerate(esquemas):
                prog.set(i, message=f"Entrenando red (CV {nombre.lower()})…")
                p = validacion.predicciones_cv(fabrica, X, y, grupos if nombre == "Agrupada" else None)
                salida[nombre] = {"p": p, "emp": umbral.desde_puntajes(y, p)}
        return {"y": y, "n": len(y), "P": P, "esquemas": salida}

    @render.text
    def res_red():
        r = entrenamiento()
        lineas = [f"n = {r['n']}, positivos = {int(r['y'].sum())}, prevalencia = {r['P']:.3f}"]
        for nombre, s in r["esquemas"].items():
            k = validacion.matriz(r["y"], s["p"])
            lineas.append(f"[{nombre}] matriz en 0,5: VP={k['VP']} FN={k['FN']} FP={k['FP']} VN={k['VN']}   "
                          f"c* empírico (cuantil 1−P) = {s['emp']['c_star']:.4f}")
        return "\n".join(lineas)

    @render.data_frame
    def tabla_red():
        r = entrenamiento()
        filas = {}
        for nombre, s in r["esquemas"].items():
            for k, v in s["emp"]["tabla"].items():
                filas[f"{nombre} · {k}"] = v
        return _tabla(filas)

    @render.plot
    def graf_red():
        r = entrenamiento()
        nombre, s = next(reversed(r["esquemas"].items()))
        return _grafico_curvas(s["emp"]["curvas"], {"c*": s["emp"]["c_star"], "0,5": 0.5},
                               f"Curvas empíricas (CV {nombre.lower()})")

    # --- cambio de prevalencia ---
    @reactive.calc
    def barrido():
        Ps = np.linspace(0.01, 0.5, 50)
        filas = []
        for k in [0.75, 1.0, 1.25, 1.5]:
            for P in Ps:
                r = umbral.desde_matriz(input.vp(), input.fn(), input.fp(), input.vn(), P=P, kappa=k)
                filas.append({"kappa": k, "P": P, "c*": r["c_star"],
                              "F(c0)": r["tabla"]["c0 (red)"]["F"], "F(c*)": r["tabla"]["c* (propuesto)"]["F"]})
        return pd.DataFrame(filas)

    @render.plot
    def graf_prev():
        df = barrido()
        fig, ax = plt.subplots(figsize=(7, 4))
        for k, g in df.groupby("kappa"):
            ax.plot(g["P"], g["c*"], label=f"κ = {k}")
        ax.axhline(0.5, ls=":", color="gray")
        ax.set_xlabel("prevalencia de uso P"); ax.set_ylabel("c*(P)"); ax.legend(); fig.tight_layout()
        return fig

    @render.data_frame
    def tabla_prev():
        df = barrido()
        return df[df["kappa"] == 1.0].iloc[::7].round(4)


app = App(app_ui, server)
