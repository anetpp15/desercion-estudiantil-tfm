"""
figuras_memoria.py
Genera las figuras complementarias de la memoria del TFM a partir de:
  * la carpeta resultados_tfm/ exportada por DesercionEstudiantil_TFM_final.ipynb (tablas CSV y figuras PNG);
  * el conjunto de datos de Kaggle (thedevastator/higher-education-predictors-of-student-retention).

Figuras que produce (carpeta figuras_memoria/):
  fig_protocolo.png, fig_sat.png                      diagramas de elaboración propia
  eda_binarias.png, eda_course.png, eda_rendimiento.png, eda_edad.png    análisis exploratorio
  fig_ic.png, fig_cd.png, fig_equidad_edad.png        síntesis de resultados a partir de las tablas
  comb_*.png                                          composiciones de figuras exportadas por el cuaderno

Uso:  python figuras_memoria.py  [ruta a resultados_tfm]   (por defecto ./resultados_tfm)
"""
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("resultados_tfm")
TAB, FIG = RES / "tablas", RES / "figuras"
OUT = Path("figuras_memoria"); OUT.mkdir(exist_ok=True)
MODELS = ["Regresión logística", "Árbol de decisión", "Random Forest", "XGBoost", "CatBoost"]
EDGE = "#3A5A78"
plt.rcParams.update({"font.size": 10})

def save(name, dpi=250):
    plt.savefig(OUT / name, dpi=dpi, bbox_inches="tight"); plt.close("all"); print("Generada:", OUT / name)

# ---------------------------------------------------------------- datos
def load_data():
    try:
        import kagglehub
        folder = Path(kagglehub.dataset_download("thedevastator/higher-education-predictors-of-student-retention"))
        path = next(p for p in folder.rglob("*.csv"))
    except Exception as e:
        print("Kaggle no disponible, se usa dataset.csv local:", type(e).__name__)
        path = Path("dataset.csv")
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig"); df.columns = df.columns.str.strip()
    assert df.shape == (4424, 35), f"Dimensiones inesperadas: {df.shape}"
    return df

# ---------------------------------------------------------------- diagramas
def box(ax, x, y, w, h, text, fc, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=EDGE, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.3))

def fig_protocolo():
    fig, ax = plt.subplots(figsize=(11, 6.4)); ax.set_xlim(0, 110); ax.set_ylim(0, 64); ax.axis("off")
    box(ax, 38, 56, 34, 6, "Conjunto de datos: 4.424 estudiantes\n(escenario principal: 28 predictoras)", "#DCEBF7", bold=True)
    arrow(ax, 55, 56, 55, 51.5); box(ax, 38, 46, 34, 5, "Partición estratificada 80/20 (semilla 42)", "#DCEBF7")
    arrow(ax, 46, 46, 27, 41.5); arrow(ax, 64, 46, 86, 41.5)
    box(ax, 4, 36, 46, 5.5, "Entrenamiento: 3.539 registros", "#E3F1E1", bold=True)
    box(ax, 70, 36, 34, 5.5, "Prueba: 885 registros\n(no interviene en ninguna decisión)", "#FBE3D6", bold=True)
    ax.add_patch(FancyBboxPatch((2, 3), 50, 30, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#F6FAF5", ec=EDGE, lw=1, ls="--"))
    ax.text(27, 31.3, "A. Validación cruzada anidada 10 × 3 (selección del modelo)", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(27, 28.4, "En cada uno de los 10 pliegues externos (9/10 ajuste, 1/10 validación):", ha="center", fontsize=8.5, style="italic")
    steps = ["1. Búsqueda aleatoria: 10 configuraciones,\nvalidación interna de 3 pliegues, G-Mean",
             "2. Umbral: probabilidades out-of-fold (3 pliegues),\nmáximo G-Mean en la rejilla 0,05 a 0,95",
             "3. Reajuste con la mejor configuración\ny evaluación en el pliegue externo"]
    y = 21
    for i, t in enumerate(steps):
        box(ax, 5, y, 44, 4.6, t, "#FFFFFF", fs=8.3)
        if i < 2: arrow(ax, 27, y, 27, y - 1.9)
        y -= 6.6
    ax.text(27, 4.6, "Criterio a priori: máximo G-Mean medio de los 10 pliegues externos", ha="center", fontsize=8.6, fontweight="bold", color="#2E5E2A")
    arrow(ax, 27, 36, 27, 33.6)
    ax.add_patch(FancyBboxPatch((56, 3), 50, 22, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#F6FAF5", ec=EDGE, lw=1, ls="--"))
    ax.text(81, 23.3, "B. Procedimiento final sobre todo el entrenamiento", ha="center", fontsize=9.5, fontweight="bold")
    box(ax, 59, 15.5, 44, 4.6, "Búsqueda (10 configuraciones, 3 pliegues)\ny umbral out-of-fold (5 pliegues)", "#FFFFFF", fs=8.3)
    arrow(ax, 81, 15.5, 81, 13.6)
    box(ax, 59, 8.6, 44, 4.6, "Modelo final reajustado con 3.539 registros\ny umbral fijado", "#FFFFFF", fs=8.3)
    arrow(ax, 50, 38.7, 58, 24.6); arrow(ax, 87, 36, 96, 26)
    ax.text(97.5, 30.5, "evaluación\núnica", fontsize=8, ha="left", color=EDGE)
    ax.text(81, 5.0, "Métricas, IC bootstrap, McNemar, DeLong,\nSHAP y equidad sobre la prueba", ha="center", fontsize=8.4)
    ax.text(55, 0.3, "El preprocesamiento y el sobremuestreo aleatorio se ajustan dentro del pipeline en cada entrenamiento; nunca ven datos de validación ni de prueba.",
            ha="center", fontsize=8, style="italic", color="#444444")
    plt.tight_layout(); save("fig_protocolo.png", 220)

def fig_sat():
    fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 110); ax.set_ylim(0, 56); ax.axis("off")
    box(ax, 2, 40, 20, 10, "1. Cierre del\nprimer semestre:\nextracción de datos\nacadémicos y\nadministrativos", "#DCEBF7", 8.8, True)
    box(ax, 28, 40, 22, 10, "2. Modelo predictivo\nprobabilidad de abandono\n+ explicación SHAP\nde cada estudiante", "#DCEBF7", 8.8, True)
    box(ax, 56, 40, 24, 10, "3. Franjas de riesgo\nbajo · revisión en torno\nal umbral · alto", "#FBE3D6", 8.8, True)
    box(ax, 86, 40, 22, 10, "4. Revisión humana\npor tutores con la\nexplicación del caso", "#E3F1E1", 8.8, True)
    arrow(ax, 22.5, 45, 27.5, 45); arrow(ax, 50.5, 45, 55.5, 45); arrow(ax, 80.5, 45, 85.5, 45)
    box(ax, 86, 20, 22, 12, "5. Intervención\nproporcional:\ntutoría, orientación,\napoyo económico", "#E3F1E1", 8.8, True)
    arrow(ax, 97, 40, 97, 32.5)
    box(ax, 56, 20, 24, 12, "6. Seguimiento\nregistro de la\nintervención y del\ndesenlace", "#F6FAF5", 8.8)
    arrow(ax, 85.5, 26, 80.5, 26)
    box(ax, 28, 20, 22, 12, "7. Auditoría periódica\nrendimiento, calibración\ny equidad por grupo\n(edad, género)", "#F6FAF5", 8.8)
    arrow(ax, 55.5, 26, 50.5, 26)
    box(ax, 2, 20, 20, 12, "8. Reentrenamiento\ncon nuevas cohortes\ny validación\ntemporal", "#F6FAF5", 8.8)
    arrow(ax, 27.5, 26, 22.5, 26); arrow(ax, 12, 32.5, 12, 39.5)
    ax.text(55, 11.5, "Garantías transversales: base jurídica y evaluación de impacto (RGPD, art. 35) · ninguna decisión automatizada sobre el estudiante (art. 22)\n"
            "información al alumnado · documentación técnica y supervisión humana (Reglamento (UE) 2024/1689) · acceso restringido a los datos",
            ha="center", va="center", fontsize=8.4, style="italic", color="#333333", bbox=dict(boxstyle="round,pad=0.6", fc="#FFF8E1", ec="#C9A227"))
    plt.tight_layout(); save("fig_sat.png", 220)

# ---------------------------------------------------------------- análisis exploratorio
def figs_eda(df):
    df = df.assign(y=(df["Target"] == "Dropout").astype(int))
    c = df.groupby("Course")["y"].agg(["mean", "size"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh([f"Titulación {k} (n = {n})" for k, n in zip(c.index, c["size"])], 100 * c["mean"], color="#4C72B0")
    ax.axvline(100 * df.y.mean(), color="#C44E52", ls="--", label=f"Media global ({100 * df.y.mean():.1f} %)")
    ax.set_xlabel("Tasa de abandono (%)"); ax.legend(loc="lower right"); plt.tight_layout(); save("eda_course.png", 300)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for lab, col in [(0, "#4C72B0"), (1, "#C44E52")]:
        name = "Dropout" if lab else "Graduate y Enrolled"
        axes[0].hist(df[df.y == lab]["Curricular units 1st sem (approved)"].clip(upper=12), bins=np.arange(-0.5, 13.5, 1), alpha=.6, density=True, color=col, label=name)
        axes[1].hist(df[df.y == lab]["Curricular units 1st sem (grade)"], bins=np.arange(0, 20.5, 1), alpha=.6, density=True, color=col, label=name)
    axes[0].set_xlabel("Unidades aprobadas en el 1.er semestre (12 = 12 o más)"); axes[0].set_ylabel("Proporción"); axes[0].legend()
    axes[1].set_xlabel("Calificación media del 1.er semestre (0 a 20)"); axes[1].legend()
    plt.tight_layout(); save("eda_rendimiento.png", 300)

    pairs = [("Tuition fees up to date", "Pagos al día"), ("Debtor", "Deudor"), ("Scholarship holder", "Becario"),
             ("Gender", "Hombre"), ("Displaced", "Desplazado"), ("Daytime/evening attendance", "Diurno")]
    r = pd.DataFrame([(l, 100 * df[df[v] == 1].y.mean(), 100 * df[df[v] == 0].y.mean()) for v, l in pairs], columns=["var", "si", "no"])
    x = np.arange(len(r)); fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(x - 0.2, r.si, 0.4, label="Sí (valor 1)", color="#C44E52"); ax.bar(x + 0.2, r.no, 0.4, label="No (valor 0)", color="#4C72B0")
    for i, (a, b) in enumerate(zip(r.si, r.no)):
        ax.text(i - 0.2, a + 1, f"{a:.0f}", ha="center", fontsize=8); ax.text(i + 0.2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(r["var"]); ax.set_ylabel("Tasa de abandono (%)"); ax.legend()
    plt.tight_layout(); save("eda_binarias.png", 300)

    df["edad"] = pd.cut(df["Age at enrollment"], bins=[0, 20, 25, 30, 40, 100], labels=["≤ 20", "21 a 25", "26 a 30", "31 a 40", "> 40"])
    e = df.groupby("edad", observed=True)["y"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar([f"{k}\n(n = {n})" for k, n in zip(e.index, e["size"])], 100 * e["mean"], color="#4C72B0")
    for i, v in enumerate(100 * e["mean"]): ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("Tasa de abandono (%)"); ax.set_xlabel("Edad al matricularse (años)")
    plt.tight_layout(); save("eda_edad.png", 300)

# ---------------------------------------------------------------- síntesis de resultados
def parse_ci(s):
    lo, hi = re.findall(r"[-+]?\d*\.?\d+", str(s))[:2]; return float(lo), float(hi)

def fig_ic():
    test = pd.read_csv(TAB / "tabla_prueba_primer_semestre.csv").set_index("Modelo").loc[MODELS]
    ci = pd.read_csv(TAB / "tabla_ic_bootstrap.csv"); ci = ci[ci["Escenario"] == "Primer semestre"].set_index("Modelo").loc[MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True); y = np.arange(len(MODELS))[::-1]
    for ax, m in [(axes[0], "G-Mean"), (axes[1], "ROC-AUC")]:
        v = test[m].to_numpy(); bounds = np.array([parse_ci(s) for s in ci[f"{m} IC95"]])
        ax.errorbar(v, y, xerr=[v - bounds[:, 0], bounds[:, 1] - v], fmt="o", color=EDGE, capsize=4)
        ax.set_title(m); ax.grid(axis="x", alpha=.3)
    axes[0].set_yticks(y); axes[0].set_yticklabels(MODELS); plt.tight_layout(); save("fig_ic.png")

def fig_cd(n_folds=10, q_alpha=2.728):
    fr = pd.read_csv(TAB / "tabla_friedman.csv"); fr = fr[fr["Escenario"] == "Primer semestre"].iloc[0]
    ranks = {m: float(fr[f"Rango medio {m}"]) for m in MODELS}; k = len(ranks)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n_folds))
    fig, ax = plt.subplots(figsize=(9, 3)); ax.axis("off"); ax.set_xlim(-0.6, 6.6); ax.set_ylim(0, 3)
    ax.hlines(2.4, 1, 5, color="k")
    for t in range(1, 6): ax.vlines(t, 2.35, 2.45, color="k"); ax.text(t, 2.55, str(t), ha="center", fontsize=9)
    ax.hlines(2.85, 1, 1 + cd, color="#C44E52", lw=2); ax.vlines([1, 1 + cd], 2.8, 2.9, color="#C44E52")
    ax.text(1 + cd / 2, 2.92, f"DC = {cd:.2f}", ha="center", fontsize=9, color="#C44E52")
    items = sorted(ranks.items(), key=lambda x: x[1]); left = [i for i in items if i[1] < 3]; right = [i for i in items if i[1] >= 3]
    for i, (m, r) in enumerate(left):
        yy = 2.0 - 0.38 * i; ax.plot([r, r, 0.95], [2.4, yy, yy], color="k", lw=0.8); ax.text(0.93, yy, f"{m} ({r:.2f})", ha="right", va="center", fontsize=9)
    for i, (m, r) in enumerate(sorted(right, key=lambda x: -x[1])):
        yy = 2.0 - 0.38 * i; ax.plot([r, r, 5.05], [2.4, yy, yy], color="k", lw=0.8); ax.text(5.07, yy, f"({r:.2f}) {m}", ha="left", va="center", fontsize=9)
    # grupos maximales sin diferencias significativas (rango máximo − mínimo < DC)
    srt = [r for _, r in items]; groups = []
    for i in range(k):
        j = max(j for j in range(i, k) if srt[j] - srt[i] < cd)
        if j > i and not any(g[0] <= i and j <= g[1] for g in groups): groups.append((i, j))
    for g, (i, j) in enumerate(groups):
        ax.hlines(2.25 - 0.13 * g, srt[i] - 0.03, srt[j] + 0.03, color=EDGE, lw=4, alpha=.7)
    plt.tight_layout(); save("fig_cd.png")

def fig_equidad():
    eq = pd.read_csv(TAB / "tabla_equidad.csv"); eq = eq[eq["Atributo"] == "Edad al matricularse"]
    groups = ["≤ 20", "21 a 25", "≥ 26"]; x = np.arange(len(MODELS)); w = 0.26; cols = ["#9DB7D5", "#4C72B0", "#1F3B63"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for j, g in enumerate(groups):
        vals = [eq[(eq["Modelo"] == m) & (eq["Grupo"] == g)]["Sensibilidad"].iloc[0] for m in MODELS]
        ax.bar(x + (j - 1) * w, vals, w, label=f"{g} años", color=cols[j])
    ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=9); ax.set_ylabel("Sensibilidad en prueba"); ax.set_ylim(0.5, 1)
    ax.legend(title="Edad al matricularse", fontsize=8); plt.tight_layout(); save("fig_equidad_edad.png")

# ---------------------------------------------------------------- composiciones
def compose(files, cols, name, pad=20):
    ims = [Image.open(FIG / f).convert("RGB") for f in files]
    w = max(i.width for i in ims); h = max(i.height for i in ims); rows = (len(ims) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w + (cols - 1) * pad, rows * h + (rows - 1) * pad), "white")
    for k, im in enumerate(ims):
        r, c = divmod(k, cols); canvas.paste(im, (c * (w + pad) + (w - im.width) // 2, r * (h + pad)))
    canvas.save(OUT / name); print("Generada:", OUT / name)

def composiciones():
    compose([f"fig_matriz_confusion_{s}.png" for s in ["regresion_logistica", "arbol_de_decision", "random_forest", "xgboost", "catboost"]], 3, "comb_confusion.png")
    compose(["fig_roc_after_first_semester.png", "fig_pr_after_first_semester.png"], 2, "comb_roc_pr.png")
    compose(["fig_roc_after_first_semester_sin_enrolled.png", "fig_pr_after_first_semester_sin_enrolled.png"], 2, "comb_sens.png")
    compose(["fig_roc_at_enrollment.png", "fig_roc_after_second_semester.png"], 2, "comb_roc_temporal.png")
    compose(["fig_pr_at_enrollment.png", "fig_pr_after_second_semester.png"], 2, "comb_pr_temporal.png")
    for case in ["verdadero_positivo", "falso_negativo", "caso_umbral"]:
        compose([f"fig_shap_waterfall_{case}_xgboost.png", f"fig_shap_waterfall_{case}_seleccionado.png"], 2, f"comb_wf_{case}.png")

if __name__ == "__main__":
    fig_protocolo(); fig_sat()
    figs_eda(load_data())
    fig_ic(); fig_cd(); fig_equidad()
    composiciones()
    print("Figuras complementarias en", OUT.resolve())
