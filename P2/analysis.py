"""
=============================================================================
  ME607 — Análise de Séries Temporais
  Consumo Mensal de Eletricidade na Alemanha
  Fonte: Open Power System Data (OPSD) — dados horários ENTSO-E agregados
  Script Python completo — gera todas as figuras e tabelas do relatório
=============================================================================
  Dependências:
      pip install pandas numpy matplotlib statsmodels pmdarima scipy

  Execução:
      python analysis.py

  Fonte dos dados (download automático, sem conta, sem API key):
      https://data.open-power-system-data.org/time_series/latest/
               time_series_60min_singleindex.csv
      Coluna utilizada: DE_load_actual_entsoe_transparency  (MWh/h)
      Período disponível: 2015-01-01 até ~2020 (varia consoante a versão)

  Saída:
      fig_serie.png, fig_perfil_sazonal.png, fig_stl.png,
      fig_fac_facp.png, fig_diagnostico.png, fig_previsao.png
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # ← must come before ANY pyplot import
                        #   forces non-interactive backend so plt.show()
                        #   never tries to open a display window

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import shapiro, jarque_bera

from pmdarima import auto_arima

# ── Estilo global dos gráficos ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

AZUL_ESCURO  = "#1F3864"
AZUL_MEDIO   = "#2C5F8A"
CINZA_CLARO  = "#EBF3FB"
LARANJA      = "#E05C00"

# =============================================================================
# 1. IMPORTAÇÃO DOS DADOS
# =============================================================================

# URL pública do Open Power System Data — sem conta, sem API key
OPSD_URL = (
    "https://data.open-power-system-data.org"
    "/time_series/latest/time_series_60min_singleindex.csv"
)
# Coluna de carga real da Alemanha (MWh por hora, fonte ENTSO-E via OPSD)
OPSD_COL = "DE_load_actual_entsoe_transparency"


def load_data() -> pd.Series:
    """
    Descarrega e prepara a série mensal de consumo de eletricidade da Alemanha
    a partir do Open Power System Data (OPSD).

    Fonte primária dos dados: ENTSO-E Transparency Platform.
    Publicado por: Open Power System Data (https://open-power-system-data.org)
    Coluna: DE_load_actual_entsoe_transparency  [MWh/h, resolução horária]
    Período: janeiro de 2015 a ~dezembro de 2019/2020 (varia com a versão)

    Procedimento:
        1. Lê o CSV horário directamente da URL pública (sem autenticação).
        2. Filtra a coluna DE_load_actual_entsoe_transparency.
        3. Remove valores nulos e converte MWh/h → GWh/mês por resample.
        4. Devolve pd.Series com frequência mensal (MS) e nome "load_gwh".

    Nota: o ficheiro tem ~1 GB; o download demora 1-3 minutos consoante
    a ligação. pandas transmite o ficheiro por blocos — não é preciso
    guardar localmente.
    """
    print("A descarregar dados OPSD (pode demorar 1-3 min) …")
    print(f"URL: {OPSD_URL}")

    # Ler apenas as colunas necessárias para poupar memória
    df = pd.read_csv(
        OPSD_URL,
        usecols=["utc_timestamp", OPSD_COL],
        parse_dates=["utc_timestamp"],
        index_col="utc_timestamp",
        low_memory=False,
    )

    # Remover linhas sem dados de carga
    df = df.dropna(subset=[OPSD_COL])

    # Converter de MWh/h (potência média horária) para GWh mensais
    #   soma horária  [MWh]  /  1 000  →  GWh mensais
    series = (
        df[OPSD_COL]
        .resample("MS")           # agrupar por início de mês
        .sum()                    # soma = energia total do mês [MWh]
        .div(1_000)               # MWh → GWh
        .rename("load_gwh")
    )

    # Descartar meses incompletos na cauda (menos de 600 horas válidas)
    hourly_counts = df[OPSD_COL].resample("MS").count()
    series = series[hourly_counts >= 600]

    print(f"Dados carregados: {len(series)} observações mensais")
    print(f"Período: {series.index[0].date()} a {series.index[-1].date()}")
    print(f"Média mensal: {series.mean():,.0f} GWh\n")
    return series


# =============================================================================
# 2. ANÁLISE DESCRITIVA E EXPLORATÓRIA
# =============================================================================

def plot_serie(series: pd.Series) -> None:
    """Figura 1 — Série temporal completa."""
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(series.index, series.values,
            color=AZUL_ESCURO, linewidth=1.2)
    ax.axvline(pd.Timestamp("2020-03-01"), color="red",
               linestyle="--", linewidth=1.2,
               label="COVID-19 (mar/2020)")
    ax.set_title("Consumo Mensal de Eletricidade — Alemanha (2006–2023)")
    ax.set_ylabel("GWh")
    ax.set_xlabel("Ano")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig("fig_serie.png", bbox_inches="tight")
    plt.close()  # show() resets figure state in non-interactive mode
    print("fig_serie.png  ✓")


def descriptive_stats(series: pd.Series) -> None:
    """Estatísticas descritivas por subperíodo."""
    pre  = series[:"2019-12-01"]
    post = series["2020-01-01":]
    print("\n── Estatísticas Descritivas ──────────────────────────")
    for label, s in [("Completo", series), ("Pré-COVID (2006–2019)", pre),
                     ("Pós-COVID (2020–2023)", post)]:
        print(f"\n{label}  (N={len(s)})")
        print(s.describe().round(1).to_string())


def plot_seasonal_profile(series: pd.Series) -> None:
    """Figura — Perfil sazonal médio."""
    monthly_means = series.groupby(series.index.month).mean()
    months = ["Jan","Fev","Mar","Abr","Mai","Jun",
              "Jul","Ago","Set","Out","Nov","Dez"]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(months, monthly_means.values,
                  color=AZUL_MEDIO, alpha=0.85, edgecolor="white")
    ax.set_title("Perfil Sazonal Médio — Consumo Mensal (GWh)")
    ax.set_ylabel("GWh (média)")
    ax.set_ylim(monthly_means.min() * 0.9, monthly_means.max() * 1.05)
    for bar, val in zip(bars, monthly_means.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 150,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout()
    plt.savefig("fig_perfil_sazonal.png", bbox_inches="tight")
    plt.close()
    print("fig_perfil_sazonal.png  ✓")


def plot_stl(series: pd.Series) -> None:
    """Figura — Decomposição STL."""
    stl_result = STL(series, period=12, robust=True).fit()
    fig = stl_result.plot()
    fig.set_size_inches(13, 8)
    fig.suptitle("Decomposição STL — Consumo de Eletricidade (Alemanha)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("fig_stl.png", bbox_inches="tight")
    plt.close()
    print("fig_stl.png  ✓")

def plot_acf_pacf(series: pd.Series) -> None:
    """Figura — FAC e FACP (original e após diff. sazonal).

    O número de lags é calculado automaticamente:
      - FACP: statsmodels exige nlags < 50% do tamanho da série
      - FAC:  usa o mesmo limite para consistência visual
    """
    d_series = series.diff(12).dropna()

    # Número máximo de lags permitido por statsmodels: < len/2
    # Usamos min(48, floor(n/2) - 1) para ambas as séries
    max_lags_orig = min(48, len(series) // 2 - 1)
    max_lags_diff = min(48, len(d_series) // 2 - 1)
    print(f"  plot_acf_pacf: lags original={max_lags_orig}, "
          f"lags diff.sazonal={max_lags_diff}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    plot_acf (series,   lags=max_lags_orig, ax=axes[0, 0],
              title="FAC — série original")
    plot_pacf(series,   lags=max_lags_orig, ax=axes[0, 1],
              title="FACP — série original", method="ywm")
    plot_acf (d_series, lags=max_lags_diff, ax=axes[1, 0],
              title=r"FAC — após $\nabla_{12}$")
    plot_pacf(d_series, lags=max_lags_diff, ax=axes[1, 1],
              title=r"FACP — após $\nabla_{12}$", method="ywm")

    for ax in axes.flat:
        ax.axhline(0, color="black", linewidth=0.7)
    plt.suptitle("Funções de Autocorrelação — Consumo de Eletricidade",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig_fac_facp.png", bbox_inches="tight")
    plt.close()
    print("fig_fac_facp.png  ✓")


# =============================================================================
# 3. TESTES DE ESTACIONARIDADE
# =============================================================================

def run_stationarity_tests(series: pd.Series) -> None:
    """Testes ADF e KPSS para a série original e diferenciadas."""

    def _report(s: pd.Series, label: str) -> None:
        adf_stat, adf_p, _, _, adf_cv, _ = adfuller(s, autolag="AIC")
        kpss_stat, kpss_p, _, kpss_cv   = kpss(s, regression="c",
                                                  nlags="auto")
        print(f"\n  {label}")
        print(f"    ADF:  estat={adf_stat:8.4f}  p={adf_p:.4f}  "
              f"CV(5%)={adf_cv['5%']:.4f}  "
              f"→ {'Rejeita H0 (estacionária)' if adf_p < 0.05 else 'Não rejeita H0'}")
        print(f"    KPSS: estat={kpss_stat:8.4f}  p={kpss_p:.4f}  "
              f"CV(5%)={kpss_cv['5%']:.4f}  "
              f"→ {'Não rejeita H0 (estacionária)' if kpss_p > 0.05 else 'Rejeita H0'}")

    print("\n── Testes de Estacionaridade ─────────────────────────")
    _report(series,                          "Original")
    _report(series.diff(12).dropna(),        "Diff. sazonal (D=1)")
    _report(series.diff(12).diff().dropna(), "Diff. sazonal + regular (D=1, d=1)")


# =============================================================================
# 4. IDENTIFICAÇÃO E ESTIMAÇÃO
# =============================================================================

def identify_model(series: pd.Series):
    """auto_arima como referência; retorna modelo selecionado."""
    print("\n── auto_arima ────────────────────────────────────────")
    model_auto = auto_arima(
        series,
        seasonal=True, m=12,
        d=None, D=1,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        information_criterion="aicc",
        stepwise=False,
        error_action="ignore",
        suppress_warnings=True,
        trace=False,
    )
    print(model_auto.summary())
    order          = model_auto.order
    seasonal_order = model_auto.seasonal_order
    print(f"\nModelo selecionado: SARIMA{order}{seasonal_order}")
    return order, seasonal_order


def compare_candidates(series: pd.Series) -> tuple:
    """Estima modelos candidatos e imprime tabela de critérios."""
    candidates = [
        ((1, 0, 1), (0, 1, 1, 12)),
        ((0, 1, 1), (0, 1, 1, 12)),
        ((1, 1, 1), (1, 1, 0, 12)),
        ((2, 0, 0), (0, 1, 1, 12)),
        ((0, 1, 2), (0, 1, 1, 12)),
    ]

    results = []
    for order, seas in candidates:
        try:
            m = SARIMAX(series, order=order, seasonal_order=seas,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
            p, d, q = order
            P, D, Q, s = seas
            results.append({
                "Modelo":  f"SARIMA({p},{d},{q})({P},{D},{Q})_12",
                "Params":  len(m.params),
                "AIC":     round(m.aic,  2),
                "AICc":    round(m.aicc, 2),
                "BIC":     round(m.bic,  2),
                "_order":  order,
                "_seas":   seas,
            })
        except Exception as e:
            print(f"Erro em {order}{seas}: {e}")

    df = pd.DataFrame(results).sort_values("AICc")
    print("\n── Tabela de Critérios de Seleção ────────────────────")
    print(df[["Modelo","Params","AIC","AICc","BIC"]].to_string(index=False))

    best = df.iloc[0]
    return best["_order"], best["_seas"]


def estimate_model(series: pd.Series, order: tuple,
                   seasonal_order: tuple) -> SARIMAX:
    """Estima o modelo selecionado e imprime o sumário."""
    fit = SARIMAX(series,
                  order=order,
                  seasonal_order=seasonal_order,
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)
    print("\n── Sumário do Modelo Selecionado ─────────────────────")
    print(fit.summary())

    coef_df = pd.DataFrame({
        "Estimativa":   fit.params.round(4),
        "Erro-padrão":  fit.bse.round(4),
        "Estat. t":     fit.tvalues.round(3),
        "Valor-p":      fit.pvalues.round(4),
    })
    print("\nCoeficientes:\n", coef_df.to_string())
    return fit


# =============================================================================
# 5. ANÁLISE DE INTERVENÇÃO
# =============================================================================

def intervention_analysis(series: pd.Series, order: tuple,
                           seasonal_order: tuple) -> SARIMAX:
    """Modelo SARIMAX com variável de pulso para COVID-19."""
    pulse = (series.index == pd.Timestamp("2020-03-01")).astype(float).reshape(-1, 1)

    fit_int = SARIMAX(series,
                      order=order,
                      seasonal_order=seasonal_order,
                      exog=pulse,
                      enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False)

    print("\n── Análise de Intervenção (COVID-19) ─────────────────")
    print(fit_int.summary())

    # Comparar modelos
    fit_base = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)
    print(f"\nAIC sem intervenção : {fit_base.aic:.2f}")
    print(f"AIC com intervenção : {fit_int.aic:.2f}")
    delta = fit_base.aic - fit_int.aic
    print(f"ΔAIC = {delta:.2f}  "
          f"({'intervenção melhora o modelo' if delta > 2 else 'melhora marginal'})")
    return fit_int


# =============================================================================
# 6. DIAGNÓSTICO DOS RESÍDUOS
# =============================================================================

def diagnose_residuals(fit, series: pd.Series) -> None:
    """Análise gráfica e testes formais dos resíduos."""
    resid = fit.resid

    # ── Figura diagnóstico ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Série dos resíduos
    axes[0, 0].plot(series.index, resid, color=AZUL_ESCURO, linewidth=0.9)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Resíduos ao longo do tempo")
    axes[0, 0].set_ylabel("Resíduo (GWh)")
    axes[0, 0].xaxis.set_major_locator(mdates.YearLocator(3))
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # FAC dos resíduos
    plot_acf(resid, lags=36, ax=axes[0, 1],
             title="FAC dos Resíduos")

    # Histograma com curva normal
    axes[1, 0].hist(resid, bins=25, density=True,
                    color=AZUL_MEDIO, alpha=0.75, edgecolor="white")
    xmin, xmax = axes[1, 0].get_xlim()
    xx = np.linspace(xmin, xmax, 200)
    axes[1, 0].plot(xx, stats.norm.pdf(xx, resid.mean(), resid.std()),
                    color="red", linewidth=1.8, label="Normal ajustada")
    axes[1, 0].set_title("Histograma dos Resíduos")
    axes[1, 0].legend()

    # QQ-plot
    stats.probplot(resid, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ-plot dos Resíduos")

    plt.suptitle("Diagnóstico dos Resíduos — Modelo SARIMA",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("fig_diagnostico.png", bbox_inches="tight")
    plt.close()
    print("fig_diagnostico.png  ✓")

    # ── Testes formais ────────────────────────────────────────
    print("\n── Testes de Diagnóstico dos Resíduos ────────────────")

    # Ljung-Box
    lb = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
    print("Ljung-Box:\n", lb.round(4).to_string())

    # Shapiro-Wilk
    sw_stat, sw_p = shapiro(resid)
    print(f"\nShapiro-Wilk:  W={sw_stat:.4f}  p={sw_p:.4f}  "
          f"→ {'Normal' if sw_p > 0.05 else 'Não Normal'} (α=5%)")

    # Jarque-Bera
    jb_stat, jb_p = jarque_bera(resid)
    print(f"Jarque-Bera:   JB={jb_stat:.4f}  p={jb_p:.4f}  "
          f"→ {'Normal' if jb_p > 0.05 else 'Não Normal'} (α=5%)")

    # ARCH-LM
    arch_stat, arch_p, _, _ = het_arch(resid, nlags=12)
    print(f"ARCH-LM (12):  stat={arch_stat:.4f}  p={arch_p:.4f}  "
          f"→ {'Sem ARCH' if arch_p > 0.05 else 'ARCH detectado'} (α=5%)")


# =============================================================================
# 7. PREVISÃO
# =============================================================================

def forecast_model(fit, series: pd.Series, h: int = 12) -> None:
    """Previsão h passos à frente com IC 80% e 95%."""
    fc       = fit.get_forecast(steps=h)
    fc_mean  = fc.predicted_mean
    fc_ci95  = fc.conf_int(alpha=0.05)
    fc_ci80  = fc.conf_int(alpha=0.20)
    # Use the index statsmodels already computed — never rebuild manually
    fut_idx  = fc_mean.index

    # ── Figura previsão ───────────────────────────────────────
    hist = series["2021":]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hist.index, hist.values,
            color=AZUL_ESCURO, linewidth=1.4, label="Histórico (2021-2023)")
    ax.plot(fut_idx, fc_mean.values,
            color=LARANJA, linewidth=1.8,
            linestyle="--", label="Previsao pontual")
    ax.fill_between(fut_idx,
                    fc_ci95.iloc[:, 0].values, fc_ci95.iloc[:, 1].values,
                    color=AZUL_MEDIO, alpha=0.20, label="IC 95%")
    ax.fill_between(fut_idx,
                    fc_ci80.iloc[:, 0].values, fc_ci80.iloc[:, 1].values,
                    color=AZUL_MEDIO, alpha=0.40, label="IC 80%")
    ax.set_title("Previsão do Consumo de Eletricidade — Alemanha (2024)")
    ax.set_ylabel("GWh")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    plt.tight_layout()
    plt.savefig("fig_previsao.png", bbox_inches="tight")
    plt.close()
    print("fig_previsao.png  ✓")

    # ── Tabela de previsões ───────────────────────────────────
    fc_table = pd.DataFrame({
        "Mês":         fut_idx.strftime("%b/%Y"),
        "Previsão":    fc_mean.round(1).values,
        "IC 95% inf.": fc_ci95.iloc[:, 0].round(1).values,
        "IC 95% sup.": fc_ci95.iloc[:, 1].round(1).values,
    })
    print("\n── Tabela de Previsões ───────────────────────────────")
    print(fc_table.to_string(index=False))


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  ME607 — Análise de Séries Temporais")
    print("  Consumo de Eletricidade na Alemanha (ENTSO-E)")
    print("=" * 60)

    # 1. Dados
    series = load_data()

    # 2. Análise exploratória
    descriptive_stats(series)
    plot_serie(series)
    plot_seasonal_profile(series)
    plot_stl(series)
    plot_acf_pacf(series)

    # 3. Testes de estacionaridade
    run_stationarity_tests(series)

    # 4. Identificação e estimação
    order, seasonal_order = compare_candidates(series)
    fit = estimate_model(series, order, seasonal_order)

    # 4b. auto_arima como verificação independente
    # identify_model(series)  # descomentar para execução completa (lenta)

    # 5. Análise de intervenção
    fit_int = intervention_analysis(series, order, seasonal_order)

    # 6. Diagnóstico
    diagnose_residuals(fit, series)

    # 7. Previsão
    forecast_model(fit, series, h=12)

    print("\n" + "=" * 60)
    print("  Análise concluída. Todas as figuras geradas.")
    print("=" * 60)