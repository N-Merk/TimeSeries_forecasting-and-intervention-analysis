"""
=============================================================================
  ME607 — Análise de Séries Temporais
  Consumo Mensal de Eletricidade na Alemanha
  Fonte: Open Power System Data (OPSD) — dados horários ENTSO-E agregados
  Script Python completo — gera todas as figuras e tabelas do relatório
=============================================================================
  Dependências:
      install pandas numpy matplotlib statsmodels pmdarima scipy itertools

  Execução:
      python analysis.py

  Fonte dos dados (download automático, sem conta, sem API key):
      https://data.open-power-system-data.org/time_series/latest/
               time_series_60min_singleindex.csv
      Coluna utilizada: DE_load_actual_entsoe_transparency  (MWh/h)
      Período disponível: 2015-01-01 até ~2020

=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from itertools import product

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.sm_exceptions import InterpolationWarning
from scipy.stats import shapiro, jarque_bera


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
# IMPORTAÇÃO DOS DADOS
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
# ANÁLISE DESCRITIVA E EXPLORATÓRIA
# =============================================================================

def plot_serie(series: pd.Series) -> None:
    """Figura 1 — Série temporal completa."""
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(series.index, series.values,
            color=AZUL_ESCURO, linewidth=1.2)
    ax.axvline(pd.Timestamp("2020-03-01"), color="red",
               linestyle="--", linewidth=1.2,
               label="COVID-19 (mar/2020)")
    #ax.set_title("Consumo Mensal de Eletricidade — Alemanha (2006–2023)")
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
    pre  = series[:"2020-02-01"]
    post = series["2020-03-01":]
    print("\n── Estatísticas Descritivas ──────────────────────────")
    for label, s in [("Completo", series), ("Pré-COVID (2015–02/2020)", pre),
                     ("Pós-COVID (03/2020–09/2020)", post)]:
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
    #ax.set_title("Perfil Sazonal Médio — Consumo Mensal (GWh)")
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

def plot_heatmap_sazonal(series: pd.Series) -> None:
    """Figura — Heatmap sazonal por mês e ano."""
    # Build pivot table: rows = years, columns = months
    df = series.to_frame(name="valor")
    df["mes"] = df.index.month
    df["ano"] = df.index.year
    pivot = df.pivot_table(index="ano", columns="mes", values="valor", aggfunc="mean")
    pivot.columns = ["Jan","Fev","Mar","Abr","Mai","Jun",
                     "Jul","Ago","Set","Out","Nov","Dez"]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")

    # Axes labels
    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_ylabel("Ano")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("GWh (média)", fontsize=8)

    # Annotate each cell
    vmin, vmax = pivot.values.min(), pivot.values.max()
    for i, year in enumerate(pivot.index):
        for j, month in enumerate(pivot.columns):
            val = pivot.loc[year, month]
            if pd.isna(val):
                continue
            # Dark text on light cells, white text on dark cells
            brightness = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = "white" if brightness > 0.6 else "black"
            ax.text(j, i, f"{val:,.0f}",
                    ha="center", va="center", fontsize=6.5, color=color)

    plt.tight_layout()
    plt.savefig("fig_heatmap_sazonal.png", bbox_inches="tight")
    plt.close()
    print("fig_heatmap_sazonal.png  ✓")


def plot_stl(series: pd.Series) -> None:
    """Figura — Decomposição STL."""
    stl_result = STL(series, period=12, robust=True).fit()
    fig = stl_result.plot()
    fig.set_size_inches(13, 8)
    #fig.suptitle("Decomposição STL — Consumo de Eletricidade (Alemanha)", fontsize=13, fontweight="bold", y=1.01)
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
    d12      = series.diff(12).dropna()          # ∇₁₂
    d1_d12   = d12.diff(1).dropna()              # ∇∇₁₂
    d1squ_d12   = d12.diff(1).dropna().diff(1).dropna()   # ∇∇∇₁₂

    # Varianca das series
    var_orig = np.var(series)
    var_d12 = np.var(d12)
    var_d1_d12 = np.var(d1_d12)
    var_d1_squ_d12 = np.var(d1squ_d12)

    # Número máximo de lags permitido por statsmodels: < len/2
    # Usamos min(48, floor(n/2) - 1) para ambas as séries
    max_lags_orig  = min(48, len(series) // 2 - 1)
    max_lags_d12   = min(48, len(d12)    // 2 - 1)
    max_lags_d1d12 = min(48, len(d1_d12) // 2 - 1)
    max_lags_d1squd12 = min(48, len(d1squ_d12) // 2 - 1)

    print(f"  plot_acf_pacf: lags original={max_lags_orig}, "
          f"lags diff.sazonal={max_lags_d12},"
          f"lags diff.sazonal_twice={max_lags_d1d12}"
          )

    fig, axes = plt.subplots(4, 2, figsize=(13, 10))
    plot_acf (series,   lags=max_lags_orig, ax=axes[0, 0],
              title="FAC — série original")
    plot_pacf(series,   lags=max_lags_orig, ax=axes[0, 1],
              title="FACP — série original", method="ywm")

    plot_acf (d12, lags=max_lags_d12, ax=axes[1, 0],
              title=r"FAC — após $\nabla_{12}$")
    plot_pacf(d12, lags=max_lags_d12, ax=axes[1, 1],
              title=r"FACP — após $\nabla_{12}$", method="ywm")

    plot_acf (d1_d12,   lags=max_lags_d1d12, ax=axes[2, 0],
              title=r"FAC — $\nabla\nabla_{12}$")
    plot_pacf(d1_d12,   lags=max_lags_d1d12, ax=axes[2, 1],
              title=r"FACP — $\nabla\nabla_{12}$", method="ywm")

    plot_acf (d1squ_d12,   lags=max_lags_d1squd12, ax=axes[3, 0],
              title=r"FAC — $\nabla^2\nabla_{12}$")
    plot_pacf(d1squ_d12,   lags=max_lags_d1squd12, ax=axes[3, 1],
              title=r"FACP — $\nabla^2\nabla_{12}$", method="ywm")

    for ax in axes.flat:
        ax.axhline(0, color="black", linewidth=0.7)
    #plt.suptitle("Funções de Autocorrelação — Consumo de Eletricidade", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig_fac_facp.png", bbox_inches="tight")
    plt.close()
    print("fig_fac_facp.png  ✓")
    print(f"var orig: {var_orig}")
    print(f"var d12: {var_d12}")
    print(f"var d1_d12: {var_d1_d12}")
    print(f"var d1squ_d12: {var_d1_squ_d12}")


# =============================================================================
# TESTES DE ESTACIONARIDADE
# =============================================================================
def run_stationarity_tests(series: pd.Series) -> None:
    """ADF e KPSS para a série original e diferenciadas."""

    def _report(s: pd.Series, label: str) -> None:
        adf_stat, adf_p, _, _, adf_cv, _ = adfuller(s, autolag="AIC")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_stat, kpss_p, _, kpss_cv = kpss(
                s,
                regression="c",
                nlags="auto"
            )

        if kpss_p >= 0.10:
            p_text = "> 0.10"
        elif kpss_p <= 0.01:
            p_text = "< 0.01"
        else:
            p_text = f"{kpss_p:.4f}"

        print(f"\n  {label}")
        print(f"    ADF:  estat={adf_stat:8.4f}  p={adf_p:.4f}  "
              f"CV(5%)={adf_cv['5%']:.4f}  "
              f"→ {'Rejeita H0 (estacionária)' if adf_p < 0.05 else 'Não rejeita H0'}")

        print(
            f"KPSS: estat={kpss_stat:8.4f}  "
            f"p={p_text}  "
            f"CV(5%)={kpss_cv['5%']:.4f}  "
            f"→ {'Não rejeita H0 (estacionária)' if kpss_p > 0.05 else 'Rejeita H0 (não estacionária)'}"
        )

    print("\n── Testes de Estacionaridade ─────────────────────────")
    _report(series,                          "Original")
    _report(series.diff(12).dropna(),        "Diff. sazonal (D=1)")
    _report(series.diff(12).diff().dropna(), "Diff. sazonal + regular (D=1, d=1)")
    _report(series.diff(12).diff().diff().dropna(), "Diff. sazonal + regular (D=1, d=2)")


# =============================================================================
# IDENTIFICAÇÃO E ESTIMAÇÃO
# =============================================================================
def compare_candidates(series: pd.Series) -> tuple:
    """Estima modelos candidatos e imprime tabela de critérios."""
    candidates = [
        ((p, 1, q), (P, 1, Q, 12))
        for p, q, P, Q in product(range(3), repeat=4)
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

    df = pd.DataFrame(results).sort_values("AICc").reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    print("\n── Tabela de Critérios de Seleção ────────────────────")
    print(df[["Rank", "Modelo","Params","AIC","AICc","BIC"]].to_string(index=False))

    best = df.iloc[0]
    return best["_order"], best["_seas"]

def estimate_model(series: pd.Series, order: tuple,
                   seasonal_order: tuple, exog: pd.DataFrame = None) -> SARIMAX:
    """Estima o modelo selecionado e imprime o sumário.

    exog: variáveis exógenas opcionais (p.ex. pulso de intervenção).
    """
    fit = SARIMAX(series,
                  order=order,
                  seasonal_order=seasonal_order,
                  exog=exog,
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
# ANÁLISE DE INTERVENÇÃO
# =============================================================================
def intervention_naive(series: pd.Series, order: tuple,
                        seasonal_order: tuple, fit_base: SARIMAX) -> SARIMAX:
    """
    5.1 — Verificação preliminar: SARIMAX com pulso unitário (mar/2020),
    estimado sobre toda a amostra. Compara com fit_base (modelo sem
    intervenção, já estimado em estimate_model) via AIC/AICc.
    """
    pulse = pd.DataFrame(
        {
            "covid_pulse":
            ((series.index.year == 2020) &
             (series.index.month == 3)).astype(float)
        },
        index=series.index
    )

    fit_int = estimate_model(series, order, seasonal_order, exog=pulse)

    print("\n── 5.1 Verificação Preliminar (Amostra Completa) ─────")
    print("\nNon-zero observations:")
    print(pulse[pulse["covid_pulse"] == 1])

    delta_aic  = fit_base.aic  - fit_int.aic
    delta_aicc = fit_base.aicc - fit_int.aicc
    print(f"\nAIC  sem/com intervenção : {fit_base.aic:.2f} / {fit_int.aic:.2f}  "
          f"(ΔAIC={delta_aic:.2f}; positivo favorece o modelo SEM intervenção)")
    print(f"AICc sem/com intervenção : {fit_base.aicc:.2f} / {fit_int.aicc:.2f}  "
          f"(ΔAICc={delta_aicc:.2f})")

    return fit_int


def intervention_counterfactual(series: pd.Series, order: tuple,
                                 seasonal_order: tuple) -> tuple:
    """
    5.2 — Teste contrafactual: SARIMA estimado apenas no período pré-COVID
    (jan/2015-fev/2020), com previsão h=len(pós-COVID) passos à frente
    comparada aos valores observados de mar-set/2020.

    Retorna (fit_pre, intervencao_df).
    """
    pre_covid  = series[:"2020-02-01"]
    post_covid = series["2020-03-01":]

    print(f"\n── 5.2 Teste Contrafactual (Modelo Pré-COVID, N={len(pre_covid)}) ──")
    fit_pre = estimate_model(pre_covid, order, seasonal_order)

    h       = len(post_covid)
    fc      = fit_pre.get_forecast(steps=h)
    fc_mean = fc.predicted_mean
    fc_ci95 = fc.conf_int(alpha=0.05)

    intervencao_df = pd.DataFrame({
        "Observado":  post_covid.round(1).values,
        "Previsto":   fc_mean.round(1).values,
        "IC95_inf":   fc_ci95.iloc[:, 0].round(1).values,
        "IC95_sup":   fc_ci95.iloc[:, 1].round(1).values,
    }, index=post_covid.index)
    intervencao_df["Desvio"] = (
        intervencao_df["Observado"] - intervencao_df["Previsto"]
    ).round(1)
    intervencao_df["Desvio_pct"] = (
        100 * intervencao_df["Desvio"] / intervencao_df["Previsto"]
    ).round(1)
    intervencao_df["Fora_IC95"] = (
        (intervencao_df["Observado"] < intervencao_df["IC95_inf"]) |
        (intervencao_df["Observado"] > intervencao_df["IC95_sup"])
    )

    print("\n── Tabela: Observado vs. Previsão Contrafactual ───────")
    print(intervencao_df.to_string())
    n_fora = intervencao_df["Fora_IC95"].sum()
    print(f"\nMeses fora do IC 95% da previsão contrafactual: {n_fora} de {h}")

    return fit_pre, intervencao_df


def plot_intervention_counterfactual(series: pd.Series, fit_pre: SARIMAX,
                                      intervencao_df: pd.DataFrame) -> None:
    """Figura — observado vs. previsão contrafactual do modelo pré-COVID."""
    pre_covid  = series[:"2020-02-01"]
    post_covid = series["2020-03-01":]
    h          = len(post_covid)
    fc         = fit_pre.get_forecast(steps=h)
    fc_mean    = fc.predicted_mean
    fc_ci95    = fc.conf_int(alpha=0.05)

    fig, ax = plt.subplots(figsize=(12, 5))
    hist_plot = pre_covid["2018":]
    ax.plot(hist_plot.index, hist_plot.values,
            color=AZUL_ESCURO, linewidth=1.3, label="Histórico (pré-COVID)")
    ax.plot(post_covid.index, post_covid.values,
            color="black", linewidth=1.6, marker="o", markersize=4,
            label="Observado (pós-COVID)")
    ax.plot(fc_mean.index, fc_mean.values,
            color=LARANJA, linewidth=1.8, linestyle="--",
            label="Previsão contrafactual")
    ax.fill_between(fc_ci95.index,
                    fc_ci95.iloc[:, 0].values, fc_ci95.iloc[:, 1].values,
                    color=AZUL_MEDIO, alpha=0.25, label="IC 95%")
    ax.axvline(pd.Timestamp("2020-03-01"), color="red",
               linestyle=":", linewidth=1.2)
    #ax.set_title("Consumo Observado vs. Previsão Contrafactual " "(Modelo Pré-COVID)")
    ax.set_ylabel("GWh")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    plt.tight_layout()
    plt.savefig("fig_intervencao_precovid.png", bbox_inches="tight")
    plt.close()
    print("fig_intervencao_precovid.png  ✓")

# =============================================================================
# MODELO COM INTERVENÇÃO TEMPORÁRIA/GRADUAL (PULSO COM DECAIMENTO)
# =============================================================================
def build_decaying_pulse(series: pd.Series, start: str = "2020-03-01",
                          delta: float = 0.7) -> pd.Series:
    idx = series.index
    t0_pos = idx.get_indexer([series[start:].index[0]])[0]
    decay = np.zeros(len(idx))
    for i in range(t0_pos, len(idx)):
        decay[i] = delta ** (i - t0_pos)
    return pd.Series(decay, index=idx, name="covid_decay")


def intervention_dynamic(series: pd.Series, order: tuple,
                          seasonal_order: tuple, fit_base: SARIMAX,
                          delta_grid=(0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)) -> tuple:
    """
    Modelo de intervenção temporária/gradual (pulso com decaimento
    exponencial), estimado sobre toda a amostra. O parâmetro delta é
    escolhido por busca em grelha, minimizando o AICc; omega e delta
    poderiam alternativamente ser estimados conjuntamente por máxima
    verossimilhança não linear, mas a grelha é suficiente e mais estável
    dado o pequeno número de observações pós-intervenção (n=7).

    Retorna (fit_dyn, delta_otimo, df_comparacao).
    """
    print("\n── Modelo de Intervenção Temporária/Gradual ───────")

    resultados = []
    melhor_aicc = np.inf
    fit_dyn = None
    delta_otimo = None

    for delta in delta_grid:
        decay = build_decaying_pulse(series, delta=delta)
        m = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    exog=decay,
                    enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False)
        resultados.append({
            "delta": delta,
            "omega": round(m.params["covid_decay"], 2),
            "p_omega": round(m.pvalues["covid_decay"], 4),
            "AIC": round(m.aic, 2),
            "AICc": round(m.aicc, 2),
        })
        if m.aicc < melhor_aicc:
            melhor_aicc = m.aicc
            fit_dyn = m
            delta_otimo = delta

    grid_df = pd.DataFrame(resultados).sort_values("delta")
    print("\nBusca em grelha para delta (decaimento):")
    print(grid_df.to_string(index=False))
    print(f"\nMelhor delta = {delta_otimo}  (AICc = {melhor_aicc:.2f})")

    print(f"\n── Modelo Final (delta={delta_otimo}) ─────────────────")
    print(fit_dyn.summary())

    # ── Comparação dos três modelos: base, pulso único, pulso decaído ──
    comparacao = pd.DataFrame({
        "Modelo": ["Sem intervenção",
                   f"Pulso com decaimento (δ={delta_otimo})"],
        "Params": [fit_base.params.shape[0], fit_dyn.params.shape[0]],
        "AIC":    [round(fit_base.aic, 2), round(fit_dyn.aic, 2)],
        "AICc":   [round(fit_base.aicc, 2), round(fit_dyn.aicc, 2)],
        "BIC":    [round(fit_base.bic, 2), round(fit_dyn.bic, 2)],
    })
    print("\nComparação dos modelos (sem intervenção vs. com decaimento):")
    print(comparacao.to_string(index=False))

    return fit_dyn, delta_otimo, comparacao


def plot_dynamic_intervention(series: pd.Series, fit_base: SARIMAX,
                               fit_dyn: SARIMAX, delta: float) -> None:
    """Figura — efeito estimado da intervenção temporária/gradual ao longo do tempo."""
    decay = build_decaying_pulse(series, delta=delta)
    omega = fit_dyn.params["covid_decay"]
    efeito_estimado = omega * decay

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Painel superior: ajuste in-sample dos dois modelos vs. observado
    fitted_base = fit_base.fittedvalues
    fitted_dyn  = fit_dyn.fittedvalues
    axes[0].plot(series.index, series.values, color="black",
                 linewidth=1.3, label="Observado")
    axes[0].plot(series.index, fitted_dyn.values, color=LARANJA,
                 linewidth=1.4, linestyle="--",
                 label=f"Ajustado (com intervenção, δ={delta})")
    axes[0].axvline(pd.Timestamp("2020-03-01"), color="red",
                    linestyle=":", linewidth=1.0)
    #axes[0].set_title("Série Observada vs. Ajustada (Modelo com Intervenção)")
    axes[0].set_ylabel("GWh")
    axes[0].legend(loc="upper right")

    # Painel inferior: efeito estimado da intervenção isoladamente
    axes[1].plot(series.index, efeito_estimado.values,
                 color=AZUL_ESCURO, linewidth=1.6)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].axvline(pd.Timestamp("2020-03-01"), color="red",
                    linestyle=":", linewidth=1.0)
    axes[1].set_title(r"Efeito Estimado da Intervenção: "
                       r"$\hat{\omega} \cdot \delta^{t-t_0}$")
    axes[1].set_ylabel("GWh")
    axes[1].set_xlabel("Ano")
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    #plt.suptitle("Modelo de Intervenção Temporária/Gradual — COVID-19", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("fig_intervencao_dinamica.png", bbox_inches="tight")
    plt.close()
    print("fig_intervencao_dinamica.png  ✓")


# =============================================================================
# DIAGNÓSTICO DOS RESÍDUOS
# =============================================================================
def diagnose_residuals(fit, series: pd.Series, modelname="None") -> None:
    """Análise gráfica e testes formais dos resíduos.

    Usa os resíduos padronizados (standardized_forecasts_error) do filtro de
    Kalman, excluindo o período de inicialização (burn-in) de d + D*s = 13
    observações. Os resíduos brutos (fit.resid) incluem artefactos de
    inicialização que inflacionam artificialmente os testes de normalidade.
    """
    # ── Resíduos completos (incluindo burn-in) ───────────────────
    std_fe = fit.standardized_forecasts_error.squeeze()
    resid_all = pd.Series(std_fe, index=series.index)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        resid_all.index,
        resid_all.values,
        color=AZUL_ESCURO,
        linewidth=0.9,
        label="Resíduos padronizados"
    )

    # linha horizontal em zero
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)

    # marca a observação 13 (índice do primeiro valor após burn-in)
    burn = 13
    burn_date = resid_all.index[burn]

    ax.axvline(
        burn_date,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Fim do burn-in ({burn} observações)"
    )

    # opcional: sombrear região de burn-in
    ax.axvspan(
        resid_all.index[0],
        burn_date,
        color="orange",
        alpha=0.15
    )

    #ax.set_title("Resíduos Padronizados (incluindo período de inicialização)")
    ax.set_ylabel("Resíduo padronizado")
    ax.legend()

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(f"fig_residual_{modelname}.png", bbox_inches="tight")
    plt.close()
    print(f"fig_residual_{modelname}.png  ✓")


    # ── Resíduos correctos: padronizados, sem burn-in ─────────
    d, D, s = 1, 1, 12
    burn = d + D * s  # = 13 observações de inicialização
    std_fe = fit.standardized_forecasts_error.squeeze()
    resid  = pd.Series(std_fe, index=series.index).iloc[burn:].dropna()
    print(f"  Resíduos após burn-in: n={len(resid)} "
          f"(excluídas as primeiras {burn} obs. de inicialização)")

    # ── Figura diagnóstico ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Série dos resíduos — usa resid.index (56 pts), não series.index (69 pts)
    axes[0, 0].plot(resid.index, resid.values, color=AZUL_ESCURO, linewidth=0.9)
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Resíduos Padronizados ao longo do tempo")
    axes[0, 0].set_ylabel("Resíduo padronizado")
    axes[0, 0].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # FAC dos resíduos — lags limitado a < 50% do tamanho
    max_lags_resid = min(24, len(resid) // 2 - 1)
    plot_acf(resid, lags=max_lags_resid, ax=axes[0, 1],
             title="FAC dos Resíduos Padronizados")

    # Histograma com curva normal
    axes[1, 0].hist(resid, bins=20, density=True,
                    color=AZUL_MEDIO, alpha=0.75, edgecolor="white")
    xmin, xmax = axes[1, 0].get_xlim()
    xx = np.linspace(xmin, xmax, 200)
    axes[1, 0].plot(xx, stats.norm.pdf(xx, 0, 1),
                    color="red", linewidth=1.8, label="N(0,1)")
    axes[1, 0].set_title("Histograma dos Resíduos Padronizados")
    axes[1, 0].legend()

    # QQ-plot
    stats.probplot(resid, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ-plot dos Resíduos Padronizados")

    #plt.suptitle("Diagnóstico dos Resíduos — Modelo SARIMA", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"fig_diagnostico_{modelname}.png", bbox_inches="tight")
    plt.close()
    print(f"fig_diagnostico_{modelname}.png  ✓")

    # ── Testes formais ────────────────────────────────────────
    print("\n── Testes de Diagnóstico dos Resíduos ────────────────")
    print(f"   (n={len(resid)}, excluídas {burn} obs. de burn-in)")

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
    arch_stat, arch_p, _, _ = het_arch(resid, nlags=min(12, len(resid)//5))
    print(f"ARCH-LM:       stat={arch_stat:.4f}  p={arch_p:.4f}  "
          f"→ {'Sem ARCH' if arch_p > 0.05 else 'ARCH detectado'} (α=5%)")

    # Model internal check (reference)
    jb_int = fit.test_normality("jarquebera")
    print(f"\nJB interno (statsmodels): stat={jb_int[0][0]:.4f}  "
          f"p={jb_int[0][1]:.4f}  [referência — exclui burn-in automaticamente]")




# =============================================================================
# PREVISÃO
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
    hist = series
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hist.index, hist.values,
            color=AZUL_ESCURO, linewidth=1.4, label="Histórico (2015-2020)")
    ax.plot(fut_idx, fc_mean.values,
            color=LARANJA, linewidth=1.8,
            linestyle="--", label="Previsao pontual")
    ax.fill_between(fut_idx,
                    fc_ci95.iloc[:, 0].values, fc_ci95.iloc[:, 1].values,
                    color=AZUL_MEDIO, alpha=0.20, label="IC 95%")
    ax.fill_between(fut_idx,
                    fc_ci80.iloc[:, 0].values, fc_ci80.iloc[:, 1].values,
                    color=AZUL_MEDIO, alpha=0.40, label="IC 80%")
    #ax.set_title("Previsão do Consumo de Eletricidade — Alemanha (2020--2021)")
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

def forecast_model_with_exog(fit, series: pd.Series, delta: float,
                              h: int = 12) -> None:
    """
    Previsão h passos à frente para o modelo com intervenção (covid_decay
    como regressor exógeno). O regressor é estendido para o horizonte de
    previsão, continuando o decaimento exponencial a partir do fim da
    amostra.
    """
    # Posição do início da intervenção (mar/2020) e extensão do decay
    # para o período de previsão
    t0_pos = series.index.get_indexer([series["2020-03-01":].index[0]])[0]
    n_total = len(series) + h
    full_decay = np.array([
        delta ** (i - t0_pos) if i >= t0_pos else 0.0
        for i in range(n_total)
    ])
    future_decay = full_decay[len(series):]
    future_idx = pd.date_range(series.index[-1], periods=h + 1, freq="MS")[1:]
    exog_future = pd.DataFrame({"covid_decay": future_decay}, index=future_idx)

    fc       = fit.get_forecast(steps=h, exog=exog_future)
    fc_mean  = fc.predicted_mean
    fc_ci95  = fc.conf_int(alpha=0.05)
    fc_ci80  = fc.conf_int(alpha=0.20)
    fut_idx  = fc_mean.index

    hist = series
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hist.index, hist.values,
            color=AZUL_ESCURO, linewidth=1.4, label="Histórico")
    ax.plot(fut_idx, fc_mean.values,
            color=LARANJA, linewidth=1.8,
            linestyle="--", label="Previsão pontual")
    ax.fill_between(fut_idx,
                    fc_ci95.iloc[:, 0].values, fc_ci95.iloc[:, 1].values,
                    color=AZUL_MEDIO, alpha=0.20, label="IC 95%")
    ax.fill_between(fut_idx,
                    fc_ci80.iloc[:, 0].values, fc_ci80.iloc[:, 1].values,
                    color=AZUL_MEDIO, alpha=0.40, label="IC 80%")
    #ax.set_title("Previsão do Consumo de Eletricidade — Alemanha " "(Modelo com Intervenção)")
    ax.set_ylabel("GWh")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    plt.tight_layout()
    plt.savefig("fig_previsao.png", bbox_inches="tight")
    plt.close()
    print("fig_previsao.png  ✓")

    fc_table = pd.DataFrame({
        "Mês":         fut_idx.strftime("%b/%Y"),
        "Previsão":    fc_mean.round(1).values,
        "IC 95% inf.": fc_ci95.iloc[:, 0].round(1).values,
        "IC 95% sup.": fc_ci95.iloc[:, 1].round(1).values,
    })
    print("\n── Tabela de Previsões (Modelo com Intervenção) ──────")
    print(fc_table.to_string(index=False))

# =============================================================================
# MAIN PIPELINE
# =============================================================================

#if __name__ == "__main__":
print("=" * 60)
print("  ME607 — Análise de Séries Temporais")
print("  Consumo de Eletricidade na Alemanha (ENTSO-E)")
print("=" * 60)

# Dados
print("-" * 60)
print("Dados")
print("-" * 60)
series = load_data()

# Análise exploratória
print("-" * 60)
print("Análise exploratória")
print("-" * 60)
descriptive_stats(series)
plot_serie(series)
plot_seasonal_profile(series)
plot_heatmap_sazonal(series)
plot_stl(series)
plot_acf_pacf(series)

# Testes de estacionaridade
print("-" * 60)
print("Testes de estacionaridade")
print("-" * 60)
run_stationarity_tests(series)

# Basemodel: Identificação e estimação
print("-" * 60)
print("Identificação e estimação")
print("-" * 60)
order, seasonal_order = compare_candidates(series)

#Candidatos
m1_order = (0, 1, 2)
m1_seasonal_order = (0, 1, 2, 12)

m2_order = (0, 1, 1)
m2_seasonal_order = (0, 1, 2, 12)

m3_order = (0, 1, 0)
m3_seasonal_order = (0, 1, 2, 12)

m4_order = (0, 1, 1)
m4_seasonal_order = (0, 1, 1, 12)

m5_order = (0, 1, 0)
m5_seasonal_order = (0, 1, 1, 12)

print("-" * 60)
print(f"SARIMA({m1_order})({m1_seasonal_order})")
estimate_model(series, m1_order, m1_seasonal_order)
print("-" * 60)
print(f"SARIMA({m2_order})({m2_seasonal_order})")
estimate_model(series, m2_order, m2_seasonal_order)
print("-" * 60)
print(f"SARIMA({m3_order})({m3_seasonal_order})")
estimate_model(series, m3_order, m3_seasonal_order)
print("-" * 60)
print(f"SARIMA({m4_order})({m4_seasonal_order})")
estimate_model(series, m4_order, m4_seasonal_order)
print("-" * 60)
print(f"SARIMA({m5_order})({m5_seasonal_order})")
estimate_model(series, m5_order, m5_seasonal_order)


#--------Modelo elegido
order = (0, 1, 1)
seasonal_order = (0, 1, 1, 12)
fit = estimate_model(series, order, seasonal_order)

#--------Residuos
print("-" * 60)
print("Base Diagnóstico / Residuos")
print("-" * 60)
diagnose_residuals(fit, series, modelname="base")


#Análise de intervenção
print("-" * 60)
print("Análise de intervenção")
print("-" * 60)
fit_int = intervention_naive(series, order, seasonal_order, fit_base=fit)
fit_pre, intervencao_df = intervention_counterfactual(series, order, seasonal_order)
plot_intervention_counterfactual(series, fit_pre, intervencao_df)

fit_dyn, delta_otimo, comparacao_df = intervention_dynamic(
    series, order, seasonal_order, delta_grid=(0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99 ), fit_base=fit
)
plot_dynamic_intervention(series, fit, fit_dyn, delta_otimo)

#--------Residuos
print("-" * 60)
print("Modelo intervenção Diagnóstico / Residuos")
print("-" * 60)
diagnose_residuals(fit_dyn, series, modelname="intervention")


#Previsão
print("-" * 60)
print("Previsão")
print("-" * 60)
forecast_model_with_exog(fit_dyn, series, delta_otimo, h=12)

print("\n" + "=" * 60)
print("  Análise concluída. Todas as figuras geradas.")
print("=" * 60)