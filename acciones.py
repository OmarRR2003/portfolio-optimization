"""
Análisis y optimización de un portafolio de N acciones.

Este módulo contiene la lógica (datos, métricas, optimización, validación y gráficos).
Se usa de dos formas:

  * Desde el notebook `acciones.ipynb` (visual, paso a paso):
        import acciones as ac
        cfg = ac.Config(["AAPL", "MSFT", "KO"])
        precios, bench = ac.preparar_precios(cfg)

  * Desde la terminal (reporte completo + CSV y gráficos en disco):
        python acciones.py AAPL MSFT GOOGL AMZN --capital 10000
        python acciones.py --help
"""

from libs import (
    DiscreteAllocation, EfficientCVaR, EfficientFrontier, Path, argparse, dataclass, date,
    Markdown, display, expected_returns, field, leaves_list, linkage, minimize, mticker, np,
    objective_functions, os, pd, plt, replace, risk_models, squareform, sys, timedelta, widgets, yf,
)

DIAS = 252  # días hábiles por año
REBALANCEOS = {"diario": "D", "mensual": "ME", "trimestral": "QE", "anual": "YE", "nunca": None}
PCT = {
    "Rend. total", "Rend. anual (CAGR)", "Volatilidad", "Máx. drawdown", "VaR 95% diario",
    "CVaR 95% diario", "Alfa anual", "Tracking error", "% días positivos",
    "Rend. esperado", "Volatilidad esperada",
}


# =========================================================================== configuración

@dataclass
class Config:
    tickers: list[str]
    inicio: str = field(default_factory=lambda: str(date.today() - timedelta(days=365 * 5)))
    fin: str = field(default_factory=lambda: str(date.today()))
    benchmark: str | None = "SPY"
    rf: float | None = None          # None = promedio de ^IRX en el periodo
    max_peso: float = 0.40           # peso máximo por activo en las optimizaciones
    mu: str = "historico"            # historico | ema | capm
    cov: str = "ledoit-wolf"         # ledoit-wolf | muestral
    gamma: float = 0.1               # regularización L2
    test: float = 0.30               # fracción final para validar fuera de muestra
    rebalanceo: str = "mensual"      # diario | mensual | trimestral | anual | nunca
    capital: float | None = None
    salida: Path = Path("resultados")
    min_cobertura: float = 0.90

    def __post_init__(self):
        if isinstance(self.tickers, str):
            self.tickers = [self.tickers]
        tickers = []
        for t in self.tickers:
            tickers += [x.strip().upper() for x in t.split(",") if x.strip()]
        self.tickers = list(dict.fromkeys(tickers))
        if not self.tickers:
            raise ValueError("Debes indicar al menos un ticker.")
        if self.benchmark is not None:
            b = self.benchmark.strip().upper()
            self.benchmark = None if b in {"", "NINGUNO", "NONE"} else b
        if not 0 <= self.test < 0.9:
            raise ValueError("test debe estar entre 0 y 0.9")
        if not 0 < self.max_peso <= 1:
            raise ValueError("max_peso debe estar entre 0 y 1")
        if self.rebalanceo not in REBALANCEOS:
            raise ValueError(f"rebalanceo debe ser uno de {list(REBALANCEOS)}")
        self.salida = Path(self.salida)


# =========================================================================== utilidades

def aviso(msg: str) -> None:
    print(f"  [aviso] {msg}", flush=True)


def formatear(df: pd.DataFrame) -> pd.DataFrame:
    """Formato de texto para la terminal (columnas = métricas)."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        fmt = "{:.2%}" if c in PCT else "{:.2f}"
        out[c] = df[c].map(lambda x, f=fmt: f.format(x) if pd.notna(x) and np.isfinite(x) else "-")
    return out


def estilo(df: pd.DataFrame, colorear=("Sharpe", "Sortino", "Rend. anual (CAGR)", "Máx. drawdown")):
    """Formato con colores para el notebook (verde = mejor)."""
    fmt = {c: ("{:.2%}" if c in PCT else "{:.2f}") for c in df.columns}
    st = df.style.format(fmt, na_rep="-")
    for c in colorear:
        if c in df.columns and df[c].notna().sum() > 1:
            st = st.background_gradient(subset=[c], cmap="RdYlGn")
    return st


def estilo_pesos(df: pd.DataFrame):
    return df.style.format(lambda x: f"{x:.1%}" if x > 0 else "-").background_gradient(
        cmap="Blues", vmin=0, vmax=max(0.4, float(df.max().max())))


def limpiar_pesos(w: pd.Series, umbral: float = 1e-4) -> pd.Series:
    w = w.clip(lower=0)
    w[w < umbral] = 0.0
    return w / w.sum()


# =========================================================================== datos

def descargar_cierres(tickers: list[str], inicio: str, fin: str) -> pd.DataFrame:
    datos = yf.download(tickers, start=inicio, end=fin, auto_adjust=True,
                        progress=False, threads=True)
    if datos is None or datos.empty:
        return pd.DataFrame()
    cierre = datos["Close"]
    if isinstance(cierre, pd.Series):
        cierre = cierre.to_frame(tickers[0])
    cierre.index = pd.to_datetime(cierre.index)
    if cierre.index.tz is not None:
        cierre.index = cierre.index.tz_localize(None)
    return cierre.sort_index()


def preparar_precios(cfg: Config) -> tuple[pd.DataFrame, pd.Series | None]:
    """Descarga y limpia precios. Devuelve (precios de los activos, precios del benchmark)."""
    print(f"Descargando {len(cfg.tickers)} activos ({cfg.inicio} → {cfg.fin})...")
    precios = descargar_cierres(cfg.tickers, cfg.inicio, cfg.fin)

    faltantes = [t for t in cfg.tickers if t not in precios.columns or precios[t].isna().all()]
    for t in faltantes:
        aviso(f"'{t}' no tiene datos (¿ticker inválido?). Se descarta.")
    precios = precios.drop(columns=[t for t in faltantes if t in precios.columns])
    if precios.empty:
        raise RuntimeError("No se obtuvo ningún dato de precios.")

    # Quedarse con los días en que opera la mayoría (evita fines de semana de cripto, etc.)
    precios = precios[precios.notna().mean(axis=1) >= 0.5]

    # Descartar activos con historial demasiado corto respecto al periodo
    cobertura = precios.notna().mean()
    cortos = cobertura[cobertura < cfg.min_cobertura].index.tolist()
    for t in cortos:
        aviso(f"'{t}' solo tiene datos en {cobertura[t]:.0%} del periodo "
              f"(mínimo {cfg.min_cobertura:.0%}). Se descarta.")
    precios = precios.drop(columns=cortos)

    precios = precios.ffill(limit=5).dropna()
    precios = precios.loc[:, precios.nunique() > 1]
    if len(precios) < 60 or precios.shape[1] == 0:
        raise RuntimeError("Muy pocos datos válidos tras la limpieza (se necesitan ≥ 60 días).")

    bench = None
    if cfg.benchmark:
        b = descargar_cierres([cfg.benchmark], cfg.inicio, cfg.fin)
        if b.empty or b.iloc[:, 0].isna().all():
            aviso(f"No se pudo descargar el benchmark '{cfg.benchmark}'. Se omite.")
        else:
            bench = b.iloc[:, 0].reindex(precios.index).ffill().bfill()
            bench.name = cfg.benchmark

    print(f"  Activos válidos: {', '.join(precios.columns)}")
    print(f"  Periodo efectivo: {precios.index[0].date()} → {precios.index[-1].date()} "
          f"({len(precios)} días)")
    return precios, bench


def tasa_libre_riesgo(cfg: Config) -> float:
    """Promedio del T-Bill a 13 semanas (^IRX) en el periodo, o 2 % si falla."""
    if cfg.rf is not None:
        return cfg.rf
    try:
        irx = descargar_cierres(["^IRX"], cfg.inicio, cfg.fin).iloc[:, 0].dropna()
        rf = float(irx.mean()) / 100
        if np.isfinite(rf) and 0 <= rf < 0.2:
            return rf
    except Exception:
        pass
    aviso("No se pudo obtener la tasa libre de riesgo (^IRX). Se usa 2 %.")
    return 0.02


def rendimientos(precios: pd.DataFrame | pd.Series):
    return precios.pct_change().dropna()


# =========================================================================== métricas

def metricas(r: pd.Series, rf: float, bench: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {}
    rf_d = (1 + rf) ** (1 / DIAS) - 1
    exceso = r - rf_d
    valor = (1 + r).cumprod()
    años = len(r) / DIAS
    cagr = valor.iloc[-1] ** (1 / años) - 1 if valor.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(DIAS)
    sharpe = exceso.mean() / r.std() * np.sqrt(DIAS) if r.std() > 0 else np.nan
    desv_neg = np.sqrt((np.minimum(exceso, 0) ** 2).mean()) * np.sqrt(DIAS)
    sortino = exceso.mean() * DIAS / desv_neg if desv_neg > 0 else np.nan
    mdd = drawdown(r).min()
    var95 = -np.percentile(r, 5)
    cola = r[r <= -var95]
    m = {
        "Rend. total": valor.iloc[-1] - 1,
        "Rend. anual (CAGR)": cagr,
        "Volatilidad": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Máx. drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "VaR 95% diario": var95,
        "CVaR 95% diario": -cola.mean() if len(cola) else np.nan,
        "Asimetría": r.skew(),
        "Curtosis": r.kurt(),
        "% días positivos": (r > 0).mean(),
    }
    if bench is not None:
        b = bench.reindex(r.index).dropna()
        ra = r.reindex(b.index)
        if len(b) > 2 and b.var() > 0:
            beta = np.cov(ra, b)[0, 1] / b.var()
            alfa = ((ra - rf_d).mean() - beta * (b - rf_d).mean()) * DIAS
            dif = ra - b
            te = dif.std() * np.sqrt(DIAS)
            m.update({
                "Beta": beta,
                "Alfa anual": alfa,
                "Tracking error": te,
                "Information ratio": dif.mean() * DIAS / te if te > 0 else np.nan,
            })
    return m


def tabla_metricas(rend: pd.DataFrame, rf: float, bench: pd.Series | None) -> pd.DataFrame:
    return pd.DataFrame({c: metricas(rend[c], rf, bench) for c in rend.columns}).T


def drawdown(r: pd.Series | pd.DataFrame):
    valor = (1 + r).cumprod()
    return valor / valor.cummax().clip(lower=1) - 1


def pares_correlacion(corr: pd.DataFrame, k: int = 5) -> tuple[pd.Series, pd.Series]:
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pares = corr.where(mask).stack()
    pares.index = [f"{a} / {b}" for a, b in pares.index]
    return pares.sort_values(ascending=False).head(k), pares.sort_values().head(k)


# =========================================================================== portafolios

def estimar_mu_S(precios: pd.DataFrame, bench: pd.Series | None, cfg: Config, rf: float):
    """Rendimientos esperados (mu) y matriz de covarianza anualizada (S)."""
    if cfg.mu == "capm" and bench is not None:
        mu = expected_returns.capm_return(
            precios, market_prices=bench.reindex(precios.index).to_frame(), risk_free_rate=rf)
    elif cfg.mu == "ema":
        mu = expected_returns.ema_historical_return(precios, span=DIAS)
    else:
        mu = expected_returns.mean_historical_return(precios)
    if cfg.cov == "ledoit-wolf":
        S = risk_models.CovarianceShrinkage(precios).ledoit_wolf()
    else:
        S = risk_models.sample_cov(precios)
    return mu, risk_models.fix_nonpositive_semidefinite(S)


def paridad_riesgo(S: pd.DataFrame) -> pd.Series:
    """Equal Risk Contribution (formulación convexa de Spinu)."""
    cov = S.values
    n = len(cov)
    res = minimize(
        lambda y: 0.5 * y @ cov @ y - np.log(y).sum() / n,
        1 / np.sqrt(np.diag(cov)),
        jac=lambda y: cov @ y - 1 / (n * y),
        method="L-BFGS-B", bounds=[(1e-10, None)] * n,
    )
    return pd.Series(res.x / res.x.sum(), index=S.index)


def hrp(rend: pd.DataFrame, S: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (López de Prado, 2016): agrupa activos por correlación
    y reparte el riesgo por bisección recursiva. No invierte la covarianza: es robusto."""
    corr = rend.corr().values
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    np.fill_diagonal(dist, 0)
    orden = leaves_list(linkage(squareform(dist, checks=False), method="single")).tolist()
    cov = S.values

    def var_cluster(idx):
        sub = cov[np.ix_(idx, idx)]
        ivp = 1 / np.diag(sub)
        ivp /= ivp.sum()
        return ivp @ sub @ ivp

    w = np.ones(len(cov))
    grupos = [orden]
    while grupos:
        siguientes = []
        for g in grupos:
            if len(g) < 2:
                continue
            a, b = g[: len(g) // 2], g[len(g) // 2:]
            va, vb = var_cluster(a), var_cluster(b)
            alfa = 1 - va / (va + vb)
            w[a] *= alfa
            w[b] *= 1 - alfa
            siguientes += [a, b]
        grupos = siguientes
    return pd.Series(w / w.sum(), index=S.index)


def construir_portafolios(precios: pd.DataFrame, bench: pd.Series | None,
                          cfg: Config, rf: float) -> dict[str, pd.Series]:
    """Devuelve {nombre de estrategia: pesos}."""
    activos = precios.columns
    n = len(activos)
    if n == 1:
        return {"Único activo": pd.Series(1.0, index=activos)}

    rend = rendimientos(precios)
    mu, S = estimar_mu_S(precios, bench, cfg, rf)
    limites = (0, max(cfg.max_peso, 1 / n + 1e-6))
    port: dict[str, pd.Series] = {}

    def intentar(nombre, fn):
        try:
            w = pd.Series(fn(), dtype=float).reindex(activos).fillna(0)
            port[nombre] = limpiar_pesos(w)
        except Exception as e:
            aviso(f"No se pudo calcular '{nombre}': {e}")

    def frontera():
        ef = EfficientFrontier(mu, S, weight_bounds=limites)
        if cfg.gamma > 0:
            ef.add_objective(objective_functions.L2_reg, gamma=cfg.gamma)
        return ef

    port["Igual peso"] = pd.Series(1 / n, index=activos)
    iv = 1 / np.sqrt(np.diag(S))
    port["Inversa volatilidad"] = pd.Series(iv / iv.sum(), index=activos)
    intentar("Paridad de riesgo", lambda: paridad_riesgo(S))
    intentar("Mínima volatilidad", lambda: frontera().min_volatility())
    if (mu > rf).any():
        intentar("Máximo Sharpe", lambda: frontera().max_sharpe(risk_free_rate=rf))
    else:
        aviso("Ningún activo supera la tasa libre de riesgo: se omite 'Máximo Sharpe'.")
    intentar("Mínimo CVaR",
             lambda: EfficientCVaR(mu, rend, beta=0.95, weight_bounds=limites).min_cvar())
    intentar("HRP", lambda: hrp(rend, S))
    return port


def simular(rend: pd.DataFrame, w: pd.Series, rebalanceo: str) -> pd.Series:
    """Rendimientos diarios del portafolio con rebalanceo periódico (entre rebalanceos
    los pesos derivan con los precios, como en la realidad)."""
    w_arr = w.reindex(rend.columns).fillna(0).values
    freq = REBALANCEOS[rebalanceo]
    if freq == "D":
        return pd.Series(rend.values @ w_arr, index=rend.index)
    grupos = [rend] if freq is None else [g for _, g in rend.groupby(pd.Grouper(freq=freq))]
    partes = []
    for bloque in grupos:
        if bloque.empty:
            continue
        valor = (1 + bloque).cumprod().values @ w_arr
        previo = np.concatenate([[1.0], valor[:-1]])
        partes.append(pd.Series(valor / previo - 1, index=bloque.index))
    return pd.concat(partes)


def simular_todos(rend: pd.DataFrame, pesos: dict[str, pd.Series], rebalanceo: str) -> pd.DataFrame:
    return pd.DataFrame({n: simular(rend, w, rebalanceo) for n, w in pesos.items()})


def desempeño_ex_ante(pesos: dict[str, pd.Series], mu: pd.Series, S: pd.DataFrame,
                      rf: float) -> pd.DataFrame:
    filas = {}
    for nombre, w in pesos.items():
        w = w.reindex(mu.index).fillna(0)
        r = float(w @ mu)
        v = float(np.sqrt(w @ S @ w))
        filas[nombre] = {
            "Rend. esperado": r,
            "Volatilidad esperada": v,
            "Sharpe": (r - rf) / v if v > 0 else np.nan,
            "N° activos": int((w > 0).sum()),
            "Activos efectivos": 1 / float((w ** 2).sum()),
        }
    return pd.DataFrame(filas).T


def contribucion_riesgo(w: pd.Series, S: pd.DataFrame) -> pd.Series:
    w = w.reindex(S.index).fillna(0)
    marginal = S.values @ w.values
    return pd.Series(w.values * marginal / (w.values @ marginal), index=S.index)


# =========================================================================== validación

@dataclass
class FueraDeMuestra:
    periodo_train: tuple
    periodo_test: tuple
    pesos: dict[str, pd.Series]        # calculados solo con entrenamiento
    rendimientos: pd.DataFrame         # diarios de cada estrategia en el periodo de prueba
    benchmark: pd.Series | None
    metricas: pd.DataFrame             # ordenadas por Sharpe


def validar_fuera_de_muestra(precios: pd.DataFrame, bench: pd.Series | None,
                             cfg: Config, rf: float) -> FueraDeMuestra | None:
    """Optimiza con el primer (1 - test) del periodo y evalúa en el resto."""
    if cfg.test <= 0:
        return None
    corte = int(len(precios) * (1 - cfg.test))
    if corte < DIAS or len(precios) - corte < 60:
        aviso("Datos insuficientes para validar fuera de muestra (se necesita ≥ 1 año de "
              "entrenamiento y ≥ 60 días de prueba). Se omite.")
        return None
    p_train, p_test = precios.iloc[:corte], precios.iloc[corte - 1:]
    pesos = construir_portafolios(p_train, bench, cfg, rf)
    r_test = rendimientos(p_test)
    rend = simular_todos(r_test, pesos, cfg.rebalanceo)
    b = rendimientos(bench).reindex(r_test.index).dropna() if bench is not None else None
    tabla = rend.copy()
    if b is not None:
        tabla[f"[{bench.name}]"] = b
    met = tabla_metricas(tabla, rf, b).sort_values("Sharpe", ascending=False)
    return FueraDeMuestra(
        (p_train.index[0].date(), p_train.index[-1].date()),
        (r_test.index[0].date(), r_test.index[-1].date()),
        pesos, rend, b, met,
    )


def elegir_sugerido(pesos: dict[str, pd.Series], oos: FueraDeMuestra | None,
                    ex_ante: pd.DataFrame) -> tuple[str, str]:
    if oos is not None:
        candidatos = [n for n in oos.metricas.index if n in pesos]
        if candidatos:
            return candidatos[0], "mejor Sharpe fuera de muestra"
    return ex_ante["Sharpe"].idxmax(), "mejor Sharpe esperado (sin validación fuera de muestra)"


def asignacion_discreta(w: pd.Series, precios: pd.DataFrame, capital: float):
    """Cuántas acciones enteras comprar. Devuelve (tabla, efectivo sobrante)."""
    ultimos = precios.iloc[-1]
    w = w[w > 0]
    da = DiscreteAllocation(w.to_dict(), ultimos[w.index], total_portfolio_value=capital)
    asignacion, sobrante = da.greedy_portfolio()
    tabla = pd.DataFrame({
        "Acciones": pd.Series(asignacion, dtype=int),
        "Precio": ultimos[list(asignacion)],
    })
    tabla["Monto"] = tabla["Acciones"] * tabla["Precio"]
    tabla["Peso real"] = tabla["Monto"] / capital
    tabla["Peso objetivo"] = w.reindex(tabla.index)
    return tabla.sort_values("Monto", ascending=False), sobrante


# =========================================================================== gráficos
# Cada función devuelve una figura de matplotlib (el notebook la muestra; la terminal la guarda).

def _pct(ax, eje="y"):
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(mticker.PercentFormatter(1))


def graf_precios(precios: pd.DataFrame, bench: pd.Series | None = None):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    (precios / precios.iloc[0] * 100).plot(ax=ax, lw=1.2, legend=len(precios.columns) <= 15)
    if bench is not None:
        (bench / bench.iloc[0] * 100).plot(ax=ax, color="k", ls="--", lw=1.8, label=bench.name)
        ax.legend(fontsize=8, ncol=2)
    ax.set(title="Precios normalizados (base 100)", ylabel="Valor de 100 invertidos", xlabel="")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def graf_distribucion(rend: pd.DataFrame, columnas: int = 4):
    n = rend.shape[1]
    filas = int(np.ceil(n / columnas))
    cols = min(n, columnas)
    fig, axes = plt.subplots(filas, cols, figsize=(3.6 * cols, 2.8 * filas), squeeze=False)
    for ax, t in zip(axes.flat, rend.columns):
        r = rend[t]
        var = np.percentile(r, 5)
        ax.hist(r, bins=60, color="steelblue", alpha=0.8)
        ax.axvline(var, color="crimson", ls="--", lw=1.2, label=f"VaR 95% = {-var:.1%}")
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=7)
        _pct(ax, "x")
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Distribución de rendimientos diarios", y=1.0)
    fig.tight_layout()
    return fig


def graf_riesgo_rendimiento(met: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(met["Volatilidad"], met["Rend. anual (CAGR)"], c=met["Sharpe"],
                    cmap="RdYlGn", s=120, edgecolor="k")
    for t, fila in met.iterrows():
        ax.annotate(t, (fila["Volatilidad"], fila["Rend. anual (CAGR)"]), fontsize=9,
                    xytext=(6, 4), textcoords="offset points")
    fig.colorbar(sc, ax=ax, label="Sharpe")
    ax.set(title="Riesgo vs. rendimiento por activo", xlabel="Volatilidad anual",
           ylabel="Rendimiento anual (CAGR)")
    _pct(ax, "x")
    _pct(ax, "y")
    ax.axhline(0, color="gray", lw=0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def graf_correlacion(corr: pd.DataFrame):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(5, 0.55 * n + 3), max(4, 0.55 * n + 2)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=90)
    ax.set_yticks(range(n), corr.index)
    if n <= 20:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Matriz de correlación de rendimientos diarios")
    fig.tight_layout()
    return fig


def graf_frontera(mu: pd.Series, S: pd.DataFrame, pesos: dict[str, pd.Series], rf: float,
                  destacar: str | None = None, n_aleatorios: int = 5000):
    fig, ax = plt.subplots(figsize=(11, 7))
    if len(mu) > 1:
        rng = np.random.default_rng(0)
        W = rng.dirichlet(np.ones(len(mu)), n_aleatorios)
        r_al = W @ mu.values
        v_al = np.sqrt(np.einsum("ij,jk,ik->i", W, S.values, W))
        sc = ax.scatter(v_al, r_al, c=(r_al - rf) / v_al, cmap="viridis", s=4, alpha=0.5)
        fig.colorbar(sc, ax=ax, label="Sharpe")
        try:  # frontera teórica (sin límites de peso ni regularización)
            w_min = pd.Series(EfficientFrontier(mu, S).min_volatility())
            fr_v, fr_r = [], []
            for objetivo in np.linspace(float(w_min @ mu), mu.max() * 0.999, 40):
                ef = EfficientFrontier(mu, S)
                try:
                    ef.efficient_return(objetivo)
                    rr, vv, _ = ef.portfolio_performance(risk_free_rate=rf)
                    fr_v.append(vv)
                    fr_r.append(rr)
                except Exception:
                    continue
            ax.plot(fr_v, fr_r, "k-", lw=2, label="Frontera eficiente")
        except Exception:
            pass
    vols = np.sqrt(np.diag(S))
    ax.scatter(vols, mu, marker="o", c="gray", edgecolor="k", s=50)
    for t, v in zip(mu.index, vols):
        ax.annotate(t, (v, mu[t]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    marcadores = "*PXDsv^<>"
    for i, (nombre, w) in enumerate(pesos.items()):
        w = w.reindex(mu.index).fillna(0)
        grande = nombre == destacar
        ax.scatter(np.sqrt(w @ S @ w), w @ mu, marker=marcadores[i % len(marcadores)],
                   s=380 if grande else 180, edgecolor="red" if grande else "k",
                   linewidth=2.5 if grande else 1, label=nombre, zorder=5)
    ax.set(title="Frontera eficiente", xlabel="Volatilidad anual",
           ylabel="Rendimiento anual esperado")
    _pct(ax, "x")
    _pct(ax, "y")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def graf_pesos(pesos: dict[str, pd.Series]):
    tabla = pd.DataFrame(pesos).T
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(tabla) + 2))
    tabla.plot.barh(stacked=True, ax=ax, colormap="tab20", width=0.75)
    ax.set(title="Pesos por estrategia", xlabel="Peso", xlim=(0, 1))
    _pct(ax, "x")
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def graf_crecimiento(rend_port: pd.DataFrame, bench: pd.Series | None = None,
                     titulo: str = "Crecimiento de 100"):
    curvas = (1 + rend_port).cumprod() * 100
    fig, ax = plt.subplots(figsize=(11, 5.5))
    curvas.plot(ax=ax, lw=1.4)
    if bench is not None:
        ((1 + bench).cumprod() * 100).plot(ax=ax, color="k", ls="--", lw=2,
                                           label=f"Benchmark ({bench.name})")
    ax.legend(fontsize=8)
    ax.set(title=titulo, ylabel="Valor", xlabel="")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def graf_drawdown(rend_port: pd.DataFrame, bench: pd.Series | None = None,
                  titulo: str = "Drawdown (caída desde el máximo)"):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    drawdown(rend_port).plot(ax=ax, lw=1.1)
    if bench is not None:
        drawdown(bench).plot(ax=ax, color="k", ls="--", lw=1.6, label=f"Benchmark ({bench.name})")
    ax.legend(fontsize=8)
    ax.set(title=titulo, xlabel="")
    _pct(ax)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def graf_contribucion(w: pd.Series, S: pd.DataFrame, titulo: str = ""):
    cr = contribucion_riesgo(w, S)
    df = pd.DataFrame({"Peso (capital)": w, "Contribución al riesgo": cr})
    df = df[df["Peso (capital)"] > 0].sort_values("Peso (capital)", ascending=False)
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(df) + 3), 4.5))
    df.plot.bar(ax=ax, color=["steelblue", "darkorange"], width=0.75)
    ax.set(title=titulo or "Capital invertido vs. riesgo aportado", xlabel="")
    _pct(ax)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# =========================================================================== interpretación
# Textos en Markdown generados a partir de los resultados: el notebook los muestra para que
# la lectura se actualice sola al cambiar acciones, periodo o parámetros.

def _nota(lineas: list[str]) -> str:
    return "> **Tus resultados**\n>\n" + "\n".join(f"> - {l}" for l in lineas if l)


def lectura_precios(precios: pd.DataFrame, bench: pd.Series | None) -> str:
    total = precios.iloc[-1] / precios.iloc[0] - 1
    años = len(precios) / DIAS
    l = [f"Periodo analizado: **{precios.index[0].date()} → {precios.index[-1].date()}** "
         f"(≈ {años:.1f} años, {precios.shape[1]} acciones).",
         f"Mejor: **{total.idxmax()}** ({total.max():+.0%}). Peor: **{total.idxmin()}** ({total.min():+.0%})."]
    if bench is not None:
        b = bench.iloc[-1] / bench.iloc[0] - 1
        l.append(f"**{(total > b).sum()} de {len(total)}** acciones superaron a {bench.name} ({b:+.0%}).")
    perdedoras = total[total < 0]
    if len(perdedoras):
        l.append(f"Terminaron en pérdida: {', '.join(perdedoras.index)}.")
    return _nota(l)


def lectura_metricas(met: pd.DataFrame, rf: float) -> str:
    acc = met[~met.index.str.startswith("[")]
    l = [f"Mejor rendimiento por riesgo (Sharpe): **{acc['Sharpe'].idxmax()}** ({acc['Sharpe'].max():.2f}); "
         f"peor: **{acc['Sharpe'].idxmin()}** ({acc['Sharpe'].min():.2f}).",
         f"Más volátil: **{acc['Volatilidad'].idxmax()}** ({acc['Volatilidad'].max():.0%} anual); "
         f"más estable: **{acc['Volatilidad'].idxmin()}** ({acc['Volatilidad'].min():.0%}).",
         f"Peor caída: **{acc['Máx. drawdown'].idxmin()}** llegó a perder {-acc['Máx. drawdown'].min():.0%} "
         f"desde su máximo."]
    bajo_rf = acc.index[acc["Rend. anual (CAGR)"] < rf].tolist()
    if bajo_rf:
        l.append(f"Rindieron menos que la tasa libre de riesgo ({rf:.1%}): {', '.join(bajo_rf)}. "
                 f"Se asumió riesgo sin recompensa.")
    if "Beta" in acc:
        l.append(f"Más sensible al mercado: **{acc['Beta'].idxmax()}** (beta {acc['Beta'].max():.2f}); "
                 f"más defensiva: **{acc['Beta'].idxmin()}** (beta {acc['Beta'].min():.2f}).")
    return _nota(l)


def lectura_correlacion(corr: pd.DataFrame) -> str:
    if len(corr) < 2:
        return _nota(["Con una sola acción no hay diversificación posible."])
    prom = corr.values[np.triu_indices(len(corr), 1)].mean()
    nivel = ("alta: las acciones se mueven muy parecido y diversifican poco" if prom > 0.6 else
             "moderada: hay beneficio de diversificación, pero limitado" if prom > 0.3 else
             "baja: buena diversificación")
    altos, bajos = pares_correlacion(corr, 1)
    return _nota([f"Correlación promedio **{prom:.2f}**, {nivel}.",
                  f"Par más parecido: **{altos.index[0]}** ({altos.iloc[0]:.2f}). "
                  f"Par más complementario: **{bajos.index[0]}** ({bajos.iloc[0]:.2f})."])


def lectura_pesos(pesos: dict[str, pd.Series], ex_ante: pd.DataFrame) -> str:
    tabla = pd.DataFrame(pesos)
    conc = ex_ante["Activos efectivos"].idxmin()
    t, e = tabla.stack().idxmax()
    return _nota([
        f"Estrategia más concentrada: **{conc}** (equivale a {ex_ante.loc[conc, 'Activos efectivos']:.1f} "
        f"acciones a igual peso, de {len(tabla)}).",
        f"Mayor peso individual: **{t}** con {tabla.loc[t, e]:.0%} en *{e}*.",
        f"Mejor Sharpe *esperado*: **{ex_ante['Sharpe'].idxmax()}** ({ex_ante['Sharpe'].max():.2f}), "
        f"aún sin validar.",
    ])


def lectura_oos(oos: FueraDeMuestra, met_hist: pd.DataFrame | None = None) -> str:
    met = oos.metricas
    estr = met[~met.index.str.startswith("[")]
    gan = estr.index[0]
    l = [f"Ganadora fuera de muestra: **{gan}** (Sharpe {estr.loc[gan, 'Sharpe']:.2f}, "
         f"CAGR {estr.loc[gan, 'Rend. anual (CAGR)']:.1%}, peor caída {estr.loc[gan, 'Máx. drawdown']:.1%})."]
    if oos.benchmark is not None:
        b = met.loc[f"[{oos.benchmark.name}]"]
        n = (estr["Sharpe"] > b["Sharpe"]).sum()
        l.append(f"**{n} de {len(estr)}** estrategias superaron el Sharpe del benchmark ({b['Sharpe']:.2f}).")
    rango = estr["Sharpe"].max() - estr["Sharpe"].min()
    if rango < 0.3:
        l.append(f"La diferencia de Sharpe entre la mejor y la peor es solo {rango:.2f}: "
                 f"probablemente **ninguna es claramente superior** (podría ser azar).")
    if met_hist is not None and "Máximo Sharpe" in estr.index and "Máximo Sharpe" in met_hist.index:
        pos_in = met_hist["Sharpe"].rank(ascending=False)["Máximo Sharpe"]
        pos_out = estr["Sharpe"].rank(ascending=False)["Máximo Sharpe"]
        l.append(f"*Máximo Sharpe* quedó #{pos_in:.0f} dentro de muestra y **#{pos_out:.0f}** fuera de muestra"
                 + (": señal de sobreajuste." if pos_out > pos_in else "."))
    return _nota(l)


def lectura_final(nombre: str, w: pd.Series, S: pd.DataFrame) -> str:
    cr = contribucion_riesgo(w, S)
    w = w[w > 0]
    t_r = cr.idxmax()
    l = [f"Mayor inversión: **{w.idxmax()}** ({w.max():.0%} del capital).",
         f"Mayor fuente de riesgo: **{t_r}** ({cr[t_r]:.0%} del riesgo con {w.get(t_r, 0):.0%} del capital)."]
    activos = cr[cr.index.isin(w.index)]
    if (activos - 1 / len(activos)).abs().max() < 0.02:
        l.append("Cada acción aporta prácticamente el mismo riesgo: el riesgo está bien repartido.")
    elif cr.max() > 0.4:
        l.append(f"Una sola acción concentra más del 40 % del riesgo: el portafolio depende mucho de {t_r}.")
    return _nota(l)


# =========================================================================== análisis completo

@dataclass
class Analisis:
    """Todos los resultados de un análisis, en un solo objeto."""
    cfg: Config
    precios: pd.DataFrame
    bench: pd.Series | None
    rf: float
    rend: pd.DataFrame
    bench_rend: pd.Series | None
    met_activos: pd.DataFrame          # incluye el benchmark como fila "[SPY]"
    corr: pd.DataFrame
    mu: pd.Series
    S: pd.DataFrame                    # covarianza según cfg.cov
    S_muestral: pd.DataFrame           # para comparar con Ledoit-Wolf
    pesos: dict[str, pd.Series]        # calculados con todo el historial
    ex_ante: pd.DataFrame
    met_hist: pd.DataFrame             # desempeño dentro de muestra
    oos: FueraDeMuestra | None
    elegido: str
    motivo: str

    @property
    def w(self) -> pd.Series:
        """Pesos del portafolio sugerido."""
        return self.pesos[self.elegido]


def analizar(cfg: Config) -> Analisis:
    """Ejecuta el análisis completo: datos, métricas, estrategias y validación."""
    precios, bench = preparar_precios(cfg)
    rf = tasa_libre_riesgo(cfg)
    rend = rendimientos(precios)
    bench_rend = rendimientos(bench) if bench is not None else None

    tabla = rend.copy()
    if bench_rend is not None:
        tabla[f"[{bench.name}]"] = bench_rend
    met_activos = tabla_metricas(tabla, rf, bench_rend)

    mu, S = estimar_mu_S(precios, bench, cfg, rf)
    _, S_muestral = estimar_mu_S(precios, bench, replace(cfg, cov="muestral"), rf)
    pesos = construir_portafolios(precios, bench, cfg, rf)
    ex_ante = desempeño_ex_ante(pesos, mu, S, rf)
    met_hist = tabla_metricas(simular_todos(rend, pesos, cfg.rebalanceo), rf, bench_rend)
    oos = validar_fuera_de_muestra(precios, bench, cfg, rf)
    elegido, motivo = elegir_sugerido(pesos, oos, ex_ante)
    return Analisis(cfg, precios, bench, rf, rend, bench_rend, met_activos, rend.corr(), mu, S,
                    S_muestral, pesos, ex_ante, met_hist, oos, elegido, motivo)


# =========================================================================== notebook / app

MAX_TICKERS = 30  # límite del panel para que la app pública no se sature


def en_voila() -> bool:
    """True si el notebook se está mostrando como app con Voilà (Voilà define estas variables)."""
    return "VOILA_SERVER_URL" in os.environ or "VOILA_BASE_URL" in os.environ


class Tablero:
    """Coordina el panel de control y las secciones del notebook.

    * Modo app (Voilà): cada sección vive en un widget Output y se vuelve a dibujar cuando
      el visitante pulsa "Analizar" con otras acciones.
    * Modo notebook (Jupyter, GitHub, nbviewer): las secciones se dibujan de forma normal,
      así los resultados quedan guardados en el archivo y se ven sin ejecutar nada.
    """

    def __init__(self, interactivo: bool | None = None):
        self.interactivo = en_voila() if interactivo is None else interactivo
        self.A: Analisis | None = None
        self._secciones: list = []
        self._boton = None

    def analizar(self, cfg: Config) -> Analisis:
        self.A = analizar(cfg)
        return self.A

    # ---- secciones
    def seccion(self, fn):
        """Decorador: registra una función que dibuja una sección usando `self.A`."""
        if self.interactivo:
            out = widgets.Output()
            self._secciones.append((out, fn))
            display(out)
            self._dibujar(out, fn)
        else:
            fn()
            plt.show()
        return fn

    def redibujar(self) -> None:
        for out, fn in self._secciones:
            self._dibujar(out, fn)

    @staticmethod
    def _dibujar(out, fn) -> None:
        out.clear_output(wait=True)
        with out:
            try:
                fn()
                plt.show()
            except Exception as e:
                plt.close("all")
                print(f"No se pudo mostrar esta sección: {e}")

    # ---- panel de control
    def panel(self):
        cfg = self.A.cfg
        ancho = widgets.Layout(width="95%")
        tickers = widgets.Textarea(
            value=", ".join(cfg.tickers), placeholder="Ej.: AAPL, MSFT, KO, XOM",
            description="Acciones", layout=widgets.Layout(width="95%", height="60px"))
        inicio = widgets.DatePicker(value=date.fromisoformat(cfg.inicio), description="Desde")
        benchmark = widgets.Text(value=cfg.benchmark or "", description="Benchmark",
                                 placeholder="SPY (vacío = ninguno)")
        capital = widgets.FloatText(value=cfg.capital or 0, description="Capital ($)")
        boton = widgets.Button(description="Analizar", button_style="primary", icon="play",
                               layout=widgets.Layout(width="160px"))
        estado = widgets.Output(layout=ancho)
        self._boton = boton

        def al_pulsar(_):
            boton.disabled, boton.description = True, "Analizando..."
            estado.clear_output()
            with estado:
                try:
                    if inicio.value is None:
                        raise ValueError("Elige una fecha de inicio.")
                    nuevo = replace(self.A.cfg, tickers=[tickers.value],
                                    inicio=str(inicio.value), benchmark=benchmark.value or None,
                                    capital=capital.value or None)
                    if len(nuevo.tickers) > MAX_TICKERS:
                        raise ValueError(f"Máximo {MAX_TICKERS} acciones.")
                    self.analizar(nuevo)
                    if self.interactivo:
                        self.redibujar()
                        print(f"Listo: {len(self.A.precios.columns)} acciones analizadas. "
                              f"Baja para ver los resultados.")
                    else:
                        print("Listo. En Jupyter, ejecuta las celdas de abajo para ver el nuevo "
                              "análisis (Run > Run Selected Cell and All Below).")
                except Exception as e:
                    print(f"Error: {e}")
                finally:
                    boton.disabled, boton.description = False, "Analizar"

        boton.on_click(al_pulsar)
        ayuda = widgets.HTML(
            "<b>Escribe las acciones que quieras analizar</b> (símbolos de Yahoo Finance, separados "
            "por coma) y pulsa <i>Analizar</i>. La descarga y optimización tarda entre 10 y 60 s.")
        return widgets.VBox([ayuda, tickers,
                             widgets.HBox([inicio, benchmark, capital]),
                             boton, estado],
                            layout=widgets.Layout(border="1px solid #ccc", padding="10px"))


# =========================================================================== terminal

def titulo(texto: str) -> None:
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def parsear_args() -> Config:
    d = Config(["X"])  # valores por defecto
    p = argparse.ArgumentParser(
        description="Análisis y optimización de un portafolio de N acciones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("tickers", nargs="+", help="Tickers separados por espacio o coma")
    p.add_argument("--inicio", default=d.inicio, help="AAAA-MM-DD")
    p.add_argument("--fin", default=d.fin, help="AAAA-MM-DD")
    p.add_argument("--benchmark", default=d.benchmark, help="Índice de referencia ('ninguno' para omitir)")
    p.add_argument("--rf", type=float, default=None,
                   help="Tasa libre de riesgo anual (ej. 0.04). Por defecto: promedio de ^IRX")
    p.add_argument("--max-peso", type=float, default=d.max_peso,
                   help="Peso máximo por activo en optimizaciones (1 = sin límite)")
    p.add_argument("--mu", choices=["historico", "ema", "capm"], default=d.mu)
    p.add_argument("--cov", choices=["ledoit-wolf", "muestral"], default=d.cov)
    p.add_argument("--gamma", type=float, default=d.gamma,
                   help="Regularización L2 (más alto = pesos más repartidos; 0 = ninguna)")
    p.add_argument("--test", type=float, default=d.test,
                   help="Fracción final del periodo para validar fuera de muestra (0 = omitir)")
    p.add_argument("--rebalanceo", choices=list(REBALANCEOS), default=d.rebalanceo)
    p.add_argument("--capital", type=float, default=None, help="Monto a invertir")
    p.add_argument("--salida", default=str(d.salida), help="Carpeta para CSV y gráficos")
    p.add_argument("--sin-graficos", action="store_true")
    p.add_argument("--min-cobertura", type=float, default=d.min_cobertura)
    a = p.parse_args()
    try:
        cfg = Config(a.tickers, a.inicio, a.fin, a.benchmark, a.rf, a.max_peso, a.mu, a.cov,
                     a.gamma, a.test, a.rebalanceo, a.capital, Path(a.salida), a.min_cobertura)
    except ValueError as e:
        p.error(str(e))
    cfg.graficos = not a.sin_graficos
    return cfg


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cfg = parsear_args()
    cfg.salida.mkdir(parents=True, exist_ok=True)

    try:
        A = analizar(cfg)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
    precios, bench, rf, corr, oos, pesos = A.precios, A.bench, A.rf, A.corr, A.oos, A.pesos
    mu, S, ex_ante, elegido, motivo = A.mu, A.S, A.ex_ante, A.elegido, A.motivo
    met_act, met_hist = A.met_activos, A.met_hist
    print(f"  Tasa libre de riesgo: {rf:.2%}")
    print(f"  Configuración: mu={cfg.mu}, cov={cfg.cov}, peso máx={cfg.max_peso:.0%}, "
          f"L2 gamma={cfg.gamma}, rebalanceo={cfg.rebalanceo}")

    titulo("1. MÉTRICAS POR ACTIVO (periodo completo)")
    print(formatear(met_act).to_string())

    if len(corr) > 1:
        titulo("2. CORRELACIONES")
        altos, bajos = pares_correlacion(corr)
        print("Pares más correlacionados (diversifican poco):")
        print(altos.round(2).to_string())
        print("\nPares menos correlacionados (diversifican más):")
        print(bajos.round(2).to_string())
        print(f"\nCorrelación promedio: {corr.values[np.triu_indices(len(corr), 1)].mean():.2f}")

    if oos is not None:
        titulo(f"3. VALIDACIÓN FUERA DE MUESTRA\n"
               f"   Entrenamiento: {oos.periodo_train[0]} → {oos.periodo_train[1]}  |  "
               f"Prueba: {oos.periodo_test[0]} → {oos.periodo_test[1]}")
        print(formatear(oos.metricas).to_string())

    titulo("4. PORTAFOLIOS CON TODO EL HISTORIAL (pesos para invertir hoy)")
    tabla_pesos = pd.DataFrame(pesos)
    print(tabla_pesos.map(lambda x: f"{x:.1%}" if x > 0 else "-").to_string())
    print("\nDesempeño esperado (ex ante; tiende a ser optimista):\n")
    print(formatear(ex_ante).to_string())

    titulo(f"5. PORTAFOLIO SUGERIDO: {elegido}  ({motivo})")
    w = pesos[elegido]
    detalle = pd.DataFrame({"Peso": w, "Contribución al riesgo": contribucion_riesgo(w, S)})
    detalle = detalle[detalle["Peso"] > 0].sort_values("Peso", ascending=False)
    print(detalle.map(lambda x: f"{x:.1%}").to_string())
    if cfg.capital:
        try:
            compra, sobrante = asignacion_discreta(w, precios, cfg.capital)
            print(f"\nCompra sugerida con ${cfg.capital:,.2f}:")
            print(compra.to_string(formatters={"Precio": "${:,.2f}".format,
                                               "Monto": "${:,.2f}".format,
                                               "Peso real": "{:.1%}".format,
                                               "Peso objetivo": "{:.1%}".format}))
            print(f"Efectivo sobrante: ${sobrante:,.2f}")
        except Exception as e:
            aviso(f"No se pudo calcular la asignación discreta: {e}")

    # Exportar
    met_act.to_csv(cfg.salida / "metricas_activos.csv")
    corr.to_csv(cfg.salida / "correlacion.csv")
    tabla_pesos.to_csv(cfg.salida / "pesos.csv")
    ex_ante.to_csv(cfg.salida / "desempeno_esperado.csv")
    met_hist.to_csv(cfg.salida / "metricas_portafolios_historico.csv")
    precios.to_csv(cfg.salida / "precios.csv")
    if oos is not None:
        oos.metricas.to_csv(cfg.salida / "metricas_fuera_de_muestra.csv")

    n_graf = 0
    if cfg.graficos:
        figuras = {
            "01_precios.png": graf_precios(precios, bench),
            "02_correlacion.png": graf_correlacion(corr),
            "03_frontera_eficiente.png": graf_frontera(mu, S, pesos, rf, destacar=elegido),
            "04_pesos.png": graf_pesos(pesos),
            "05_contribucion_riesgo.png": graf_contribucion(w, S, f"{elegido}: capital vs. riesgo"),
        }
        if oos is not None:
            figuras["06_fuera_de_muestra.png"] = graf_crecimiento(
                oos.rendimientos, oos.benchmark, "Fuera de muestra: crecimiento de 100")
            figuras["07_drawdown.png"] = graf_drawdown(oos.rendimientos, oos.benchmark)
        for nombre, fig in figuras.items():
            fig.savefig(cfg.salida / nombre, dpi=130)
            plt.close(fig)
        n_graf = len(figuras)

    titulo("ARCHIVOS GENERADOS")
    print(f"Carpeta: {cfg.salida.resolve()}  ({len(list(cfg.salida.glob('*.csv')))} CSV, "
          f"{n_graf} gráficos)")
    print("\nNota: el desempeño pasado no garantiza resultados futuros. Esto es un análisis "
          "cuantitativo, no una recomendación de inversión.")


if __name__ == "__main__":
    plt.switch_backend("Agg")
    main()
