"""
analog.py — Modelo de pronóstico por analogía (Analog KNN).

Autor: Uriel
Descripción: Implementación del método análogo con selección KNN y regresión,
             empaquetado como estimador compatible con:
             • Nixtla StatsForecast  →  sf = StatsForecast(models=[AnalogKNN()])
             • sklearn               →  model.fit(y); model.predict(h)
             • Pipeline propio       →  mismo .fit() / .predict()

Uso con StatsForecast
---------------------
    from statsforecast import StatsForecast
    from analog_holidays.analog import AnalogKNN

    sf = StatsForecast(
        models=[AnalogKNN(season_length=24, k=10, typereg='PCR')],
        freq='h',
    )
    sf.fit(df)
    preds = sf.predict(h=48)

Uso standalone
--------------
    from analog_holidays.analog import AnalogKNN

    model = AnalogKNN(season_length=24)
    model.fit(y=serie_numpy)
    forecast = model.predict(h=48)['mean']
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

# ── Regresiones (lazy import para no romper si falta alguna) ──────────
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.pipeline import make_pipeline

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


# =====================================================================
# Funciones auxiliares de regresión (privadas)
# =====================================================================

def _olsstep(X, Y, X2, pi_step=0.001):
    model = sm.OLS(Y, X)
    results = model.fit()
    pred = results.predict(X2)

    i = 0
    pvalues = [(idx, p) for idx, p in enumerate(results.pvalues)]
    pvalues.sort(key=lambda t: t[1], reverse=True)
    (i, pi) = pvalues[0]

    while pi > pi_step:
        X = sm.add_constant(X)
        X2 = sm.add_constant(X2)
        X = np.delete(X, obj=i, axis=1)
        X2 = np.delete(X2, obj=i, axis=1)
        model = sm.OLS(Y, X)
        results = model.fit()
        pvalues = [(idx, p) for idx, p in enumerate(results.pvalues)]
        pvalues.sort(key=lambda t: t[1], reverse=True)
        (i, pi) = pvalues[0]
        pred = results.predict(X2)

    if len(pred) == 0:
        model = sm.OLS(Y, X)
        results = model.fit()
        pred = results.predict(X2)
    return pred


def _merge_regressor_params(
    default_params: Optional[Mapping[str, object]] = None,
    regressor_params: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    """Merge default model kwargs with Optuna-tuned overrides."""
    merged = dict(default_params or {})
    if regressor_params:
        merged.update({str(key): value for key, value in regressor_params.items()})
    return merged


def _fit_predict(ModelClass, X, Y, X2, regressor_params: Optional[Mapping[str, object]] = None, **kwargs):
    """Ajusta un regresor sklearn genérico y devuelve la predicción."""
    model = ModelClass(**_merge_regressor_params(kwargs, regressor_params))
    model.fit(X, Y)
    return model.predict(X2)


def _fit_predict_lightgbm(X, Y, X2, regressor_params: Optional[Mapping[str, object]] = None):
    """Fit LightGBM only when the optional dependency is installed."""
    if LGBMRegressor is None:
        raise ImportError("Install lightgbm to use typereg='LGBM'.")

    return _fit_predict(
        LGBMRegressor,
        X,
        Y,
        X2,
        regressor_params=regressor_params,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def _voting_ensemble(X, Y, X2):
    gb = GradientBoostingRegressor(random_state=42)
    rf = RandomForestRegressor(random_state=42)
    br = BaggingRegressor(random_state=42)
    ab = AdaBoostRegressor(random_state=42)
    for m in (gb, rf, br, ab):
        m.fit(X, Y)
    voting = VotingRegressor([("gb", gb), ("rf", rf), ("br", br), ("ab", ab)])
    voting.fit(X, Y)
    return voting.predict(X2)


def _voting_linear(X, Y, X2):
    lr = LinearRegression()
    ri = Ridge(alpha=0.1)
    la = Lasso(alpha=0.1)
    pc = make_pipeline(PCA(n_components=min(1, X.shape[1])), LinearRegression())
    for m in (lr, ri, la, pc):
        m.fit(X, Y)
    voting = VotingRegressor([("lr", lr), ("ri", ri), ("la", la), ("pc", pc)])
    voting.fit(X, Y)
    return voting.predict(X2)





# Mapa de regresores disponibles
_REGRESSORS = {
    'OLSstep': lambda X, Y, X2, nc, rp=None: _olsstep(X, Y, X2),
    'RF': lambda X, Y, X2, nc, rp=None: _fit_predict(
        RandomForestRegressor, X, Y, X2, regressor_params=rp, random_state=42, n_jobs=-1),
    'Boosting': lambda X, Y, X2, nc, rp=None: _fit_predict(
        GradientBoostingRegressor, X, Y, X2, regressor_params=rp, random_state=42),
    'Bagging': lambda X, Y, X2, nc, rp=None: _fit_predict(
        BaggingRegressor, X, Y, X2, regressor_params=rp, random_state=42),
    'AdaBoost': lambda X, Y, X2, nc, rp=None: _fit_predict(
        AdaBoostRegressor, X, Y, X2, regressor_params=rp, random_state=42),
    'LinearReg': lambda X, Y, X2, nc, rp=None: _fit_predict(
        LinearRegression, X, Y, X2, regressor_params=rp),
    'BayesRidge': lambda X, Y, X2, nc, rp=None: _fit_predict(
        BayesianRidge, X, Y, X2, regressor_params=rp, compute_score=True),
    'LassoReg': lambda X, Y, X2, nc, rp=None: _fit_predict(
        Lasso, X, Y, X2, regressor_params=rp, alpha=0.1, max_iter=10000),
    'RidgeReg': lambda X, Y, X2, nc, rp=None: _fit_predict(
        Ridge, X, Y, X2, regressor_params=rp, alpha=0.1),
    'PLS': lambda X, Y, X2, nc, rp=None: (
        (lambda X_, Y_, X2_, nc_:
            (lambda Xarr, Yarr:
                PLSRegression(
                    n_components=min(
                        nc_,
                        Xarr.shape[0],
                        Xarr.shape[1],
                        Yarr.shape[1] if Yarr.ndim > 1 else 1
                    )
                ).fit(Xarr, Yarr).predict(X2_).flatten()
            )(np.asarray(X_), np.asarray(Y_))
        )(X, Y, X2, nc)
    ),
    'PCR': lambda X, Y, X2, nc, rp=None: (
        make_pipeline(
            PCA(n_components=min(nc, np.asarray(X).shape[0], np.asarray(X).shape[1])),
            LinearRegression(),
        ).fit(X, Y).predict(X2)),
    'VotingEnsemble': lambda X, Y, X2, nc, rp=None: _voting_ensemble(X, Y, X2),
    'VotingLinear': lambda X, Y, X2, nc, rp=None: _voting_linear(X, Y, X2),
}

if LGBMRegressor is not None:
    _REGRESSORS['LGBM'] = lambda X, Y, X2, nc, rp=None: _fit_predict_lightgbm(
        X,
        Y,
        X2,
        regressor_params=rp,
    )


# =====================================================================
# Función core — búsqueda de análogos + regresión
# =====================================================================

def analog_knn_core(
    serie: np.ndarray,
    vsele: int,
    k: int = 10,
    tol: float = 0.8,
    n_components: int = 3,
    typedist: str = 'pearson',
    typereg: str = 'OLSstep',
) -> Tuple[np.ndarray, float, float, bool, List[int]]:
    """
    Pronóstico por analogía: busca k ventanas similares en el pasado y
    usa regresión sobre sus "futuros" para predecir el siguiente periodo.

    Parameters
    ----------
    serie : np.ndarray
        Serie de tiempo histórica completa.
    vsele : int
        Tamaño de ventana (ej. 24 para datos horarios = 1 día).
    k : int
        Número de vecinos más cercanos.
    tol : float
        Tolerancia de solapamiento (0–1). Vecinos cuya distancia de inicio
        sea < tol*vsele se filtran como redundantes.
    n_components : int
        Componentes para PLS / PCR.
    typedist : str
        Distancia: 'pearson', 'euclidian', 'dtw'.
    typereg : str
        Regresor: 'OLSstep', 'RF', 'Boosting', 'Bagging', 'LinearReg',
        'AdaBoost', 'BayesRidge', 'LassoReg', 'RidgeReg', 'PLS', 'PCR',
        'VotingEnsemble', 'VotingLinear'.

    Returns
    -------
    prediction : np.ndarray  — pronóstico de longitud vsele
    t_sel      : float       — tiempo de selección de vecinos (s)
    t_reg      : float       — tiempo de regresión (s)
    fail       : bool        — True si se usó persistencia como respaldo
    positions  : list[int]   — índices donde inician los k análogos
    neighbors2 : np.ndarray  — futuros de los k vecinos (k × vsele)
    """
    t0 = time.time()
    n = len(serie)

    if n < 2 * vsele + 1:
        # Serie demasiado corta → persistencia
        return (
            np.full(vsele, serie[-1]),
            0.0, 0.0, True, [],
            np.full((1, vsele), serie[-1]),
        )

    # ── PASO 1: Selección de vecinos (vectorizado) ─────────────────────
    Y = serie[n - vsele:n]
    n_windows = n - 2 * vsele

    if typedist == 'dtw':
        # DTW: mantener loop (uso raro, librería externa)
        try:
            from dtw import dtw as dtw_func
        except ImportError:
            raise ImportError("pip install dtw-python  para usar typedist='dtw'")
        distances = []
        for i in range(n_windows):
            dist = dtw_func(Y, serie[i:i + vsele]).distance
            if dist > 0:
                distances.append((i, dist))
        distances.sort(key=lambda t: t[1])

    elif typedist == 'euclidian':
        # Euclidiana vectorizada
        windows = np.lib.stride_tricks.sliding_window_view(
            serie[:n - vsele], vsele,
        )[:n_windows]
        dists = np.linalg.norm(windows - Y, axis=1)
        mask = dists > 0
        idx = np.where(mask)[0]
        vals = dists[mask]
        order = np.argsort(vals)
        distances = list(zip(idx[order].tolist(), vals[order].tolist()))

    else:  # pearson — vectorizado vía sliding window + dot product
        Y_std = np.std(Y)
        if Y_std == 0:
            distances = []
        else:
            windows = np.lib.stride_tricks.sliding_window_view(
                serie[:n - vsele], vsele,
            )[:n_windows]
            Y_c = Y - Y.mean()
            W_means = windows.mean(axis=1, keepdims=True)
            W_stds = windows.std(axis=1)
            valid = W_stds > 0
            corr_all = np.full(n_windows, np.nan)
            corr_all[valid] = (
                (windows[valid] - W_means[valid]) @ Y_c
            ) / (vsele * W_stds[valid] * Y_std)
            usable = np.isfinite(corr_all) & (corr_all > 0)
            idx = np.where(usable)[0]
            vals = corr_all[usable]
            order = np.argsort(-vals)
            distances = list(zip(idx[order].tolist(), vals[order].tolist()))

    neighbors, neighbors2, positions = [], [], []
    count = 0
    for pos, _ in distances:
        if count == 0:
            positions.append(pos)
            neighbors.append(serie[pos:pos + vsele])
            neighbors2.append(serie[pos + vsele:pos + 2 * vsele])
        else:
            overlap = any(abs(pos - p) < tol * vsele for p in positions)
            if overlap:
                count -= 1
            else:
                positions.append(pos)
                neighbors.append(serie[pos:pos + vsele])
                neighbors2.append(serie[pos + vsele:pos + 2 * vsele])
        count += 1
        if count == k:
            break

    neighbors = np.array(neighbors)
    neighbors2 = np.array(neighbors2)
    t_sel = time.time() - t0

    # ── PASO 2: Regresión ─────────────────────────────────────────────
    X = neighbors.T.tolist()
    X2 = neighbors2.T.tolist()
    Y_list = Y.tolist()

    reg_func = _REGRESSORS.get(typereg)
    if reg_func is None:
        raise ValueError(
            f"typereg='{typereg}' no reconocido. "
            f"Opciones: {sorted(_REGRESSORS.keys())}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = np.asarray(reg_func(X, Y_list, X2, n_components))

    t_reg = time.time() - t_sel - t0
    fail = False

    if len(prediction) == 0:
        prediction = np.full(vsele, serie[-1])
        fail = True

    return prediction, t_sel, t_reg, fail, positions, neighbors2


# =====================================================================
# Clase principal — compatible con StatsForecast + sklearn + standalone
# =====================================================================

class AnalogKNN:
    """
    Pronóstico por analogía (Analog Method) con selección KNN + regresión.

    Busca en el historial los ``k`` periodos más similares al presente y
    usa lo que ocurrió después de cada uno para predecir el futuro
    mediante un modelo de regresión.

    Parameters
    ----------
    season_length : int, default=24
        Tamaño de la ventana de selección (en periodos).
        Para datos horarios: 24 = 1 día completo.
    k : int, default=10
        Número de vecinos análogos.
    tol : float, default=0.8
        Tolerancia de solapamiento entre vecinos (0–1).
    n_components : int, default=3
        Componentes para regresores PLS / PCR.
    typedist : {'pearson', 'euclidian', 'dtw'}, default='pearson'
        Medida de distancia/similitud entre ventanas.
    typereg : str, default='PCR'
        Modelo de regresión:
        'OLSstep', 'RF', 'Boosting', 'Bagging', 'LinearReg', 'AdaBoost',
        'BayesRidge', 'LassoReg', 'RidgeReg', 'PLS', 'PCR',
        'VotingEnsemble', 'VotingLinear'.
    alias : str or None
        Nombre que aparece en las columnas de StatsForecast.
        Si None → 'AnalogKNN'.

    Examples
    --------
    **StatsForecast:**

    >>> from statsforecast import StatsForecast
    >>> from analog_holidays.analog import AnalogKNN
    >>> sf = StatsForecast(
    ...     models=[AnalogKNN(season_length=24, k=10, typereg='PCR')],
    ...     freq='h',
    ... )
    >>> sf.fit(df)
    >>> preds = sf.predict(h=48)

    **Standalone:**

    >>> model = AnalogKNN(season_length=24)
    >>> model.fit(y=np.array([...]))
    >>> result = model.predict(h=48)
    >>> forecast = result['mean']

    **Pipeline propio:**

    >>> model = AnalogKNN(season_length=24)
    >>> model.fit(y=serie_numpy)
    >>> forecast_24h = model.predict(h=24)['mean']
    >>> forecast_48h = model.predict(h=48)['mean']
    """

    def __init__(
        self,
        season_length: int = 24,
        k: int = 10,
        tol: float = 0.8,
        n_components: int = 3,
        typedist: str = 'pearson',
        typereg: str = 'PCR',
        alias: Optional[str] = None,
    ):
        self.season_length = season_length
        self.k = k
        self.tol = tol
        self.n_components = n_components
        self.typedist = typedist
        self.typereg = typereg
        self.alias = alias or 'AnalogKNN'

        # Estado interno tras .fit()
        self._y: Optional[np.ndarray] = None
        self._fitted: bool = False

        # ── Protocolo interno de StatsForecast ────────────────────────
        self.uses_exog: bool = False
        self._conformal_method = None
        self._conformity_scores = None
        self._store_cs = False

    # ── Nombre para StatsForecast ─────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AnalogKNN(season_length={self.season_length}, k={self.k}, "
            f"typedist='{self.typedist}', typereg='{self.typereg}')"
        )

    def new(self) -> 'AnalogKNN':
        """Devuelve una copia fresca con los mismos hiper-parámetros
        (requerido por StatsForecast para cross-validation)."""
        return AnalogKNN(**{k: v for k, v in self.get_params().items()})

    def forward(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[list] = None,
        fitted: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Alias de forecast() — requerido por StatsForecast."""
        return self.forecast(y, h, X, X_future, level, fitted)

    def predict_in_sample(self, level: Optional[list] = None) -> Dict[str, np.ndarray]:
        """Devuelve fitted values in-sample (stub: devuelve media)."""
        if self._y is None:
            return {'fitted': np.array([])}
        return {'fitted': np.full_like(self._y, np.nan)}

    # ── fit / predict / forecast ──────────────────────────────────────

    def fit(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> 'AnalogKNN':
        """
        Almacena la serie histórica para su uso posterior en predict().

        Parameters
        ----------
        y : np.ndarray
            Serie de tiempo histórica (1-D).
        X : ignored
            No se usa; presente por compatibilidad con StatsForecast.

        Returns
        -------
        self
        """
        self._y = np.asarray(y, dtype=np.float64)
        self._fitted = True
        return self

    def predict(
        self,
        h: int,
        X: Optional[np.ndarray] = None,
        level: Optional[list] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Genera pronóstico de h pasos adelante.

        Si h > season_length, realiza pronósticos encadenados (rolling):
        cada bloque de season_length pasos se predice, se anexa a la
        historia y se repite hasta cubrir h.

        Parameters
        ----------
        h : int
            Horizonte de pronóstico (número de periodos).
        X : ignored
            No se usa; presente por compatibilidad con StatsForecast.
        level : list[float] or None
            Niveles de confianza en porcentaje, ej. [80, 95].
            Cuando se indica, se calculan intervalos de predicción
            usando los cuantiles empíricos de los futuros de los k
            vecinos análogos.

        Returns
        -------
        dict con claves:
          'mean' : np.ndarray de longitud h.
          'lo-{L}', 'hi-{L}' : np.ndarray — límites inferior/superior
              para cada L en *level*.  Solo presentes si level != None.
        """
        if not self._fitted or self._y is None:
            raise RuntimeError("Llama a .fit(y) antes de .predict(h)")

        vs = self.season_length
        serie = self._y.copy()
        forecasts: list = []
        # Almacenar futuros de vecinos para intervalos
        all_neighbors2: list = []

        remaining = h
        while remaining > 0:
            block_h = min(remaining, vs)

            pred, _, _, fail, _, nb2 = analog_knn_core(
                serie=serie,
                vsele=vs,
                k=self.k,
                tol=self.tol,
                n_components=self.n_components,
                typedist=self.typedist,
                typereg=self.typereg,
            )

            forecasts.extend(pred[:block_h].tolist())
            all_neighbors2.append(nb2[:, :block_h])  # (k, block_h)
            # Extender la historia para el siguiente bloque (rolling)
            serie = np.concatenate([serie, pred[:block_h]])
            remaining -= block_h

        result = {'mean': np.array(forecasts[:h])}

        # ── Intervalos de predicción ──────────────────────────────────
        if level is not None and len(level) > 0:
            # Concatenar futuros de vecinos: (k, h)
            nb2_full = np.concatenate(all_neighbors2, axis=1)[:, :h]
            for lv in level:
                alpha = (100 - lv) / 2.0
                lo = np.percentile(nb2_full, alpha, axis=0)
                hi = np.percentile(nb2_full, 100 - alpha, axis=0)
                result[f'lo-{lv}'] = lo
                result[f'hi-{lv}'] = hi

        return result

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[list] = None,
        fitted: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Atajo fit + predict (interfaz directa de StatsForecast).

        Parameters
        ----------
        y : np.ndarray — serie histórica
        h : int        — horizonte de pronóstico
        X, X_future    — ignorados (compatibilidad)
        level          — lista de niveles de confianza, ej. [80, 95]
        fitted         — ignorado

        Returns
        -------
        dict con clave 'mean': np.ndarray de longitud h,
        y opcionalmente 'lo-{L}', 'hi-{L}' si level != None.
        """
        self.fit(y, X)
        return self.predict(h, X_future, level=level)

    # ── Métodos utilitarios ───────────────────────────────────────────

    def predict_single(
        self,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, bool, List[int]]:
        """
        Pronóstico de un solo bloque (season_length pasos), devolviendo
        también tiempos y posiciones de los análogos encontrados.

        Útil para depuración y visualización en notebooks.

        Parameters
        ----------
        y : np.ndarray — serie histórica

        Returns
        -------
        prediction : np.ndarray
        t_sel      : float (segundos)
        t_reg      : float (segundos)
        fail       : bool
        positions  : list[int]
        """
        res = analog_knn_core(
            serie=np.asarray(y, dtype=np.float64),
            vsele=self.season_length,
            k=self.k,
            tol=self.tol,
            n_components=self.n_components,
            typedist=self.typedist,
            typereg=self.typereg,
        )
        # Devolver solo los 5 primeros elementos (excluir neighbors2)
        return res[:5]

    def get_params(self) -> dict:
        """Devuelve los hiperparámetros (compatible con sklearn)."""
        return {
            'season_length': self.season_length,
            'k': self.k,
            'tol': self.tol,
            'n_components': self.n_components,
            'typedist': self.typedist,
            'typereg': self.typereg,
            'alias': self.alias,
        }

    def set_params(self, **params) -> 'AnalogKNN':
        """Establece hiperparámetros (compatible con sklearn)."""
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise ValueError(f"Parámetro desconocido: {key}")
        return self

    @staticmethod
    def available_regressors() -> list:
        """Lista de nombres de regresores disponibles."""
        return sorted(_REGRESSORS.keys())

    @staticmethod
    def available_distances() -> list:
        """Lista de distancias disponibles."""
        return ['pearson', 'euclidian', 'dtw']
