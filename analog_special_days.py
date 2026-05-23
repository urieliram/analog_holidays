"""
analog_special_days.py - Analog variant for pre-labeled special days.

Author: Uriel
Description: Experimental implementation of the analog method where
             X/X2 candidates are selected from a historical
             `special_days` mask instead of pure similarity.

Method semantics
----------------
For each special event detected in `special_days`, the following block of
length `season_length` is used as X2 and the immediately preceding block is
used as X. The target block Y remains the most recent window in the series,
exactly as in `analog.py`.

Standalone usage
----------------
    from analog_special_days import AnalogSpecialDays

    model = AnalogSpecialDays(season_length=24, typereg='PCR')
    model.fit(y=series_numpy, special_days=mask_special_days)
    forecast = model.predict(h=48)['mean']

Usage with StatsForecast
------------------------
    from statsforecast import StatsForecast
    from analog_special_days import AnalogSpecialDays

    sf = StatsForecast(
        models=[AnalogSpecialDays(season_length=24, typereg='PCR')],
        freq='h',
    )
    sf.fit(df_y, X=df_special_days)
    preds = sf.predict(h=48)
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from analog import _REGRESSORS
except ImportError:
    from train.analog import _REGRESSORS


def _coerce_special_days(values, expected_len: Optional[int] = None) -> np.ndarray:
    """Convert the special-day input into a 1-D vector."""
    if values is None:
        raise ValueError("You must provide the `special_days` series.")

    if hasattr(values, 'columns'):
        columns = list(values.columns)
        if 'special_days' in columns:
            arr = np.asarray(values['special_days'])
        elif len(columns) == 1:
            arr = np.asarray(values.iloc[:, 0])
        else:
            raise ValueError(
                "X must contain a single column or a column named "
                "'special_days'."
            )
    else:
        arr = np.asarray(values)

    if arr.ndim == 2:
        if 1 in arr.shape:
            arr = arr.reshape(-1)
        else:
            raise ValueError(
                "`special_days` must be a 1-D vector or a single-column matrix."
            )
    elif arr.ndim != 1:
        raise ValueError("`special_days` must be a 1-D vector.")

    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(
            "`special_days` and `y` must have the same length during fit()."
        )

    return arr.astype(np.float64, copy=False)


def _select_special_positions(
    special_days: np.ndarray,
    vsele: int,
    special_day_value: float,
    min_special_points: Optional[int],
    min_event_gap: Optional[int],
    max_events: Optional[int],
) -> List[int]:
    """Select historical positions whose future block is flagged as special."""
    history_len = len(special_days)
    if history_len < 2 * vsele + 1:
        return []

    flagged = np.isclose(special_days, special_day_value)
    future_windows = np.lib.stride_tricks.sliding_window_view(
        flagged[vsele:],
        vsele,
    )

    threshold = vsele if min_special_points is None else int(min_special_points)
    threshold = max(1, min(vsele, threshold))

    window_scores = future_windows.sum(axis=1)
    candidate_positions = np.where(window_scores >= threshold)[0].tolist()

    gap = vsele if min_event_gap is None else max(1, int(min_event_gap))
    positions: List[int] = []
    last_future_start: Optional[int] = None

    for pos in candidate_positions:
        future_start = pos + vsele
        if last_future_start is not None and future_start - last_future_start < gap:
            continue
        positions.append(int(pos))
        last_future_start = future_start

    if max_events is not None and len(positions) > int(max_events):
        positions = positions[-int(max_events):]

    return positions


def _normalize_typedist(typedist: str) -> str:
    """Normalize distance aliases and validate supported options."""
    normalized = str(typedist).lower()
    if normalized == 'euclidean':
        normalized = 'euclidian'

    valid = {'pearson', 'euclidian', 'dtw'}
    if normalized not in valid:
        raise ValueError(
            f"typedist='{typedist}' is not recognized. Options: {sorted(valid)}"
        )
    return normalized


def _select_k_similar_positions(
    serie: np.ndarray,
    positions: List[int],
    target_window: np.ndarray,
    vsele: int,
    k: Optional[int],
    typedist: str = 'pearson',
    dtw_window: Optional[float] = None,
) -> List[int]:
    """Keep the k special candidates whose X blocks are most similar to Y."""
    if k is None or len(positions) <= int(k):
        return positions

    limit = max(0, int(k))
    if limit == 0:
        return []

    candidate_positions = [int(pos) for pos in positions]
    windows = np.array(
        [serie[pos:pos + vsele] for pos in candidate_positions],
        dtype=np.float64,
    )
    target_window = np.asarray(target_window, dtype=np.float64)
    typedist = _normalize_typedist(typedist)

    distances = np.linalg.norm(windows - target_window, axis=1)

    if typedist == 'dtw':
        try:
            from dtw import dtw as dtw_func
        except ImportError as exc:
            raise ImportError("Install dtw-python to use typedist='dtw'.") from exc

        _dtw_win_type = "none"
        _dtw_win_args: dict = {}
        if dtw_window is not None and 0.0 < dtw_window < 1.0:
            _dtw_win_type = "sakoechiba"
            _dtw_win_args = {"window_size": max(1, int(round(dtw_window * vsele)))}
        dtw_distances = np.array(
            [
                dtw_func(
                    target_window,
                    window,
                    window_type=_dtw_win_type,
                    window_args=_dtw_win_args,
                ).distance
                for window in windows
            ],
            dtype=np.float64,
        )
        ranked_indices = sorted(
            range(len(candidate_positions)),
            key=lambda idx: (
                float(dtw_distances[idx]),
                float(distances[idx]),
                -candidate_positions[idx],
            ),
        )
        return [candidate_positions[idx] for idx in ranked_indices[:limit]]

    if typedist == 'euclidian':
        ranked_indices = sorted(
            range(len(candidate_positions)),
            key=lambda idx: (
                float(distances[idx]),
                -candidate_positions[idx],
            ),
        )
        return [candidate_positions[idx] for idx in ranked_indices[:limit]]

    target_std = np.std(target_window)
    correlations = np.full(len(candidate_positions), np.nan, dtype=np.float64)
    if target_std > 0 and len(windows) > 0:
        target_centered = target_window - target_window.mean()
        window_stds = windows.std(axis=1)
        valid = window_stds > 0
        if np.any(valid):
            centered_windows = windows[valid] - windows[valid].mean(axis=1, keepdims=True)
            correlations[valid] = (
                centered_windows @ target_centered
            ) / (vsele * window_stds[valid] * target_std)

    ranked_indices = sorted(
        range(len(candidate_positions)),
        key=lambda idx: (
            0 if np.isfinite(correlations[idx]) else 1,
            -float(correlations[idx]) if np.isfinite(correlations[idx]) else float(distances[idx]),
            float(distances[idx]),
            -candidate_positions[idx],
        ),
    )
    return [candidate_positions[idx] for idx in ranked_indices[:limit]]


def analog_special_days_core(
    serie: np.ndarray,
    special_days: np.ndarray,
    vsele: int,
    k: Optional[int] = None,
    typedist: str = 'pearson',
    n_components: int = 3,
    typereg: str = 'PCR',
    special_day_value: float = 1.0,
    min_special_points: Optional[int] = None,
    min_event_gap: Optional[int] = None,
    max_events: Optional[int] = None,
    dtw_window: Optional[float] = None,
) -> Tuple[np.ndarray, float, float, bool, List[int], np.ndarray]:
    """
    Analog forecast using preselected special days.

    Parameters
    ----------
    serie : np.ndarray
        Full historical series. It may include already forecast blocks
        when rolling over multiple horizons.
    special_days : np.ndarray
        Binary historical mask. A future X2 block is considered a candidate
        when it contains at least `min_special_points` values equal to
        `special_day_value`. If `min_special_points=None`, the full block of
        length `vsele` must be marked.
    vsele : int
        Window length for X, Y, and X2.
    k : int or None
        Number of special neighbors to keep after ranking similarity
        between X and Y. If None, use all filtered candidates.
    typedist : str
        Distance or similarity used to rank special candidates: 'pearson',
        'euclidian', or 'dtw'.
    n_components : int
        Number of components for PLS or PCR.
    typereg : str
        Regressor to use.
    special_day_value : float
        Value that marks a special period.
    min_special_points : int or None
        Minimum number of marked points inside X2.
    min_event_gap : int or None
        Minimum separation between consecutive special-event starts.
    max_events : int or None
        Maximum number of special events to use. If None, use all.

    Returns
    -------
    prediction : np.ndarray
    t_sel      : float
    t_reg      : float
    fail       : bool
    positions  : list[int]
    neighbors2 : np.ndarray
    """
    t0 = time.time()
    serie = np.asarray(serie, dtype=np.float64)
    special_days = _coerce_special_days(special_days)
    typedist = _normalize_typedist(typedist)

    n = len(serie)
    history_len = len(special_days)

    if history_len > n:
        raise ValueError(
            "`special_days` cannot be longer than the series used in predict()."
        )

    if n < vsele or history_len < 2 * vsele + 1:
        return (
            np.full(vsele, serie[-1]),
            0.0,
            0.0,
            True,
            [],
            np.full((1, vsele), serie[-1]),
        )

    Y = serie[n - vsele:n]
    history_serie = serie[:history_len]
    candidate_positions = _select_special_positions(
        special_days=special_days,
        vsele=vsele,
        special_day_value=special_day_value,
        min_special_points=min_special_points,
        min_event_gap=min_event_gap,
        max_events=max_events,
    )
    positions = _select_k_similar_positions(
        serie=history_serie,
        positions=candidate_positions,
        target_window=Y,
        vsele=vsele,
        k=k,
        typedist=typedist,
        dtw_window=dtw_window,
    )

    neighbors = np.array(
        [history_serie[pos:pos + vsele] for pos in positions],
        dtype=np.float64,
    )
    neighbors2 = np.array(
        [history_serie[pos + vsele:pos + 2 * vsele] for pos in positions],
        dtype=np.float64,
    )
    t_sel = time.time() - t0

    if len(neighbors) == 0:
        return (
            np.full(vsele, serie[-1]),
            t_sel,
            0.0,
            True,
            [],
            np.full((1, vsele), serie[-1]),
        )

    reg_func = _REGRESSORS.get(typereg)
    if reg_func is None:
        raise ValueError(
            f"typereg='{typereg}' is not recognized. "
            f"Options: {sorted(_REGRESSORS.keys())}"
        )

    X = neighbors.T.tolist()
    X2 = neighbors2.T.tolist()
    Y_list = Y.tolist()

    fail = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prediction = np.asarray(reg_func(X, Y_list, X2, n_components)).reshape(-1)
    except Exception:
        prediction = np.full(vsele, serie[-1])
        fail = True

    t_reg = time.time() - t0 - t_sel

    if len(prediction) == 0 or prediction.shape[0] != vsele:
        prediction = np.full(vsele, serie[-1])
        fail = True

    return prediction, t_sel, t_reg, fail, positions, neighbors2


def _build_pairwise_interval_scenarios(
    serie: np.ndarray,
    history_len: int,
    positions: List[int],
    neighbors2: np.ndarray,
    vsele: int,
    typereg: str,
    n_components: int,
) -> np.ndarray:
    """Build per-analog bivariate scenarios for interval estimation.

    The point forecast remains the original multivariate analog regression.
    Confidence intervals are instead derived from k pairwise regressions,
    mirroring the scenario construction used by RiskAnalog.
    """
    if len(positions) == 0 or len(neighbors2) == 0:
        return np.empty((0, vsele), dtype=np.float64)

    reg_func = _REGRESSORS.get(typereg)
    if reg_func is None:
        raise ValueError(
            f"typereg='{typereg}' is not recognized. "
            f"Options: {sorted(_REGRESSORS.keys())}"
        )

    history_serie = np.asarray(serie[:history_len], dtype=np.float64)
    target_window = np.asarray(serie[-vsele:], dtype=np.float64)
    scenarios: list[np.ndarray] = []

    for idx, pos in enumerate(positions[: len(neighbors2)]):
        x_hist = history_serie[pos:pos + vsele]
        x_future = np.asarray(neighbors2[idx], dtype=np.float64)

        if len(x_hist) != vsele or len(x_future) == 0:
            continue

        x_train = x_hist.reshape(-1, 1)
        x_pred = x_future.reshape(-1, 1)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                scenario = np.asarray(
                    reg_func(x_train, target_window, x_pred, n_components)
                ).reshape(-1)
        except Exception:
            scenario = x_future.copy()

        if scenario.shape[0] != x_future.shape[0] or not np.isfinite(scenario).all():
            scenario = x_future.copy()

        scenarios.append(scenario)

    if not scenarios:
        return np.asarray(neighbors2, dtype=np.float64)

    return np.asarray(scenarios, dtype=np.float64)


class AnalogSpecialDays:
    """
    Analog method variant that uses pre-labeled special events.

    X/X2 candidates are built from the `special_days` mask instead of
    correlation- or distance-based similarity. The target window Y and the
    rolling multi-horizon flow follow the `AnalogKNN` logic.
    """

    def __init__(
        self,
        season_length: int = 24,
        k: Optional[int] = None,
        typedist: str = 'pearson',
        n_components: int = 3,
        typereg: str = 'PCR',
        special_day_value: float = 1.0,
        min_special_points: Optional[int] = None,
        min_event_gap: Optional[int] = None,
        max_events: Optional[int] = None,
        alias: Optional[str] = None,
        dtw_window: Optional[float] = None,
    ):
        self.season_length = season_length
        self.k = k
        self.typedist = _normalize_typedist(typedist)
        self.n_components = n_components
        self.typereg = typereg
        self.special_day_value = special_day_value
        self.min_special_points = min_special_points
        self.min_event_gap = min_event_gap
        self.max_events = max_events
        self.alias = alias or 'AnalogSpecialDays'
        self.dtw_window = dtw_window

        self._y: Optional[np.ndarray] = None
        self._special_days: Optional[np.ndarray] = None
        self._fitted: bool = False

        self.uses_exog: bool = True
        self._conformal_method = None
        self._conformity_scores = None
        self._store_cs = False

    def __repr__(self) -> str:
        return (
            f"AnalogSpecialDays(season_length={self.season_length}, "
            f"k={self.k}, typedist='{self.typedist}', typereg='{self.typereg}', "
            f"max_events={self.max_events})"
        )

    def new(self) -> 'AnalogSpecialDays':
        return AnalogSpecialDays(**{k: v for k, v in self.get_params().items()})

    def forward(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[list] = None,
        fitted: bool = False,
    ) -> Dict[str, np.ndarray]:
        return self.forecast(y, h, X, X_future, level, fitted)

    def predict_in_sample(self, level: Optional[list] = None) -> Dict[str, np.ndarray]:
        if self._y is None:
            return {'fitted': np.array([])}
        return {'fitted': np.full_like(self._y, np.nan)}

    def fit(
        self,
        y: np.ndarray,
        X: Optional[np.ndarray] = None,
        special_days: Optional[np.ndarray] = None,
    ) -> 'AnalogSpecialDays':
        self._y = np.asarray(y, dtype=np.float64)
        mask_source = special_days if special_days is not None else X
        self._special_days = _coerce_special_days(mask_source, expected_len=len(self._y))
        self._fitted = True
        return self

    def predict(
        self,
        h: int,
        X: Optional[np.ndarray] = None,
        level: Optional[list] = None,
    ) -> Dict[str, np.ndarray]:
        if not self._fitted or self._y is None or self._special_days is None:
            raise RuntimeError("Call .fit(y, special_days=...) before .predict(h)")

        vs = self.season_length
        serie = self._y.copy()
        forecasts: list = []
        interval_scenarios_by_block: list = []
        history_len = len(self._special_days)

        remaining = h
        while remaining > 0:
            block_h = min(remaining, vs)

            pred, _, _, _, positions, nb2 = analog_special_days_core(
                serie=serie,
                special_days=self._special_days,
                vsele=vs,
                k=self.k,
                typedist=self.typedist,
                n_components=self.n_components,
                typereg=self.typereg,
                special_day_value=self.special_day_value,
                min_special_points=self.min_special_points,
                min_event_gap=self.min_event_gap,
                max_events=self.max_events,
                dtw_window=self.dtw_window,
            )

            block_interval_scenarios = _build_pairwise_interval_scenarios(
                serie=serie,
                history_len=history_len,
                positions=positions,
                neighbors2=nb2,
                vsele=vs,
                typereg=self.typereg,
                n_components=self.n_components,
            )

            forecasts.extend(pred[:block_h].tolist())
            interval_scenarios_by_block.append(block_interval_scenarios[:, :block_h])
            serie = np.concatenate([serie, pred[:block_h]])
            remaining -= block_h

        result = {'mean': np.array(forecasts[:h])}

        if level is not None and len(level) > 0 and len(interval_scenarios_by_block) > 0:
            min_k = min(arr.shape[0] for arr in interval_scenarios_by_block)
            if min_k > 0:
                scenarios_full = np.concatenate(
                    [arr[:min_k] for arr in interval_scenarios_by_block],
                    axis=1,
                )[:, :h]
                for lv in level:
                    alpha = (100 - lv) / 2.0
                    result[f'lo-{lv}'] = np.percentile(scenarios_full, alpha, axis=0)
                    result[f'hi-{lv}'] = np.percentile(scenarios_full, 100 - alpha, axis=0)

        return result

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        X: Optional[np.ndarray] = None,
        X_future: Optional[np.ndarray] = None,
        level: Optional[list] = None,
        fitted: bool = False,
        special_days: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        mask_source = special_days if special_days is not None else X
        self.fit(y, mask_source, special_days=special_days)
        return self.predict(h, X_future, level=level)

    def predict_single(
        self,
        y: np.ndarray,
        special_days: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float, bool, List[int]]:
        mask_source = special_days if special_days is not None else self._special_days
        res = analog_special_days_core(
            serie=np.asarray(y, dtype=np.float64),
            special_days=_coerce_special_days(mask_source),
            vsele=self.season_length,
            k=self.k,
            typedist=self.typedist,
            n_components=self.n_components,
            typereg=self.typereg,
            special_day_value=self.special_day_value,
            min_special_points=self.min_special_points,
            min_event_gap=self.min_event_gap,
            max_events=self.max_events,
            dtw_window=self.dtw_window,
        )
        return res[:5]

    def get_params(self) -> dict:
        return {
            'season_length': self.season_length,
            'k': self.k,
            'typedist': self.typedist,
            'n_components': self.n_components,
            'typereg': self.typereg,
            'special_day_value': self.special_day_value,
            'min_special_points': self.min_special_points,
            'min_event_gap': self.min_event_gap,
            'max_events': self.max_events,
            'alias': self.alias,
            'dtw_window': self.dtw_window,
        }

    def set_params(self, **params) -> 'AnalogSpecialDays':
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    @staticmethod
    def available_regressors() -> list:
        return sorted(_REGRESSORS.keys())

    @staticmethod
    def available_distances() -> list:
        return ['pearson', 'euclidian', 'dtw']