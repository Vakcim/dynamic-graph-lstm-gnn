"""Masked forecast metrics in original units, with streaming prediction records."""

from dataclasses import dataclass

import numpy as np


PREDICTION_COLUMNS = (
    "origin_time", "target_time", "node_id", "node", "observed",
    "target_feature", "prediction_feature", "target_raw", "prediction_raw",
)


def _matrix(value, name, n_nodes, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != n_nodes:
        raise ValueError(f"{name} must have shape [batch, {n_nodes}]")
    return array


def _times(values, batch_size, name):
    if values is None:
        return [None] * batch_size
    if isinstance(values, (str, bytes)) or np.isscalar(values):
        values = [values]
    result = list(values)
    if len(result) != batch_size:
        raise ValueError(f"{name} must have one entry per forecast row")
    return [str(value) if value is not None else None for value in result]


@dataclass(frozen=True)
class MetricBatch:
    """Validated predictions and untouched targets for one evaluation batch."""

    prediction_feature: np.ndarray
    prediction_raw: np.ndarray
    target_feature: np.ndarray
    target_raw: np.ndarray
    mask: np.ndarray
    origin_times: tuple
    target_times: tuple


class ForecastMetrics:
    """Accumulate errors over observed node-time pairs without storing predictions.

    Predictions must already be inverse-normalized into feature units. For
    MOEX, these units are log(RV), and raw predictions are exp(log_prediction).
    For synthetic data and METR-LA, feature and raw prediction scales coincide.
    Missing targets are never imputed. Nonfinite model outputs are errors even
    at masked positions, rather than being silently excluded from evaluation.
    """

    def __init__(self, dataset, node_order):
        self.dataset = str(dataset).lower()
        if self.dataset not in {"synthetic", "metr_la", "moex"}:
            raise ValueError("dataset must be synthetic, metr_la or moex")
        self.node_order = tuple(map(str, node_order))
        if not self.node_order or len(set(self.node_order)) != len(self.node_order):
            raise ValueError("node_order must contain unique, nonempty node names")
        n = len(self.node_order)
        self.count = np.zeros(n, dtype=np.int64)
        self.absolute_sum = np.zeros(n, dtype=np.float64)
        self.squared_sum = np.zeros(n, dtype=np.float64)
        self.qlike_sum = np.zeros(n, dtype=np.float64) if self.dataset == "moex" else None
        self.forecast_rows = 0
        self.observed_rows = 0
        self._seen_target_times = set()
        self._all_target_times_given = True

    def update(self, pred_feature, y_feature, y_raw, y_mask, *,
               origin_times=None, target_times=None):
        """Update once per batch and return records-ready validated arrays.

        If supplied, target times must be unique across this accumulator.
        A fresh accumulator should be used for each dataset, split and seed.
        """
        n = len(self.node_order)
        prediction = _matrix(pred_feature, "pred_feature", n)
        feature = _matrix(y_feature, "y_feature", n)
        raw = _matrix(y_raw, "y_raw", n)
        original_mask = np.asarray(y_mask)
        if original_mask.dtype.kind != "b" and not np.isin(original_mask, [0, 1]).all():
            raise ValueError("y_mask must contain boolean values or zeros and ones")
        mask = _matrix(original_mask, "y_mask", n, dtype=bool)
        if prediction.shape != feature.shape or prediction.shape != raw.shape or prediction.shape != mask.shape:
            raise ValueError("Predictions, targets and mask must have identical shapes")
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Model produced a nonfinite feature-space prediction")
        if np.any(mask & (~np.isfinite(feature) | ~np.isfinite(raw))):
            raise ValueError("An observed target is nonfinite; fix the target mask")
        origins = _times(origin_times, len(prediction), "origin_times")
        targets = _times(target_times, len(prediction), "target_times")
        supplied = [value for value in targets if value is not None]
        if len(set(supplied)) != len(supplied) or self._seen_target_times.intersection(supplied):
            raise ValueError("A target time would be evaluated more than once")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                prediction_raw = np.exp(prediction) if self.dataset == "moex" else prediction.copy()
                if not np.isfinite(prediction_raw).all():
                    raise FloatingPointError("Model produced a nonfinite raw prediction")
                if self.dataset == "moex":
                    if np.any(prediction_raw <= 0):
                        raise FloatingPointError("exp(log-RV prediction) underflowed to zero")
                    if np.any(raw[mask] < 0):
                        raise ValueError("Observed realized variance must be nonnegative")
                errors = np.zeros_like(prediction_raw)
                np.subtract(prediction_raw, raw, out=errors, where=mask)
                absolute = np.abs(errors).sum(axis=0, dtype=np.float64)
                squared = np.square(errors).sum(axis=0, dtype=np.float64)
                qlike = None
                if self.dataset == "moex":
                    losses = np.zeros_like(prediction_raw)
                    # This form supports a zero realized-variance proxy without log(y).
                    losses[mask] = prediction[mask] + raw[mask] / prediction_raw[mask]
                    qlike = losses.sum(axis=0, dtype=np.float64)
        except FloatingPointError as exc:
            raise FloatingPointError(
                "Forecast metrics overflowed or received an invalid variance prediction; "
                "predictions were not clipped or silently dropped"
            ) from exc
        if not (np.isfinite(absolute).all() and np.isfinite(squared).all() and
                (qlike is None or np.isfinite(qlike).all())):
            raise FloatingPointError("A batch metric is nonfinite")
        new_absolute = self.absolute_sum + absolute
        new_squared = self.squared_sum + squared
        new_qlike = self.qlike_sum + qlike if qlike is not None else None
        if not (np.isfinite(new_absolute).all() and np.isfinite(new_squared).all() and
                (new_qlike is None or np.isfinite(new_qlike).all())):
            raise FloatingPointError("Accumulated forecast metrics overflowed")
        self.absolute_sum, self.squared_sum = new_absolute, new_squared
        if new_qlike is not None:
            self.qlike_sum = new_qlike
        self.count += mask.sum(axis=0, dtype=np.int64)
        self.forecast_rows += len(prediction)
        self.observed_rows += int(mask.any(axis=1).sum())
        self._seen_target_times.update(supplied)
        self._all_target_times_given &= len(supplied) == len(prediction)
        return MetricBatch(prediction, prediction_raw, feature, raw, mask,
                           tuple(origins), tuple(targets))

    def per_node(self):
        """Return node-level metrics, retaining nodes with no evaluated targets."""
        records = []
        for i, node in enumerate(self.node_order):
            count = int(self.count[i])
            record = {"node_id": i, "node": node, "name": node,
                      "observed_pairs": count, "n_observed": count,
                      "mae": float(self.absolute_sum[i] / count) if count else None,
                      "rmse": float(np.sqrt(self.squared_sum[i] / count)) if count else None}
            if self.qlike_sum is not None:
                record["qlike"] = float(self.qlike_sum[i] / count) if count else None
            records.append(record)
        return records

    def summary(self):
        """Return pooled metrics and unweighted means of observed node metrics.

        QLIKE is log(v_hat) + RV / v_hat, without target-only constants. Its
        absolute value may be negative and is comparable only on matched data.
        Macro RMSE means mean(node RMSE), distinct from pooled RMSE.
        """
        total = int(self.count.sum())
        nodes = self.per_node()
        observed = [record for record in nodes if record["observed_pairs"]]
        result = {
            "dataset": self.dataset,
            "forecast_rows": self.forecast_rows,
            "observed_rows": self.observed_rows,
            "unique_target_times": len(self._seen_target_times) if self._all_target_times_given else None,
            "observed_pairs": total,
            "n_observed": total,
            "n_origins": self.forecast_rows,
            "nodes": len(nodes),
            "nodes_with_targets": len(observed),
            "mae": float(self.absolute_sum.sum() / total) if total else None,
            "rmse": float(np.sqrt(self.squared_sum.sum() / total)) if total else None,
            "macro_mae": float(np.mean([record["mae"] for record in observed])) if observed else None,
            "macro_rmse": float(np.mean([record["rmse"] for record in observed])) if observed else None,
        }
        if self.qlike_sum is not None:
            result.update(qlike=float(self.qlike_sum.sum() / total) if total else None,
                          macro_qlike=float(np.mean([record["qlike"] for record in observed])) if observed else None)
        return result


def prediction_rows(batch, node_order):
    """Yield CSV-ready dictionaries; masked targets become blank, never zero.

    Use csv.DictWriter(..., fieldnames=PREDICTION_COLUMNS).writerows(...)
    once per batch instead of accumulating all METR-LA predictions in memory.
    """
    nodes = tuple(map(str, node_order))
    if batch.prediction_feature.shape[1] != len(nodes):
        raise ValueError("node_order does not match the metric batch")
    for row in range(len(batch.prediction_feature)):
        for node_id, node in enumerate(nodes):
            observed = bool(batch.mask[row, node_id])
            yield {
                "origin_time": batch.origin_times[row],
                "target_time": batch.target_times[row],
                "node_id": node_id, "node": node, "observed": int(observed),
                "target_feature": float(batch.target_feature[row, node_id]) if observed else None,
                "prediction_feature": float(batch.prediction_feature[row, node_id]),
                "target_raw": float(batch.target_raw[row, node_id]) if observed else None,
                "prediction_raw": float(batch.prediction_raw[row, node_id]),
            }
