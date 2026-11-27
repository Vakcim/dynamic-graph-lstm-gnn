"""Shared preprocessing and CSV loading for synthetic data, METR-LA and MOEX.

Import this module from the project src directory. Importing performs no I/O.
CSV formats, chronological partitions and missing-value policies match the
original Prepare_datasets notebook.
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import prepare_moex as moex_worker

UNIFIED_CSV_VERSION = "unified-graph-series-1.2"


SPLITS = ("train", "val", "test")


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _json_value(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _frame_hash(frame):
    hasher = hashlib.sha256()
    hasher.update(json.dumps(list(map(str, frame.columns))).encode())
    hasher.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return hasher.hexdigest()


def build_sample_index(values, raw_values, segments, splits, history=60):
    """Keep calendar positions; never advance a target past a missing observation."""
    if history < 1:
        raise ValueError("history must be positive")
    n = len(values)
    target_observed = (np.isfinite(values.to_numpy(dtype=float)) &
                       np.isfinite(raw_values.to_numpy(dtype=float)))
    good_y = target_observed.any(axis=1)
    segment_values = np.asarray(segments)
    breaks = np.r_[0, np.cumsum(segment_values[1:] != segment_values[:-1])]
    records = []
    for target in range(1, n):
        left = target - history
        if left < 0:
            reason = "insufficient_history"
        elif breaks[target] != breaks[left]:
            reason = "segment_boundary_in_window"
        elif not good_y[target]:
            reason = "all_targets_missing"
        else:
            reason = ""
        records.append({"origin_row": target - 1, "target_row": target,
                        "origin_time": values.index[target - 1], "target_time": values.index[target],
                        "split": str(splits[target]), "observed_target_nodes": int(target_observed[target].sum()),
                        "kept": not reason, "reason": reason})
    return pd.DataFrame(records, columns=["origin_row", "target_row", "origin_time", "target_time", "split", "observed_target_nodes", "kept", "reason"])


def export_dataset(values, raw_values, segments, out, history=60,
                   train_fraction=.7, val_fraction=.15, train_end=None, val_end=None, metadata=None):
    """Write one scalar per node per time; preserve missing rows and split by target time.

    Date boundaries are inclusive. Without dates, chronological row fractions
    define disjoint partitions. The CSV partitions contain no copied context;
    get_window reads preceding partitions when constructing validation/test X.
    """
    out = Path(out).resolve()
    values, raw_values = values.copy(), raw_values.copy()
    if not isinstance(values, pd.DataFrame) or not isinstance(raw_values, pd.DataFrame):
        raise TypeError("values and raw_values must be DataFrames")
    if len(values) < 3 or not len(values.columns):
        raise ValueError("At least three time rows and one node are required")
    if not values.index.equals(raw_values.index) or not values.columns.equals(raw_values.columns):
        raise ValueError("Raw and transformed values must have identical rows and node order")
    if values.index.has_duplicates or values.index.hasnans or not values.index.is_monotonic_increasing:
        raise ValueError("Time index must be unique, increasing and nonmissing")
    values.columns = values.columns.map(str)
    raw_values.columns = raw_values.columns.map(str)
    if values.columns.has_duplicates or "time" in values.columns:
        raise ValueError("Node names must be unique and different from time")
    values = values.astype(float).replace([np.inf, -np.inf], np.nan)
    raw_values = raw_values.astype(float).replace([np.inf, -np.inf], np.nan)
    if isinstance(segments, pd.Series):
        if not segments.index.equals(values.index):
            raise ValueError("Segment index must match the time index")
        segments = segments.to_numpy()
    segments = np.asarray(segments)
    if len(segments) != len(values) or pd.isna(segments).any():
        raise ValueError("Each time row needs one nonmissing segment identifier")
    if history < 1:
        raise ValueError("history must be positive")
    if (train_end is None) != (val_end is None):
        raise ValueError("Specify both date boundaries or neither")
    if train_end is not None:
        if not isinstance(values.index, pd.DatetimeIndex):
            raise TypeError("Date boundaries require a DatetimeIndex")
        train_cut, val_cut = pd.Timestamp(train_end), pd.Timestamp(val_end)
        if train_cut >= val_cut:
            raise ValueError("train_end must precede val_end")
        labels = np.where(values.index <= train_cut, "train", np.where(values.index <= val_cut, "val", "test"))
    else:
        if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
            raise ValueError("Chronological fractions must be positive and sum to less than one")
        train_stop = int(len(values) * train_fraction)
        val_stop = train_stop + int(len(values) * val_fraction)
        labels = np.array(["train"] * train_stop + ["val"] * (val_stop - train_stop) + ["test"] * (len(values) - val_stop))
    if set(labels) != set(SPLITS):
        raise ValueError("Each chronological split must contain at least one time row")
    segment_frame = pd.DataFrame({"segment_id": segments}, index=values.index)
    data_hash = _sha_bytes(( _frame_hash(values) + _frame_hash(raw_values) + _frame_hash(segment_frame)).encode())
    train_means = values.loc[labels == "train"].mean(axis=0)
    if not np.isfinite(train_means.to_numpy()).all():
        raise ValueError("Every node needs at least one finite training observation for input fallback")
    settings = _json_value({"history": int(history), "horizon": 1, "train_fraction": train_fraction,
                           "val_fraction": val_fraction, "train_end": train_end, "val_end": val_end,
                           "input_missing_policy": "window_forward_fill_then_train_mean",
                           "target_missing_policy": "never_imputed_use_y_mask",
                           "metadata": metadata or {}})
    if out.exists() and any(out.iterdir()):
        manifest_path = out / "manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError("Nonempty output lacks manifest.json; choose a new output directory")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("version") != UNIFIED_CSV_VERSION or saved.get("source_data_sha256") != data_hash or saved.get("settings") != settings:
            raise FileExistsError("Existing output has different data/settings; choose a new output directory")
        for name, expected in saved.get("file_sha256", {}).items():
            path = out / name
            if not path.is_file() or _sha_bytes(path.read_bytes()) != expected:
                raise RuntimeError("Existing output checksum mismatch: " + str(path))
        return saved
    out.mkdir(parents=True, exist_ok=True)
    calendar = pd.DataFrame({"time": values.index, "segment_id": segments, "split": labels})
    samples = build_sample_index(values, raw_values, segments, labels, history)
    schema = {"node_order": list(values.columns), "history": int(history), "horizon": 1,
              "n_nodes": len(values.columns), "n_features": 1, "x_shape": [int(history), len(values.columns), 1],
              "time_kind": "datetime" if isinstance(values.index, pd.DatetimeIndex) else "integer" if pd.api.types.is_integer_dtype(values.index.dtype) else "string",
              "series_columns": ["time", *list(values.columns)],
              "target_rule": "next row of the common time/calendar index, within one segment",
              "input_missing_policy": "forward fill only inside the selected history window, then per-node mean fitted only on train; return original X_mask separately",
              "target_missing_policy": "never imputed; y_mask selects finite transformed and raw targets",
              "normalization": "none; fit any scaler only on train"}
    def write(frame, path, index=False):
        frame.to_csv(path, index=index, index_label="time" if index else None,
                     float_format="%.17g", date_format="%Y-%m-%d %H:%M:%S")
    write(calendar, out / "calendar.csv")
    write(samples, out / "samples.csv")
    counts, rows = {}, {}
    for split in SPLITS:
        folder = out / split
        folder.mkdir()
        selected = labels == split
        write(values.loc[selected], folder / "series.csv", index=True)
        write(raw_values.loc[selected], folder / "raw_values.csv", index=True)
        write(values.loc[selected].notna().astype(np.uint8), folder / "observed_mask.csv", index=True)
        counts[split] = int((samples["split"].eq(split) & samples["kept"]).sum())
        rows[split] = int(selected.sum())
    (out / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "train_input_mean.json").write_text(json.dumps(train_means.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"version": UNIFIED_CSV_VERSION, "settings": settings, "source_data_sha256": data_hash,
                "node_order": list(values.columns), "rows": rows, "usable_samples": counts,
                "ready_for_training": all(counts.values()), "pandas": pd.__version__, "numpy": np.__version__,
                "file_sha256": {str(path.relative_to(out)): _sha_bytes(path.read_bytes()) for path in sorted(out.rglob("*")) if path.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def get_window(dataset_root, split, sample_index=0):
    """Read one usable sample, including lawful past context from earlier splits."""
    if split not in SPLITS:
        raise ValueError("split must be train, val or test")
    root = Path(dataset_root)
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    node_order = schema["node_order"]
    sample_table = pd.read_csv(root / "samples.csv")
    kept = sample_table.loc[sample_table["split"].eq(split) & sample_table["kept"].eq(True)].reset_index(drop=True)
    if sample_index < 0 or sample_index >= len(kept):
        raise IndexError("No usable sample at this split/index")
    selected = kept.iloc[int(sample_index)]
    target, history = int(selected["target_row"]), int(schema["history"])
    values = pd.concat([pd.read_csv(root / part / "series.csv", float_precision="round_trip") for part in SPLITS], ignore_index=True)
    raw = pd.concat([pd.read_csv(root / part / "raw_values.csv", float_precision="round_trip") for part in SPLITS], ignore_index=True)
    times = values["time"]
    if schema["time_kind"] == "datetime":
        times = pd.to_datetime(times)
    calendar = pd.read_csv(root / "calendar.csv")
    if len(calendar) != len(values) or not values["time"].astype(str).equals(raw["time"].astype(str)):
        raise RuntimeError("Saved raw and transformed CSV rows do not align")
    frame = values[node_order]
    means = pd.Series(json.loads((root / "train_input_mean.json").read_text(encoding="utf-8"))).reindex(node_order)
    segment_ids = calendar["segment_id"].to_numpy()
    if len(set(segment_ids[target - history:target + 1])) != 1:
        raise RuntimeError("The saved sample crosses a segment boundary")
    window = frame.iloc[target - history:target]
    filled = window.ffill().fillna(means)
    x = filled.to_numpy(dtype=np.float32).reshape(history, len(node_order), 1)
    x_mask = frame.iloc[target - history:target].notna().to_numpy().reshape(history, len(node_order), 1)
    y = frame.iloc[target].to_numpy(dtype=np.float32)
    y_raw = raw.iloc[target][node_order].to_numpy(dtype=np.float64)
    y_mask = np.isfinite(y) & np.isfinite(y_raw)
    if not np.isfinite(x).all() or not y_mask.any():
        raise RuntimeError("A saved usable window has invalid filled inputs or no observed target")
    return {"X": x, "X_mask": x_mask, "y": y, "y_raw": y_raw,
            "y_mask": y_mask, "y_for_loss": np.where(y_mask, y, 0).astype(np.float32), "node_order": node_order,
            "history_times": times.iloc[target - history:target].to_list(),
            "origin_time": times.iloc[target - 1], "target_time": times.iloc[target], "split": split}


def _read_legacy_metr_la(path):
    """Read the known single-block METR-LA layout without pandas attributes decoding."""
    import tables

    def text(value):
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    with tables.open_file(path, mode="r") as handle:
        try:
            group = handle.get_node("/df")
            axes = [handle.get_node(group, name) for name in (
                "axis0", "axis1", "block0_items", "block0_values")]
        except tables.NoSuchNodeError as exc:
            raise ValueError("Unsupported legacy METR-LA HDF5 layout") from exc
        if (text(group._v_attrs.pandas_type) != "frame" or
                int(group._v_attrs.ndim) != 2 or int(group._v_attrs.nblocks) != 1):
            raise ValueError("Expected a single-block pandas fixed DataFrame")
        if not all(isinstance(node, tables.Array) for node in axes):
            raise ValueError("Expected plain arrays in the legacy METR-LA file")
        column_node, time_node, item_node, value_node = axes
        if text(time_node._v_attrs.kind) != "datetime64":
            raise ValueError("Expected nanosecond datetime64 timestamps")
        columns, times = column_node.read(), time_node.read()
        items, values = item_node.read(), value_node.read()
        if columns.ndim != 1 or times.ndim != 1 or not np.array_equal(items, columns):
            raise ValueError("Legacy METR-LA axes and block items do not match")
        if not np.issubdtype(times.dtype, np.integer):
            raise ValueError("Expected integer nanosecond timestamps")
        if values.shape != (len(times), len(columns)) or values.dtype.kind not in "iuf":
            raise ValueError("Legacy METR-LA values do not match the expected numeric layout")
        # The actual array already has time-by-sensor order; do not transpose it.
        # Rebuild timestamps instead of reading the old Python 2 frequency attribute.
        return pd.DataFrame(values, index=pd.to_datetime(times, unit="ns"),
                            columns=[text(value) for value in columns])


def _read_metr_la_frame(path):
    """Use pandas normally, with a narrow fallback for byte-valued legacy metadata."""
    try:
        return pd.read_hdf(path)
    except TypeError as exc:
        if "a bytes-like object is required" not in str(exc):
            raise
        return _read_legacy_metr_la(path)


def load_metr_la(h5_path):
    """Return physical-unit speeds and a reproducible data-quality audit.

    The original DCRNN convention treats zero speeds as missing for evaluation.
    Here nonpositive and nonfinite readings are unobserved, not zero-filled;
    this is an explicit benchmark convention rather than proof that every zero
    is a sensor failure. Positive speeds are preserved without normalization.
    All sensor columns are retained. Gaps in the common five-minute calendar
    are inserted as missing rows, never silently removed from the timeline.
    """
    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(part)
    try:
        frame = _read_metr_la_frame(path)
    except ImportError as exc:
        raise ImportError(
            "Reading the original METR-LA HDF5 file requires PyTables. "
            "Install tables in the notebook kernel with: %pip install tables"
        ) from exc
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Expected a nonempty sensor-by-time DataFrame in HDF5")
    if frame.columns.has_duplicates:
        raise ValueError("Source has duplicate sensor columns")
    times = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Source timestamps must be unique, nonmissing, and increasing")
    aligned = ((times.minute % 5 == 0) & (times.second == 0) &
               (times.microsecond == 0) & (times.nanosecond == 0))
    if not aligned.all():
        raise ValueError("METR-LA timestamps must lie on a five-minute grid")
    sensors = [str(column) for column in frame.columns]
    if len(set(sensors)) != len(sensors):
        raise ValueError("Sensor IDs collide after conversion to text")
    values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float, copy=True)
    finite = np.isfinite(values)
    zeros = int(((values == 0) & finite).sum())
    negatives = int(((values < 0) & finite).sum())
    nonfinite = int((~finite).sum())
    observed = finite & (values > 0)
    values[~observed] = np.nan
    cleaned = pd.DataFrame(values, index=times, columns=sensors)
    expected = pd.date_range(times[0], times[-1], freq="5min")
    missing_times = expected.difference(times)
    cleaned = cleaned.reindex(expected)
    cleaned.index.name = "time"
    audit = {
        "source": "METR-LA; original DCRNN authors' HDF5",
        "source_repository": "https://github.com/liyaguang/DCRNN",
        "source_file": path.name,
        "source_sha256": sha.hexdigest(),
        "source_rows": len(frame),
        "sensors": len(sensors),
        "frequency": "5min",
        "zero_readings_masked": zeros,
        "negative_readings_masked": negatives,
        "nonfinite_readings_masked": nonfinite,
        "missing_calendar_rows_inserted": len(missing_times),
        "missing_calendar_rows_examples": [stamp.isoformat() for stamp in missing_times[:10]],
        "output_rows": len(cleaned),
        "first_time": cleaned.index[0].isoformat(),
        "last_time": cleaned.index[-1].isoformat(),
        "source_timezone": str(times.tz) if times.tz is not None else "naive source labels preserved",
        "zero_policy": "nonpositive readings are missing under the benchmark convention",
        "units": "original speed units, unchanged",
        "normalization": "none",
        "imputation": "none",
        "node_selection": "none; all original sensors retained",
        "graph_pickle_loaded": False,
    }
    return cleaned, audit


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_synthetic(path, time_column="step"):
    """Read the supplied simulation without regenerating or changing its values."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic CSV not found: {path}")
    frame = pd.read_csv(path, float_precision="round_trip")
    if time_column not in frame.columns:
        raise ValueError(f"Time column {time_column!r} not found. Available: {list(frame.columns)}")
    steps = pd.to_numeric(frame.pop(time_column), errors="raise")
    if not np.isfinite(steps).all() or not np.equal(steps, np.floor(steps)).all():
        raise ValueError("Synthetic steps must be finite integers")
    frame.index = pd.Index(steps.astype(np.int64), name="time")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Synthetic steps must be unique and increasing")
    frame = frame.apply(pd.to_numeric, errors="raise").astype(float)
    if not len(frame.columns) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Synthetic node values must be nonempty and finite")
    segments = pd.Series(np.r_[False, np.diff(frame.index) != 1].cumsum(), index=frame.index)
    return frame, segments, {"source_file": path.name, "source_sha256": file_hash(path),
                            "time_column": time_column, "feature": "simulated scalar state",
                            "generation": "existing CSV preserved; simulation is not regenerated"}


def export_moex(raw_root, out, *, history=60, train_end='2023-12-31',
                val_end='2024-12-31', test_end='2025-12-31', n_stocks=10,
                min_coverage=0.95, max_gap_days=7, epsilon=1e-12, run_id=None):
    """Screen stocks and aggregate daily RV through the imported MOEX module."""
    raw_root, out = Path(raw_root), Path(out)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"MOEX raw cache not found: {raw_root}")
    with tempfile.TemporaryDirectory(prefix="moex_csv_stage_") as temporary:
        temporary = Path(temporary)
        arguments = ["--raw", str(raw_root), "--out", str(temporary / "daily"),
                     "--train-end", train_end, "--validation-end", val_end,
                     "--test-end", test_end, "--n-stocks", str(n_stocks),
                     "--min-coverage", str(min_coverage), "--history", "1",
                     "--max-gap-days", str(max_gap_days), "--epsilon", str(epsilon)]
        if run_id is not None:
            arguments.extend(["--run-id", run_id])
        try:
            worker_manifest = moex_worker.prepare(moex_worker.parser().parse_args(arguments))
        except RuntimeError:
            # The strict legacy window check is not the masked protocol used here.
            manifest_path = temporary / "daily/manifest.json"
            if not manifest_path.is_file():
                raise
            worker_manifest = json.loads(manifest_path.read_text())
            if worker_manifest.get("ready_for_training", True):
                raise
        daily = temporary / "daily"
        stocks = pd.read_csv(daily / "stocks.csv")
        nodes = stocks["SECID"].astype(str).to_list()
        values = pd.read_csv(daily / "log_rv_wide.csv", index_col=0, parse_dates=True,
                             float_precision="round_trip").reindex(columns=nodes)
        raw_values = pd.read_csv(daily / "rv_wide.csv", index_col=0, parse_dates=True,
                                 float_precision="round_trip").reindex(columns=nodes)
        calendar = pd.read_csv(daily / "calendar.csv", index_col=0, parse_dates=True)
        segments = calendar["segment_id"].reindex(values.index)
        metadata = {"feature": "log(max(within-session RV, epsilon))",
                    "raw_target": "within-session RV; no overnight return",
                    "epsilon": epsilon,
                    "worker_script_sha256": worker_manifest["script_sha256"],
                    "raw_run_file_sha256": worker_manifest["raw_run_file_sha256"],
                    "source_requests_sha256": file_hash(daily / "source_requests.csv"),
                    "stock_selection": "train only: >=95% valid core days, top median ruble turnover, one issue per issuer",
                    "stock_count": n_stocks, "limitations": worker_manifest["limitations"]}
        result = export_dataset(values, raw_values, segments, out, history=history,
                                train_end=train_end, val_end=val_end, metadata=metadata)
        reports = out / "reports"
        reports.mkdir(exist_ok=True)
        for name in ["selection_audit.csv", "stocks.csv", "daily_panel.csv", "quality_by_split.csv", "source_requests.csv"]:
            destination = reports / name
            shutil.copyfile(daily / name, destination)
            result["file_sha256"][str(destination.relative_to(out))] = file_hash(destination)
        (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
