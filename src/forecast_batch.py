"""Run one forecasting model across prepared datasets and save batch summaries."""

import gc
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forecast_data import load_data
from forecast_experiment import run_experiment


SUMMARY_COLUMNS = [
    "dataset", "model", "seed", "split", "primary_metric", "primary_score",
    "mae", "rmse", "qlike", "normalized_mse", "n_observed", "n_origins",
    "best_epoch", "parameters", "fit_seconds", "comparison_id", "run_id", "output_dir",
]
STATUS_COLUMNS = ["dataset", "model", "seed", "status", "error", "output_dir"]


def run_batch(data_root, model_kind, results_root, *,
              datasets=("synthetic", "moex", "metr_la"), seeds=(0,), config=None,
              evaluate_test=True, run_tag="pilot_v1", inspect_dataset=None):
    """Process datasets sequentially; preserve progress and report every failure.

    The existing runner owns training, selection, test evaluation and caching.
    This wrapper does not change the four core modules or their experiment IDs.
    KeyboardInterrupt propagates after the partial summary is saved.
    """
    datasets, seeds = tuple(datasets), tuple(seeds)
    if (not datasets or len(set(datasets)) != len(datasets) or
            any(name not in ("synthetic", "moex", "metr_la") for name in datasets)):
        raise ValueError("Choose distinct prepared datasets: synthetic, moex, metr_la")
    if (not seeds or len(set(seeds)) != len(seeds) or
            any(isinstance(seed, bool) or int(seed) != seed or seed < 0 for seed in seeds)):
        raise ValueError("Use distinct nonnegative integer seeds")
    seeds = tuple(int(seed) for seed in seeds)
    data_root, results_root = Path(data_root).resolve(), Path(results_root).resolve()
    started = datetime.now(timezone.utc)
    summary_dir = results_root / "summaries" / model_kind / started.strftime("%Y%m%dT%H%M%S_%fZ")
    summary_dir.mkdir(parents=True, exist_ok=False)
    reports, rows = [], []
    status = [{"dataset": name, "model": model_kind, "seed": seed,
               "status": "pending", "error": "", "output_dir": ""}
              for name in datasets for seed in seeds]
    metadata = {
        "model": model_kind, "datasets": list(datasets), "seeds": list(seeds),
        "config": dict(config or {}), "evaluate_test": bool(evaluate_test),
        "run_tag": str(run_tag), "started_at_utc": started.isoformat(),
        "data_root": str(data_root), "results_root": str(results_root),
        "batch_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "running",
    }
    clock = time.perf_counter()

    def save_progress():
        summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
        statuses = pd.DataFrame(status, columns=STATUS_COLUMNS)
        for name, frame in (("summary.csv", summary), ("status.csv", statuses),
                            ("errors.csv", statuses.loc[statuses["status"].eq("failed")])):
            temporary = summary_dir / (name + ".tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(summary_dir / name)
        metadata["elapsed_seconds"] = time.perf_counter() - clock
        temporary = summary_dir / "batch.json.tmp"
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(summary_dir / "batch.json")

    save_progress()
    try:
        for dataset_name in datasets:
            print(f"\n=== {model_kind}: {dataset_name} ===", flush=True)
            items = [item for item in status if item["dataset"] == dataset_name]
            data = None
            try:
                data = load_data(data_root / dataset_name, verify_hashes=True)
                if inspect_dataset is not None:
                    inspect_dataset(data, dataset_name)
            except Exception as exc:
                traceback.print_exc()
                for item in items:
                    item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                save_progress()
                del data
                gc.collect()
                continue
            try:
                for item in items:
                    item["status"] = "running"
                    save_progress()
                    try:
                        report = run_experiment(
                            data, model_kind, dataset_name, results_root,
                            seed=item["seed"], config=config, evaluate_test=evaluate_test, run_tag=run_tag,
                        )
                        reports.append(report)
                        for split in ("val", "test") if evaluate_test else ("val",):
                            metrics = report["metrics"][split]
                            rows.append({
                                "dataset": dataset_name, "model": model_kind, "seed": item["seed"],
                                "split": split, "primary_metric": report["primary_metric"],
                                "primary_score": metrics[report["primary_metric"]],
                                "mae": metrics["mae"], "rmse": metrics["rmse"],
                                "qlike": metrics.get("qlike"), "normalized_mse": metrics["normalized_mse"],
                                "n_observed": metrics["n_observed"], "n_origins": metrics["n_origins"],
                                "best_epoch": report["best_epoch"], "parameters": report["trainable_parameters"],
                                "fit_seconds": report["fit_seconds"], "comparison_id": report["comparison_id"],
                                "run_id": report["run_id"], "output_dir": report["output_dir"],
                            })
                        item.update(status="completed", output_dir=report["output_dir"])
                    except Exception as exc:
                        item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                        traceback.print_exc()
                    save_progress()
            finally:
                del data
                gc.collect()
        metadata["status"] = "completed" if all(item["status"] == "completed" for item in status) else "incomplete"
    finally:
        if metadata["status"] == "running":
            metadata["status"] = "interrupted"
            for item in status:
                if item["status"] == "running":
                    item["status"] = "interrupted"
        save_progress()
    print("Batch summaries:", summary_dir, flush=True)
    return {"reports": reports, "summary": pd.DataFrame(rows, columns=SUMMARY_COLUMNS),
            "status": pd.DataFrame(status, columns=STATUS_COLUMNS), "summary_dir": summary_dir}
