"""Shared training, matched evaluation and experiment logging for four notebooks."""

import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from forecast_metrics import ForecastMetrics
from forecast_models import LSTMGNN, MultivariateLSTM, build_adjacency

PROTOCOL_VERSION = "lstm-gnn-pilot-1.0"
DEFAULT_CONFIG = {
    "hidden_size": 16, "batch_size": 32, "max_epochs": 15, "patience": 4,
    "learning_rate": 0.001, "weight_decay": 0.0001, "gradient_clip": 1.0,
    "top_k": 3, "cpu_threads": 2, "device": "cpu", "min_delta": 0.0,
}
MODEL_KINDS = ("persistence", "lstm", "lstm_gnn_static", "lstm_gnn_dynamic",
               "lstm_gnn_fixed_edges_dynamic_weights")


def _sha_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def seed_everything(seed, cpu_threads):
    """Use the same initialization and batch generator for the two GNN variants."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Persistence(nn.Module):
    """Predict the last available input value; there are no learned parameters."""

    def forward(self, x):
        return x[:, -1, :, 0]


def fixed_graph(data, top_k):
    """Fit a graph using exactly the last L training observations, then freeze it."""
    origin = int(np.flatnonzero(data.split_labels == "train")[-1])
    x, _ = data.input_window(origin)
    adjacency = build_adjacency(torch.from_numpy(x), top_k=top_k).detach().cpu()
    return adjacency, {"anchor_origin_row": origin, "anchor_time": str(data.times[origin]),
                       "history": data.history, "rule": "row-normalized top-k absolute Pearson correlation",
                       "A_ij": "message from node j to node i", "self_edges": False,
                       "estimated_on": "last history-length window within train only"}


def make_model(model_kind, n_nodes, config, adjacency=None):
    if model_kind == "persistence":
        return Persistence()
    if model_kind == "lstm":
        return MultivariateLSTM(n_nodes, hidden=config["hidden_size"])
    mode = {"lstm_gnn_static": "static", "lstm_gnn_dynamic": "dynamic",
            "lstm_gnn_fixed_edges_dynamic_weights": "fixed_edges_dynamic_weights"}[model_kind]
    return LSTMGNN(n_nodes, hidden=config["hidden_size"], top_k=config["top_k"],
                   graph_mode=mode, static_adjacency=adjacency)


def _parameter_hash(model):
    hasher = hashlib.sha256()
    for name, value in model.named_parameters():
        hasher.update(name.encode())
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def _loader(data, split, config, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(data.dataset(split), batch_size=config["batch_size"], shuffle=shuffle,
                      num_workers=0, generator=generator, drop_last=False)


def _masked_mse(prediction, target, mask):
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("The model produced a nonfinite normalized prediction")
    count = mask.sum()
    if not count:
        raise ValueError("A batch has no observed target")
    squared = (prediction - target).square()
    return squared.masked_select(mask).sum(), count


def evaluate_model(model, data, dataset_name, split, config, device, output=None):
    """Evaluate exactly the saved sample index and masks; never refit on val/test."""
    model.eval()
    accumulator = ForecastMetrics(dataset_name, data.nodes)
    squared_sum, count_sum = 0.0, 0
    losses, saved = [], []
    with torch.no_grad():
        for batch in _loader(data, split, config):
            prediction = model(batch["X"].to(device))
            squared, count = _masked_mse(prediction, batch["y_norm"].to(device), batch["y_mask"].to(device))
            squared_sum += float(squared.item())
            count_sum += int(count.item())
            target_rows = batch["target_row"].numpy()
            origin_rows = batch["origin_row"].numpy()
            feature_prediction = data.inverse_target(prediction.detach().cpu().numpy())
            metric_batch = accumulator.update(
                feature_prediction, batch["y_feature"].numpy(), batch["y_raw"].numpy(), batch["y_mask"].numpy(),
                origin_times=[str(data.times[row]) for row in origin_rows],
                target_times=[str(data.times[row]) for row in target_rows])
            if output is not None:
                errors = np.zeros_like(metric_batch.prediction_raw)
                np.subtract(metric_batch.prediction_raw, metric_batch.target_raw,
                            out=errors, where=metric_batch.mask)
                absolute = np.abs(errors).sum(axis=1)
                squared_errors = np.square(errors).sum(axis=1)
                counts = metric_batch.mask.sum(axis=1)
                qlike = None
                if dataset_name == "moex":
                    q = np.zeros_like(errors)
                    observed = metric_batch.mask
                    q[observed] = (metric_batch.prediction_feature[observed] +
                                   metric_batch.target_raw[observed] / metric_batch.prediction_raw[observed])
                    qlike = q.sum(axis=1)
                for row in range(len(target_rows)):
                    record = {"origin_row": int(origin_rows[row]), "target_row": int(target_rows[row]),
                              "origin_time": metric_batch.origin_times[row], "target_time": metric_batch.target_times[row],
                              "n_observed": int(counts[row]), "sum_absolute_error": float(absolute[row]),
                              "sum_squared_error": float(squared_errors[row]),
                              "mae": float(absolute[row] / counts[row]),
                              "rmse": float(np.sqrt(squared_errors[row] / counts[row]))}
                    if qlike is not None:
                        record.update(sum_qlike=float(qlike[row]), qlike=float(qlike[row] / counts[row]))
                    losses.append(record)
                saved.append((origin_rows, target_rows, metric_batch))
    if not count_sum:
        raise ValueError("The split has no evaluated observations: " + split)
    summary = accumulator.summary()
    summary["normalized_mse"] = squared_sum / count_sum
    if output is not None:
        output = Path(output)
        pd.DataFrame(accumulator.per_node()).to_csv(output / f"{split}_per_node.csv", index=False)
        pd.DataFrame(losses).to_csv(output / f"{split}_losses.csv", index=False)
        arrays = {
            "origin_row": np.concatenate([part[0] for part in saved]),
            "target_row": np.concatenate([part[1] for part in saved]),
            "node_order": np.asarray(data.nodes, dtype=str),
        }
        for name in ("prediction_feature", "prediction_raw", "target_feature", "target_raw", "mask"):
            arrays[name] = np.concatenate([getattr(part[2], name) for part in saved])
        np.savez_compressed(output / f"{split}_predictions.npz", **arrays)
        _write_json(output / f"{split}_metrics.json", summary)
    return summary


def _code_hashes():
    folder = Path(__file__).resolve().parent
    return {name: _sha_file(folder / name) for name in (
        "forecast_data.py", "forecast_models.py", "forecast_metrics.py", "forecast_experiment.py")}


def run_experiment(data, model_kind, dataset_name, results_root, *, seed=0,
                   config=None, evaluate_test=False, run_tag="pilot_v1"):
    """Train/select on train/val; optionally score test once without retraining."""
    if model_kind not in MODEL_KINDS or dataset_name not in ("synthetic", "metr_la", "moex"):
        raise ValueError("Unknown model or dataset")
    settings = dict(DEFAULT_CONFIG)
    if config:
        if set(config) - set(settings):
            raise ValueError("Unknown training settings: " + str(set(config) - set(settings)))
        settings.update(config)
    if any(not np.isfinite(float(value)) for name, value in settings.items() if name != "device"):
        raise ValueError("Numeric training settings must be finite")
    if any(int(settings[name]) < 1 for name in ("hidden_size", "batch_size", "max_epochs", "patience", "cpu_threads")):
        raise ValueError("Dimensions, epochs, patience and CPU threads must be positive")
    if settings["learning_rate"] <= 0 or settings["gradient_clip"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    if settings["min_delta"] < 0 or int(settings["top_k"]) != settings["top_k"] or settings["top_k"] < 0:
        raise ValueError("min_delta and integer top_k must be nonnegative")
    if any(not len(data.dataset(split)) for split in ("train", "val")):
        raise ValueError("Training and validation need at least one usable sample")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "torch": str(torch.__version__)}
    protocol = {"protocol": PROTOCOL_VERSION, "dataset": dataset_name, "data": data.provenance,
                "config": settings, "code_sha256": _code_hashes(), "versions": versions,
                "run_tag": str(run_tag)}
    comparison_id = _json_hash(protocol)[:12]
    identity = {**protocol, "model": model_kind, "seed": seed}
    run_id = _json_hash(identity)[:12]
    output = Path(results_root).resolve() / dataset_name / model_kind / f"seed_{seed:03d}" / run_id
    run_file = output / "run.json"
    existing = None
    if output.exists() and any(output.iterdir()):
        if not run_file.is_file():
            raise FileExistsError("Incomplete result directory; choose a new run_tag: " + str(output))
        existing = json.loads(run_file.read_text(encoding="utf-8"))
        if existing.get("identity") != identity or existing.get("status") != "completed":
            raise FileExistsError("Result identity/status mismatch; choose a new run_tag")
        for name, expected in existing["file_sha256"].items():
            if not (output / name).is_file() or _sha_file(output / name) != expected:
                raise RuntimeError("Saved experiment checksum mismatch: " + name)
        if not evaluate_test or "test" in existing["metrics"]:
            print("Reusing completed run:", output)
            return existing
    else:
        output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, int(settings["cpu_threads"]))
    requested_device = settings["device"]
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available()
                          else "cpu" if requested_device == "auto" else requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select device='cpu'")
    adjacency, graph_info = (None, None)
    if model_kind.startswith("lstm_gnn"):
        adjacency, graph_info = fixed_graph(data, settings["top_k"])
    model = make_model(model_kind, len(data.nodes), settings, adjacency).to(device)
    initial_hash = _parameter_hash(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    primary = "qlike" if dataset_name == "moex" else "mae"
    best_epoch, fit_seconds, history = None, 0.0, []
    if existing is not None:
        if model_kind != "persistence":
            checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state"])
        best_epoch, fit_seconds = existing["best_epoch"], existing["fit_seconds"]
        print("Using selected checkpoint for test; training is not repeated")
    elif model_kind != "persistence":
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        train_loader = _loader(data, "train", settings, shuffle=True, seed=seed)
        best_score, stale = float("inf"), 0
        started = time.perf_counter()
        for epoch in range(1, int(settings["max_epochs"]) + 1):
            model.train()
            total_squared, total_count = 0.0, 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch["X"].to(device))
                squared, count = _masked_mse(prediction, batch["y_norm"].to(device), batch["y_mask"].to(device))
                loss = squared / count
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                total_squared += float(squared.item())
                total_count += int(count.item())
            validation = evaluate_model(model, data, dataset_name, "val", settings, device)
            score = validation[primary]
            record = {"epoch": epoch, "train_normalized_mse": total_squared / total_count,
                      "val_normalized_mse": validation["normalized_mse"],
                      "val_primary": score, "elapsed_seconds": time.perf_counter() - started}
            history.append(record)
            print(f"{model_kind} seed={seed} epoch={epoch}: train MSE={record['train_normalized_mse']:.5g}; val {primary}={score:.6g}", flush=True)
            if score < best_score - settings["min_delta"]:
                best_score, best_epoch, stale = score, epoch, 0
                torch.save({"model_state": model.state_dict(), "best_epoch": best_epoch}, output / "best_model.pt")
            else:
                stale += 1
            if stale >= settings["patience"]:
                break
        fit_seconds = time.perf_counter() - started
        checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
    if existing is None:
        pd.DataFrame(history, columns=["epoch", "train_normalized_mse", "val_normalized_mse", "val_primary", "elapsed_seconds"]).to_csv(output / "history.csv", index=False)
        data.quality().to_csv(output / "target_coverage.csv", index=False)
        _write_json(output / "normalization.json", data.normalization)
        if adjacency is not None:
            np.save(output / "static_graph.npy", adjacency.numpy())
            _write_json(output / "graph.json", graph_info)
        metrics = {"val": evaluate_model(model, data, dataset_name, "val", settings, device, output)}
    else:
        metrics = existing["metrics"]
    if evaluate_test:
        metrics["test"] = evaluate_model(model, data, dataset_name, "test", settings, device, output)
    report = {"identity": identity, "status": "completed", "model": model_kind,
              "dataset": dataset_name, "seed": seed, "comparison_id": comparison_id,
              "run_id": run_id, "output_dir": str(output), "primary_metric": primary,
              "metrics": metrics, "best_epoch": best_epoch, "fit_seconds": fit_seconds,
              "trainable_parameters": parameter_count, "initial_parameter_sha256": initial_hash,
              "device": str(device), "graph": graph_info,
              "architecture_note": "LSTM is multivariate; both GNNs share a per-node temporal encoder and differ only in graph updates",
              "graph_update_note": "dynamic mode updates both edges and weights; effects are not yet separated",
              "file_sha256": {path.name: _sha_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "run.json"}}
    _write_json(run_file, report)
    print("Saved:", output)
    return report


def collect_results(results_root, dataset_name, split="val", run_tag="pilot_v1"):
    """Show runs with comparison IDs; do not mix different protocols when ranking models."""
    rows = []
    for path in (Path(results_root) / dataset_name).glob("*/seed_*/*/run.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "completed" or report["identity"]["run_tag"] != run_tag or split not in report["metrics"]:
            continue
        metrics = report["metrics"][split]
        row = {"model": report["model"], "seed": report["seed"], "split": split,
               "comparison_id": report["comparison_id"], "mae": metrics["mae"], "rmse": metrics["rmse"],
               "n_observed": metrics["n_observed"], "best_epoch": report["best_epoch"],
               "parameters": report["trainable_parameters"], "fit_seconds": report["fit_seconds"],
               "output_dir": report["output_dir"]}
        if "qlike" in metrics:
            row["qlike"] = metrics["qlike"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "seed", "comparison_id"]).reset_index(drop=True) if rows else pd.DataFrame()
