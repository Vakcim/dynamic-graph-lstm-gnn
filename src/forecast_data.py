"""Read prepared CSVs once and construct normalized forecast windows lazily."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ForecastData:
    """One cached time-by-node panel; no overlapping-window tensor is stored."""

    def __init__(self, root, verify_hashes=True):
        self.root = Path(root).resolve()
        self.schema = json.loads((self.root / "schema.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.nodes = list(self.schema["node_order"])
        self.node_order = self.nodes
        self.history = int(self.schema["history"])
        if self.schema["n_features"] != 1 or self.schema["horizon"] != 1:
            raise ValueError("This loader supports one feature per node and a one-step horizon")
        if verify_hashes:
            for relative, expected in self.manifest["file_sha256"].items():
                path = (self.root / relative).resolve()
                if self.root not in path.parents or not path.is_file() or _sha_file(path) != expected:
                    raise RuntimeError("Prepared data checksum mismatch: " + relative)
        features, raw, times, labels = [], [], [], []
        for split in SPLITS:
            folder = self.root / split
            f = pd.read_csv(folder / "series.csv", float_precision="round_trip")
            r = pd.read_csv(folder / "raw_values.csv", float_precision="round_trip")
            mask = pd.read_csv(folder / "observed_mask.csv")
            expected_columns = ["time", *self.nodes]
            if any(list(frame.columns) != expected_columns for frame in (f, r, mask)):
                raise ValueError("CSV node order does not match schema.json")
            if not f["time"].equals(r["time"]) or not f["time"].equals(mask["time"]):
                raise ValueError("Feature, raw-target and mask times do not align")
            f_values = f[self.nodes].to_numpy(dtype=np.float64, copy=True)
            r_values = r[self.nodes].to_numpy(dtype=np.float64, copy=True)
            observed = np.isfinite(f_values)
            if not np.array_equal(mask[self.nodes].to_numpy(), observed.astype(np.uint8)):
                raise ValueError("The observed mask does not match feature values")
            if np.isinf(f_values).any() or np.isinf(r_values).any():
                raise ValueError("Prepared values must contain finite numbers or NaN")
            features.append(f_values)
            raw.append(r_values)
            times.append(f["time"].to_numpy())
            labels.extend([split] * len(f))
        self.feature_values = np.concatenate(features)
        self.features = self.feature_values
        self.values = self.feature_values.astype(np.float32)
        self.raw_values = np.concatenate(raw)
        self.raw = self.raw_values
        self.split_labels = np.asarray(labels)
        time_values = np.concatenate(times)
        self.times = pd.DatetimeIndex(pd.to_datetime(time_values)) if self.schema["time_kind"] == "datetime" else pd.Index(time_values)
        if self.times.has_duplicates or not self.times.is_monotonic_increasing:
            raise ValueError("The common time index must be unique and increasing")
        calendar = pd.read_csv(self.root / "calendar.csv")
        calendar_times = pd.DatetimeIndex(pd.to_datetime(calendar["time"])) if self.schema["time_kind"] == "datetime" else pd.Index(calendar["time"])
        if not self.times.equals(calendar_times) or not np.array_equal(calendar["split"], self.split_labels):
            raise ValueError("CSV partitions do not match the saved common calendar")
        self.segments = calendar["segment_id"].to_numpy()
        train = self.feature_values[self.split_labels == "train"]
        self.mean = np.nanmean(train, axis=0)
        self.std = np.nanstd(train, axis=0, ddof=0)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("Every node needs finite observed training data")
        self.std[self.std == 0] = 1.0
        self.feature_mean, self.feature_std = self.mean, self.std
        fallback = json.loads((self.root / "train_input_mean.json").read_text(encoding="utf-8"))
        self.fallback_mean = np.asarray([fallback[node] for node in self.nodes], dtype=np.float64)
        if not np.allclose(self.fallback_mean, self.mean, rtol=1e-12, atol=1e-12):
            raise ValueError("Saved input fallback is inconsistent with observed training means")
        samples = pd.read_csv(self.root / "samples.csv")
        self.samples = samples.loc[samples["kept"].eq(True)].copy().reset_index(drop=True)
        target_rows = self.samples["target_row"].to_numpy(dtype=np.int64)
        origin_rows = self.samples["origin_row"].to_numpy(dtype=np.int64)
        if (len(set(target_rows)) != len(target_rows) or (target_rows < self.history).any() or
                (target_rows >= len(self.values)).any() or not np.array_equal(origin_rows, target_rows - 1)):
            raise ValueError("Invalid saved forecast sample positions")
        breaks = np.r_[0, np.cumsum(self.segments[1:] != self.segments[:-1])]
        if ((breaks[target_rows] != breaks[target_rows - self.history]).any() or
                not np.array_equal(self.samples["split"], self.split_labels[target_rows])):
            raise ValueError("Saved samples cross a segment or split boundary incorrectly")
        self.target_rows = {split: self.samples.loc[self.samples["split"].eq(split), "target_row"].to_numpy(dtype=np.int64)
                            for split in SPLITS}
        self.provenance = {"manifest_sha256": _sha_file(self.root / "manifest.json"),
                           "schema_sha256": _sha_file(self.root / "schema.json"),
                           "source_data_sha256": self.manifest["source_data_sha256"],
                           "history": self.history, "node_order": self.nodes,
                           "normalization": {"fit": "observed training rows only", "mean": self.mean.tolist(),
                                             "std": self.std.tolist(), "ddof": 0},
                           "input_missing_policy": "window forward fill then saved train mean",
                           "target_missing_policy": "never imputed; loss and metrics require y_mask"}
        self.normalization = self.provenance["normalization"]

    def input_window(self, origin_row):
        """Return normalized X and its original mask for one history-only origin."""
        origin = int(origin_row)
        left = origin - self.history + 1
        if left < 0 or origin >= len(self.values):
            raise IndexError("The origin has insufficient history or lies outside the panel")
        if len(set(self.segments[left:origin + 1])) != 1:
            raise ValueError("The requested input window crosses a market-closure segment")
        window = self.values[left:origin + 1].copy()
        observed = np.isfinite(window)
        positions = np.where(observed, np.arange(self.history)[:, None], -1)
        np.maximum.accumulate(positions, axis=0, out=positions)
        filled = window[np.maximum(positions, 0), np.arange(len(self.nodes))[None, :]]
        absent = positions < 0
        filled[absent] = np.broadcast_to(self.fallback_mean, filled.shape)[absent]
        x = ((filled.astype(np.float64) - self.mean) / self.std).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("The filled input window contains a nonfinite value")
        return x[..., None], observed[..., None]

    @property
    def training_features(self):
        """Observed training features for fitting fixed graph parameters."""
        return self.feature_values[self.split_labels == "train"]

    def dataset(self, split):
        if split not in SPLITS:
            raise ValueError("split must be train, val or test")
        return ForecastDataset(self, split)

    def inverse_target(self, normalized):
        """Return the feature scale; MOEX still needs exp() to obtain raw RV."""
        return np.asarray(normalized, dtype=np.float64) * self.std + self.mean

    def quality(self):
        """Report the actual scored target count for every node, including zeros."""
        records = []
        for split in SPLITS:
            rows = self.split_labels == split
            targets = self.target_rows[split]
            observed = np.isfinite(self.feature_values[targets]) & np.isfinite(self.raw_values[targets])
            for node, name in enumerate(self.nodes):
                count = int(observed[:, node].sum())
                records.append({"split": split, "node": name, "rows": int(rows.sum()),
                                "observed_feature_rows": int(np.isfinite(self.feature_values[rows, node]).sum()),
                                "usable_samples": len(targets), "evaluated_targets": count,
                                "evaluated_target_fraction": count / len(targets) if len(targets) else 0.0})
        return pd.DataFrame(records)


class ForecastDataset:
    """Map-style NumPy dataset compatible with PyTorch's default DataLoader."""

    def __init__(self, bundle, split):
        self.bundle, self.split = bundle, split
        self.target_rows = bundle.target_rows[split]

    def __len__(self):
        return len(self.target_rows)

    def __getitem__(self, item):
        b = self.bundle
        target = int(self.target_rows[item])
        x, observed = b.input_window(target - 1)
        y_feature = b.feature_values[target].copy()
        y_raw = b.raw_values[target].copy()
        mask = np.isfinite(y_feature) & np.isfinite(y_raw)
        y_norm = np.where(mask, (y_feature - b.mean) / b.std, 0).astype(np.float32)
        if not np.isfinite(x).all() or not mask.any():
            raise ValueError("The saved forecast sample has invalid input or no target")
        return {"X": x, "X_mask": observed, "y_norm": y_norm,
                "y_feature": y_feature, "y_raw": y_raw, "y_mask": mask,
                "origin_row": np.int64(target - 1), "target_row": np.int64(target)}


def load_data(root, verify_hashes=True):
    return ForecastData(root, verify_hashes=verify_hashes)
