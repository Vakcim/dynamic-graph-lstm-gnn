#!/usr/bin/env python3
"""Prepare reproducible MOEX RV forecasting CSVs from moex_raw_download.ipynb.

Run from the project root or notebooks directory; --help lists all settings.
The script uses only the existing raw cache, pandas, NumPy and the standard
library. It exports complete 60-session windows by default, with one log-RV
feature per stock and next-session targets. Normalize using training data only.
"""

import numpy as np
import pandas as pd

CANDLE_COLUMNS = ["begin", "end", "open", "close", "high", "low", "value", "volume"]


def _calendar(session_dates):
    dates = pd.DatetimeIndex(pd.to_datetime(list(session_dates)))
    if dates.tz is not None:
        dates = dates.tz_convert("Europe/Moscow").tz_localize(None)
    dates = dates.normalize()
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Session dates must be nonmissing, unique, and increasing")
    return dates


def summarize_candles(candles, session_dates, epsilon=1e-12):
    """Aggregate only complete 10:00-18:40 Moscow-time grids into daily RV.

    The first return starts at the first candle's open; subsequent prices are
    candle closes, so inter-bar changes are included. No previous-day close is
    used. The fixed epsilon floors only log(RV), never the stored raw RV.
    Missing or invalid candles invalidate the day; they are not filled.
    session_dates must be supplied from the common market calendar.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    missing = set(CANDLE_COLUMNS) - set(candles.columns)
    if missing and len(candles):
        raise ValueError("Missing candle columns: " + ", ".join(sorted(missing)))
    dates = _calendar(session_dates)
    frame = candles.reindex(columns=CANDLE_COLUMNS).copy()
    frame["begin"] = pd.to_datetime(frame["begin"], errors="raise")
    frame["end"] = pd.to_datetime(frame["end"], errors="raise")
    for name in ("begin", "end"):
        if frame[name].dt.tz is not None:
            frame[name] = frame[name].dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    if frame["begin"].isna().any():
        raise ValueError("A candle with a missing begin timestamp cannot be assigned to a day")
    minutes = frame["begin"].dt.hour * 60 + frame["begin"].dt.minute
    frame = frame.loc[(minutes >= 600) & (minutes < 1120)].copy()
    frame = frame.drop_duplicates(subset=CANDLE_COLUMNS)
    frame["date"] = frame["begin"].dt.normalize()
    grouped = {day: group for day, group in frame.groupby("date", sort=False)}
    results = []
    feature_names = ("rv", "log_rv", "turnover", "volume", "intraday_return", "log_range")
    for day in dates:
        group = grouped.get(day)
        row = {"date": day, "n_bars": 0, "valid": False, "invalid_reason": "missing_day"}
        row.update({name: np.nan for name in feature_names})
        if group is None or group.empty:
            results.append(row)
            continue
        row["n_bars"] = int(group["begin"].nunique())
        if group["begin"].duplicated().any():
            row["invalid_reason"] = "conflicting_duplicate"
            results.append(row)
            continue
        group = group.sort_values("begin")
        expected = pd.date_range(day + pd.Timedelta(hours=10), periods=52, freq="10min")
        if not pd.DatetimeIndex(group["begin"]).equals(expected):
            row["invalid_reason"] = "incomplete_or_misaligned_grid"
            results.append(row)
            continue
        end_ok = group["end"].notna() & (group["end"] >= group["begin"]) & (
            group["end"] < group["begin"] + pd.Timedelta(minutes=10))
        if not end_ok.all():
            row["invalid_reason"] = "invalid_bar_end"
            results.append(row)
            continue
        values = group[["open", "close", "high", "low", "value", "volume"]].apply(
            pd.to_numeric, errors="coerce")
        price = values[["open", "close", "high", "low"]]
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            row["invalid_reason"] = "nonfinite_numeric_value"
        elif not (price > 0).to_numpy().all():
            row["invalid_reason"] = "nonpositive_price"
        elif not ((values["high"] >= values[["open", "close"]].max(axis=1)) &
                  (values["low"] <= values[["open", "close"]].min(axis=1)) &
                  (values["low"] <= values["high"])).all():
            row["invalid_reason"] = "inconsistent_ohlc"
        elif not (values[["value", "volume"]] >= 0).to_numpy().all():
            row["invalid_reason"] = "negative_turnover_or_volume"
        else:
            prices = np.r_[values["open"].iloc[0], values["close"].to_numpy()]
            returns = np.diff(np.log(prices))
            rv = float(returns @ returns)
            row.update(valid=True, invalid_reason="", rv=rv,
                       log_rv=float(np.log(max(rv, epsilon))),
                       turnover=float(values["value"].sum()),
                       volume=float(values["volume"].sum()),
                       intraday_return=float(np.log(prices[-1]) - np.log(prices[0])),
                       log_range=float(np.log(values["high"].max()) -
                                       np.log(values["low"].min())))
        results.append(row)
    return pd.DataFrame(results, columns=["date", "n_bars", "valid", "invalid_reason", *feature_names])


import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

BASE = "https://iss.moex.com/iss"
VERSION = "moex-rv-preprocessing-1.0"
FEATURES = ["rv", "log_rv", "turnover", "volume", "intraday_return", "log_range"]
SHORT_SESSION_SOURCE = [
    "https://www.moex.com/n45532",
    "https://www.moex.com/n45611",
    "https://www.moex.com/n46118",
    "https://www.moex.com/n46282",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def table(payload, name):
    block = payload.get(name, {})
    columns, rows = block.get("columns"), block.get("data")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("Missing ISS table: " + name)
    if len(columns) != len(set(columns)):
        raise ValueError("Repeated ISS columns: " + name)
    if any(not isinstance(row, list) or len(row) != len(columns) for row in rows):
        raise ValueError("Malformed ISS rows: " + name)
    return [dict(zip(columns, row)) for row in rows]


class Cache:
    """Read the original downloader cache; never issue network requests."""

    def __init__(self, root):
        self.root = root
        self.used = {}

    def get(self, path, **params):
        url = BASE + "/" + path.lstrip("/") + "?" + urlencode(sorted(params.items()))
        key = digest(url.encode())
        folder = self.root / "requests" / key[:2] / key
        try:
            meta = json.loads((folder / "request.json").read_text(encoding="utf-8"))
            packed = (folder / "response.json.gz").read_bytes()
        except FileNotFoundError as exc:
            raise RuntimeError("Missing cached request. Complete the raw download: " + url) from exc
        if meta.get("url") != url or digest(packed) != meta.get("gzip_sha256"):
            raise RuntimeError("Cache metadata/gzip mismatch: " + str(folder))
        body = gzip.decompress(packed)
        if digest(body) != meta.get("body_sha256"):
            raise RuntimeError("Cache body checksum mismatch: " + str(folder))
        payload = json.loads(body)
        if not isinstance(payload, dict) or "error" in payload or "errors" in payload:
            raise RuntimeError("Unsuccessful ISS response: " + str(folder))
        self.used[key] = {"request_key": key, "body_sha256": meta["body_sha256"],
                          "gzip_sha256": meta["gzip_sha256"], "url": url,
                          "received_utc": meta.get("received_utc")}
        return payload

    def pages(self, path, block, **params):
        offset, seen = 0, set()
        while True:
            payload = self.get(path, start=offset, **params)
            rows = table(payload, block)
            cursor = table(payload, block + ".cursor") if block + ".cursor" in payload else []
            total = int(cursor[0]["TOTAL"]) if cursor else None
            if cursor and int(cursor[0]["INDEX"]) != offset:
                raise RuntimeError("ISS cursor does not match requested offset")
            if not rows:
                if total is not None and offset < total:
                    raise RuntimeError("Premature empty ISS page")
                return
            signature = digest(json.dumps(rows, sort_keys=True).encode())
            if signature in seen:
                raise RuntimeError("Repeated ISS page")
            seen.add(signature)
            yield rows
            offset += len(rows)
            if total is not None and offset >= total:
                return


def resolve_run(raw, run_id):
    if run_id:
        run = raw / "runs" / run_id
        candidates = [run] if run.is_dir() else []
    else:
        candidates = sorted(path.parent for path in (raw / "runs").glob("*/summary.json"))
    completed = []
    for run in candidates:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        checkpoint = json.loads((run / "checkpoint.json").read_text(encoding="utf-8"))
        if summary.get("status") == checkpoint.get("status") == "completed":
            completed.append(run)
    if len(completed) != 1:
        raise RuntimeError("Expected one completed download; specify --run-id if several exist")
    return completed[0]


def read_metadata(cache, universe):
    result = {}
    for secid in universe:
        payload = cache.get("securities/" + quote(secid, safe="") + ".json")
        description = {str(row["name"]): row.get("value") for row in table(payload, "description")}
        result[secid] = {"SECID": secid, "TYPE": description.get("TYPE", ""),
                         "EMITTER_ID": description.get("EMITTER_ID", ""),
                         "ISIN": description.get("ISIN", ""),
                         "SHORTNAME": description.get("SHORTNAME", secid)}
    return result


def read_calendar(cache, configuration, ordinary, end):
    """Infer market sessions from positive daily turnover of ordinary shares."""
    board = configuration["board"]
    prefix = f"history/engines/stock/markets/shares/boards/{board}/securities.json"
    availability = {secid: set() for secid in ordinary}
    calendar, mismatched_currency = [], set()
    dates = pd.date_range(configuration["date_from"], end, freq="D")
    for number, day in enumerate(dates, 1):
        active = False
        for batch in cache.pages(prefix, "history", date=day.strftime("%Y-%m-%d"),
                                 tradingsession=configuration["daily_tradingsession"], limit=100):
            for row in batch:
                if row.get("BOARDID") != board or str(row.get("TRADEDATE"))[:10] != day.strftime("%Y-%m-%d"):
                    raise RuntimeError("Daily rows do not match the requested date/board")
                secid = str(row.get("SECID"))
                if secid not in ordinary:
                    continue
                currency = row.get("CURRENCYID")
                if currency and currency not in {"SUR", "RUB"}:
                    mismatched_currency.add((secid, day))
                value = pd.to_numeric(row.get("VALUE"), errors="coerce")
                if np.isfinite(value) and value > 0:
                    active = True
                    availability[secid].add(day)
        if active:
            calendar.append(day)
        if number % 250 == 0 or number == len(dates):
            print(f"Calendar: {number}/{len(dates)} days", flush=True)
    return pd.DatetimeIndex(calendar), availability, mismatched_currency


def read_candles(cache, jobs, board, start, end):
    rows = []
    for job in jobs:
        if pd.Timestamp(job["from"]) > end or pd.Timestamp(job["till"]) < start:
            continue
        prefix = f"engines/stock/markets/shares/boards/{board}/securities/{quote(job['SECID'], safe='')}/candles.json"
        for batch in cache.pages(prefix, "candles", **{
                "from": job["from"] + " 00:00:00", "till": job["till"] + " 23:59:59", "interval": 10}):
            for row in batch:
                stamp = pd.Timestamp(row["begin"])
                if not pd.Timestamp(job["from"]) <= stamp.normalize() <= pd.Timestamp(job["till"]):
                    raise RuntimeError("Candle outside its acquisition job")
                minute = stamp.hour * 60 + stamp.minute
                if start <= stamp.normalize() <= end and 600 <= minute < 1120:
                    rows.append(row)
    return pd.DataFrame(rows, columns=CANDLE_COLUMNS)


def aggregate(cache, jobs, board, calendar, full_sessions, end, epsilon):
    dates = calendar[calendar <= end]
    candles = read_candles(cache, jobs, board, dates.min(), end)
    result = summarize_candles(candles, dates, epsilon).set_index("date")
    short = ~full_sessions.reindex(result.index).to_numpy()
    result.loc[short, "valid"] = False
    result.loc[short, "invalid_reason"] = "known_short_session"
    result.loc[short, FEATURES] = np.nan
    return result


def make_windows(logs, rv, calendar, train_end, validation_end, history, max_gap_days):
    """Export only complete, contiguous common windows with next-session targets."""
    dates = pd.DatetimeIndex(logs.index)
    nodes = list(logs.columns)
    segments = calendar["segment_id"].to_numpy()
    xcols = [f"x_lag{lag:03d}_{node}" for lag in range(history - 1, -1, -1) for node in nodes]
    ycols = [f"y_log_rv_{node}" for node in nodes] + [f"y_rv_{node}" for node in nodes]
    records, audit = [], []
    values, raw = logs.to_numpy(), rv.to_numpy()
    for target in range(1, len(dates)):
        origin = target - 1
        split = "train" if dates[target] <= train_end else "validation" if dates[target] <= validation_end else "test"
        reason = ""
        left = target - history
        if left < 0:
            reason = "insufficient_history"
        elif segments[left] != segments[target]:
            reason = "market_closure_in_window"
        elif not np.isfinite(values[left:target]).all():
            reason = "invalid_input_day"
        elif not np.isfinite(values[target]).all():
            reason = "invalid_target_day"
        row = {"origin_date": dates[origin], "target_date": dates[target],
               "segment_id": int(segments[target]), "split": split}
        audit.append({**row, "kept": not reason, "reason": reason})
        if not reason:
            records.append([*row.values(), *values[left:target].ravel(),
                            *values[target], *raw[target]])
    columns = ["origin_date", "target_date", "segment_id", "split", *xcols, *ycols]
    return pd.DataFrame(records, columns=columns), pd.DataFrame(audit), xcols, ycols


def write_csv(frame, path, index=False):
    frame.to_csv(path, index=index, date_format="%Y-%m-%d", float_format="%.17g")


def prepare(args):
    raw, output = args.raw.resolve(), args.out.resolve()
    if output == raw or raw in output.parents:
        raise ValueError("Output must be outside the raw cache")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new --out")
    run = resolve_run(raw, args.run_id)
    configuration = json.loads((run / "configuration.json").read_text(encoding="utf-8"))
    if configuration.get("board") != "TQBR" or configuration.get("candle_interval") != 10:
        raise ValueError("This script expects the TQBR/10-minute downloader configuration")
    first, last = pd.Timestamp(configuration["date_from"]), pd.Timestamp(configuration["date_till"])
    train_end, validation_end, test_end = map(pd.Timestamp, (args.train_end, args.validation_end, args.test_end))
    if not first < train_end < validation_end < test_end <= last:
        raise ValueError("Require raw_start < train_end < validation_end < test_end <= raw_end")
    if not 0 < args.min_coverage <= 1 or args.n_stocks < 1 or args.history < 1 or args.max_gap_days < 1:
        raise ValueError("Invalid coverage, stock count, history, or gap limit")
    output.mkdir(parents=True, exist_ok=True)
    cache = Cache(raw)
    universe = [row["SECID"] for row in json.loads((run / "universe.json").read_text(encoding="utf-8"))["securities"]]
    metadata = read_metadata(cache, universe)
    ordinary = {secid for secid, item in metadata.items() if item["TYPE"] == "common_share"}
    calendar, availability, wrong_currency = read_calendar(cache, configuration, ordinary, test_end)
    if not len(calendar):
        raise RuntimeError("No market sessions found")
    known_short = (calendar >= "2022-03-24") & (calendar <= "2022-03-30")
    full = pd.Series(~known_short, index=calendar)
    gaps = calendar.to_series().diff().dt.days.fillna(0).astype(int)
    cal = pd.DataFrame({"date": calendar, "full_session_expected": full.to_numpy(),
                        "gap_days": gaps.to_numpy(),
                        "segment_id": (gaps > args.max_gap_days).cumsum().to_numpy()}).set_index("date")
    eligible_days = calendar[(calendar <= train_end) & full.to_numpy()]
    if not len(eligible_days):
        raise RuntimeError("No full training sessions found")
    write_csv(cal, output / "calendar.csv", index=True)
    jobs_by_security = {secid: [] for secid in universe}
    for job in json.loads((run / "candle_plan.json").read_text(encoding="utf-8"))["jobs"]:
        jobs_by_security[job["SECID"]].append(job)
    audit = []
    for number, secid in enumerate(sorted(universe), 1):
        item = dict(metadata[secid], expected_train_days=len(eligible_days),
                    valid_train_days=np.nan, train_coverage=np.nan, median_turnover_rub=np.nan,
                    selected=False, exclusion_reason="")
        issuer = str(item["EMITTER_ID"] or "").strip()
        daily_available = len(availability.get(secid, set()) & set(eligible_days))
        item["daily_train_coverage"] = daily_available / len(eligible_days)
        if secid not in ordinary:
            item["exclusion_reason"] = "not_common_share_or_unknown_type"
        elif not issuer or issuer == "0":
            item["exclusion_reason"] = "unknown_issuer"
        elif any(name == secid and day <= train_end for name, day in wrong_currency):
            item["exclusion_reason"] = "non_ruble_trades"
        elif item["daily_train_coverage"] < args.min_coverage:
            item["exclusion_reason"] = "insufficient_daily_train_history"
        else:
            days = aggregate(cache, jobs_by_security[secid], "TQBR", calendar, full, train_end, args.epsilon)
            good = days.reindex(eligible_days)["valid"].fillna(False).astype(bool)
            item["valid_train_days"] = int(good.sum())
            item["train_coverage"] = float(good.mean())
            item["median_turnover_rub"] = days.reindex(eligible_days).loc[good, "turnover"].median()
            if item["train_coverage"] < args.min_coverage:
                item["exclusion_reason"] = "insufficient_complete_10m_days"
            elif not np.isfinite(item["median_turnover_rub"]) or item["median_turnover_rub"] <= 0:
                item["exclusion_reason"] = "nonpositive_median_turnover"
        audit.append(item)
        if number % 20 == 0 or number == len(universe):
            print(f"Train-only screening: {number}/{len(universe)}", flush=True)
    audit = pd.DataFrame(audit).set_index("SECID")
    ranked = audit.loc[audit["exclusion_reason"].eq("")].sort_values(
        ["median_turnover_rub", "SHORTNAME"], ascending=[False, True], kind="stable")
    chosen, issuers = [], set()
    for secid, row in ranked.iterrows():
        issuer = str(row["EMITTER_ID"])
        if issuer in issuers:
            audit.loc[secid, "exclusion_reason"] = "another_issue_of_same_issuer"
        elif len(chosen) < args.n_stocks:
            chosen.append(secid)
            issuers.add(issuer)
            audit.loc[secid, "selected"] = True
        else:
            audit.loc[secid, "exclusion_reason"] = "outside_top_turnover_rank"
    write_csv(audit.reset_index(), output / "selection_audit.csv")
    if len(chosen) < args.n_stocks:
        raise RuntimeError(f"Only {len(chosen)} eligible stocks; see selection_audit.csv. Thresholds were not relaxed")
    nodes = audit.loc[chosen].reset_index()
    nodes.insert(0, "node_id", np.arange(len(nodes)))
    write_csv(nodes, output / "stocks.csv")
    panel = []
    for secid in chosen:
        print("Aggregating selected security: " + secid, flush=True)
        days = aggregate(cache, jobs_by_security[secid], "TQBR", calendar, full, test_end, args.epsilon)
        panel.append(days.reset_index().assign(SECID=secid))
    panel = pd.concat(panel, ignore_index=True).sort_values(["date", "SECID"])
    panel["split"] = np.where(panel["date"] <= train_end, "train",
                               np.where(panel["date"] <= validation_end, "validation", "test"))
    write_csv(panel, output / "daily_panel.csv")
    logs = panel.pivot(index="date", columns="SECID", values="log_rv").reindex(index=calendar, columns=chosen)
    rv = panel.pivot(index="date", columns="SECID", values="rv").reindex(index=calendar, columns=chosen)
    write_csv(logs, output / "log_rv_wide.csv", index=True)
    write_csv(rv, output / "rv_wide.csv", index=True)
    write_csv(logs.notna().astype(int), output / "observed_mask.csv", index=True)
    windows, window_audit, xcols, ycols = make_windows(logs, rv, cal, train_end, validation_end,
                                                    args.history, args.max_gap_days)
    write_csv(window_audit, output / "window_audit.csv")
    counts = {}
    for split in ("train", "validation", "test"):
        part = windows.loc[windows["split"].eq(split)].reset_index(drop=True)
        write_csv(part, output / (split + ".csv"))
        counts[split] = len(part)
    quality = panel.groupby(["SECID", "split"], sort=False).agg(
        market_sessions=("valid", "size"), valid_days=("valid", "sum"))
    quality["valid_fraction"] = quality["valid_days"] / quality["market_sessions"]
    write_csv(quality.reset_index(), output / "quality_by_split.csv")
    source_manifest = pd.DataFrame(list(cache.used.values())).sort_values("request_key")
    write_csv(source_manifest, output / "source_requests.csv")
    schema = {"node_order": chosen, "x_columns": xcols,
              "y_log_rv_columns": [f"y_log_rv_{node}" for node in chosen],
              "y_rv_columns": [f"y_rv_{node}" for node in chosen],
              "x_shape": ["samples", args.history, len(chosen), 1],
              "x_order": "oldest_to_newest_then_stocks_csv_node_order"}
    (output / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    config = vars(args).copy()
    config.update(raw=str(raw), out=str(output), run_id=run.name)
    ready = all(counts.values())
    manifest = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "configuration": config, "selected_stocks": chosen, "window_counts": counts,
                "ready_for_training": ready,
                "script_sha256": digest(Path(__file__).read_bytes()),
                "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
                "raw_run_file_sha256": {name: digest((run / name).read_bytes()) for name in (
                    "configuration.json", "universe.json", "candle_plan.json", "summary.json", "checkpoint.json")},
                "session_window_moscow": "10:00 inclusive to 18:40 exclusive; 52 ten-minute begins",
                "target": "next market session within-session RV; no overnight return",
                "log_transform": "log(max(rv, epsilon)); raw RV is retained",
                "known_short_session_range": ["2022-03-24", "2022-03-30"],
                "short_session_sources": SHORT_SESSION_SOURCE,
                "calendar_rule": "positive ordinary-share daily turnover; known short sessions retained but unusable",
                "limitations": ["Security specifications are acquisition-time snapshots, not historical metadata.",
                                "Fixed strict core grid can reject other shortened sessions or source outages; review calendar and quality CSVs.",
                                "No SECID stitching, corporate-event adjustment, imputation, graph construction or normalization is performed.",
                                "Previously explored 2020-2025 observations are not a new independent holdout."],
                "output_sha256": {path.name: digest(path.read_bytes()) for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Selected stocks:", ", ".join(chosen))
    print("Complete windows:", counts)
    print("Output:", output)
    if not ready:
        raise RuntimeError("At least one split has no complete windows; inspect window_audit.csv. No missing values were filled")
    return manifest


def parser():
    project = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    p = argparse.ArgumentParser(description="Build train-only-selected MOEX RV CSVs from the raw downloader cache")
    p.add_argument("--raw", type=Path, default=project / "data/raw/moex_iss_2020_2025")
    p.add_argument("--out", type=Path, default=project / "data/processed/moex_rv_v1")
    p.add_argument("--run-id", default=None)
    p.add_argument("--train-end", default="2023-12-31")
    p.add_argument("--validation-end", default="2024-12-31")
    p.add_argument("--test-end", default="2025-12-31")
    p.add_argument("--n-stocks", type=int, default=10)
    p.add_argument("--min-coverage", type=float, default=0.95)
    p.add_argument("--history", type=int, default=60)
    p.add_argument("--max-gap-days", type=int, default=7)
    p.add_argument("--epsilon", type=float, default=1e-12)
    return p


if __name__ == "__main__":
    prepare(parser().parse_args())
