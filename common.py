import json
import re
import pandas as pd
import xmltodict

from collections import defaultdict
from pathlib import Path
from typing import Callable, Any


def collect_results(
    path: Path,
    matcher: Callable[[Path], dict[str, Any] | None],
    renew_cache: bool = False,
):
    cache = path / "cache.csv"

    if renew_cache:
        cache.unlink(missing_ok=True)

    if cache.is_file():
        return pd.read_csv(cache)

    records = []
    for result_dir in path.iterdir():
        if result_dir.is_dir():
            metadata = matcher(result_dir)
            if metadata is not None:
                records.extend(collect_result_single(result_dir, metadata))

    df = pd.DataFrame.from_records(records)
    df.to_csv(cache, index=False)

    return df


def collect_result_single(path: Path, metadata: dict[str, Any]):
    records = []
    for region_path in path.iterdir():
        if region_path.is_dir():
            records.append(collect_result_region(region_path, metadata))
    return records


def compute_quantiles(
    s: pd.Series,
    remove_lower_outliers: bool = False,
    remove_upper_outliers: bool = False,
    suffix: str = "",
):
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if remove_lower_outliers:
        lower_bound = s.quantile(0.25) - 1.5 * iqr
        s = s[(s >= lower_bound)]
    if remove_upper_outliers:
        upper_bound = s.quantile(0.75) + 1.5 * iqr
        s = s[(s <= upper_bound)]
    return {
        f"avg{suffix}": s.mean(),
        f"p0{suffix}": s.min(),
        f"p25{suffix}": s.quantile(0.25),
        f"p50{suffix}": s.median(),
        f"p75{suffix}": s.quantile(0.75),
        f"p90{suffix}": s.quantile(0.9),
        f"p95{suffix}": s.quantile(0.95),
        f"p99{suffix}": s.quantile(0.99),
        f"p100{suffix}": s.max(),
    }


def collect_result_region(region_path: Path, metadata: dict[str, Any]):
    record = {"path": region_path.as_posix(), "region": region_path.name, **metadata}

    # Parse histograms file
    procedures = set()
    histograms_file = region_path / "histograms.json"
    proc_pattern = re.compile(r"procedures\.([\w\d]+)")
    with open(histograms_file, "r") as f:
        data = json.load(f)
        total = 0

        def get_procedures_and_count(name):
            nonlocal total
            record[name] = data[name]["NUM_SAMPLES"]
            for proc, count in data[name]["HISTOGRAM"].items():
                if count > 0:
                    match = proc_pattern.search(proc)
                    if match:
                        procedures.add(match.group(1))
                        record[f"{name}.{match.group(1)}"] = count
                total += count

        get_procedures_and_count("rejected")
        get_procedures_and_count("aborted")
        get_procedures_and_count("unexpected")
        get_procedures_and_count("completed")
        record["total"] = total

    found_raw_file = False

    # Parse other files
    for file in region_path.iterdir():
        if file.is_file():
            # Parse config file
            if file.name.endswith("config.xml"):
                with open(file, "r") as f:
                    config = xmltodict.parse(f.read())["configuration"]
                    record.update(
                        {
                            "scalefactor": int(config["scalefactor"]),
                            "terminals": int(config["terminals"]),
                            "time": int(config["works"]["work"]["time"]),
                            "rate": int(config["works"]["work"]["rate"]),
                        }
                    )
                    if "warmup" in config["works"]["work"]:
                        record["warmup"] = int(config["works"]["work"]["warmup"])
                    else:
                        record["warmup"] = 0

            # Parse summary file
            if file.name.endswith("summary.json"):
                with open(file, "r") as f:
                    summary = json.load(f)
                    latency = summary["Latency Distribution"]
                    record["throughput"] = summary["Goodput (requests/second)"]
                    if "avg" not in record:
                        record.update(
                            {
                                "avg": latency["Average Latency (microseconds)"] / 1000,
                                "p0": latency["Minimum Latency (microseconds)"] / 1000,
                                "p25": latency["25th Percentile Latency (microseconds)"]
                                / 1000,
                                "p50": latency["Median Latency (microseconds)"] / 1000,
                                "p75": latency["75th Percentile Latency (microseconds)"]
                                / 1000,
                                "p90": latency["90th Percentile Latency (microseconds)"]
                                / 1000,
                                "p95": latency["95th Percentile Latency (microseconds)"]
                                / 1000,
                                "p99": latency["99th Percentile Latency (microseconds)"]
                                / 1000,
                                "p100": latency["Maximum Latency (microseconds)"]
                                / 1000,
                            }
                        )

            # Parse procedure files
            for proc in procedures:
                if file.name.endswith(f"{proc}.csv"):
                    df = pd.read_csv(file)
                    df = df[df["Average Latency (millisecond)"] > 0].median()
                    record[f"throughput.{proc}"] = df["Throughput (requests/second)"]
                    if f"avg.{proc}" not in record:
                        record.update(
                            {
                                f"avg.{proc}": df["Average Latency (millisecond)"],
                                f"p0.{proc}": df["Minimum Latency (millisecond)"],
                                f"p25.{proc}": df[
                                    "25th Percentile Latency (millisecond)"
                                ],
                                f"p50.{proc}": df["Median Latency (millisecond)"],
                                f"p75.{proc}": df[
                                    "75th Percentile Latency (millisecond)"
                                ],
                                f"p90.{proc}": df[
                                    "90th Percentile Latency (millisecond)"
                                ],
                                f"p95.{proc}": df[
                                    "95th Percentile Latency (millisecond)"
                                ],
                                f"p99.{proc}": df[
                                    "99th Percentile Latency (millisecond)"
                                ],
                                f"p100.{proc}": df["Maximum Latency (millisecond)"],
                            }
                        )
                    break

            # Parse raw file
            if file.name.endswith("raw.csv"):
                found_raw_file = True
                df = pd.read_csv(file)
                record.update(
                    compute_quantiles(
                        df["Latency (microseconds)"] / 1000.0,
                        remove_lower_outliers=True,
                    )
                )
                for proc in procedures:
                    record.update(
                        compute_quantiles(
                            df[df["Transaction Name"] == proc]["Latency (microseconds)"]
                            / 1000.0,
                            remove_lower_outliers=True,
                            suffix=f".{proc}",
                        )
                    )

            # Parse error file
            if file.name.endswith("errors.csv"):
                df = pd.read_csv(file)
                error_procs = df["transaction"].unique()
                for proc in error_procs:
                    proc_df = df[df.transaction == proc]
                    record.update(
                        {
                            f"ood_index_page.{proc}": proc_df[
                                proc_df.validation == "index"
                            ]["count"].sum(),
                            f"ood_table.{proc}": proc_df[proc_df.validation == "table"][
                                "count"
                            ].sum(),
                            f"ood_tuple.{proc}": proc_df[proc_df.validation == "tuple"][
                                "count"
                            ].sum(),
                            f"other_aborts.{proc}": proc_df[
                                (proc_df.deadlock == False)
                                & (proc_df.validation.isna())
                            ]["count"].sum(),
                            f"deadlock.{proc}": proc_df[proc_df.deadlock == True][
                                "count"
                            ].sum(),
                        }
                    )
                record.update(
                    {
                        "ood_index_page": df[df.validation == "index"]["count"].sum(),
                        "ood_table": df[df.validation == "table"]["count"].sum(),
                        "ood_tuple": df[df.validation == "tuple"]["count"].sum(),
                        "other_aborts": df[
                            (df.deadlock == False) & (df.validation.isna())
                        ]["count"].sum(),
                        "deadlock": df[df.deadlock == True]["count"].sum(),
                    }
                )

    if not found_raw_file:
        print(
            f'WARNING: Raw file not found in "{region_path}", fallbacked to values in summary file.'
        )

    # Previous version of benchbase incorrectly measure the transactions during warming up,
    # so we need to re-calculate the throughput of results before fix
    timestamp = pd.to_datetime(
        metadata.get("suffix", ""), format="%Y%m%d-%H%M%S", errors="coerce"
    )
    if timestamp is not pd.NaT and timestamp < pd.to_datetime("2024-02-16 16:00:00"):
        record["throughput"] = record["completed"] / (record["time"] + record["warmup"])

    return record


def normalize_region_name(df):
    """Previous version of benchbase uses different region names, so we need to normalize them."""
    df["region"].replace(
        {
            "1-us-east-1": "1-us-east-1-0",
            "2-us-east-2": "2-us-east-2-0",
            "3-us-west-1": "3-us-west-1-0",
        },
        inplace=True,
    )


def scale_lightness(rgb, scale_l):
    """Scale the lightness of the given RGB color.

    Source: https://stackoverflow.com/a/60562502
    """
    import colorsys

    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)
