"""Downloading, chunked CSV loading, dtype optimisation and CSV vs Parquet benchmark."""
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.utils import df_memory_mb, file_size_mb, save_json


def download_csv(url: str = config.CSV_URL, dest: Path = config.CSV_PATH) -> Path:
    """Download the corpus CSV unless it is already present."""
    if dest.exists():
        print(f"CSV already present: {dest} ({file_size_mb(dest)} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {dest} (about 337 MB)")

    def progress(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r  {pct:3d}%", end="", flush=True)

    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    tmp.rename(dest)
    print("\n  done.")
    return dest


def clean_placeholders(chunk: pd.DataFrame) -> pd.DataFrame:
    """Turn placeholder strings (ЖОҚ, НЕТ, undefined, '-', ...) into real NaN."""
    obj_cols = chunk.select_dtypes(include="object").columns
    for col in obj_cols:
        stripped = chunk[col].str.strip()
        chunk[col] = stripped.where(~stripped.isin(config.PLACEHOLDERS), np.nan)
    return chunk


def clean_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fix invalid values found by the audit; returns the cleaned frame and a log of changes.

    - database_section: a few acts carry bare numeric codes ("3700", "3800") instead of a
      section name. They match no real section, so they become missing.
    - multi-valued and text fields: collapse repeated whitespace ("...).  Арнаулы").
    """
    log = {}
    if "database_section" in df:
        numeric = df["database_section"].str.fullmatch(r"\d+", na=False)
        log["database_section_numeric_codes"] = df.loc[numeric, "database_section"].value_counts().to_dict()
        df.loc[numeric, "database_section"] = np.nan
    for col in config.MULTI_VALUE_COLUMNS + config.TEXT_COLUMNS:
        if col in df:
            before = df[col].copy()
            df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()
            log[f"{col}_whitespace_fixed"] = int((before.fillna("") != df[col].fillna("")).sum())
    return df, log


def read_csv_chunked(path: Path = config.CSV_PATH, chunksize: int = config.CHUNKSIZE,
                     nrows: int | None = None) -> pd.DataFrame:
    """Read the CSV chunk by chunk, cleaning placeholders in each chunk."""
    chunks = []
    total = 0
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize, dtype=str,
                                          keep_default_na=True, nrows=nrows)):
        chunks.append(clean_placeholders(chunk))
        total += len(chunk)
        print(f"\r  chunk {i + 1}: {total:,} rows", end="", flush=True)
    print()
    return pd.concat(chunks, ignore_index=True)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to compact dtypes: datetimes, booleans, categories, small ints."""
    df = df.copy()
    for col in config.DATE_COLUMNS:
        if col in df:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    for col in config.BOOL_COLUMNS:
        if col in df:
            df[col] = df[col].map({"t": True, "f": False, "True": True, "False": False}).astype("boolean")
    if "source_id" in df:
        df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce", downcast="integer")
    if "state_registry_number" in df:
        df["state_registry_number"] = pd.to_numeric(df["state_registry_number"], errors="coerce",
                                                    downcast="float")
    for col in config.CATEGORY_COLUMNS:
        if col in df:
            df[col] = df[col].astype("category")
    # derived columns used throughout the analysis
    if "date_adopted" in df:
        df["year"] = df["date_adopted"].dt.year.astype("Int16")
        df["decade"] = (df["year"] // 10 * 10).astype("Int16")
    df["text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
    return df


def benchmark_formats(df: pd.DataFrame, csv_path: Path = config.CSV_PATH,
                      parquet_path: Path = config.PARQUET_PATH, nrows: int | None = None) -> dict:
    """Compare CSV and Parquet: file size, read time and in-memory size."""
    results = {}

    start = time.perf_counter()
    naive = pd.read_csv(csv_path, nrows=nrows, low_memory=False)   # default pandas behaviour, all object
    results["csv_read_time_s"] = round(time.perf_counter() - start, 2)
    results["csv_naive_memory_mb"] = df_memory_mb(naive)
    results["csv_file_mb"] = file_size_mb(csv_path)
    del naive

    df.to_parquet(parquet_path, index=False, compression="snappy")
    start = time.perf_counter()
    pq = pd.read_parquet(parquet_path)
    results["parquet_read_time_s"] = round(time.perf_counter() - start, 2)
    results["parquet_memory_mb"] = df_memory_mb(pq)
    results["parquet_file_mb"] = file_size_mb(parquet_path)
    return results


def load_and_convert(nrows: int | None = None) -> pd.DataFrame:
    """Full step 1: read chunks, optimise dtypes, save Parquet, record the measurements."""
    config.ensure_dirs()
    if not config.CSV_PATH.exists():
        download_csv()

    print("Reading CSV in chunks...")
    raw = read_csv_chunked(nrows=nrows)
    mem_before = df_memory_mb(raw)
    raw, cleaning_log = clean_values(raw)
    save_json(cleaning_log, config.RESULTS_DIR / "cleaning_log.json")
    print(f"Cleaning: {cleaning_log}")

    optimized = optimize_dtypes(raw)
    # `text` is a new derived column, so compare like with like
    mem_after = df_memory_mb(optimized.drop(columns=["text", "year", "decade"]))
    del raw

    report = {
        "rows": len(optimized),
        "columns": len(optimized.columns),
        "memory_object_dtypes_mb": mem_before,
        "memory_optimized_mb": mem_after,
        "memory_reduction_pct": round(100 * (1 - mem_after / mem_before), 1),
        **benchmark_formats(optimized, nrows=nrows),
    }
    save_json(report, config.RESULTS_DIR / "loading_benchmark.json")
    print_report(report)
    return optimized


def load_parquet() -> pd.DataFrame:
    """Load the processed corpus; tells the user what to do if step 1 was not run."""
    if not config.PARQUET_PATH.exists():
        raise FileNotFoundError("Parquet file not found. Run menu item 1 (Load data) first.")
    return pd.read_parquet(config.PARQUET_PATH)


def print_report(report: dict) -> None:
    for key, value in report.items():
        print(f"  {key:<28} {value}")
