"""Metadata quality audit: missing values, date formats, multi-valued fields, placeholders."""
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src import config
from src.utils import load_json, save_json

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def split_multi_value(value) -> list[str]:
    """Split a multi-valued field such as legal_relation_area.

    Values are joined with ", ", but some tags contain commas themselves, e.g.
    "Нормативтік құқықтық актілер: дайындау, қабылдау, өзгерту". A new tag always starts
    with an upper-case letter, so a piece starting in lower case is glued back to the
    previous tag.
    """
    if not isinstance(value, str) or not value.strip():
        return []
    tags: list[str] = []
    for piece in value.split(","):
        piece = re.sub(r"\s+", " ", piece).strip()
        if not piece:
            continue
        if tags and not piece[0].isupper():
            tags[-1] = f"{tags[-1]}, {piece}"
        else:
            tags.append(piece)
    return tags


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    return (pd.DataFrame({"missing": missing, "missing_pct": (missing / len(df) * 100).round(2)})
            .sort_values("missing_pct", ascending=False))


def placeholder_report(csv_path: Path, nrows: int | None = None) -> dict:
    """Count placeholder strings in the raw CSV (before they were turned into NaN)."""
    counts: dict[str, dict[str, int]] = {}
    # keep_default_na=False so pandas does not silently swallow "null", "N/A", "" etc.
    for chunk in pd.read_csv(csv_path, chunksize=config.CHUNKSIZE, dtype=str, nrows=nrows,
                             keep_default_na=False):
        for col in chunk.columns:
            values = chunk[col].str.strip()
            values = values[values != ""]          # empty cells are plain missing values
            found = values[values.isin(config.PLACEHOLDERS)].value_counts()
            for placeholder, n in found.items():
                counts.setdefault(col, {})
                counts[col][placeholder] = counts[col].get(placeholder, 0) + int(n)
    return counts


def date_format_report(csv_path: Path, nrows: int | None = None) -> dict:
    """How many raw date values are ISO, empty or in some other format."""
    report = {}
    for col in config.DATE_COLUMNS:
        try:
            raw = pd.read_csv(csv_path, usecols=[col], dtype=str, nrows=nrows)[col]
        except ValueError:          # column not in file
            continue
        non_null = raw.dropna().str.strip()
        iso = non_null.str.match(ISO_DATE)
        report[col] = {
            "null": int(raw.isna().sum()),
            "iso_yyyy_mm_dd": int(iso.sum()),
            "other_format": int((~iso).sum()),
            "other_examples": non_null[~iso].unique()[:5].tolist(),
        }
    return report


def multi_value_report(df: pd.DataFrame, column: str, top: int = 15) -> dict:
    tag_lists = df[column].astype("object").map(split_multi_value)
    n_tags = tag_lists.map(len)
    counter = Counter(tag for tags in tag_lists for tag in tags)
    return {
        "acts_with_values": int((n_tags > 0).sum()),
        "acts_with_multiple_values": int((n_tags > 1).sum()),
        "max_values_per_act": int(n_tags.max()),
        "mean_values_per_act": round(float(n_tags[n_tags > 0].mean()), 2) if (n_tags > 0).any() else 0,
        "unique_values": len(counter),
        "top_values": counter.most_common(top),
    }


def run_audit(df: pd.DataFrame, nrows: int | None = None) -> dict:
    """Full audit; prints a summary and saves reports/results/audit.json."""
    missing = missing_report(df)
    years = df["year"].dropna()
    report = {
        "rows": len(df),
        "duplicate_law_id": int(df["law_id"].duplicated().sum()),
        "duplicate_titles": int(df["title"].duplicated().sum()),
        "short_generic_titles": df["title"].value_counts().head(10).to_dict(),
        "year_range": [int(years.min()), int(years.max())] if len(years) else None,
        "acts_per_decade": {int(k): int(v) for k, v in df["decade"].value_counts().sort_index().items()},
        "missing": missing.to_dict(orient="index"),
        "date_formats": date_format_report(config.CSV_PATH, nrows) if config.CSV_PATH.exists() else {},
        "placeholders": placeholder_report(config.CSV_PATH, nrows) if config.CSV_PATH.exists() else {},
        "multi_value": {col: multi_value_report(df, col) for col in config.MULTI_VALUE_COLUMNS if col in df},
        "cleaning_applied": load_json(config.RESULTS_DIR / "cleaning_log.json", default={}),
    }
    save_json(report, config.RESULTS_DIR / "audit.json")
    missing.to_csv(config.RESULTS_DIR / "missing_values.csv")

    print(f"Rows: {report['rows']:,}   years: {report['year_range']}")
    print(f"Duplicate law_id: {report['duplicate_law_id']}   duplicate titles: {report['duplicate_titles']:,}")
    print("\nMissing values (top 10):")
    print(missing.head(10).to_string())
    print("\nPlaceholders found in raw CSV:")
    for col, found in report["placeholders"].items():
        print(f"  {col:<28} {found}")
    print(f"\nCleaning applied at load time: {report['cleaning_applied']}")
    for col, info in report["multi_value"].items():
        print(f"\n{col}: {info['unique_values']} unique values, "
              f"{info['acts_with_multiple_values']:,} acts with several values "
              f"(max {info['max_values_per_act']})")
    return report
