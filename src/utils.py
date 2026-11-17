"""Small shared helpers: JSON persistence, program state, timing and memory measurement."""
import json
import time
import tracemalloc
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import STATE_PATH


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        print(f"[warn] {path.name} is corrupted, ignoring it.")
        return default


def load_state() -> dict:
    """Which pipeline steps have been completed, and when."""
    return load_json(STATE_PATH, default={})


def mark_step_done(step: str, **info) -> None:
    state = load_state()
    state[step] = {"done_at": datetime.now().isoformat(timespec="seconds"), **info}
    save_json(state, STATE_PATH)


def tagged(path: Path, tag: str = "") -> Path:
    """Output path for one text source: tag '' = title + description, '_full' = full texts."""
    return path.with_name(f"{path.stem}{tag}{path.suffix}")


def df_memory_mb(df: pd.DataFrame) -> float:
    return round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2)


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1024 ** 2, 2)


@contextmanager
def measure(result: dict, key_prefix: str = ""):
    """Record wall time (s) and peak Python memory (MB) of the wrapped block into `result`."""
    tracemalloc.start()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result[f"{key_prefix}time_s"] = round(elapsed, 3)
        result[f"{key_prefix}peak_mem_mb"] = round(peak / 1024 ** 2, 2)
