"""Full-text case study on a small stratified sample of acts.

The corpus holds only titles and short descriptions. To check how well they represent an act,
the full texts of a small stratified sample are downloaded from adilet.zan.kz (src/fetch_pdfs.py)
and saved as data/fulltext/<law_id>.txt. This module then compares the full text with the
title + description of the same act: length, vocabulary, overlap, and the section the best
model predicts from each.

Workflow:
  1. make_sample()      -> data/fulltext/sample_to_collect.csv (which acts to fetch)
  2. python -m src.fetch_pdfs -> data/fulltext/<law_id>.txt (or copy a text by hand, UTF-8)
  3. run_case_study()   -> reports/results/fulltext_case_study.csv + .json, two figures
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import remove_stopwords, stem, tokenize
from src.utils import save_json

PER_CELL = 3                 # acts per (decade, section) cell
N_SECTIONS = 8               # the largest sections
DECADES = [1990, 2000, 2010, 2020]
COLLECTORS = ["Temirlan", "Dilnaz", "Zhanel"]   # rows are split among the team


def kazakh_url(url: str) -> str:
    """The dataset links to the Russian page; the Kazakh text is under /kaz/."""
    return url.replace("/rus/docs/", "/kaz/docs/") if isinstance(url, str) else url


def make_sample(df: pd.DataFrame, per_cell: int = PER_CELL, n_sections: int = N_SECTIONS) -> pd.DataFrame:
    """Stratified list of acts to collect: per_cell acts per decade in the n_sections largest sections.

    Every act gets one fixed random key, drawn over the whole corpus in law_id order. Taking the
    first acts by that key in each cell means a larger list always CONTAINS a smaller one: texts
    already collected stay on the list when per_cell or n_sections grows.
    """
    data = df.dropna(subset=["adilet_url", "decade", config.TARGET_COLUMN]).sort_values("law_id").copy()
    data["key"] = np.random.default_rng(config.RANDOM_STATE).random(len(data))
    data["section"] = data[config.TARGET_COLUMN].astype(str)
    top_sections = data["section"].value_counts().head(n_sections).index
    data = data[data["section"].isin(top_sections) & data["decade"].isin(DECADES)]
    sample = data.sort_values("key").groupby(["decade", "section"], observed=True).head(per_cell)

    collected = set(load_fulltexts())
    out = pd.DataFrame({
        "law_id": sample["law_id"],
        "decade": sample["decade"].astype(int),
        "section": sample["section"],
        "title": sample["title"],
        "url_kaz": sample["adilet_url"].map(kazakh_url),
    }).sort_values(["decade", "section", "law_id"]).reset_index(drop=True)
    out["collected"] = out["law_id"].isin(collected)
    # stable per act (CRC of the law_id), so nobody's assignments change when the list grows
    out["collector"] = [COLLECTORS[zlib.crc32(law_id.encode()) % len(COLLECTORS)] for law_id in out["law_id"]]

    config.FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.FULLTEXT_SAMPLE_PATH, index=False, encoding="utf-8-sig")
    per_person = out.groupby("collector").size().to_dict()
    print(f"{len(out)} acts to collect ({out['collected'].sum()} already collected) -> {config.FULLTEXT_SAMPLE_PATH}")
    print(f"Split: {per_person}")
    print("Download the texts with  python -m src.fetch_pdfs  (see data/fulltext/README.md), "
          "then run the case study.")
    return out


def load_fulltexts(folder: Path | None = None) -> dict[str, str]:
    """All collected texts as {law_id: text}; empty or unreadable files are skipped."""
    folder = folder or config.FULLTEXT_DIR
    if not folder.exists():
        return {}
    texts = {}
    for path in sorted(folder.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8-sig").strip()
        except UnicodeDecodeError:
            print(f"[warn] {path.name} is not UTF-8; save it as UTF-8 and run again.")
            continue
        if text:
            texts[path.stem] = text
    return texts


def compare_act(title_text: str, full_text: str) -> dict:
    """How the title + description of one act relates to its full text."""
    short = remove_stopwords(tokenize(title_text))
    full = remove_stopwords(tokenize(full_text))
    short_stems, full_stems = {stem(t) for t in short}, {stem(t) for t in full}
    return {
        "short_tokens": len(short),
        "full_tokens": len(full),
        "length_ratio": round(len(full) / max(1, len(short)), 1),
        "full_types": len(set(full)),
        "full_stems": len(full_stems),
        "stem_reduction_pct": round(100 * (1 - len(full_stems) / max(1, len(set(full)))), 1),
        # share of the title's (stemmed) content words that also occur in the body
        "title_coverage_pct": round(100 * len(short_stems & full_stems) / max(1, len(short_stems)), 1),
        "jaccard_stems": round(len(short_stems & full_stems) / max(1, len(short_stems | full_stems)), 3),
    }


def run_case_study(df: pd.DataFrame) -> pd.DataFrame:
    """Compare every collected full text with its act's title + description."""
    texts = load_fulltexts()
    if not texts:
        raise FileNotFoundError(
            f"No full texts in {config.FULLTEXT_DIR}. Run the sample step first, then save each act "
            "as <law_id>.txt.")
    meta = df.set_index("law_id")
    unknown = [law_id for law_id in texts if law_id not in meta.index]
    if unknown:
        print(f"[warn] skipped {len(unknown)} file(s) whose name is not a law_id in the corpus: {unknown[:5]}")

    try:
        from src.models import load_best_model
        model = load_best_model()
    except FileNotFoundError:
        model = None
        print("[info] no trained model yet (menu item 7); section predictions are skipped.")

    rows = []
    for law_id, full_text in texts.items():
        if law_id not in meta.index:
            continue
        act = meta.loc[law_id]
        row = {"law_id": law_id, "decade": act.get("decade"),
               "section": str(act.get(config.TARGET_COLUMN)), **compare_act(act["text"], full_text)}
        if model is not None:
            row["pred_from_title"] = model.predict([act["text"]])[0]
            row["pred_from_fulltext"] = model.predict([full_text])[0]
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("None of the collected files matched a law_id in the corpus.")

    summary = {
        "acts": len(table),
        "median_short_tokens": float(table["short_tokens"].median()),
        "median_full_tokens": float(table["full_tokens"].median()),
        "median_length_ratio": float(table["length_ratio"].median()),
        "mean_title_coverage_pct": round(float(table["title_coverage_pct"].mean()), 1),
        "mean_stem_reduction_pct_fulltext": round(float(table["stem_reduction_pct"].mean()), 1),
    }
    if model is not None:
        summary["title_accuracy"] = round(float((table["pred_from_title"] == table["section"]).mean()), 3)
        summary["fulltext_accuracy"] = round(float((table["pred_from_fulltext"] == table["section"]).mean()), 3)
        summary["same_prediction_pct"] = round(
            100 * float((table["pred_from_title"] == table["pred_from_fulltext"]).mean()), 1)

    table.to_csv(config.RESULTS_DIR / "fulltext_case_study.csv", index=False, encoding="utf-8-sig")
    save_json(summary, config.RESULTS_DIR / "fulltext_case_study.json")
    plot_case_study(table)

    print(f"Full-text case study on {summary['acts']} acts:")
    for key, value in summary.items():
        print(f"  {key:<34} {value}")
    if summary["acts"] < 20:
        print("  (fewer than 20 acts: treat these numbers as indicative only)")
    return table


def plot_case_study(table: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(table["short_tokens"], table["full_tokens"], color="#2a6f97")
    axes[0].set_yscale("log")
    axes[0].set(title="Title + description vs full text", xlabel="content tokens, title + description",
                ylabel="content tokens, full text (log)")
    axes[1].hist(table["title_coverage_pct"], bins=np.arange(0, 105, 10), color="#2a6f97", edgecolor="white")
    axes[1].set(title="Title words found in the full text", xlabel="% of title stems in the body",
                ylabel="acts")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "20_fulltext_case_study.png", dpi=110)
    plt.close(fig)
