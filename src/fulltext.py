"""Full texts of a stratified sample of acts, and their comparison with title + description.

The corpus holds only titles and short descriptions. The full texts of a stratified sample are
downloaded from adilet.zan.kz (src/fetch_pdfs.py) and saved as data/fulltext/<law_id>.txt.

Workflow:
  1. make_sample()            -> data/fulltext/sample_to_collect.csv (which acts to fetch)
  2. python -m src.fetch_pdfs -> data/fulltext/<law_id>.txt (or copy a text by hand, UTF-8)
  3. fulltext_corpus()        -> the sampled acts with their cleaned full text as `text`, which
                                 main.run_fulltext_analysis() sends through every analysis step
  4. compare_sources()        -> title + description vs full text on the same acts (figure 21)
  5. run_case_study()         -> per-act comparison (figure 20)
"""
import random
import re
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import STOPWORDS, fix_lookalikes, remove_stopwords, stem, tokenize
from src.utils import df_memory_mb, file_size_mb, save_json

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
          "then run the full-text analysis (menu 10 -> 2).")
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


# The PDF repeats the act's type and date in Russian on one line: "Закон от 07.06.1999 № 389".
RUSSIAN_HEADER = re.compile(r"^[А-ЯЁа-яё ]+ от \d{2}\.\d{2}\.\d{4}.*$", re.MULTILINE)
FULLTEXT_MIN_DOCS = 30        # acts per decade needed to compare decades in the full-text sample
FULLTEXT_MIN_SUPPORT = 20     # acts per section needed in the training part of the full-text split


def clean_fulltext(text: str) -> str:
    """Remove the Russian header line and dash rulers left over from the PDF layout, and join words
    hyphenated across a line break ("бағ-" + "дарлама" -> "бағдарлама")."""
    text = RUSSIAN_HEADER.sub("", fix_lookalikes(text))
    text = re.sub(r"([а-яәғқңөұүһі])-\n([а-яәғқңөұүһі])", r"\1\2", text, flags=re.IGNORECASE)
    text = re.sub(r"^-{5,}$", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{2,}", "\n", text).strip()


def fulltext_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """The acts that have a downloaded full text, with all their metadata.

    `text` becomes the cleaned full text; the original title + description is kept in `text_short`,
    so both sources can be analysed on exactly the same acts.
    """
    texts = load_fulltexts()
    if not texts:
        raise FileNotFoundError(f"No full texts in {config.FULLTEXT_DIR}. Download them first "
                                "(python -m src.fetch_pdfs).")
    sub = df[df["law_id"].isin(texts)].copy()
    sub["text_short"] = sub["text"]
    sub["text"] = sub["law_id"].map(lambda law_id: clean_fulltext(texts[law_id]))
    sub = sub[sub["text"].str.len() > 0]
    print(f"Full-text corpus: {len(sub)} acts, "
          f"{sub['text'].str.len().sum() / 1e6:.1f}M characters, decades: "
          f"{ {int(k): int(v) for k, v in sub['decade'].value_counts().sort_index().items()} }")
    return sub


def _source_stats(token_lists: list[list[str]], heaps_points: int = 60) -> tuple[dict, list, Counter]:
    """Vocabulary, stemming, Zipf and Heaps figures for one text source."""
    from src.lexical import derive_char_ngrams, derive_stems, fit_heaps, fit_zipf

    raw, content = Counter(), Counter()
    total = sum(len(t) for t in token_lists)
    step = max(1, total // heaps_points)
    curve, seen = [], 0
    for tokens in token_lists:
        raw.update(tokens)
        content.update(t for t in tokens if t not in STOPWORDS)
        before, seen = seen, seen + len(tokens)
        if seen // step > before // step:
            curve.append((seen, len(raw)))
    curve.append((seen, len(raw)))
    stems = derive_stems(content)
    zipf, heaps = fit_zipf(content), fit_heaps(curve)
    stats = {
        "acts": len(token_lists),
        "tokens": total,
        "tokens_per_act": round(total / max(1, len(token_lists)), 1),
        "wordform_types": len(content),
        "stem_types": len(stems),
        "stem_reduction_pct": round(100 * (1 - len(stems) / max(1, len(content))), 1),
        "char_ngram_types": len(derive_char_ngrams(content)),
        "hapax_pct": round(100 * sum(1 for f in content.values() if f == 1) / max(1, len(content)), 1),
        "zipf_s": zipf["s"], "zipf_r2": zipf["r2"],
        "heaps_K": heaps["K"], "heaps_beta": heaps["beta"],
    }
    return stats, curve, content


def _equal_size_vocab(token_lists: list[list[str]], n_tokens: int, seed: int = config.RANDOM_STATE) -> dict:
    """Vocabulary after the first n_tokens content tokens, acts taken in a random order.

    Vocabulary grows with corpus size, so the full texts are cut to as many tokens as the titles
    have; only then is the comparison of vocabulary size and stemming fair.
    """
    order = list(range(len(token_lists)))
    random.Random(seed).shuffle(order)
    kept: list[str] = []
    for i in order:
        kept.extend(t for t in token_lists[i] if t not in STOPWORDS)
        if len(kept) >= n_tokens:
            break
    kept = kept[:n_tokens]
    forms = set(kept)
    stems = {stem(t) for t in forms}
    return {"tokens": len(kept), "wordform_types": len(forms), "stem_types": len(stems),
            "stem_reduction_pct": round(100 * (1 - len(stems) / max(1, len(forms))), 1)}


def compare_sources(fdf: pd.DataFrame) -> pd.DataFrame:
    """Title + description vs full text on the same acts: vocabulary, stemming, Zipf, Heaps, top words."""
    ordered = fdf.sort_values("date_adopted")
    sources = {"title + description": ordered["text_short"], "full text": ordered["text"]}
    tokens = {name: [tokenize(t) for t in col] for name, col in sources.items()}

    rows, curves, content = {}, {}, {}
    for name, token_lists in tokens.items():
        rows[name], curves[name], content[name] = _source_stats(token_lists)
    n_short = sum(1 for doc in tokens["title + description"] for t in doc if t not in STOPWORDS)
    equal = _equal_size_vocab(tokens["full text"], n_short)
    rows["full text, cut to title size"] = {**{k: None for k in rows["full text"]}, "acts": None, **equal}

    top = {name: {w for w, _ in c.most_common(100)} for name, c in content.items()}
    overlap = len(top["title + description"] & top["full text"])
    table = pd.DataFrame(rows).T
    table.to_csv(config.RESULTS_DIR / "fulltext_vs_title.csv", encoding="utf-8-sig")
    save_json({"table": table.to_dict(orient="index"),
               "top100_shared_words": overlap,
               "top_words": {name: c.most_common(30) for name, c in content.items()},
               "heaps_curves": curves},
              config.RESULTS_DIR / "fulltext_vs_title.json")

    from src import visualization as viz
    viz.plot_source_comparison(table, curves)
    print("\nTitle + description vs full text (same acts):")
    print(table.to_string())
    print(f"Top-100 content words shared by both sources: {overlap}")
    return table


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
        full_text = clean_fulltext(full_text)      # same cleaning as the rest of the full-text analysis
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

    print(f"Per-act comparison on {summary['acts']} acts:")
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


FULLTEXT_PARQUET_PATH = config.PROCESSED_DIR / "fulltext.parquet"
# Frequent Russian function words; in a Kazakh act they signal untranslated Russian passages.
RUSSIAN_MARKERS = {"и", "в", "на", "по", "с", "о", "от", "для", "не", "что", "или", "к", "из",
                   "года", "статья", "статьи", "пункт", "является", "который", "которые"}


def store_fulltext_parquet(fdf: pd.DataFrame) -> dict:
    """Save the cleaned full-text corpus as Parquet with compact dtypes; compare with the .txt files.

    The same memory-aware handling as the main corpus (requirement 11): category / int32 dtypes
    instead of object / int64, and one columnar Parquet file instead of hundreds of text files.
    """
    paths = [config.FULLTEXT_DIR / f"{law_id}.txt" for law_id in fdf["law_id"]]
    start = time.perf_counter()
    for path in paths:
        path.read_text(encoding="utf-8-sig")
    txt_read_s = time.perf_counter() - start

    plain = pd.DataFrame({
        "law_id": fdf["law_id"].astype(str),
        "decade": fdf["decade"].astype("int64"),
        "section": fdf[config.TARGET_COLUMN].astype(str),
        "n_chars": fdf["text"].str.len().astype("int64"),
        "text": fdf["text"],
    })
    compact = plain.assign(decade=plain["decade"].astype("int16"),
                           section=plain["section"].astype("category"),
                           n_chars=plain["n_chars"].astype("int32"))
    FULLTEXT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    compact.to_parquet(FULLTEXT_PARQUET_PATH, index=False)
    start = time.perf_counter()
    pd.read_parquet(FULLTEXT_PARQUET_PATH)
    parquet_read_s = time.perf_counter() - start

    meta = plain.drop(columns="text")
    result = {
        "acts": len(plain),
        "txt_files_mb": round(sum(p.stat().st_size for p in paths) / 1024 ** 2, 2),
        "parquet_mb": file_size_mb(FULLTEXT_PARQUET_PATH),
        "txt_read_s": round(txt_read_s, 3),
        "parquet_read_s": round(parquet_read_s, 3),
        "metadata_memory_mb_default_dtypes": df_memory_mb(meta),
        "metadata_memory_mb_optimized_dtypes": df_memory_mb(compact.drop(columns="text")),
        "total_memory_mb_optimized": df_memory_mb(compact),
    }
    save_json(result, config.RESULTS_DIR / "fulltext_storage.json")
    print(f"\nFull texts: {result['txt_files_mb']} MB in {len(paths)} .txt files -> "
          f"{result['parquet_mb']} MB Parquet; read {result['txt_read_s']} s vs {result['parquet_read_s']} s; "
          f"metadata memory {result['metadata_memory_mb_default_dtypes']} -> "
          f"{result['metadata_memory_mb_optimized_dtypes']} MB with category/int dtypes")
    return result


def audit_fulltexts(fdf: pd.DataFrame) -> dict:
    """Data quality of the full-text sample: coverage, length, very short texts, duplicates, Russian."""
    sample = pd.read_csv(config.FULLTEXT_SAMPLE_PATH, encoding="utf-8-sig") \
        if config.FULLTEXT_SAMPLE_PATH.exists() else pd.DataFrame(columns=["law_id"])
    raw = load_fulltexts()
    tokens = fdf["text"].map(tokenize)
    n_tokens = tokens.map(len)
    russian_share = tokens.map(lambda t: sum(w in RUSSIAN_MARKERS for w in t) / max(1, len(t)))
    headers_removed = sum(bool(RUSSIAN_HEADER.search(raw[law_id])) for law_id in fdf["law_id"] if law_id in raw)
    hyphen_joins = sum(len(re.findall(r"[а-яәғқңөұүһі]-\n[а-яәғқңөұүһі]", raw[law_id], re.IGNORECASE))
                       for law_id in fdf["law_id"] if law_id in raw)
    report = {
        "acts_on_list": len(sample),
        "acts_with_text": len(fdf),
        "coverage_pct": round(100 * len(fdf) / max(1, len(sample)), 1),
        "texts_not_in_corpus": sorted(set(raw) - set(fdf["law_id"])),
        "tokens": {"min": int(n_tokens.min()), "median": float(n_tokens.median()),
                   "mean": round(float(n_tokens.mean()), 1), "max": int(n_tokens.max())},
        "very_short_acts_lt_50_tokens": int((n_tokens < 50).sum()),
        "duplicate_texts": int(fdf["text"].duplicated().sum()),
        "russian_header_lines_removed": int(headers_removed),
        "hyphenated_line_breaks_joined": int(hyphen_joins),
        "acts_with_russian_passages_gt_2pct": int((russian_share > 0.02).sum()),
        "acts_per_decade": {int(k): int(v) for k, v in fdf["decade"].value_counts().sort_index().items()},
    }
    save_json(report, config.RESULTS_DIR / "fulltext_audit.json")
    print("\nFull-text audit:")
    for key, value in report.items():
        print(f"  {key:<38} {value}")
    return report
