"""All plots. Every function saves a PNG into reports/figures and returns its path."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # no window needed; works in console, Colab and IDEs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

# Fonts that contain the Kazakh letters ә ғ қ ң ө ұ ү һ і (needed for WordCloud).
FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"), Path("/Library/Fonts/Arial.ttf"),
]


def find_font() -> str | None:
    for path in FONT_CANDIDATES:
        if path.exists():
            return str(path)
    return None


def _save(fig, name: str) -> Path:
    path = config.FIGURES_DIR / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_loading_benchmark(bench: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].bar(["object dtypes", "optimised"], [bench["memory_object_dtypes_mb"], bench["memory_optimized_mb"]],
                color=["#999999", "#2a6f97"])
    axes[0].set(title="DataFrame memory", ylabel="MB")
    axes[1].bar(["CSV", "Parquet"], [bench["csv_file_mb"], bench["parquet_file_mb"]], color=["#999999", "#2a6f97"])
    axes[1].set(title="File size on disk", ylabel="MB")
    axes[2].bar(["CSV", "Parquet"], [bench["csv_read_time_s"], bench["parquet_read_time_s"]],
                color=["#999999", "#2a6f97"])
    axes[2].set(title="Read time", ylabel="seconds")
    return _save(fig, "01_loading_benchmark")


def plot_missing(missing: pd.DataFrame) -> Path:
    data = missing[missing["missing_pct"] > 0].sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(data))))
    ax.barh(data.index, data["missing_pct"], color="#c44e52")
    ax.set(title="Missing values per field (after placeholder cleaning)", xlabel="% of acts missing")
    return _save(fig, "02_missing_values")


def plot_acts_per_decade(docs_per_decade: dict) -> Path:
    items = sorted((int(k), v) for k, v in docs_per_decade.items() if int(k) >= 0)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar([f"{d}s" for d, _ in items], [v for _, v in items], color="#2a6f97")
    ax.set(title="Number of acts per decade of adoption", xlabel="decade", ylabel="acts")
    for i, (_, v) in enumerate(items):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    return _save(fig, "03_acts_per_decade")


def plot_vocabulary(vocab: dict) -> Path:
    names = list(vocab)
    types = [vocab[n]["types"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(names, types, color=["#999999", "#7a9cc6", "#2a6f97", "#dd8452"])
    ax.set(title="Vocabulary size (distinct units) by representation", ylabel="types")
    ax.bar_label(bars, labels=[f"{t:,}" for t in types], fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save(fig, "04_vocabulary_size")


def plot_zipf(rank_freq: dict, zipf: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, label, color in [("raw", "word forms", "#2a6f97"), ("stems", "suffix stems", "#dd8452")]:
        freqs = np.array(rank_freq[key], dtype=float)
        ranks = np.arange(1, len(freqs) + 1)
        ax.loglog(ranks, freqs, ".", ms=2, color=color, label=label, alpha=0.6)
    fit = zipf["all_raw"]
    if fit["s"] is not None:
        r = np.arange(1, len(rank_freq["raw"]) + 1)
        ax.loglog(r, np.exp(fit["log_c"]) * r ** (-fit["s"]), "k--", lw=1,
                  label=f"fit: s={fit['s']}, R²={fit['r2']}")
    ax.set(title="Zipf's law: rank vs frequency (log-log)", xlabel="rank", ylabel="frequency")
    ax.legend()
    return _save(fig, "05_zipf")


def plot_zipf_by_decade(zipf: dict) -> Path:
    decades = sorted(k for k in zipf if k.isdigit())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot([f"{d}s" for d in decades], [zipf[d]["s"] for d in decades], "o-", color="#2a6f97")
    ax.axhline(1.0, color="grey", ls=":", label="classic Zipf (s = 1)")
    ax.set(title="Zipf exponent by decade (content words)", xlabel="decade", ylabel="s")
    ax.legend()
    return _save(fig, "06_zipf_by_decade")


def plot_heaps(heaps: dict) -> Path:
    curve = np.array(heaps["curve"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve[:, 0], curve[:, 1], color="#2a6f97", label="observed")
    if heaps.get("K"):
        ax.plot(curve[:, 0], heaps["K"] * curve[:, 0] ** heaps["beta"], "k--", lw=1,
                label=f"V = {heaps['K']}·N^{heaps['beta']}")
    ax.set(title="Heaps' law: vocabulary growth (chronological order)", xlabel="tokens seen",
           ylabel="distinct word forms")
    ax.legend()
    return _save(fig, "07_heaps")


def plot_top_items(items: list, title: str, name: str, n: int = 25) -> Path:
    items = items[:n][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(items) + 1))
    ax.barh([str(w) for w, _ in items], [f for _, f in items], color="#2a6f97")
    ax.set(title=title, xlabel="frequency")
    return _save(fig, name)


def plot_wordclouds(top_by_decade: dict) -> Path | None:
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[warn] wordcloud not installed; skipping word clouds.")
        return None
    decades = sorted(top_by_decade)
    cols = min(3, len(decades))
    rows = int(np.ceil(len(decades) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    font = find_font()
    for ax, decade in zip(axes.flat, decades):
        freqs = dict(top_by_decade[decade])
        if freqs:
            wc = WordCloud(width=600, height=360, background_color="white", font_path=font,
                           colormap="viridis").generate_from_frequencies(freqs)
            ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{decade}s")
        ax.axis("off")
    for ax in list(axes.flat)[len(decades):]:
        ax.axis("off")
    return _save(fig, "10_wordclouds_by_decade")


def plot_term_history(word: str, history: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot([f"{d}s" for d in history.index], history.values, "o-", color="#2a6f97")
    ax.set(title=f"'{word}' across decades", xlabel="decade", ylabel="frequency per million tokens")
    safe = "".join(c for c in word if c.isalnum())[:30] or "term"
    return _save(fig, f"term_{safe}")


def plot_model_comparison(table: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    names = table.index.tolist()
    colors = ["#999999", "#7a9cc6", "#2a6f97", "#dd8452"][:len(names)]
    for ax, col, title in [(axes[0], "f1_macro", "F1-macro (test)"),
                           (axes[1], "fit_time_s", "Training time, s"),
                           (axes[2], "fit_peak_mem_mb", "Peak memory during training, MB")]:
        bars = ax.bar(names, table[col], color=colors)
        ax.bar_label(bars, fmt="%.3g", fontsize=8)
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, "11_model_comparison")


def plot_confusion(cm: pd.DataFrame, title: str) -> Path:
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(0.45 * len(cm) + 4, 0.45 * len(cm) + 3))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    short = [s[:30] for s in cm.index]
    ax.set_xticks(range(len(cm)), short, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cm)), short, fontsize=7)
    ax.set(title=title, xlabel="predicted", ylabel="true")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, label="share of true class")
    return _save(fig, "12_confusion_matrix")


def plot_error_by_group(errors: pd.DataFrame, column: str, title: str, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(errors.index.astype(str), errors[column], color="#c44e52")
    ax.set(title=title, ylabel="error rate")
    return _save(fig, name)


def _short(name: str, n: int = 45) -> str:
    """Shorten long official names: drop parentheses and trim."""
    name = str(name).split(" (")[0]
    return name if len(name) <= n else name[: n - 1] + "…"


def plot_sql_results(results: dict[str, pd.DataFrame], min_decade: int = 1990) -> list[Path]:
    """Charts built directly from SQL query results (loaded into pandas)."""
    results = {name: df for name, df in results.items() if len(df)}   # skip empty results
    paths = []

    if "section_growth" in results:
        df = results["section_growth"]
        df = df[df["decade"] >= min_decade]
        top = df.groupby("section")["n_acts"].sum().nlargest(8).index
        pivot = df[df["section"].isin(top)].pivot(index="decade", columns="section", values="n_acts").fillna(0)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for col in pivot.columns:
            ax.plot([f"{d}s" for d in pivot.index], pivot[col], "o-", label=_short(col, 35))
        ax.set(title="Acts per decade in the 8 largest sections (SQL: section_growth)", ylabel="acts",
               xlabel="decade (2020s = 2020-2026, still incomplete)")
        ax.legend(fontsize=7, loc="upper left")
        paths.append(_save(fig, "15_sql_section_growth"))

    if "area_share_by_decade" in results:
        df = results["area_share_by_decade"]
        df = df[df["decade"] >= min_decade]
        pivot = df.pivot(index="decade", columns="area", values="share_pct").fillna(0)
        pivot = pivot[pivot.sum().sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        bottom = np.zeros(len(pivot))
        for col in pivot.columns:
            ax.bar([f"{d}s" for d in pivot.index], pivot[col], bottom=bottom, label=_short(col, 40))
            bottom += pivot[col].values
        ax.set(title="Share of the top-10 legal relation areas by decade (SQL: area_share_by_decade)",
               ylabel="% of area links in decade")
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        paths.append(_save(fig, "16_sql_area_share"))

    if "repealed_by_form" in results:
        df = results["repealed_by_form"].nlargest(15, "n_acts").sort_values("repealed_pct")
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(df["act_form"], df["repealed_pct"], color="#c44e52")
        ax.bar_label(bars, labels=[f"{p}% of {n:,}" for p, n in zip(df["repealed_pct"], df["n_acts"])], fontsize=7)
        ax.set(title="Repeal rate of the 15 most common act forms (SQL: repealed_by_form)", xlabel="% repealed")
        paths.append(_save(fig, "17_sql_repealed_by_form"))

    if "top_bodies" in results:
        df = results["top_bodies"].head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([_short(b) for b in df["body"]], df["n_acts"], color="#2a6f97")
        ax.set(title="Most productive adopting bodies (SQL: top_bodies)", xlabel="acts")
        paths.append(_save(fig, "18_sql_top_bodies"))

    if "text_length_by_section" in results:
        df = results["text_length_by_section"].sort_values("avg_tokens")
        fig, ax = plt.subplots(figsize=(8, 0.25 * len(df) + 1))
        colors = ["#c44e52" if d < 0 else "#2a6f97" for d in df["diff_from_mean"]]
        ax.barh([_short(s, 40) for s in df["section"]], df["diff_from_mean"], color=colors)
        ax.set(title="Text length vs corpus mean by section (SQL: text_length_by_section)",
               xlabel="difference from mean, tokens")
        paths.append(_save(fig, "19_sql_text_length_by_section"))
    return paths


def draw_er_diagram(path: Path = config.DOCS_DIR / "er_diagram.png") -> Path:
    """Draw the ER diagram of sql/schema.sql with matplotlib (no external tools needed)."""
    tables = {
        "bodies": (0.05, 0.62, ["body_id PK", "name"]),
        "acts": (0.36, 0.40, ["act_id PK", "law_id", "title", "date_adopted", "year, decade",
                              "is_repealed", "legal_force", "region", "body_id FK", "section_id FK", "n_tokens"]),
        "sections": (0.05, 0.12, ["section_id PK", "name"]),
        "act_areas": (0.68, 0.66, ["act_id PK/FK", "area_id PK/FK"]),
        "areas": (0.88, 0.66, ["area_id PK", "name"]),
        "act_forms": (0.68, 0.16, ["act_id PK/FK", "form_id PK/FK"]),
        "forms": (0.88, 0.16, ["form_id PK", "name"]),
    }
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = {}
    for name, (x, y, cols) in tables.items():
        h = 0.045 * (len(cols) + 1)
        w = 0.15
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, fc="#f2f6fa", ec="#2a6f97", lw=1.5))
        ax.add_patch(plt.Rectangle((x, y + h - 0.045), w, 0.045, fc="#2a6f97", ec="#2a6f97"))
        ax.text(x + w / 2, y + h - 0.0225, name, ha="center", va="center", color="white", weight="bold")
        for i, col in enumerate(cols):
            ax.text(x + 0.008, y + h - 0.045 * (i + 1.5), col, va="center", fontsize=9)
        boxes[name] = (x, y, w, h)

    def connect(a, b, label):
        ax_, ay, aw, ah = boxes[a]
        bx, by, bw, bh = boxes[b]
        start = (ax_ + aw, ay + ah / 2) if bx > ax_ else (ax_, ay + ah / 2)
        end = (bx, by + bh / 2) if bx > ax_ else (bx + bw, by + bh / 2)
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-", color="#555", lw=1.2))
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.015, label, fontsize=8,
                ha="center", color="#555")

    connect("bodies", "acts", "1 : N")
    connect("sections", "acts", "1 : N")
    connect("acts", "act_areas", "1 : N")
    connect("act_areas", "areas", "N : 1")
    connect("acts", "act_forms", "1 : N")
    connect("act_forms", "forms", "N : 1")
    ax.set_title("ER diagram - KazakhLawCorpus (SQLite)", fontsize=13, weight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path
