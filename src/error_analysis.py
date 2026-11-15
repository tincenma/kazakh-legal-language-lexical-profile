"""Error analysis of the best model: confused section pairs, errors by text length and year."""
import pandas as pd
from sklearn.metrics import confusion_matrix

from src import config
from src.models import PREDICTIONS_PATH, best_model_name
from src.preprocessing import tokenize
from src.utils import save_json
from src import visualization as viz


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError("No predictions found. Run menu item 7 (Train & compare) first.")
    return pd.read_parquet(PREDICTIONS_PATH)


def top_confusions(df: pd.DataFrame, pred_col: str, n: int = 15) -> pd.DataFrame:
    wrong = df[df["label"] != df[pred_col]]
    return (wrong.groupby(["label", pred_col]).size().rename("count")
            .sort_values(ascending=False).head(n).reset_index())


def error_rate_by(df: pd.DataFrame, pred_col: str, group: pd.Series) -> pd.DataFrame:
    wrong = (df["label"] != df[pred_col])
    return pd.DataFrame({"n": group.value_counts().sort_index(),
                         "error_rate": wrong.groupby(group, observed=True).mean().round(3)})


def run_error_analysis(model: str | None = None, n_examples: int = 20) -> dict:
    df = load_predictions()
    model = model or best_model_name() or "char_ngram_tfidf"
    pred_col = f"pred_{model}"
    if pred_col not in df:
        raise KeyError(f"No predictions for model {model!r}. Available: "
                       f"{[c[5:] for c in df.columns if c.startswith('pred_')]}")

    labels = sorted(df["label"].unique())
    cm = pd.DataFrame(confusion_matrix(df["label"], df[pred_col], labels=labels), index=labels, columns=labels)
    cm.to_csv(config.RESULTS_DIR / "confusion_matrix.csv")
    viz.plot_confusion(cm, f"Confusion matrix - {model}")

    confusions = top_confusions(df, pred_col)
    n_tokens = df["text"].map(lambda t: len(tokenize(t)))
    length_bins = pd.cut(n_tokens, bins=[0, 5, 10, 20, 40, 10_000],
                         labels=["1-5", "6-10", "11-20", "21-40", "40+"], include_lowest=True)
    by_length = error_rate_by(df, pred_col, length_bins)
    by_year = error_rate_by(df, pred_col, df["year"])
    viz.plot_error_by_group(by_length, "error_rate", "Error rate by text length (tokens)", "13_errors_by_length")
    viz.plot_error_by_group(by_year, "error_rate", "Error rate by year of adoption (test set)", "14_errors_by_year")

    # sample of misclassified acts, and how often the word-form and char n-gram models disagree
    examples = df[df["label"] != df[pred_col]].sample(min(n_examples, (df["label"] != df[pred_col]).sum()),
                                                     random_state=config.RANDOM_STATE)
    examples[["law_id", "title", "label", pred_col]].to_csv(config.RESULTS_DIR / "error_examples.csv", index=False)
    disagreement = {}
    if "pred_baseline_wordform_tfidf" in df and "pred_char_ngram_tfidf" in df:
        word_ok = df["pred_baseline_wordform_tfidf"] == df["label"]
        char_ok = df["pred_char_ngram_tfidf"] == df["label"]
        disagreement = {"only_wordform_correct": int((word_ok & ~char_ok).sum()),
                        "only_char_correct": int((~word_ok & char_ok).sum()),
                        "both_correct": int((word_ok & char_ok).sum()),
                        "both_wrong": int((~word_ok & ~char_ok).sum())}

    summary = {
        "model": model,
        "test_rows": len(df),
        "error_rate": round(float((df["label"] != df[pred_col]).mean()), 4),
        "top_confusions": confusions.to_dict(orient="records"),
        "error_by_length": by_length.reset_index(names="length").to_dict(orient="records"),
        "wordform_vs_char": disagreement,
    }
    save_json(summary, config.RESULTS_DIR / "error_analysis.json")

    print(f"Model: {model}   error rate: {summary['error_rate']:.1%} on {len(df):,} test acts")
    print("\nMost frequent confusions (true -> predicted):")
    print(confusions.to_string(index=False, max_colwidth=45))
    print("\nError rate by text length:")
    print(by_length.to_string())
    if disagreement:
        print(f"\nWord-form vs char n-gram model: {disagreement}")
    return summary
