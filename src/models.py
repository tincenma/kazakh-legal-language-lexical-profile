"""Comparison of text representations on a leakage-free split.

Task: predict an act's `database_section` (thematic section) from its title + description.
  - baseline 1: majority class
  - baseline 2: TF-IDF on word forms (no morphology handling)
  - method 1:   TF-IDF on suffix stems
  - method 2:   TF-IDF on character 3-5-grams
All use the same linear classifier, so differences come from the representation only.
Criteria: F1-macro (main), accuracy, fit / predict time, peak memory, number of features.
"""
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config
from src.preprocessing import analyzer_char, analyzer_stem, analyzer_wordform
from src.utils import load_json, measure, save_json

BEST_MODEL_PATH = config.MODELS_DIR / "best_model.joblib"
PREDICTIONS_PATH = config.RESULTS_DIR / "test_predictions.parquet"


def normalize_title(title: str) -> str:
    """Title key for duplicate detection: lower case, no digits, numbers or punctuation."""
    if not isinstance(title, str):
        return ""
    title = re.sub(r"[\d№N]+", " ", title.lower())
    title = re.sub(r"[^\w\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def leakage_free_split(df: pd.DataFrame, split_year: int = config.SPLIT_YEAR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Temporal split + removal of test acts whose (normalised) title also occurs in train.

    Many acts share boilerplate titles ("... күшін жою туралы" = "on repealing ..."). Without
    the title filter the model would just memorise titles it has already seen.
    """
    data = df.dropna(subset=[config.TARGET_COLUMN, "year"]).copy()
    data["label"] = data[config.TARGET_COLUMN].astype(str)
    data["title_key"] = data["title"].map(normalize_title)

    train = data[data["year"] < split_year]
    test = data[data["year"] >= split_year]
    n_test_before = len(test)
    test = test[~test["title_key"].isin(set(train["title_key"]))]
    n_removed = n_test_before - len(test)

    # keep sections that are frequent enough in train and present in test
    support = train["label"].value_counts()
    keep = set(support[support >= config.MIN_CLASS_SUPPORT].index) & set(test["label"])
    train, test = train[train["label"].isin(keep)], test[test["label"].isin(keep)]

    info = {
        "split_year": split_year,
        "train_rows": len(train), "test_rows": len(test),
        "test_removed_duplicate_titles": n_removed,
        "n_classes": len(keep),
        "train_years": [int(train["year"].min()), int(train["year"].max())] if len(train) else None,
        "test_years": [int(test["year"].min()), int(test["year"].max())] if len(test) else None,
    }
    return train, test, info


def build_models() -> dict[str, Pipeline]:
    def svm():
        return LinearSVC(C=0.5, class_weight="balanced", random_state=config.RANDOM_STATE)

    return {
        "baseline_majority": Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        "baseline_wordform_tfidf": Pipeline([
            ("vec", TfidfVectorizer(analyzer=analyzer_wordform, min_df=2, sublinear_tf=True)),
            ("clf", svm())]),
        "stem_tfidf": Pipeline([
            ("vec", TfidfVectorizer(analyzer=analyzer_stem, min_df=2, sublinear_tf=True)),
            ("clf", svm())]),
        "char_ngram_tfidf": Pipeline([
            ("vec", TfidfVectorizer(analyzer=analyzer_char, min_df=3, max_features=300_000,
                                    sublinear_tf=True)),
            ("clf", svm())]),
    }


def evaluate_model(name: str, model: Pipeline, train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, np.ndarray]:
    row = {"model": name}
    with measure(row, "fit_"):
        model.fit(train["text"], train["label"])
    with measure(row, "predict_"):
        pred = model.predict(test["text"])
    row["f1_macro"] = round(f1_score(test["label"], pred, average="macro"), 4)
    row["f1_weighted"] = round(f1_score(test["label"], pred, average="weighted"), 4)
    row["accuracy"] = round(accuracy_score(test["label"], pred), 4)
    row["n_features"] = len(model.named_steps["vec"].vocabulary_) if "vec" in model.named_steps else 0
    return row, pred


def compare_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Train every model, print and save the comparison table, keep the best model."""
    train, test, split_info = leakage_free_split(df)
    print(f"Split: {split_info}")
    if split_info["train_rows"] == 0 or split_info["test_rows"] == 0:
        raise ValueError("Empty train or test set: check SPLIT_YEAR / MIN_CLASS_SUPPORT in config.py")

    rows, predictions, models = [], {}, build_models()
    for name, model in models.items():
        print(f"  training {name}...", flush=True)
        row, pred = evaluate_model(name, model, train, test)
        rows.append(row)
        predictions[name] = pred
        print(f"    F1-macro={row['f1_macro']}  acc={row['accuracy']}  fit={row['fit_time_s']}s  "
              f"mem={row['fit_peak_mem_mb']}MB  features={row['n_features']:,}")

    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(config.RESULTS_DIR / "model_comparison.csv")
    best = table["f1_macro"].idxmax()
    print("\n" + table.to_string())
    print(f"\nBest model by F1-macro: {best}")

    joblib.dump(models[best], BEST_MODEL_PATH)
    out = test[["law_id", "title", "text", "year", "label"]].copy()
    for name, pred in predictions.items():
        out[f"pred_{name}"] = pred
    out.to_parquet(PREDICTIONS_PATH, index=False)

    report = classification_report(test["label"], predictions[best], output_dict=True, zero_division=0)
    save_json({"split": split_info, "best_model": best, "comparison": table.to_dict(orient="index"),
               "per_class_best": report}, config.RESULTS_DIR / "model_comparison.json")
    return table


def load_best_model() -> Pipeline:
    try:
        return joblib.load(BEST_MODEL_PATH)
    except FileNotFoundError:
        raise FileNotFoundError("No trained model. Run menu item 7 (Train & compare) first.") from None


def predict_text(text: str, model: Pipeline | None = None, top_k: int = 3) -> list[tuple[str, float]]:
    """Top-k predicted sections for a user text, with decision scores."""
    model = model or load_best_model()
    clf = model.named_steps["clf"]
    if hasattr(clf, "decision_function"):
        scores = model.decision_function([text])[0]
        order = np.argsort(scores)[::-1][:top_k]
        return [(clf.classes_[i], round(float(scores[i]), 3)) for i in order]
    return [(model.predict([text])[0], 1.0)]


def best_model_name() -> str | None:
    info = load_json(config.RESULTS_DIR / "model_comparison.json", default={})
    return info.get("best_model")
