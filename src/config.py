"""Project-wide paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
MODELS_DIR = PROCESSED_DIR / "models"
SQL_DIR = ROOT / "sql"
DOCS_DIR = ROOT / "docs"

CSV_URL = (
    "https://huggingface.co/datasets/Arailym-tleubayeva/KazakhLawCorpus-clean"
    "/resolve/main/data/kazakhlawcorpus_kk_only.csv"
)
# Set KLP_CSV_PATH to use a CSV stored somewhere else (e.g. on Google Drive in Colab).
CSV_PATH = Path(os.environ.get("KLP_CSV_PATH", RAW_DIR / "kazakhlawcorpus_kk_only.csv"))
PARQUET_PATH = PROCESSED_DIR / "corpus.parquet"
DB_PATH = PROCESSED_DIR / "kazakh_law.db"
STOPWORDS_PATH = DATA_DIR / "stopwords_kk.txt"
STATE_PATH = RESULTS_DIR / "state.json"

CHUNKSIZE = 50_000

# Values that mean "missing" in this corpus (ЖОҚ = "none" in Kazakh, НЕТ = "none" in Russian).
PLACEHOLDERS = ["ЖОҚ", "Жоқ", "жоқ", "НЕТ", "Нет", "нет", "undefined", "null", "None",
                "N/A", "n/a", "NaN", "nan", "-", "--", "---", ""]

DATE_COLUMNS = ["date_adopted", "date_modified", "date_published", "ministry_registration_date"]
BOOL_COLUMNS = ["is_archive", "is_repealed"]
CATEGORY_COLUMNS = ["legal_force", "region", "place_of_adoption", "database_section",
                    "act_form", "adopting_body", "regulatory_body", "developer_body"]
MULTI_VALUE_COLUMNS = ["legal_relation_area", "act_form"]
TEXT_COLUMNS = ["title", "description"]

# Classification task: predict database_section from title + description.
TARGET_COLUMN = "database_section"
MIN_CLASS_SUPPORT = 200       # drop rare sections from the classification task
SPLIT_YEAR = 2018             # train: adopted before this year, test: this year and later
RANDOM_STATE = 42


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes into."""
    for path in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, RESULTS_DIR, MODELS_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)
