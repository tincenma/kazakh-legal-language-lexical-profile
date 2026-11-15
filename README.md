# Lexical Profile of Kazakh Legal Language

Final project for **Programming for Data Analysis and Database**
(MSc *Applied Artificial Intelligence*, group **AAI-2605**). Cluster 3: Kazakh Legal Data Analytics, topic #17.

> **Goal:** study the vocabulary, n-grams and terminology of Kazakh legal texts and how they change across decades.
> **Focus:** Zipf's law, how stemming changes vocabulary size, and a decade-by-decade comparison.

**Team**

| Member | Responsibilities |
|---|---|
| Adylbekova Zhanel | _TBD_ |
| Amazholova Dilnaz | _TBD_ |
| Ilyassov Temirlan | _TBD_ |

> Each member's contribution must also be stated in the report and the presentation.

---

## 1. Dataset

**KazakhLawCorpus-clean** (Hugging Face):
<https://huggingface.co/datasets/Arailym-tleubayeva/KazakhLawCorpus-clean>

- A CSV of about **336 MB** with Kazakh legal acts: full text plus metadata (title, issuing body, adoption date, `legal_relation_area`, and so on).
- The raw file is **not committed** because of its size. Section 4 explains how to download it.
- Known quirks the audit has to handle: missing values, mixed date formats, the multi-valued `legal_relation_area` field, and placeholder values.

---

## 2. Project structure

```
kazakh-legal-language-lexical-profile/
├── main.py                  # entry point: interactive console menu
├── requirements.txt         # pinned library versions
├── README.md
├── src/
│   ├── config.py            # paths, constants (pathlib)
│   ├── data_loader.py       # chunked CSV reading, dtype optimisation, CSV → Parquet, memory measurement
│   ├── audit.py             # metadata audit: missing values, date formats, multi-valued fields, placeholders
│   ├── database.py          # normalised SQLite schema, loading, running SQL → pandas
│   ├── preprocessing.py     # Kazakh tokenisation, stop words, suffix stemmer, char n-grams
│   ├── lexical.py           # vocabulary size, frequencies, Zipf fit, n-grams, decade comparison
│   ├── models.py            # baseline vs methods, leakage-free split, metrics, timing/memory
│   ├── error_analysis.py    # misclassification inspection
│   └── visualization.py     # Matplotlib plots, WordClouds
├── sql/
│   ├── schema.sql           # DDL for the normalised schema
│   └── queries.sql          # 8+ analytical queries
├── data/
│   ├── raw/                 # original CSV (git-ignored)
│   ├── processed/           # Parquet, SQLite .db (git-ignored)
│   └── stopwords_kk.txt     # Kazakh stop-word list
├── docs/
│   └── er_diagram.png       # ER diagram
├── reports/
│   ├── figures/             # generated plots
│   ├── results/             # metrics tables (.csv / .json)
│   ├── report.pdf           # written report
│   └── presentation.pptx    # defense slides
└── notebooks/
    └── eda.ipynb            # exploratory analysis (optional)
```

---

## 3. Pipeline

### 3.1 Loading large data (requirement 11)
- Read the CSV with `pandas.read_csv(chunksize=...)`.
- Optimise dtypes: `category` for low-cardinality metadata, `datetime64` for dates, and downcast numeric columns.
- Measure memory **before and after** with `DataFrame.memory_usage(deep=True)`.
- Convert to **Parquet** and compare CSV vs Parquet on load time, file size and RAM.

### 3.2 Metadata audit
- Missing values per field (count and %).
- Date formats found in the data, unparseable dates, and the range of adoption years.
- Multi-valued `legal_relation_area`: split on its separator and count values per act.
- Placeholder values (such as `-`, `нет`, `null`, empty strings) and how each is handled.
- Output: `reports/results/audit.json` plus plots.

### 3.3 Relational database (requirement 10)
A normalised **SQLite** schema:

| Table | Content |
|---|---|
| `acts` | act_id, title, adoption_date, year, decade, body_id, text length |
| `bodies` | body_id, name (issuing authority) |
| `areas` | area_id, name (legal relation area) |
| `act_areas` | act_id, area_id: the **many-to-many** link |

The ER diagram is in `docs/er_diagram.png`. `sql/queries.sql` holds at least 8 queries that use
`JOIN`, `GROUP BY ... HAVING`, subqueries/**CTE**s and **window functions**
(`RANK`, `LAG`, running totals by decade, and similar). Each result is loaded into pandas with `pd.read_sql_query`.

### 3.4 Kazakh text preprocessing
Kazakh is agglutinative, so one lemma can appear in dozens of surface forms. Three text representations are compared:
1. **Word forms**: lower-cased tokens after stop-word removal.
2. **Suffix stemming**: a rule-based stripper for Kazakh plural, case, possessive and verbal suffixes.
3. **Character n-grams**: sub-word units (for example 3–5 characters) that capture morphology without rules.

### 3.5 Lexical profile (core of the topic)
- Vocabulary size and type/token ratio per representation, showing **how much stemming reduces vocabulary**.
- **Zipf's law**: a rank–frequency log–log plot with a fitted exponent, overall and per decade.
- Most frequent unigrams, bigrams and trigrams (NLTK), and legal terminology.
- **Decade comparison**: terms that appear, disappear or change in frequency; word clouds per decade.

### 3.6 Method comparison (requirement 9 and cluster tasks)
- **Task:** classify an act's `legal_relation_area` from its text (multi-valued labels handled explicitly).
- **Baseline:** most-frequent class, plus a word-form BoW model.
- **Methods:** TF-IDF on stemmed tokens vs TF-IDF on character n-grams, using a linear classifier.
- **Leakage-free split:** split by **adoption date** (train on older acts, test on newer ones) and **drop titles that appear in both sets**.
- **Criteria:** **F1-macro**, training and inference time, peak memory (`tracemalloc`) and vocabulary size.
- **Error analysis:** confusion between areas, typical misclassified acts, and the effect of short or noisy texts.

### 3.7 User interaction
The console menu lets the user type a query or a piece of text and see:
- its preprocessed tokens and stems,
- the predicted legal relation area,
- term frequencies by decade for a chosen word.

---

## 4. Installation and running

**Requirements:** Python 3.10 or newer.

```bash
git clone <repo-url>
```

```bash
cd kazakh-legal-language-lexical-profile
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```
(on Linux/macOS: `source .venv/bin/activate`)

```bash
pip install -r requirements.txt
```

**Get the data.** Download the CSV from the Hugging Face page above into `data/raw/`, or let the program fetch it (menu item 1).

**Run:**

```bash
python main.py
```

### Console menu

```
==== Kazakh Legal Lexical Profile ====
1 – Load data (CSV → Parquet, dtype optimisation)
2 – Metadata audit
3 – Build / refresh SQLite database
4 – Run SQL analytics
5 – Lexical profile (vocabulary, Zipf, n-grams)
6 – Decade comparison
7 – Train & compare models (baseline vs methods)
8 – Analyse a text / query
9 – Error analysis
0 – Exit
```

All results are saved to `reports/figures/` and `reports/results/`.
Google Colab, VS Code and PyCharm Community Edition are supported.

---

## 5. How the course requirements are met

| # | Requirement | Where |
|---|---|---|
| 1 | At least 5 functions besides `main` | every module in `src/` |
| 2 | Lists / dicts | token lists, frequency dicts, config and result dicts |
| 3 | `if / elif / else` | menu dispatch, audit rules, preprocessing rules |
| 4 | Loops | chunk iteration, menu `while` loop, per-decade loops |
| 5 | File I/O | CSV/Parquet/JSON/TXT read and write in `data/` and `reports/` |
| 6 | `try / except` | missing files, invalid menu input, date parsing, DB errors |
| 7 | UI | interactive console menu (`main.py`) |
| 8 | Modularity | `src/` package; `os`, `pathlib`, `datetime`, `json`, `time` |
| 9 | Approach comparison | baseline vs stemming vs char n-grams; CSV vs Parquet |
| 10 | DB and SQL | SQLite, ER diagram, 8+ queries with JOIN/CTE/window functions |
| 11 | Data volume | 336 MB CSV, chunksize, dtype optimisation, Parquet |

---

## 6. Results

_To be filled in after the experiments: memory and timing tables, vocabulary reduction from stemming, Zipf exponents per decade, and F1-macro for each method._

---

## 7. Deliverables

- [ ] Source code in this GitHub repository, with a meaningful commit history
- [ ] `requirements.txt` with pinned versions
- [ ] Data link (Hugging Face) or processed files
- [ ] Written report: data, methods, results with metric tables, limitations, conclusions, references
- [ ] Presentation (`.pptx` / `.pdf`) uploaded to Moodle **2–3 days before the exam**
- [ ] Each member's contribution stated in the report and the slides

## 8. References

- Tleubayeva A. *KazakhLawCorpus-clean*, Hugging Face Datasets.
- Zipf G. K. *Human Behavior and the Principle of Least Effort*, 1949.
- Bird S., Klein E., Loper E. *Natural Language Processing with Python* (NLTK).
