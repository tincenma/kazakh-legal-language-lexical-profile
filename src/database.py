"""Normalized SQLite database: building, loading and running the analytical queries."""
import re
import sqlite3
from pathlib import Path

import pandas as pd

from src import config
from src.audit import split_multi_value
from src.preprocessing import remove_stopwords, stem, tokenize


def lookup_table(values: pd.Series, id_col: str) -> pd.DataFrame:
    """Unique non-null values -> table with integer ids starting at 1."""
    unique = sorted(values.dropna().astype(str).unique())
    return pd.DataFrame({id_col: range(1, len(unique) + 1), "name": unique})


def link_table(df: pd.DataFrame, column: str, lookup: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Explode a multi-valued column into (act_id, <id_col>) pairs."""
    pairs = (df[["act_id", column]]
             .assign(tag=df[column].astype("object").map(split_multi_value))
             .explode("tag")
             .dropna(subset=["tag"]))
    name_to_id = dict(zip(lookup["name"], lookup[id_col]))
    pairs[id_col] = pairs["tag"].map(name_to_id)
    return pairs[["act_id", id_col]].drop_duplicates()


def build_database(df: pd.DataFrame, db_path: Path = config.DB_PATH) -> dict:
    """Create the schema and load acts, bodies, sections, areas, forms and the link tables."""
    df = df.reset_index(drop=True).copy()
    df["act_id"] = df.index + 1

    bodies = lookup_table(df["adopting_body"], "body_id")
    sections = lookup_table(df["database_section"], "section_id")
    areas = pd.DataFrame({"name": sorted({t for v in df["legal_relation_area"].astype("object")
                                          for t in split_multi_value(v)})})
    areas.insert(0, "area_id", range(1, len(areas) + 1))
    forms = pd.DataFrame({"name": sorted({t for v in df["act_form"].astype("object")
                                          for t in split_multi_value(v)})})
    forms.insert(0, "form_id", range(1, len(forms) + 1))

    acts = pd.DataFrame({
        "act_id": df["act_id"],
        "law_id": df["law_id"],
        "title": df["title"],
        "date_adopted": df["date_adopted"].dt.strftime("%Y-%m-%d"),
        "year": df["year"],
        "decade": df["decade"],
        "is_repealed": df["is_repealed"].astype("Int8"),
        "legal_force": df["legal_force"].astype("object"),
        "region": df["region"].astype("object"),
        "body_id": df["adopting_body"].astype("object").map(dict(zip(bodies["name"], bodies["body_id"]))),
        "section_id": df["database_section"].astype("object").map(
            dict(zip(sections["name"], sections["section_id"]))),
        "n_tokens": df["text"].map(lambda t: len(tokenize(t))),
    })

    act_areas = link_table(df, "legal_relation_area", areas, "area_id")
    act_forms = link_table(df, "act_form", forms, "form_id")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript((config.SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
        for name, table in [("bodies", bodies), ("sections", sections), ("areas", areas),
                            ("forms", forms), ("acts", acts), ("act_areas", act_areas),
                            ("act_forms", act_forms)]:
            table.to_sql(name, conn, if_exists="append", index=False, chunksize=10_000)
    counts = {name: len(t) for name, t in [("acts", acts), ("bodies", bodies), ("sections", sections),
                                           ("areas", areas), ("forms", forms),
                                           ("act_areas", act_areas), ("act_forms", act_forms)]}
    print("Rows loaded:", counts)
    return counts


def load_fulltexts_to_db(fdf: pd.DataFrame, db_path: Path = config.DB_PATH) -> int:
    """Store the cleaned full texts (menu 10) in the `fulltexts` table, linked to `acts` by act_id."""
    if not db_path.exists():
        raise FileNotFoundError("Database not found. Run menu item 3 (Build database) first.")
    with sqlite3.connect(db_path) as conn:
        ids = pd.read_sql_query("SELECT act_id, law_id FROM acts", conn)
        rows = fdf[["law_id", "text"]].merge(ids, on="law_id", how="inner")
        content = rows["text"].map(lambda t: set(remove_stopwords(tokenize(t))))
        table = pd.DataFrame({
            "act_id": rows["act_id"],
            "n_chars": rows["text"].str.len(),
            "n_tokens": rows["text"].map(lambda t: len(tokenize(t))),
            "n_content_types": content.map(len),
            "n_stem_types": content.map(lambda forms: len({stem(w) for w in forms})),
            "text": rows["text"],
        })
        conn.executescript((config.SQL_DIR / "fulltext_schema.sql").read_text(encoding="utf-8"))
        table.to_sql("fulltexts", conn, if_exists="append", index=False, chunksize=500)
    print(f"Loaded {len(table)} full texts into the fulltexts table ({db_path.name}).")
    return len(table)


def parse_queries(path: Path = config.SQL_DIR / "queries.sql") -> list[dict]:
    """Split queries.sql into a list of {name, description, sql}."""
    text = path.read_text(encoding="utf-8")
    queries = []
    for block in re.split(r"^-- name:\s*", text, flags=re.M)[1:]:
        lines = block.splitlines()
        name = lines[0].strip()
        description = ""
        if len(lines) > 1 and lines[1].startswith("-- description:"):
            description = lines[1].split(":", 1)[1].strip()
        sql = "\n".join(l for l in lines[1:] if not l.startswith("--")).strip()
        queries.append({"name": name, "description": description, "sql": sql})
    return queries


def run_query(sql: str, db_path: Path = config.DB_PATH, params=None) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError("Database not found. Run menu item 3 (Build database) first.")
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def summarize_findings(results: dict[str, pd.DataFrame]) -> dict:
    """Turn the SQL result tables into the key facts used in the report."""
    results = {name: df for name, df in results.items() if len(df)}   # small samples can give empty results
    findings = {}
    if "acts_per_decade" in results:
        d = results["acts_per_decade"]
        total = d["n_acts"].sum()
        findings["acts_since_2010_pct"] = round(100 * d.loc[d["decade"] >= 2010, "n_acts"].sum() / total, 1)
        findings["acts_before_1990"] = int(d.loc[d["decade"] < 1990, "n_acts"].sum())
    if "section_growth" in results:
        g = results["section_growth"]
        last = g[(g["decade"] == g["decade"].max()) & g["change"].notna()]
        if len(last):
            up, down = last.nlargest(1, "change").iloc[0], last.nsmallest(1, "change").iloc[0]
            findings["largest_growth_last_decade"] = {"section": up["section"], "change": int(up["change"])}
            findings["largest_decline_last_decade"] = {"section": down["section"], "change": int(down["change"])}
    if "area_share_by_decade" in results:
        s = results["area_share_by_decade"]
        top_area = s.groupby("area")["n"].sum().idxmax()
        share = s[s["area"] == top_area].set_index("decade")["share_pct"]
        findings["top_area"] = {"area": top_area, "share_by_decade_pct": {int(k): float(v) for k, v in share.items()}}
    if "repealed_by_form" in results:
        r = results["repealed_by_form"].iloc[0]
        findings["most_repealed_form"] = {"form": r["act_form"], "repealed_pct": float(r["repealed_pct"]),
                                          "n_acts": int(r["n_acts"])}
    if "top_bodies" in results and "acts_per_decade" in results:
        b = results["top_bodies"].iloc[0]
        findings["top_body"] = {"body": b["body"].split(" (")[0], "n_acts": int(b["n_acts"]),
                                "share_pct": round(float(100 * b["n_acts"] / results["acts_per_decade"]["n_acts"].sum()), 1)}
    return findings


def run_all_queries(db_path: Path = config.DB_PATH, save: bool = True,
                    queries_path: Path = config.SQL_DIR / "queries.sql") -> dict[str, pd.DataFrame]:
    """Run every query in a .sql file (queries.sql by default), print a preview, save each as CSV."""
    results = {}
    for q in parse_queries(queries_path):
        try:
            df = run_query(q["sql"], db_path)
        except sqlite3.Error as e:
            print(f"[error] {q['name']}: {e}")
            continue
        results[q["name"]] = df
        print(f"\n=== {q['name']}: {q['description']} ({len(df)} rows)")
        print(df.head(8).to_string(max_colwidth=50))
        if save:
            df.to_csv(config.RESULTS_DIR / f"sql_{q['name']}.csv", index=False)
    return results
