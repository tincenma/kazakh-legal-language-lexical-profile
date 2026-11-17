"""Lexical Profile of Kazakh Legal Language - console entry point.

    python main.py                      interactive menu
    python main.py --run-all            run every step non-interactively
    python main.py --run-all --nrows 20000   quick run on the first 20k rows
"""
import argparse
import sqlite3
import sys

from src import config
from src import visualization as viz
from src.utils import load_json, load_state, mark_step_done

for stream in (sys.stdout, sys.stdin):          # Kazakh letters in the Windows console / pipes
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

MENU = {
    "1": "Load data (CSV -> Parquet, dtype optimization)",
    "2": "Metadata audit",
    "3": "Build / refresh SQLite database",
    "4": "Run SQL analytics",
    "5": "Lexical profile (vocabulary, Zipf, n-grams)",
    "6": "Decade comparison",
    "7": "Train & compare models (baseline vs methods)",
    "8": "Analyze a text / query",
    "9": "Error analysis",
    "10": "Full-text analysis (all steps on downloaded full acts)",
    "0": "Exit",
}

# The loaded corpus is kept here so it is read from disk only once per session.
session: dict = {"df": None, "nrows": None}


def get_corpus():
    if session["df"] is None:
        from src.data_loader import load_parquet
        print("Loading corpus from Parquet...")
        session["df"] = load_parquet()
    return session["df"]


def step_load():
    from src.data_loader import load_and_convert
    session["df"] = load_and_convert(nrows=session["nrows"])
    viz.plot_loading_benchmark(load_json(config.RESULTS_DIR / "loading_benchmark.json"))
    mark_step_done("load", rows=len(session["df"]))


def step_audit():
    from src.audit import missing_report, run_audit
    df = get_corpus()
    run_audit(df, nrows=session["nrows"])
    viz.plot_missing(missing_report(df))
    mark_step_done("audit")


def step_database():
    from src.database import build_database
    counts = build_database(get_corpus())
    path = viz.draw_er_diagram()
    print(f"ER diagram saved to {path}")
    mark_step_done("database", **counts)


def step_sql():
    from src.database import run_all_queries, summarize_findings
    from src.utils import save_json
    results = run_all_queries()
    print(f"\n{len(results)} queries executed; results saved to {config.RESULTS_DIR}")
    viz.plot_sql_results(results)
    findings = summarize_findings(results)
    save_json(findings, config.RESULTS_DIR / "sql_findings.json")
    print("\nKey findings from SQL:")
    for key, value in findings.items():
        print(f"  {key}: {value}")
    mark_step_done("sql", n_queries=len(results))


def plot_lexical(profile: dict) -> None:
    viz.plot_acts_per_decade(profile["docs_per_decade"])
    viz.plot_vocabulary(profile["vocabulary"])
    viz.plot_zipf(profile["rank_freq"], profile["zipf"])
    viz.plot_heaps(profile["heaps"])
    viz.plot_top_items(profile["top_words"], "Most frequent content words", "08_top_words")
    viz.plot_top_items(profile["top_bigrams"], "Most frequent bigrams", "09_top_bigrams")
    print("\nTop 15 content words:", ", ".join(w for w, _ in profile["top_words"][:15]))


def step_lexical():
    from src.lexical import build_profile
    plot_lexical(build_profile(get_corpus()))
    mark_step_done("lexical")


def plot_decades(profile: dict, tag: str = "") -> None:
    from src.lexical import term_history
    viz.plot_zipf_by_decade(profile["zipf"])
    viz.plot_wordclouds(profile["top_words_by_decade"])
    for shift in profile["decade_shifts"]:
        print(f"\n{shift['from']}s -> {shift['to']}s")
        print("  rising :", ", ".join(w for w, _ in shift["rising"][:10]))
        print("  falling:", ", ".join(w for w, _ in shift["falling"][:10]))
        print("  new    :", ", ".join(shift["new_terms"][:10]) or "-")
    for word in ["кеңес", "министрлік", "цифрлық"]:     # Soviet / ministry / digital
        history = term_history(word, tag)
        if history is not None:
            viz.plot_term_history(word, history)


def step_decades():
    from src.lexical import load_profile
    plot_decades(load_profile())
    mark_step_done("decades")


def step_models():
    from src.models import compare_methods
    table = compare_methods(get_corpus())
    viz.plot_model_comparison(table)
    mark_step_done("models", best=table["f1_macro"].idxmax())


def step_analyze_text():
    from src.lexical import term_history
    from src.models import predict_text
    from src.preprocessing import preprocess

    while True:
        try:
            text = input("\nEnter a Kazakh text or a single word (empty line to return): ").strip()
        except EOFError:                # input closed (Ctrl+Z / Ctrl+D): back to the menu
            print()
            return
        if not text:
            return
        if not preprocess(text, "wordform"):
            print("  no Kazakh content words found (only stop words, numbers or Latin text).")
            continue
        print("  word forms:", preprocess(text, "wordform"))
        print("  stems     :", preprocess(text, "stem"))
        print("  char 3-5  :", preprocess(text, "char")[:15], "...")
        try:
            for label, score in predict_text(text):
                print(f"  section: {label}  (score {score})")
        except FileNotFoundError as e:
            print(f"  [skip prediction] {e}")
        if len(text.split()) == 1:
            try:
                history = term_history(text)
            except FileNotFoundError as e:
                print(f"  [skip history] {e}")
                continue
            if history is None:
                print("  word not found in the top-20k vocabulary.")
            else:
                print("  frequency per million tokens by decade:")
                print("  " + history.to_string().replace("\n", "\n  "))
                print(f"  plot: {viz.plot_term_history(text, history)}")


def step_errors():
    from src.error_analysis import run_error_analysis
    run_error_analysis()
    mark_step_done("errors")


def ask_int(prompt: str, default: int, low: int, high: int) -> int:
    """Read an integer in [low, high]; Enter keeps the default."""
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()
        if not answer:
            return default
        try:
            value = int(answer)
        except ValueError:
            print(f"  please enter a whole number from {low} to {high}")
            continue
        if low <= value <= high:
            return value
        print(f"  please enter a number from {low} to {high}")


def run_fulltext_analysis() -> None:
    """Every analysis of the title + description corpus, repeated on the downloaded full texts,
    plus a comparison of the two sources on exactly the same acts."""
    import pandas as pd
    from src import fulltext as ft
    from src.error_analysis import run_error_analysis
    from src.lexical import build_profile
    from src.models import compare_methods

    fdf = ft.fulltext_corpus(get_corpus())
    try:
        print("\n=== Lexical profile of the full texts ===")
        viz.set_source("_full")
        profile = build_profile(fdf, tag="_full", min_docs=ft.FULLTEXT_MIN_DOCS, heaps_every=10_000)
        plot_lexical(profile)
        plot_decades(profile, "_full")

        print("\n=== Title + description vs full text, same acts ===")
        viz.set_source("")
        ft.compare_sources(fdf)

        print("\n=== Models on the full texts ===")
        viz.set_source("_full")
        full = compare_methods(fdf, tag="_full", min_support=ft.FULLTEXT_MIN_SUPPORT)
        viz.plot_model_comparison(full)
        run_error_analysis(tag="_full")

        print("\n=== Same models, same acts and split, on title + description ===")
        viz.set_source("_sample_title")
        short = compare_methods(fdf.assign(text=fdf["text_short"]), tag="_sample_title",
                                min_support=ft.FULLTEXT_MIN_SUPPORT)
        viz.plot_model_comparison(short)
    finally:
        viz.set_source("")
    both = pd.DataFrame({"f1_title": short["f1_macro"], "f1_full": full["f1_macro"],
                         "fit_time_title_s": short["fit_time_s"], "fit_time_full_s": full["fit_time_s"],
                         "mem_title_mb": short["fit_peak_mem_mb"], "mem_full_mb": full["fit_peak_mem_mb"]})
    both.to_csv(config.RESULTS_DIR / "fulltext_vs_title_models.csv")
    viz.plot_fulltext_models(both)
    print("\nF1-macro, same acts and split:")
    print(both.to_string())

    print("\n=== Per-act comparison ===")
    ft.run_case_study(get_corpus())


def step_fulltext():
    """1: make or extend the list of acts to collect. 2: run every analysis on the full texts."""
    from src.fulltext import N_SECTIONS, PER_CELL, load_fulltexts, make_sample
    collected = len(load_fulltexts())
    print(f"{collected} full text(s) collected so far.")
    choice = input("1 – make or extend the list of acts   2 – run the full-text analysis: ").strip()
    if choice == "1":
        per_cell = ask_int("Acts per decade and section", PER_CELL, 1, 20)
        sections = ask_int("Number of largest sections", N_SECTIONS, 1, 33)
        print(f"About {per_cell * sections * 4} acts; already collected texts stay on the list.")
        make_sample(get_corpus(), per_cell, sections)
    elif choice == "2":
        run_fulltext_analysis()
        mark_step_done("fulltext", acts=collected)
    else:
        print(f"'{choice}' is not an option.")


ACTIONS = {"1": step_load, "2": step_audit, "3": step_database, "4": step_sql, "5": step_lexical,
           "6": step_decades, "7": step_models, "8": step_analyze_text, "9": step_errors,
           "10": step_fulltext}
PIPELINE = ["1", "2", "3", "4", "5", "6", "7", "9"]


def run_step(choice: str) -> bool:
    """Run one menu action; returns False if it failed."""
    try:
        ACTIONS[choice]()
        return True
    except FileNotFoundError as e:
        print(f"[missing file] {e}")
    except PermissionError as e:
        print(f"[file in use] {e.filename}: close it (e.g. in Excel) and try again.")
    except sqlite3.Error as e:
        print(f"[database error] {e}")
    except (ValueError, KeyError) as e:
        print(f"[error] {e}")
    except KeyboardInterrupt:
        print("\n[interrupted]")
    except Exception as e:                       # last resort: keep the menu alive
        print(f"[unexpected error] {type(e).__name__}: {e}")
    return False


def show_menu():
    state = load_state()
    step_names = {"1": "load", "2": "audit", "3": "database", "4": "sql", "5": "lexical",
                  "6": "decades", "7": "models", "9": "errors", "10": "fulltext"}
    print("\n==== Kazakh Legal Lexical Profile ====")
    for key, label in MENU.items():
        done = " [done]" if step_names.get(key) in state else ""
        print(f"{key} – {label}{done}")


def interactive():
    while True:
        show_menu()
        try:
            choice = input("Choose an option: ").strip()
        except (EOFError, KeyboardInterrupt):    # Ctrl+C / Ctrl+Z at the prompt exits cleanly
            print("\nBye!")
            break
        if choice == "0":
            print("Bye!")
            break
        elif choice in ACTIONS:
            run_step(choice)
        else:
            print(f"'{choice}' is not a menu option, please enter 0-10.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-all", action="store_true", help="run the whole pipeline without the menu")
    parser.add_argument("--nrows", type=int, default=None, help="use only the first N rows (quick test)")
    args = parser.parse_args()
    if args.nrows is not None and args.nrows <= 0:
        parser.error("--nrows must be a positive number")

    config.ensure_dirs()
    session["nrows"] = args.nrows
    if args.run_all:
        for choice in PIPELINE:
            print(f"\n######## {choice}. {MENU[choice]} ########")
            if not run_step(choice):
                sys.exit(1)
        print(f"\nAll done. Figures: {config.FIGURES_DIR}\nTables: {config.RESULTS_DIR}")
    else:
        interactive()


if __name__ == "__main__":
    main()
