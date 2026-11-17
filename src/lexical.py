"""Lexical profile: vocabulary size per representation, Zipf's and Heaps' laws, n-grams, decades."""
import itertools
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from nltk.util import ngrams

from src import config
from src.preprocessing import KAZAKH_LETTERS, STOPWORDS, char_ngrams, fix_lookalikes, stem, tokenize
from src.utils import load_json, save_json, tagged

MIN_DOCS_PER_DECADE = 300     # decades with fewer acts are too small to compare reliably
PROFILE_PATH = config.RESULTS_DIR / "lexical_profile.json"
TERM_DECADE_PATH = config.RESULTS_DIR / "term_by_decade.parquet"



# A capitalized word right before an administrative unit word is a place name:
# "Нұра ауданының мәслихаты", "Шиелі кенті", "Достық ауылдық округі".
ADMIN_UNITS = r"(?:аудан|облыс|қала|кент|ауыл|село|округ)\w*"
TOPONYM_RE = re.compile(rf"\b([А-ЯЁӘҒҚҢӨҰҮҺІ][{KAZAKH_LETTERS}]+(?:-[{KAZAKH_LETTERS}]+)?)\s+{ADMIN_UNITS}")
NOT_TOPONYMS = {"қазақстан", "республика", "республикасы", "республикасының"}
TOPONYMS_PATH = config.RESULTS_DIR / "toponyms.txt"


def extract_toponyms(df: pd.DataFrame, min_count: int = 3, tag: str = "") -> set[str]:
    """Place names (stems) found in titles, descriptions, adopting bodies and place of adoption.

    Some villages are named after common words ("Еңбек ауылдық округі" - "labour"), so a
    candidate is kept only if the corpus almost always writes it capitalized.
    """
    counts = Counter()
    for col in ["title", "description", "adopting_body", "place_of_adoption"]:
        if col not in df:
            continue
        for value in df[col].dropna().astype(str):
            for name in TOPONYM_RE.findall(fix_lookalikes(value)):
                counts[stem(name.lower())] += 1
    candidates = {t for t, c in counts.items() if c >= min_count and t not in NOT_TOPONYMS}

    upper, lower = Counter(), Counter()
    word_re = re.compile(rf"(?<![.!?]\s)\b([{KAZAKH_LETTERS}А-ЯЁӘҒҚҢӨҰҮҺІ][{KAZAKH_LETTERS}]+)")
    for text in df["text"].dropna():
        for i, w in enumerate(word_re.findall(fix_lookalikes(text))):
            if i == 0:
                continue                       # sentence start is always capitalized
            s = stem(w.lower())
            if s in candidates:
                (upper if w[0].isupper() else lower)[s] += 1
    # Institution names are capitalized too ("Еңбек ... министрлігі"), so a real place name must
    # almost never appear in lower case: at most 1 lower-case use per 10 capitalized ones.
    toponyms = {t for t in candidates if lower[t] <= 0.1 * upper[t]}
    tagged(TOPONYMS_PATH, tag).write_text("\n".join(sorted(toponyms)), encoding="utf-8")
    return toponyms


def without_toponyms(counts: Counter, toponyms: set[str]) -> Counter:
    return Counter({w: f for w, f in counts.items() if stem(w) not in toponyms})


def count_corpus(df: pd.DataFrame, heaps_every: int = 20_000) -> dict:
    """One pass over the corpus collecting every counter we need.

    Returns raw-token counts, content-word counts per decade, bigram/trigram counts per
    decade and the Heaps curve (vocabulary size vs tokens seen, in chronological order).
    """
    raw = Counter()
    words_by_decade: dict[int, Counter] = {}
    bigrams_by_decade: dict[int, Counter] = {}
    trigrams_by_decade: dict[int, Counter] = {}
    docs_by_decade: Counter = Counter()
    heaps = []
    seen_tokens = 0

    ordered = df.sort_values("date_adopted")[["decade", "text"]]
    for decade, text in ordered.itertuples(index=False):
        decade = int(decade) if pd.notna(decade) else -1
        tokens = tokenize(text)
        raw.update(tokens)
        content = [t for t in tokens if t not in STOPWORDS]

        words_by_decade.setdefault(decade, Counter()).update(content)
        bigrams_by_decade.setdefault(decade, Counter()).update(ngrams(content, 2))
        trigrams_by_decade.setdefault(decade, Counter()).update(ngrams(content, 3))
        docs_by_decade[decade] += 1

        before = seen_tokens
        seen_tokens += len(tokens)
        if seen_tokens // heaps_every > before // heaps_every:
            heaps.append((seen_tokens, len(raw)))

    heaps.append((seen_tokens, len(raw)))
    return {"raw": raw, "words": words_by_decade, "bigrams": bigrams_by_decade,
            "trigrams": trigrams_by_decade, "docs": docs_by_decade, "heaps": heaps}


def derive_stems(word_counts: Counter) -> Counter:
    """Stem counts derived from word-form counts (stem each word type once)."""
    stems = Counter()
    for word, freq in word_counts.items():
        stems[stem(word)] += freq
    return stems


def derive_char_ngrams(word_counts: Counter) -> Counter:
    grams = Counter()
    for word, freq in word_counts.items():
        for g in char_ngrams(word):
            grams[g] += freq
    return grams


def vocab_stats(counts: Counter) -> dict:
    tokens = sum(counts.values())
    types = len(counts)
    hapax = sum(1 for f in counts.values() if f == 1)
    return {"types": types, "tokens": tokens,
            "ttr": round(types / tokens, 5) if tokens else 0,
            "hapax": hapax, "hapax_pct": round(100 * hapax / types, 1) if types else 0}


def vocabulary_comparison(raw: Counter, content: Counter) -> pd.DataFrame:
    """Effect of each preprocessing step on vocabulary size."""
    rows = {
        "word forms (all)": vocab_stats(raw),
        "word forms (no stop words)": vocab_stats(content),
        "suffix stems": vocab_stats(derive_stems(content)),
        "char 3-5-grams": vocab_stats(derive_char_ngrams(content)),
    }
    table = pd.DataFrame(rows).T
    base = table.loc["word forms (no stop words)", "types"]
    table["types_vs_wordforms_pct"] = (100 * table["types"] / base).round(1)
    return table


def fit_zipf(counts: Counter, min_freq: int = 2) -> dict:
    """Fit log f = log C - s * log r by least squares (hapax tail excluded)."""
    freqs = np.array(sorted((f for f in counts.values() if f >= min_freq), reverse=True), dtype=float)
    if len(freqs) < 10:
        return {"s": None, "r2": None, "n_types": len(freqs)}
    ranks = np.arange(1, len(freqs) + 1)
    x, y = np.log(ranks), np.log(freqs)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"s": round(-slope, 3), "log_c": round(intercept, 3), "r2": round(r2, 4), "n_types": len(freqs)}


def fit_heaps(points: list[tuple[int, int]]) -> dict:
    """Fit V = K * N^beta on the Heaps curve."""
    pts = np.array([p for p in points if p[0] > 0], dtype=float)
    if len(pts) < 3:
        return {"K": None, "beta": None}
    beta, log_k = np.polyfit(np.log(pts[:, 0]), np.log(pts[:, 1]), 1)
    return {"K": round(math.exp(log_k), 2), "beta": round(beta, 3)}


def term_decade_matrix(words_by_decade: dict[int, Counter], top_n: int = 20_000) -> pd.DataFrame:
    """Frequency per million tokens of the top-N terms in every decade."""
    total = Counter()
    for c in words_by_decade.values():
        total.update(c)
    vocab = [w for w, _ in total.most_common(top_n)]
    data = {}
    for decade, counts in sorted(words_by_decade.items()):
        if decade < 0:
            continue
        n = sum(counts.values()) or 1
        data[str(decade)] = [counts.get(w, 0) / n * 1e6 for w in vocab]   # parquet needs str columns
    return pd.DataFrame(data, index=vocab).round(2)


def decade_shifts(words_by_decade: dict[int, Counter], docs: Counter, top: int = 20,
                  min_count: int = 20, min_docs: int = MIN_DOCS_PER_DECADE) -> list[dict]:
    """For each pair of consecutive decades: terms that grew or declined most (smoothed log ratio)."""
    decades = sorted(d for d in words_by_decade if d >= 0 and docs[d] >= min_docs)
    shifts = []
    for prev, cur in zip(decades, decades[1:]):
        a, b = words_by_decade[prev], words_by_decade[cur]
        na, nb = sum(a.values()), sum(b.values())
        scores = {}
        for w in set(a) | set(b):
            if a[w] + b[w] < min_count:
                continue
            scores[w] = math.log(((b[w] + 1) / (nb + 1)) / ((a[w] + 1) / (na + 1)))
        # ties broken by the word itself, so the order is the same on every run
        ranked = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
        new_terms = [w for w, f in b.most_common() if a[w] == 0 and f >= min_count][:top]
        shifts.append({
            "from": prev, "to": cur,
            "rising": [(w, round(s, 2)) for w, s in ranked[::-1][:top]],
            "falling": [(w, round(s, 2)) for w, s in ranked[:top]],
            "new_terms": new_terms,
        })
    return shifts


def merge(counters) -> Counter:
    total = Counter()
    for c in counters:
        total.update(c)
    return total


def build_profile(df: pd.DataFrame, tag: str = "", min_docs: int = MIN_DOCS_PER_DECADE,
                  heaps_every: int = 20_000) -> dict:
    """Full lexical analysis; saves lexical_profile<tag>.json and term_by_decade<tag>.parquet.

    tag '' analyses the title + description of every act; tag '_full' the downloaded full texts
    (fewer acts, so a lower min_docs per decade).
    """
    print("Counting tokens (one pass over the corpus)...")
    counts = count_corpus(df, heaps_every)
    content = merge(counts["words"].values())

    vocab_table = vocabulary_comparison(counts["raw"], content)
    vocab_table.to_csv(tagged(config.RESULTS_DIR / "vocabulary_comparison.csv", tag))
    print("\nVocabulary size by representation:")
    print(vocab_table.to_string())

    valid_decades = sorted(d for d in counts["words"] if d >= 0 and counts["docs"][d] >= min_docs)
    zipf = {"all_raw": fit_zipf(counts["raw"]), "all_content": fit_zipf(content),
            "all_stems": fit_zipf(derive_stems(content))}
    for d in valid_decades:
        zipf[str(d)] = fit_zipf(counts["words"][d])
    heaps = fit_heaps(counts["heaps"])
    print(f"\nZipf exponent s (raw tokens): {zipf['all_raw']['s']}  R²={zipf['all_raw']['r2']}")
    print(f"Heaps' law: K={heaps['K']}, beta={heaps['beta']}")

    term_decade_matrix(counts["words"]).to_parquet(tagged(TERM_DECADE_PATH, tag))

    # Place names (district councils' acts) swamp the decade comparison, so they are removed there.
    toponyms = extract_toponyms(df, tag=tag)
    words_no_places = {d: without_toponyms(c, toponyms) for d, c in counts["words"].items()}
    toponym_share = {str(d): round(100 * (1 - sum(words_no_places[d].values()) / max(1, sum(counts["words"][d].values()))), 2)
                     for d in valid_decades}
    print(f"Toponyms found: {len(toponyms)}; share of content tokens by decade (%): {toponym_share}")

    def bigrams_no_places(c: Counter, n: int) -> list:
        kept = ((g, f) for g, f in c.most_common() if not any(stem(w) in toponyms for w in g))
        return [(" ".join(g), f) for g, f in list(itertools.islice(kept, n))]

    profile = {
        "docs_per_decade": {str(k): v for k, v in sorted(counts["docs"].items())},
        "decades_compared": valid_decades,
        "vocabulary": vocab_table.to_dict(orient="index"),
        "zipf": zipf,
        "heaps": {**heaps, "curve": counts["heaps"]},
        "top_words": content.most_common(100),
        "top_bigrams": [(" ".join(g), f) for g, f in merge(counts["bigrams"].values()).most_common(50)],
        "top_trigrams": [(" ".join(g), f) for g, f in merge(counts["trigrams"].values()).most_common(50)],
        "n_toponyms": len(toponyms),
        "toponym_token_share_pct": toponym_share,
        "top_words_by_decade": {str(d): words_no_places[d].most_common(50) for d in valid_decades},
        "top_bigrams_by_decade": {str(d): bigrams_no_places(counts["bigrams"][d], 30) for d in valid_decades},
        "decade_shifts": decade_shifts(words_no_places, counts["docs"], min_docs=min_docs),
        # rank-frequency data for plotting (kept small)
        "rank_freq": {
            "raw": sorted(counts["raw"].values(), reverse=True)[:50_000],
            "stems": sorted(derive_stems(content).values(), reverse=True)[:50_000],
        },
    }
    save_json(profile, tagged(PROFILE_PATH, tag))
    return profile


def load_profile(tag: str = "") -> dict:
    profile = load_json(tagged(PROFILE_PATH, tag))
    if profile is None:
        raise FileNotFoundError("Lexical profile not found. Run menu item 5 (or 10 for full texts) first.")
    return profile


def term_history(word: str, tag: str = "") -> pd.Series | None:
    """Frequency per million of a word (or its stem) across decades."""
    path = tagged(TERM_DECADE_PATH, tag)
    if not path.exists():
        raise FileNotFoundError("Run menu item 5 (Lexical profile) first.")
    matrix = pd.read_parquet(path)
    word = word.strip().lower()
    if word in matrix.index:
        return matrix.loc[word]
    # fall back to all forms sharing the same stem
    target = stem(word)
    forms = [w for w in matrix.index if stem(w) == target]
    if not forms:
        return None
    print(f"  '{word}' not found; summing {len(forms)} forms with stem '{target}': {forms[:8]}")
    return matrix.loc[forms].sum()
