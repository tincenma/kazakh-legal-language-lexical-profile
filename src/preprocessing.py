"""Kazakh text preprocessing: tokenization, stop words, suffix stemming, character n-grams.

Kazakh is agglutinative: one stem takes chains of suffixes (plural + possessive + case),
e.g. заң "law" -> заңдар -> заңдарының -> заңдарынан. We compare three representations:
  1. word forms   - surface tokens as they appear
  2. stems        - a rule-based suffix stripper (below)
  3. char n-grams - sub-word units that catch shared morphemes without any rules
"""
import re
from functools import lru_cache

from src import config

KAZAKH_LETTERS = "а-яёәғқңөұүһі"
TOKEN_RE = re.compile(rf"[{KAZAKH_LETTERS}]+(?:-[{KAZAKH_LETTERS}]+)*")

# Older acts often type a Latin letter that looks like a Cyrillic one, most often Latin "i" for
# Kazakh "і" ("жөнiндегi"). The tokenizer accepts only Cyrillic, so such words were cut apart.
# A Latin look-alike is replaced only when it touches a Cyrillic letter, so real Latin words stay.
LOOKALIKES = {"i": "і", "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
              "k": "к", "h": "һ", "I": "І", "A": "А", "E": "Е", "O": "О", "P": "Р", "C": "С",
              "X": "Х", "Y": "У", "K": "К", "H": "Н", "T": "Т", "M": "М", "B": "В"}
_CYRILLIC = "А-ЯЁӘҒҚҢӨҰҮҺІа-яёәғқңөұүһі"
_LATIN = "".join(LOOKALIKES)
LOOKALIKE_RE = re.compile(rf"(?<=[{_CYRILLIC}])[{_LATIN}]+|[{_LATIN}]+(?=[{_CYRILLIC}])")


def fix_lookalikes(text: str) -> str:
    """Replace Latin look-alike letters inside Cyrillic words with the Cyrillic letter."""
    return LOOKALIKE_RE.sub(lambda m: "".join(LOOKALIKES[c] for c in m.group()), text)

# Suffixes grouped by grammatical role. Stripping goes from the end of the word
# (case -> possessive -> plural -> derivational), always taking the longest match.
CASE_SUFFIXES = [
    # ablative, locative, dative, accusative, genitive, instrumental
    "нан", "нен", "дан", "ден", "тан", "тен",
    "нда", "нде", "да", "де", "та", "те",
    "на", "не", "ға", "ге", "қа", "ке",
    "ны", "ні", "ды", "ді", "ты", "ті",
    "ның", "нің", "дың", "дің", "тың", "тің",
    "мен", "бен", "пен",
    "дағы", "дегі", "тағы", "тегі", "ндағы", "ндегі",
]
POSSESSIVE_SUFFIXES = [
    "ымыз", "іміз", "ыңыз", "іңіз", "мыз", "міз", "ңыз", "ңіз",
    "сы", "сі", "ым", "ім", "ың", "ің", "ы", "і",
]
PLURAL_SUFFIXES = ["лар", "лер", "дар", "дер", "тар", "тер"]
DERIVATIONAL_SUFFIXES = [
    "лық", "лік", "дық", "дік", "тық", "тік",
    "шылық", "шілік", "шы", "ші", "сыз", "сіз",
]
SUFFIX_GROUPS = [CASE_SUFFIXES, POSSESSIVE_SUFFIXES, PLURAL_SUFFIXES, DERIVATIONAL_SUFFIXES]
MIN_STEM_LEN = 3
# Stems whose ending only looks like a suffix (қазақс-тан is not an ablative). Stripping stops here.
STEM_EXCEPTIONS = {"қазақстан", "республика", "жоғары", "астана", "алматы", "тәртіп", "бағдарлама",
                   "құқық", "кодекс", "министр", "әкімдік", "мәслихат", "қала", "облыс"}


def load_stopwords(path=config.STOPWORDS_PATH) -> set[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip() and not line.startswith("#")}
    except FileNotFoundError:
        print(f"[warn] stop-word file not found: {path}; continuing without stop words.")
        return set()


STOPWORDS = load_stopwords()


def tokenize(text: str) -> list[str]:
    """Lower-case and keep Kazakh/Cyrillic words of 2+ letters (numbers and Latin dropped).

    Latin look-alike letters inside Cyrillic words are fixed first (see fix_lookalikes)."""
    if not isinstance(text, str):
        return []
    return [t for t in TOKEN_RE.findall(fix_lookalikes(text).lower()) if len(t) > 1]


def remove_stopwords(tokens: list[str], stopwords: set[str] = STOPWORDS) -> list[str]:
    return [t for t in tokens if t not in stopwords]


@lru_cache(maxsize=500_000)
def stem(word: str) -> str:
    """Strip at most one suffix per group, from the outermost group inwards."""
    for group in SUFFIX_GROUPS:
        if word in STEM_EXCEPTIONS:
            break
        for suffix in sorted(group, key=len, reverse=True):
            if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM_LEN:
                word = restore_consonant(word[: -len(suffix)])
                break
    return word


# Final voiceless consonants become voiced before a vowel suffix: кітап -> кітабы, министрлік -> министрлігі.
# After removing the suffix we undo this so all forms map to one stem.
ALTERNATION = {"б": "п", "г": "к", "ғ": "қ"}


def restore_consonant(word: str) -> str:
    if word and word[-1] in ALTERNATION:
        return word[:-1] + ALTERNATION[word[-1]]
    return word


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stem(t) for t in tokens]


def char_ngrams(token: str, n_min: int = 3, n_max: int = 5) -> list[str]:
    """Character n-grams of one word, padded with boundary marks: заң -> <за, заң, аң> ..."""
    padded = f"<{token}>"
    return [padded[i:i + n] for n in range(n_min, n_max + 1) for i in range(len(padded) - n + 1)]


def preprocess(text: str, mode: str = "wordform") -> list[str]:
    """Text -> list of units for the chosen representation: 'wordform', 'stem' or 'char'."""
    tokens = remove_stopwords(tokenize(text))
    if mode == "wordform":
        return tokens
    elif mode == "stem":
        return stem_tokens(tokens)
    elif mode == "char":
        return [g for t in tokens for g in char_ngrams(t)]
    else:
        raise ValueError(f"Unknown mode: {mode!r} (use 'wordform', 'stem' or 'char')")


# Analyzer functions for scikit-learn vectorizers (module level so they can be pickled).
def analyzer_wordform(text: str) -> list[str]:
    return preprocess(text, "wordform")


def analyzer_stem(text: str) -> list[str]:
    return preprocess(text, "stem")


def analyzer_char(text: str) -> list[str]:
    return preprocess(text, "char")
