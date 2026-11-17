# Full texts: a stratified sample of acts

The corpus contains only titles and short descriptions. The full texts of a stratified sample
(up to 1,024 acts: 32 per decade in each of the 8 largest sections) are downloaded from
adilet.zan.kz, and menu 10 → 2 runs every analysis on them.

The site has no bulk download, and its robots.txt disallows the `/api/` that serves act texts,
so the corpus is not crawled. Instead `src/fetch_pdfs.py` drives a real browser through the
site's own interface, signed in with a team member's account: for each act on the sample list it
opens the act's page, clicks **Жүктеу** (download) → **PDF жүктеп алу**, and saves the PDF the
site builds. Only the acts on the list are fetched, one at a time, with a 4–8 s pause between
them. Copying a text by hand (below) gives the same result.

## Current status

**548 of the 1,024 listed acts are collected** (136, 130, 136 and 146 in the 1990s, 2000s,
2010s and 2020s). By collector: Temirlan 333 of 334 (`V15011690_8` could not be downloaded:
its PDF button times out), Dilnaz 108 of 341, Zhanel 107 of 349. All reported results use
these 548 texts.

## What is in this folder

| File | Contents | In git |
|---|---|---|
| `sample_to_collect.csv` | the list: `law_id`, decade, section, title, `url_kaz`, `collected`, `collector` | yes |
| `<law_id>.txt` | the act's text extracted from the PDF, page-number footers removed (UTF-8) | yes |
| `pdf/<law_id>.pdf` | the downloaded PDF | no (re-downloadable) |

## How to collect

1. `sample_to_collect.csv` lists the acts to collect; the `collector` column splits them among
   the team.
2. Install the browser once:
   ```bash
   pip install -r requirements.txt
   python -m playwright install chromium
   ```
3. Sign in once. PDF download is open only to registered users. This opens a browser window:
   sign in there yourself, then press Enter in the terminal.
   ```bash
   python -m src.fetch_pdfs --login
   ```
   The session is kept in `~/.adilet_browser_profile`, outside the repository. The script
   never sees or stores the password.
4. Download:
   ```bash
   python -m src.fetch_pdfs                       # every act without a .txt yet
   python -m src.fetch_pdfs --collector Temirlan  # only your rows
   python -m src.fetch_pdfs --limit 3 --show      # a few, with the browser visible
   ```
   Each act is saved as `pdf/<law_id>.pdf`, and its text as `<law_id>.txt` in this folder.
   Acts that fail are listed at the end; running the command again retries only those.
   `--redo` downloads everything again.
5. Run `python main.py` and choose **10 → 2** to run the full-text analysis.

**By hand instead.** Open `url_kaz` in a browser, copy the act's text, and save it as
`<law_id>.txt` in this folder, encoded as **UTF-8** (example: `P950001664_.txt`). The
analysis reads hand-saved and downloaded files the same way.

**Changing the size.** Menu **10 → 1** asks how many acts per decade and section (default 3)
and how many sections (default 8); the current list was made with 32 × 8. The list is drawn
with a fixed random key per act, so a larger list always contains the smaller one: texts you
have already collected stay on it, and each act keeps its collector. The `collected` column
is refilled from the `.txt` files present each time the list is made.

| Acts per cell × sections | Acts | Download time (about 12–16 s per act, one person) |
|---|---|---|
| 3 × 8 | 96 | 20–25 min |
| 10 × 8 | 320 | 1–1.5 h |
| 32 × 8 (current) | 1,024 | 3.5–4.5 h, or about 1.5 h each when split among three |

The model comparison needs a few hundred acts: its test set is the acts adopted in 2018 or later.

## What the analysis does with the texts

Before analysis each text is cleaned: Latin look-alike letters inside Cyrillic words are fixed
(Latin “i” for Kazakh “і”, in 300 of the 548 texts), the Russian header line repeating the act's
type and date (“Закон от …”, 373 texts) and dash rulers are removed, and words hyphenated across
a line break are joined (1,849).

Menu 10 → 2 then runs, on the full texts and on the title + description of the same acts:

- **audit and storage**: coverage, length, very short texts, duplicates, Russian passages
  (`fulltext_audit.json`); the cleaned texts as Parquet, compared with the `.txt` files
  (`fulltext_storage.json`, 24.1 MB → 7.8 MB);
- **SQL**: a `fulltexts` table in the SQLite database and three queries
  (`sql_fulltext_length_by_section_decade.csv`, `sql_longest_fulltext_per_decade.csv`,
  `sql_stemming_effect_by_section.csv`);
- **lexical profile and decades**, as menu 5–6 (outputs with the suffix `_full`);
- **source comparison**: vocabulary, stemming, Zipf and Heaps for both sources, with the full
  texts also cut to the titles' size (`fulltext_vs_title.csv`, figure 21);
- **models and error analysis**, as menu 7 and 9, with the same temporal split for both sources
  (`model_comparison_full.csv`, `model_comparison_sample_title.csv`,
  `fulltext_vs_title_models.csv`, figure 22);
- **per act** (`fulltext_case_study.csv`, figure 20): title vs full-text length and their
  ratio, the full text's vocabulary and how much stemming reduces it, the share of title stems
  found in the body, and the section the main model predicts from each.

Results are in `reports/results/` and `reports/figures/`; the findings are in the main
README, section 6.8.
