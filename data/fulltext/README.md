# Full-text case study: a sample of full acts

The corpus contains only titles and short descriptions. To check how well these represent the
acts, the full texts of a small stratified sample are downloaded from adilet.zan.kz.

The site has no bulk download, and its robots.txt disallows the `/api/` that serves act texts,
so the corpus is not crawled. Instead `src/fetch_pdfs.py` drives a real browser through the
site's own interface, signed in with a team member's account: for each act on the sample list it
opens the act's page, clicks **Жүктеу** (download) → **PDF жүктеп алу**, and saves the PDF the
site builds. Only the acts on the list are fetched, one at a time, with a 4–8 s pause between
them. Copying a text by hand (below) gives the same result.

## How to collect

1. `sample_to_collect.csv` lists 96 acts by default: 3 per decade (1990s–2020s) in each of the
   8 largest sections. The `collector` column splits them among the team.
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
   Each act is saved as `pdf/<law_id>.pdf`, and its text (page-number footers removed) as
   `<law_id>.txt` in this folder. Acts that fail are listed at the end; running the command
   again retries only those. `--redo` downloads everything again.
5. Run `python main.py` and choose **10 → 2** to run the case study.

**By hand instead.** Open `url_kaz` in a browser, copy the act's text, and save it as
`<law_id>.txt` in this folder, encoded as **UTF-8** (example: `P950001664_.txt`). The case
study reads hand-saved and downloaded files the same way.

**Changing the size.** Menu **10 → 1** asks how many acts per decade and section, and how many
sections. The list is drawn with a fixed random key per act, so a larger list always contains
the smaller one: texts you have already collected stay on it, and each act keeps its collector.
The `collected` column is refilled from the files present.

| Acts per cell × sections | Acts | Download time (about 10–15 s per act) |
|---|---|---|
| 2 × 4 | 32 | 5–8 min |
| 3 × 8 (default) | 96 | 15–25 min |
| 5 × 8 | 160 | 25–40 min |

Use at least 20 acts for meaningful numbers.
Results go to `reports/results/fulltext_case_study.csv` and `.json`, and to
`reports/figures/20_fulltext_case_study.png`.

## What it measures, per act

- content tokens in the title + description vs the full text, and their ratio;
- vocabulary of the full text, and how much stemming reduces it;
- the share of the title's stemmed content words that also occur in the body;
- the section the best model predicts from the title vs from the full text.
