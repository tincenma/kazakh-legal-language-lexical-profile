"""Download the PDF of each act in the full-text sample from adilet.zan.kz with a real browser.

For every row of data/fulltext/sample_to_collect.csv the script opens url_kaz, clicks the
"Жүктеу" (download) tab, then the "PDF жүктеп алу" card, and saves the file the site builds as
data/fulltext/pdf/<law_id>.pdf. The text of each PDF is then written to data/fulltext/<law_id>.txt,
which is what src/fulltext.py reads.

PDF download is open only to signed-in users, so the browser keeps a profile in
~/.adilet_browser_profile (outside the repository: it holds your session cookies). You sign in
yourself once with --login; the script never sees or stores your password.

Only the acts on the sample list are fetched, one at a time, with a pause between them.

Setup (once):
    pip install playwright pypdf
    python -m playwright install chromium
    python -m src.fetch_pdfs --login            # sign in in the window, then press Enter here

Run:
    python -m src.fetch_pdfs                    # all rows not collected yet
    python -m src.fetch_pdfs --collector Dilnaz # only your rows
    python -m src.fetch_pdfs --limit 3 --show   # try a few with the browser visible
"""
import argparse
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

from src import config

SITE = "https://adilet.zan.kz/kaz"
PROFILE_DIR = Path.home() / ".adilet_browser_profile"
PDF_DIR = config.FULLTEXT_DIR / "pdf"

DOWNLOAD_TAB = '[data-tour="doc-tab-download"]'
PDF_BUTTON = 'button:has(h4:text-is("PDF жүктеп алу"))'
WELCOME_BACKDROP = '[data-testid="welcome-modal-backdrop"]'
WELCOME_MODAL = f'div:has(> {WELCOME_BACKDROP})'
LOGIN_REQUIRED = "text=Тек тіркелген пайдаланушыларға қолжетімді"   # "only for registered users"

PAUSE_SECONDS = (4.0, 8.0)        # random pause between acts, to keep the load on the site low
TIMEOUT_MS = 90_000               # building a long act's PDF can take a while


class LoginRequired(RuntimeError):
    pass


def open_browser(pw, show: bool):
    """Chromium with the saved profile, so a sign-in from --login carries over."""
    PROFILE_DIR.mkdir(exist_ok=True)
    return pw.chromium.launch_persistent_context(
        str(PROFILE_DIR), headless=not show, accept_downloads=True, locale="kk-KZ",
        viewport={"width": 1366, "height": 900})


def close_welcome_modal(page) -> None:
    """On a first visit the site shows a language / guided-tour dialog over the page; skip it."""
    backdrop = page.locator(WELCOME_BACKDROP)
    try:
        backdrop.wait_for(state="visible", timeout=5_000)
    except Exception:
        return                                    # no dialog this time
    skip = page.locator(f"{WELCOME_MODAL} button", has_text="жалғастыру")   # "continue" (no tour)
    if skip.count():
        skip.first.click()
    else:
        page.keyboard.press("Escape")
    backdrop.wait_for(state="hidden", timeout=10_000)


def download_pdf(page, url: str, target: Path) -> None:
    """Open one act, click Жүктеу -> PDF жүктеп алу, and save the download to target."""
    page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
    close_welcome_modal(page)
    page.locator(DOWNLOAD_TAB).click(timeout=TIMEOUT_MS)
    button = page.locator(PDF_BUTTON)
    button.wait_for(state="visible", timeout=TIMEOUT_MS)

    # The site builds the PDF in the page; it arrives as a download (or, on some setups, a new tab).
    downloads, popups = [], []

    def on_download(d):
        downloads.append(d)

    def on_popup(p):
        popups.append(p)

    page.on("download", on_download)
    page.context.on("page", on_popup)
    try:
        button.click()
        login_box = page.locator(LOGIN_REQUIRED).first
        deadline = time.monotonic() + TIMEOUT_MS / 1000
        while not downloads and not popups:
            if login_box.is_visible():
                raise LoginRequired("not signed in: run  python -m src.fetch_pdfs --login  first")
            if time.monotonic() > deadline:
                raise TimeoutError("no PDF arrived within the timeout")
            page.wait_for_timeout(500)
    finally:
        page.remove_listener("download", on_download)
        page.context.remove_listener("page", on_popup)

    target.parent.mkdir(parents=True, exist_ok=True)
    if downloads:
        download = downloads[0]
        if download.failure():
            raise RuntimeError(f"download failed: {download.failure()}")
        download.save_as(target)
    else:                                          # PDF opened in a new tab: fetch its bytes
        popup = popups[0]
        popup.wait_for_load_state()
        response = page.context.request.get(popup.url)
        target.write_bytes(response.body())
        popup.close()


PAGE_NUMBER = re.compile(r"^\s*\d+\s*/\s*\d+\s*$", re.MULTILINE)   # "3 / 7" footer on each page


def pdf_to_text(pdf_path: Path) -> str:
    """Plain text of a text-layer PDF, page by page, without the page-number footers."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    text = PAGE_NUMBER.sub("", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def login() -> None:
    """Open a visible browser on the saved profile so you can sign in by hand."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        context = open_browser(pw, show=True)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(SITE)
        input("Sign in to adilet.zan.kz in the browser window, then press Enter here... ")
        context.close()
    print(f"Session saved in {PROFILE_DIR}")


def fetch(collector: str | None = None, limit: int | None = None, show: bool = False,
          redo: bool = False) -> None:
    from playwright.sync_api import sync_playwright

    if not config.FULLTEXT_SAMPLE_PATH.exists():
        raise FileNotFoundError(f"{config.FULLTEXT_SAMPLE_PATH} not found; make the list first (menu 10 -> 1).")
    sample = pd.read_csv(config.FULLTEXT_SAMPLE_PATH, encoding="utf-8-sig")
    if collector:
        sample = sample[sample["collector"].str.lower() == collector.lower()]
    if not redo:
        done = {p.stem for p in config.FULLTEXT_DIR.glob("*.txt") if p.stat().st_size > 0}
        sample = sample[~sample["law_id"].isin(done)]
    if limit:
        sample = sample.head(limit)
    if sample.empty:
        print("Nothing to fetch: every selected act already has a .txt file.")
        return
    print(f"Fetching {len(sample)} act(s) -> {PDF_DIR}")

    ok, failed = 0, []
    with sync_playwright() as pw:
        context = open_browser(pw, show)
        page = context.pages[0] if context.pages else context.new_page()
        for i, row in enumerate(sample.itertuples(index=False), 1):
            pdf_path = PDF_DIR / f"{row.law_id}.pdf"
            txt_path = config.FULLTEXT_DIR / f"{row.law_id}.txt"
            print(f"[{i}/{len(sample)}] {row.law_id} ... ", end="", flush=True)
            try:
                if redo or not pdf_path.exists():
                    download_pdf(page, row.url_kaz, pdf_path)
                text = pdf_to_text(pdf_path)
                if not text:
                    raise RuntimeError("PDF has no text layer")
                txt_path.write_text(text, encoding="utf-8")
                ok += 1
                print(f"ok ({len(text):,} chars)")
            except LoginRequired as exc:
                print(exc)
                failed.extend(sample["law_id"].iloc[i - 1:])
                break
            except Exception as exc:  # keep going; the failed acts are listed at the end
                failed.append(row.law_id)
                print(f"FAILED: {str(exc).splitlines()[0]}")
            if i < len(sample):
                time.sleep(random.uniform(*PAUSE_SECONDS))
        context.close()

    print(f"\nDone: {ok} saved, {len(failed)} not saved.")
    if failed:
        print("Not saved (run again to retry): " + ", ".join(failed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--login", action="store_true", help="open a browser to sign in once")
    parser.add_argument("--collector", help="only this collector's rows (e.g. Temirlan)")
    parser.add_argument("--limit", type=int, help="fetch at most this many acts")
    parser.add_argument("--show", action="store_true", help="show the browser window")
    parser.add_argument("--redo", action="store_true", help="download again even if files exist")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1251
    if args.login:
        login()
    else:
        fetch(args.collector, args.limit, args.show, args.redo)


if __name__ == "__main__":
    main()
