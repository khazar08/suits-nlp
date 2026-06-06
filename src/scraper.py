"""
Scrapes Suits episode scripts from Springfield! Springfield!
and produces structured dialogue rows.

Important: Springfield transcripts have NO speaker labels —
dialogue is raw text separated by <br/> with no character attribution.
Speaker attribution happens later in the NLP pipeline (Step 2).
For graph construction we use character-mention extraction per scene.

Run via pipeline.py — or directly:
    python src/scraper.py --out data/
"""

import re
import time
import argparse
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.springfieldspringfield.co.uk"
SERIES_URL = f"{BASE_URL}/episode_scripts.php?tv-show=suits"
EPISODE_URL = f"{BASE_URL}/view_episode_scripts.php?tv-show=suits&episode={{code}}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
REQUEST_DELAY = 1.5  # seconds — polite scraping

# All recurring characters across 9 seasons (first-name keys → canonical full name)
CHARACTER_MAP = {
    "Harvey":   "Harvey Specter",
    "Mike":     "Mike Ross",
    "Louis":    "Louis Litt",
    "Donna":    "Donna Paulsen",
    "Jessica":  "Jessica Pearson",
    "Rachel":   "Rachel Zane",
    "Katrina":  "Katrina Bennett",
    "Alex":     "Alex Williams",
    "Samantha": "Samantha Wheeler",
    "Scottie":  "Dana Scott",
    "Trevor":   "Trevor Evans",
    "Jenny":    "Jenny Griffith",
    "Hardman":  "Daniel Hardman",
    "Forstman": "Charles Forstman",
    "Zoe":      "Zoe Lawford",
    "Gretchen": "Gretchen Bodinski",
    "Harold":   "Harold Gunderson",
    "Norma":    "Norma",
    "Kyle":     "Kyle Durant",
    "Sheila":   "Sheila Sazs",
    "Cahill":   "Sean Cahill",
    "Robert":   "Robert Zane",
    "Esther":   "Esther Litt",
    "Leonard":  "Leonard Bailey",
    "Oliver":   "Oliver Grady",
    "Faye":     "Faye Richardson",
    "Gibbs":    "Anita Gibbs",
    "Thomas":   "Thomas Kessler",
    "Jack":     "Jack Soloff",
    "Benjamin": "Benjamin",
    "Paula":    "Paula Agard",
    "Stu":      "Stu Buzzini",
    "Brian":    "Brian Altman",
}

SUITS_CHARACTERS = sorted(CHARACTER_MAP.keys(), key=len, reverse=True)  # longest first

# Pre-compiled regex to detect character name mentions inside dialogue text
_char_alts = "|".join(re.escape(c) for c in SUITS_CHARACTERS)
MENTION_RE = re.compile(rf"\b({_char_alts})\b")

# Lines to discard: very short, all-caps headings, stage directions
_DISCARD_RE = re.compile(r"^\[.*\]$|^[A-Z\s]{2,30}$")


def get_episode_list() -> list[dict]:
    resp = requests.get(SERIES_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    episodes = []
    for a in soup.select("a[href*='view_episode_scripts']"):
        href = a.get("href", "")
        m = re.search(r"episode=(s\d{2}e\d{2})", href)
        if not m:
            continue
        code = m.group(1)
        season = int(code[1:3])
        ep_num = int(code[4:6])
        episodes.append({
            "code": code,
            "season": season,
            "episode_num": ep_num,
            "episode_id": f"S{season:02d}E{ep_num:02d}",
            "title": a.text.strip(),
        })

    return sorted(episodes, key=lambda x: (x["season"], x["episode_num"]))


def fetch_lines(code: str) -> list[str]:
    """
    Fetch an episode page and return dialogue as a list of sentence-level strings.
    Springfield uses <br/> as sentence separators inside a single div.
    """
    url = EPISODE_URL.format(code=code)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    container = (
        soup.find("div", class_="scrolling-script-container")
        or soup.find("div", {"id": "script-text"})
        or soup.find("article")
    )
    if not container:
        return []

    # Split on <br/> — each fragment becomes a candidate dialogue line
    raw_lines: list[str] = []
    for part in container.decode_contents().split("<br/>"):
        text = BeautifulSoup(part, "lxml").get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            raw_lines.append(text)

    return raw_lines


def parse_lines(
    raw_lines: list[str],
    season: int,
    episode_num: int,
    episode_id: str,
    title: str,
) -> list[dict]:
    rows = []
    line_num = 0

    for raw in raw_lines:
        # Drop very short lines, all-caps headings, and bracketed stage directions
        if len(raw) < 8 or _DISCARD_RE.match(raw):
            continue

        line_num += 1
        mentions = sorted(set(MENTION_RE.findall(raw)))
        canonical = [CHARACTER_MAP.get(m, m) for m in mentions]

        rows.append({
            "episode_id": episode_id,
            "season": season,
            "episode_num": episode_num,
            "episode_title": title,
            "line_num": line_num,
            "speaker": "UNKNOWN",           # no speaker labels in this source
            "line": raw,
            "char_mentions": "|".join(canonical),   # pipe-separated character names
            "mention_count": len(canonical),
        })

    return rows


def scrape_all(data_dir: Path, delay: float = REQUEST_DELAY) -> pd.DataFrame:
    raw_dir = data_dir / "raw"
    transcripts_dir = raw_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching episode list from Springfield! Springfield!...")
    episodes = get_episode_list()
    print(f"Found {len(episodes)} episodes across {max(e['season'] for e in episodes)} seasons\n")

    all_rows: list[dict] = []

    for ep in tqdm(episodes, desc="Scraping episodes"):
        try:
            raw_lines = fetch_lines(ep["code"])
            rows = parse_lines(
                raw_lines,
                ep["season"], ep["episode_num"],
                ep["episode_id"], ep["title"],
            )
            all_rows.extend(rows)

            # Save raw lines for re-parsing without re-scraping
            raw_path = transcripts_dir / f"{ep['episode_id']}.txt"
            raw_path.write_text("\n".join(raw_lines), encoding="utf-8")

        except requests.RequestException as e:
            tqdm.write(f"  Network error on {ep['episode_id']}: {e}")
        except Exception as e:
            tqdm.write(f"  Unexpected error on {ep['episode_id']}: {e}")

        time.sleep(delay)

    df = pd.DataFrame(all_rows)
    out_path = data_dir / "suits_dialogue_raw.csv"
    df.to_csv(out_path, index=False)

    lines_with_mentions = (df["mention_count"] > 0).sum()
    print(f"\nSaved {len(df):,} dialogue lines → {out_path}")
    print(f"Lines mentioning at least one character: {lines_with_mentions:,} ({100*lines_with_mentions/len(df):.1f}%)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="Output data directory")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY)
    args = parser.parse_args()
    scrape_all(Path(args.out), delay=args.delay)
