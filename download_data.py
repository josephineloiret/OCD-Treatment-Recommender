"""
Download the real Reddit mental-health posts used to train the screener.

Source: solomonk/reddit_mental_health_posts on the Hugging Face Hub
License: Public Domain (PDDL); reuse subject to Reddit API terms.

Run:  python download_data.py
Files are saved to data/<condition>.csv. Some networks (e.g. corporate
Wi-Fi with content filtering) block huggingface.co; use an unfiltered
connection if the download fails.
"""

import os
import urllib.request

CONDITIONS = ["ocd", "depression", "adhd", "ptsd", "aspergers"]
BASE = "https://huggingface.co/datasets/solomonk/reddit_mental_health_posts/resolve/main"
OUT_DIR = "data"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for cond in CONDITIONS:
        url = f"{BASE}/{cond}.csv?download=true"
        dest = os.path.join(OUT_DIR, f"{cond}.csv")
        print(f"Downloading {cond}.csv ...", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            size = os.path.getsize(dest) / 1e6
            print(f"  saved {dest} ({size:.1f} MB)")
        except Exception as e:
            print(f"  FAILED ({e}). Try an unfiltered network (huggingface.co may be blocked).")


if __name__ == "__main__":
    main()
