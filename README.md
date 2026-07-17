# CS621 — YouTube Comment Analysis

**Voices in the Comments:** comparing audience sentiment and engagement for two
gaming influencers — **videogamedunkey** (7.57M subscribers) and **Game Grumps**
(5.45M subscribers) — using the YouTube Data API v3 and the standard data
science pipeline (ingestion → engineering → analytics → visualization →
delivery).

20,080 top comments were collected from each channel's 10 most-viewed videos
and cleaned to 3,006 rows. Analyses include VADER sentiment, time series
relative to each video's upload, a posting-hour spatial proxy, emoji emotion
mapping, and head-to-head comparison of the two audiences.

## Repository contents

| Path | What it is |
|---|---|
| `yt-comment-analysis/youtube_comment_analysis.py` | The full pipeline program (~330 lines) |
| `yt-comment-analysis/data/raw/` | Raw dataset — 20 JSON files, one per video, exactly as returned by the API |
| `yt-comment-analysis/data/video_meta.json` | Upload date for every video (used by the cleaning rules) |
| `CS621_yt_comments_analysis.pdf` | 12-slide results presentation (PDF export of the PowerPoint) |
| `YouTube_Comment_Analysis_Report.pdf` | Written report (PDF) |

## How to run the Python program

Requires **Python 3.10+**.

1. Install the dependencies:

   ```
   cd yt-comment-analysis
   pip install -r requirements.txt
   ```

2. Run the program:

   ```
   python youtube_comment_analysis.py
   ```

That's it — the raw JSON dataset ships with this repository, so the program
skips ingestion and goes straight to cleaning, analytics, and charts. It
prints the cleaning funnel and per-channel statistics to the terminal, then
opens seven charts one at a time (cleaning funnel, sentiment pie, weekly time
series, length boxplot + scatter, posting-hour heatmap, emoji emotions, word
cloud). **Close each chart window to advance to the next one.**

### Re-downloading the data (optional)

To rebuild the dataset from scratch instead of using the included JSON:

1. Get a YouTube Data API v3 key (free) from the
   [Google Cloud Console](https://console.cloud.google.com): create a project,
   enable *YouTube Data API v3*, and create an API key.
2. Set it as an environment variable — on Windows:

   ```
   setx YT_API_KEY "your-key-here"
   ```

   (then open a new terminal so the variable is picked up)
3. Delete or rename `data/raw/` and run the program again. Live ingestion
   ranks each channel's full uploads catalog by view count, fetches the top 10
   videos' comment threads, and saves fresh JSON. A full run costs under 1,000
   of the 10,000 free daily quota units.

### Notes

- `langdetect` and `vaderSentiment` are optional but recommended — without
  them the program falls back to a cruder language heuristic and a small
  word-list sentiment scorer.
- Every run is reproducible: language detection is seeded and the word-cloud
  layout uses a fixed random seed, so the same data always produces the same
  results.
