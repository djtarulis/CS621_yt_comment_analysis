"""
YouTube Comment Analytics for videogamedunkey and Game Grumps

Pipeline covered
    1. Data ingestion   - YouTube Data API v3 -> raw JSON files
    2. Data engineering - lists/dicts -> pandas DataFrame, 4 cleaning rules
    3. Analytics        - basic stats, sentiment, time series, emoji,
                          posting-hour (spatial proxy), comparisons
    4. Visualization    - bar, pie, boxplot, scatter, time series,
                          hour heatmap, word cloud

Usage
    python youtube_comment_analysis.py

Runs the whole pipeline top to bottom. If data/raw/ already holds JSON
files from a previous run, ingestion is skipped and those are used;
otherwise live ingestion runs (requires an API key in the YT_API_KEY
environment variable). Each chart opens in its own window - close it
to continue to the next one.

Optional extras (auto-detected): langdetect, vaderSentiment.
"""

import json
import os
import random
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------- config
CHANNELS = {"dunkey": "@videogamedunkey", "Game Grumps": "@GameGrumps"}
COLORS = {"dunkey": "#2a78d6", "Game Grumps": "#1baf7a"}
RAW_DIR = "data/raw"
COMMENTS_PER_VIDEO = 1000     # max comments
VIDEOS_PER_CHANNEL = 10       # max videos
VIDEO_META = "data/video_meta.json"   # meta data for date posted
OUTDATED_MONTHS = 18

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿❤]")
EMOJI_EMOTION = {"\U0001F602": "joy", "\U0001F923": "joy", "\U0001F480": "awe",
                 "\U0001F525": "awe", "❤": "love", "\U0001F60D": "love",
                 "\U0001F62D": "sadness", "\U0001F621": "anger"}

# ------------------------------ API auth
def api_get(endpoint, params):
    """One GET request against the YouTube Data API v3."""
    import requests  # imported here so offline runs never need it
    params["key"] = os.environ["YT_API_KEY"]
    r = requests.get(f"https://www.googleapis.com/youtube/v3/{endpoint}",
                     params=params, timeout=30)
    
    # guard agains bad request
    if r.status_code != 200:
        try:
            reason = r.json()["error"]["errors"][0].get("reason", "unknown")
        except Exception:
            reason = "unknown"
        raise RuntimeError(f"API error {r.status_code} on {endpoint}: {reason}")
    return r.json()

def popular_video_ids(handle, top_n=VIDEOS_PER_CHANNEL):
    """Get most-viewed video IDs by view count."""
    ch = api_get("channels", {"part": "contentDetails", "forHandle": handle})
    uploads = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    ids, token = [], None
    while True:
        params = {"part": "contentDetails", "playlistId": uploads,
                  "maxResults": 50}
        if token:
            params["pageToken"] = token
        data = api_get("playlistItems", params)
        ids += [it["contentDetails"]["videoId"] for it in data["items"]]
        token = data.get("nextPageToken")
        if not token:
            break
    views = {}
    for i in range(0, len(ids), 50):
        batch = api_get("videos", {"part": "statistics",
                                   "id": ",".join(ids[i:i+50])})
        for it in batch["items"]:
            views[it["id"]] = int(it["statistics"].get("viewCount", 0))
    return sorted(views, key=views.get, reverse=True)[:top_n]

def fetch_comments(video_id):
    """Page through commentThreads for one video (100 comments per call)."""
    comments, token = [], None
    while len(comments) < COMMENTS_PER_VIDEO:
        # relevance order surfaces each video's top comments, which cluster in
        # its launch window - matching the per-video outdated rule. Newest-first
        # would return mostly recent chatter that the rule then discards.
        params = {"part": "snippet", "videoId": video_id, "maxResults": 100,
                  "order": "relevance", "textFormat": "plainText"}
        if token:
            params["pageToken"] = token
        data = api_get("commentThreads", params)
        for it in data.get("items", []):
            s = it["snippet"]["topLevelComment"]["snippet"]
            comments.append({"comment_id": it["id"], "video_id": video_id,
                             "author": s["authorDisplayName"],
                             "text": s.get("textOriginal", ""),
                             "likes": s["likeCount"],
                             "published_at": s["publishedAt"]})
        token = data.get("nextPageToken")
        if not token:
            break
    return comments

def save_video_meta(video_ids):
    """Record each video's upload date"""
    meta = json.load(open(VIDEO_META)) if os.path.exists(VIDEO_META) else {}
    new = [v for v in video_ids if v not in meta]
    for i in range(0, len(new), 50):          # videos endpoint takes 50 ids/call
        batch = api_get("videos", {"part": "snippet", "id": ",".join(new[i:i+50])})
        for it in batch["items"]:
            meta[it["id"]] = it["snippet"]["publishedAt"]
    with open(VIDEO_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

def ingest():
    """Collect comments on each channel's top videos, saved as JSON files.
        Not needed once JSON is saved locally
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, handle in CHANNELS.items():
        video_ids = popular_video_ids(handle)
        save_video_meta(video_ids)
        for vid in video_ids:
            path = os.path.join(RAW_DIR, f"{name.replace(' ', '')}_{vid}.json")
            if os.path.exists(path):       # already fetched on a previous run
                continue
            try:
                rows = fetch_comments(vid)
            except RuntimeError as e:      # e.g. comments disabled on one video
                print(f"  {name}: skipping {vid} ({e})")
                continue
            for r in rows:
                r["influencer"] = name
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=1)
            print(f"  {name}: saved {len(rows):4d} comments -> {path}")

# ------------------------ Engineering
def looks_english(text):
    """Language filter: langdetect when available, ASCII-ratio otherwise."""
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0      # langdetect is randomized by default;
        return detect(text) == "en"   # seeding keeps every run reproducible
    except Exception:
        letters = [c for c in text if c.isalpha()]
        return not letters or sum(c.isascii() for c in letters) / len(letters) > 0.9

def build_dataset():
    """Locally saved raw JSON -> cleaned DataFrame. Returns (df, funnel dict)."""
    rows = []
    for fn in sorted(os.listdir(RAW_DIR)):
        if fn.endswith(".json"):
            with open(os.path.join(RAW_DIR, fn), encoding="utf-8") as f:
                rows += json.load(f)          # JSON -> list of dicts
    df = pd.DataFrame(rows)                   # list of dicts -> DataFrame
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, format="ISO8601")
    funnel = {"raw": len(df)}

    df["text"] = df["text"].fillna("").str.strip()
    df = df[df["text"] != ""]                                    # rule 1: empty
    funnel["after_empty"] = len(df)
    df = df[df["text"].map(looks_english)]                       # rule 2: language
    funnel["after_language"] = len(df)
    df = df.drop_duplicates(subset=["influencer", "text"])       # rule 3: duplicates
    funnel["after_duplicates"] = len(df)
    # drop comments posted > OUTDATED_MONTHS after their
    # video's upload Videos missing from the
    # metadata file fall back to a fixed calendar cutoff.
    uploads = json.load(open(VIDEO_META)) if os.path.exists(VIDEO_META) else {}
    upload = pd.to_datetime(df["video_id"].map(uploads), utc=True, format="ISO8601")
    window = pd.DateOffset(months=OUTDATED_MONTHS)
    fixed_cutoff = pd.Timestamp.now(tz="UTC") - window
    keep = upload.notna() & (df["published_at"] <= upload + window)
    keep |= upload.isna() & (df["published_at"] >= fixed_cutoff)
    df = df[keep]
    funnel["after_outdated"] = len(df)

    df = df.copy()
    df["length"] = df["text"].str.len()       # derived columns
    df["emojis"] = df["text"].map(lambda t: EMOJI_RE.findall(t))
    print(f"  Cleaning funnel: {funnel}")
    return df, funnel

# --------------------------- Analytics
POS_WORDS = {"insane", "funny", "hilarious", "clutch", "classic", "wholesome",
             "love", "pog", "amazing", "best", "laughed"}
NEG_WORDS = {"boring", "worst", "hate", "bad", "annoying", "skip"}

_VADER = None                 # built once, reused for every comment

def sentiment_score(text):
    global _VADER
    try:
        if _VADER is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _VADER = SentimentIntensityAnalyzer()
        return _VADER.polarity_scores(text)["compound"]
    except Exception:
        words = set(re.findall(r"[a-z']+", text.lower()))
        return (len(words & POS_WORDS) - len(words & NEG_WORDS)) / 3.0

def add_sentiment(df):
    """Attach label per comment (positive / neutral / negative)."""
    df["polarity"] = df["text"].map(sentiment_score)
    df["sentiment"] = pd.cut(df["polarity"], [-2, -0.05, 0.05, 2],
                             labels=["negative", "neutral", "positive"])
    return df

def basic_stats(df):
    """Print the per-influencer descriptive statistics table."""
    for name, g in df.groupby("influencer"):
        print(f"\n  {name}: {len(g)} comments | {g['author'].nunique()} unique "
              f"reviewers | mean length {g['length'].mean():.0f} chars | "
              f"median likes {g['likes'].median():.0f} | "
              f"{(g['emojis'].str.len() > 0).mean():.0%} contain emoji")

# ------------------------------------------------------ stage 4: visualization
def plot_funnel(funnel):
    fig, ax = plt.subplots()
    ax.bar(range(len(funnel)), funnel.values(), color="#2a78d6")
    ax.set_xticks(range(len(funnel)), funnel.keys(), rotation=20)
    ax.set_title("Cleaning funnel: comments remaining after each rule")
    plt.show()

def plot_sentiment_pie(df):
    counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct="%1.0f%%",
           colors=["#2a78d6", "#c3c2b7", "#e34948"])
    ax.set_title("Sentiment split, all cleaned comments")
    plt.show()

def plot_timeseries(df):
    fig, ax = plt.subplots()
    for name, g in df.groupby("influencer"):
        weekly = g.set_index("published_at").resample("W")["comment_id"].count()
        ax.plot(weekly.index, weekly.values, label=name, color=COLORS[name])
    ax.legend(), ax.set_title("Comments per week")
    plt.show()

def plot_lengths(df):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    groups = [g["length"] for _, g in df.groupby("influencer")]
    a1.boxplot(groups, tick_labels=list(df.groupby("influencer").groups))
    a1.set_title("Comment length (chars)")
    for name, g in df.groupby("influencer"):
        a2.scatter(g["length"], g["likes"], s=8, alpha=0.4,
                   color=COLORS[name], label=name)
    a2.set_xlabel("length"), a2.set_ylabel("likes"), a2.legend()
    a2.set_title("Length vs. likes")
    plt.show()

def plot_hour_heatmap(df):
    """Posting hour by influencer - the spatial (audience time zone) proxy."""
    pivot = (df.assign(hour=df["published_at"].dt.hour)
               .pivot_table(index="influencer", columns="hour",
                            values="comment_id", aggfunc="count", fill_value=0))
    fig, ax = plt.subplots(figsize=(9, 2.6))
    im = ax.imshow(pivot, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(pivot)), pivot.index)
    ax.set_xlabel("UTC hour posted"), fig.colorbar(im, label="comments")
    ax.set_title("Posting-hour heatmap (spatial proxy)")
    plt.show()

def plot_emoji(df):
    emotions = Counter(EMOJI_EMOTION.get(e) for row in df["emojis"] for e in row
                       if e in EMOJI_EMOTION)
    fig, ax = plt.subplots()
    ax.bar(list(emotions), emotions.values(), color="#2a78d6")
    ax.set_title("Emoji emotion counts")
    plt.show()

# common English words excluded from the word cloud
STOPWORDS = set("""
about actually after again against almost along already also although always
another anyone anything around back because been before being between both
came come comes could didnt does doesnt doing done dont down during each
either else even ever every everyone everything feel feels felt first from
gets getting goes going gonna gotta have having here himself however into
itself just know known last later least less like liked likes literally look
looking looks made make makes making many maybe mean means might more most
much must myself need never next nothing often once only onto other over
people probably really right said same saying says seen should since some
someone something sometimes soon still such sure take takes than that thats
their them then there these they thing things think this those though thought
through time times took under until upon used very want wanted wants watch
watched watching ways well went were what whatever when where which while
whole whose will with within without would your youre yours
can't didn't doesn't don't he's i'm i've isn't it's let's she's that's
there's they're wasn't we're what's won't you're
""".split())

def plot_wordcloud(df):
    words = Counter(w for t in df["text"]
                    for w in re.findall(r"[A-Za-z']{4,}", t.lower())
                    if w not in STOPWORDS)
    top, rng = words.most_common(25), random.Random(4)
    cells = [(c / 5 + 0.1, r / 5 + 0.1) for r in range(5) for c in range(5)]
    rng.shuffle(cells)                       # one word per grid cell, jittered
    fig, ax = plt.subplots(figsize=(8, 4))
    for (word, count), (x, y) in zip(top, cells):
        ax.text(x + rng.uniform(-.03, .03), y + rng.uniform(-.03, .03), word,
                ha="center", fontsize=8 + count / max(1, top[0][1]) * 22,
                color=rng.choice(list(COLORS.values())), fontweight="bold")
    ax.axis("off"), ax.set_title("Word cloud (size = frequency)")
    plt.show()

# ------------------------------------------------------------------ main
def main():
    have_json = os.path.isdir(RAW_DIR) and any(
        fn.endswith(".json") for fn in os.listdir(RAW_DIR))
    if have_json:
        print(f"Using locally saved JSON in {RAW_DIR}")
    else:
        if "YT_API_KEY" not in os.environ:
            raise SystemExit(f"No JSON in {RAW_DIR}/ and no YT_API_KEY set - "
                             "nothing to analyze.")
        print("No local JSON found - running live ingestion.")
        ingest()

    df, funnel = build_dataset()
    df = add_sentiment(df)
    basic_stats(df)

    plot_funnel(funnel)
    plot_sentiment_pie(df)
    plot_timeseries(df)
    plot_lengths(df)
    plot_hour_heatmap(df)
    plot_emoji(df)
    plot_wordcloud(df)

if __name__ == "__main__":
    main()
