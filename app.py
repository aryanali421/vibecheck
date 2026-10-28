
import os, re, json, html
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from urllib.parse import quote_plus

import requests
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq

st.set_page_config(page_title="VibeCheck", page_icon="🎬", layout="wide")

MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "").strip()

MOODS = ["Make me laugh", "Make me feel something", "Make me think", "Keep me on edge",
         "Inspire me", "Something romantic", "Take me somewhere else", "Comfort watch",
         "A good mystery", "Dark and intense"]
GENRES = ["Any", "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
          "Family", "Fantasy", "Horror", "Mystery", "Psychological", "Romance", "Sci-Fi", "Thriller"]

KIND_RULE = {
    "Movies": 'Movies only. Every item must have "type": "movie".',
    "Shows": 'TV series only. Every item must have "type": "series".',
    "Either": "A mix of movies and TV series.",
}

# ------------------------------------------------------------
# Data: OMDb + Groq
# ------------------------------------------------------------
def similar(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def omdb(**params):
    r = requests.get("https://www.omdbapi.com/",
                     params={"apikey": OMDB_API_KEY, **params}, timeout=10)
    r.raise_for_status()
    d = r.json()
    return d if d.get("Response") == "True" else None

@st.cache_data(ttl=86400, show_spinner=False)
def lookup(title, kind, year):
    if not OMDB_API_KEY:
        return {}
    t = "movie" if kind == "movie" else "series"
    try:
        d = omdb(t=title, type=t, y=year, plot="short") if year else None
        d = d or omdb(t=title, type=t, plot="short")
        if not d:
            s = omdb(s=title, type=t)
            if s:
                hits = [h for h in s.get("Search", [])
                        if similar(h.get("Title", ""), title) > 0.85]
                if hits:
                    d = omdb(i=hits[0]["imdbID"], plot="short")
        if d and similar(d.get("Title", ""), title) > 0.6:
            return d
    except Exception:
        pass
    return {}

@st.cache_resource
def groq_client():
    return Groq(api_key=GROQ_API_KEY)

def generate(mood, category, genre, prefs, avoid):
    avoid_line = ("Do NOT suggest any of these (already shown): " + "; ".join(avoid)) if avoid else ""
    prompt = f"""Recommend 8 real, existing titles.

Mood: {mood}
Format: {KIND_RULE.get(category, KIND_RULE["Either"])}
Genre: {genre}
Extra notes from the viewer: {prefs.strip() or "none"}
{avoid_line}

Rules:
- Only real titles with their real release year. No invented titles.
- Mix well-known picks with a couple of less obvious ones.
- "reason" must be specific to this mood, 1-2 sentences, no spoilers, no filler like "perfect for" or "a must-watch".
- "pitch" is a one-sentence premise.

Return ONLY JSON:
{{"recommendations":[{{"title":"","year":2010,"type":"movie","reason":"","pitch":""}}]}}"""

    kwargs = dict(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a film-literate friend recommending what to watch. Be specific and honest. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=4000,
    )
    c = groq_client()
    try:
        res = c.chat.completions.create(
            **kwargs, response_format={"type": "json_object"}, reasoning_effort="low")
    except Exception:
        res = c.chat.completions.create(**kwargs)

    raw = (res.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        a, b = raw.find("{"), raw.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("The model sent back something unreadable. Try again.")
        data = json.loads(raw[a:b + 1])

    items, seen_titles = [], set()
    for r in data.get("recommendations", []):
        title = str(r.get("title", "")).strip()
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        try:
            year = int(r.get("year"))
        except (TypeError, ValueError):
            year = None
        kind = "series" if str(r.get("type", "")).lower() in ("series", "tv", "show") else "movie"
        items.append({"title": title, "year": year, "type": kind,
                      "reason": str(r.get("reason", "")), "pitch": str(r.get("pitch", ""))})
    if not items:
        raise ValueError("No picks came back. Try again.")
    return items

def build_picks(items):
    with ThreadPoolExecutor(max_workers=8) as pool:
        details = list(pool.map(lambda i: lookup(i["title"], i["type"], i["year"]), items))
    picks = list(zip(items, details))
    if OMDB_API_KEY:
        verified = [p for p in picks if p[1]]
        if len(verified) >= 3:
            picks = verified
    return picks[:6]

# ------------------------------------------------------------
# HTML helpers (kept on single lines so markdown doesn't mangle them)
# ------------------------------------------------------------
def esc(v):
    return html.escape(str(v or ""), quote=True)

def clean(v):
    return "" if v in (None, "N/A") else str(v)

MARK_SVG = (
    '<svg class="vc-mark" viewBox="0 0 48 48" width="40" height="40" fill="none" aria-hidden="true">'
    '<rect x="3" y="3" width="42" height="42" rx="12" stroke="#ece7df" stroke-width="2.5"/>'
    '<circle cx="11" cy="11" r="1.6" fill="#ece7df" opacity=".5"/>'
    '<circle cx="37" cy="11" r="1.6" fill="#ece7df" opacity=".5"/>'
    '<circle cx="11" cy="37" r="1.6" fill="#ece7df" opacity=".5"/>'
    '<circle cx="37" cy="37" r="1.6" fill="#ece7df" opacity=".5"/>'
    '<path class="vc-tick" pathLength="1" d="M14 25.5 L21.5 33 L35 15.5" stroke="#e8b04a" '
    'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

def poster_html(d, title):
    poster = clean(d.get("Poster"))
    img = (f'<img src="{esc(poster)}" alt="" loading="lazy" referrerpolicy="no-referrer">'
           if poster.startswith("https://") else "")
    return f'<div class="vc-poster"><div class="vc-ph">{esc(title)}</div>{img}</div>'

def body_html(item, d):
    title = d.get("Title") or item["title"]
    year = clean(d.get("Year")) or (item["year"] or "")
    kind = "Series" if item["type"] == "series" else "Film"
    meta = [str(year), kind]
    if clean(d.get("Runtime")):
        meta.append(clean(d.get("Runtime")))
    if clean(d.get("imdbRating")):
        meta.append("★ " + clean(d.get("imdbRating")))
    meta_html = " · ".join(esc(m) for m in meta if m)

    imdb_id = d.get("imdbID", "")
    imdb = (f"https://www.imdb.com/title/{imdb_id}/" if re.fullmatch(r"tt\d+", str(imdb_id))
            else "https://www.imdb.com/find/?q=" + quote_plus(title))
    trailer = "https://www.youtube.com/results?search_query=" + quote_plus(f"{title} {year} trailer")

    synopsis = item["pitch"] or clean(d.get("Plot"))
    details = (f'<details><summary>Premise</summary><p>{esc(synopsis)}</p></details>' if synopsis else "")

    return (
        f'<div class="vc-meta">{meta_html}</div>'
        f'<h3 class="vc-title">{esc(title)}</h3>'
        f'<div class="vc-genre">{esc(clean(d.get("Genre")))}</div>'
        f'<p class="vc-reason">{esc(item["reason"])}</p>'
        f'{details}'
        f'<div class="vc-links"><a href="{esc(imdb)}" target="_blank" rel="noopener noreferrer">IMDb</a>'
        f'<a href="{esc(trailer)}" target="_blank" rel="noopener noreferrer">Trailer</a></div>'
    ), title, imdb

SKELETON = (
    '<div class=
