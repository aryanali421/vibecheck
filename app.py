import os, re, json, html
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from urllib.parse import quote_plus

import requests
import streamlit as st
from groq import Groq

st.set_page_config(page_title="VibeCheck", page_icon="🎬", layout="wide")


def get_secret(name):
    v = os.environ.get(name, "").strip()
    if v:
        return v
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = get_secret("GROQ_API_KEY")
OMDB_API_KEY = get_secret("OMDB_API_KEY")

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
    '<div class="vc-card vc-skel">'
    '<div class="vc-poster"></div>'
    '<div class="vc-body">'
    '<div class="vc-bar" style="width:40%"></div>'
    '<div class="vc-bar" style="width:75%;height:22px"></div>'
    '<div class="vc-bar" style="width:30%"></div>'
    '<div class="vc-bar" style="width:100%"></div>'
    '<div class="vc-bar" style="width:85%"></div>'
    '</div></div>'
)


def card_html(item, d):
    body, title, _ = body_html(item, d)
    return (f'<div class="vc-card">{poster_html(d, title)}'
            f'<div class="vc-body">{body}</div></div>')


def render_grid(cards):
    st.markdown('<div class="vc-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# Styles
# ------------------------------------------------------------
CSS = """<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 1200px;}
.vc-head {display:flex; align-items:center; gap:14px; margin-bottom:4px;}
.vc-name {font-size:2rem; font-weight:700; letter-spacing:-0.02em; color:#ece7df; margin:0;}
.vc-sub {color:#9a948a; margin:0 0 1.4rem 0; font-size:1rem;}
.vc-tick {stroke-dasharray:1; stroke-dashoffset:1; animation: vc-draw 1.1s ease-out .2s forwards;}
@keyframes vc-draw {to {stroke-dashoffset:0;}}
.vc-grid {display:grid; grid-template-columns:repeat(auto-fill, minmax(420px, 1fr)); gap:18px; margin-top:12px;}
.vc-card {display:flex; gap:16px; background:#171615; border:1px solid #2a2826; border-radius:16px; padding:14px; transition:border-color .2s, transform .2s;}
.vc-card:hover {border-color:#e8b04a66; transform:translateY(-2px);}
.vc-poster {position:relative; flex:0 0 120px; height:178px; border-radius:10px; overflow:hidden; background:#23211f;}
.vc-poster img {position:absolute; inset:0; width:100%; height:100%; object-fit:cover;}
.vc-ph {position:absolute; inset:0; display:flex; align-items:center; justify-content:center; text-align:center; padding:10px; font-size:.8rem; color:#6d675e;}
.vc-body {flex:1; min-width:0;}
.vc-meta {font-size:.78rem; color:#e8b04a; letter-spacing:.03em; margin-bottom:4px;}
.vc-title {font-size:1.15rem; font-weight:650; color:#ece7df; margin:0 0 2px 0; line-height:1.25;}
.vc-genre {font-size:.8rem; color:#8c867c; margin-bottom:8px;}
.vc-reason {font-size:.92rem; line-height:1.5; color:#cfc9bf; margin:0 0 8px 0;}
.vc-body details {margin-bottom:8px;}
.vc-body summary {cursor:pointer; font-size:.82rem; color:#9a948a;}
.vc-body details p {font-size:.88rem; color:#b5afa5; margin:6px 0 0 0;}
.vc-links {display:flex; gap:8px;}
.vc-links a {font-size:.78rem; text-decoration:none; color:#ece7df; border:1px solid #3a3733; padding:4px 10px; border-radius:999px; transition:all .15s;}
.vc-links a:hover {border-color:#e8b04a; color:#e8b04a;}
.vc-skel .vc-poster {background:linear-gradient(90deg,#1d1b1a,#2a2826,#1d1b1a); background-size:200% 100%; animation: vc-shimmer 1.3s infinite;}
.vc-bar {height:12px; border-radius:6px; margin-bottom:10px; background:linear-gradient(90deg,#1d1b1a,#2a2826,#1d1b1a); background-size:200% 100%; animation: vc-shimmer 1.3s infinite;}
@keyframes vc-shimmer {0% {background-position:200% 0;} 100% {background-position:-200% 0;}}
@media (max-width: 520px) {.vc-grid {grid-template-columns:1fr;} .vc-poster {flex-basis:96px; height:142px;}}
</style>"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    f'<div class="vc-head">{MARK_SVG}<h1 class="vc-name">VibeCheck</h1></div>'
    '<p class="vc-sub">Tell me your mood. I\'ll tell you what to watch.</p>',
    unsafe_allow_html=True)

# ------------------------------------------------------------
# State
# ------------------------------------------------------------
st.session_state.setdefault("picks", [])
st.session_state.setdefault("shown", [])
st.session_state.setdefault("error", "")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Add it in Streamlit Cloud under Settings → Secrets.")
    st.stop()

# ------------------------------------------------------------
# Controls
# ------------------------------------------------------------
c1, c2, c3 = st.columns([2, 1.3, 1.5])
with c1:
    mood = st.selectbox("Mood", MOODS)
with c2:
    category = st.radio("Format", ["Movies", "Shows", "Either"], horizontal=True, index=2)
with c3:
    genre = st.selectbox("Genre", GENRES)

prefs = st.text_input("Anything else? (optional)",
                      placeholder="e.g. nothing too long, loved Parasite, no jump scares")

b1, b2, _ = st.columns([1, 1, 4])
go = b1.button("Find my picks", type="primary", use_container_width=True)
more = b2.button("Show different ones", use_container_width=True,
                 disabled=not st.session_state["picks"])


def run(avoid):
    placeholder = st.empty()
    placeholder.markdown('<div class="vc-grid">' + SKELETON * 6 + "</div>", unsafe_allow_html=True)
    try:
        items = generate(mood, category, genre, prefs, avoid)
        picks = build_picks(items)
        st.session_state["picks"] = picks
        st.session_state["shown"] = avoid + [i["title"] for i, _ in picks]
        st.session_state["error"] = ""
    except Exception as e:
        st.session_state["error"] = str(e) or "Something went wrong. Try again."
    placeholder.empty()


if go:
    run([])
elif more:
    run(list(st.session_state["shown"]))

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
if st.session_state["error"]:
    st.error(st.session_state["error"])
elif st.session_state["picks"]:
    render_grid([card_html(item, d) for item, d in st.session_state["picks"]])
    if not OMDB_API_KEY:
        st.caption("No OMDB_API_KEY set, so posters and ratings are off. Picks still work.")
