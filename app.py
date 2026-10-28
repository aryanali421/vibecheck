import os, re, json, html
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from urllib.parse import quote_plus

import requests
import streamlit as st
import streamlit.components.v1 as components
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


MODEL = "gpt-oss-120b"
GROQ_API_KEY = get_secret("GROQ_API_KEY")
OMDB_API_KEY = get_secret("OMDB_API_KEY")

# Each mood owns a color. The whole UI (and the cursor) shifts to it.
MOOD_COLORS = {
    "Make me laugh": "#ffcf3f",
    "Make me feel something": "#ff6b8b",
    "Make me think": "#6ea8ff",
    "Keep me on edge": "#ff5a3c",
    "Inspire me": "#3fe0b0",
    "Something romantic": "#ff7eb6",
    "Take me somewhere else": "#9b7bff",
    "Comfort watch": "#ffb27a",
    "A good mystery": "#7fd4e8",
    "Dark and intense": "#e03a5a",
}
MOODS = list(MOOD_COLORS.keys())
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
# HTML helpers (single lines so markdown doesn't mangle them)
# ------------------------------------------------------------
def esc(v):
    return html.escape(str(v or ""), quote=True)


def clean(v):
    return "" if v in (None, "N/A") else str(v)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return ", ".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))


MARK_SVG = (
    '<svg class="vc-mark" viewBox="0 0 48 48" width="44" height="44" fill="none" aria-hidden="true">'
    '<rect x="3" y="3" width="42" height="42" rx="12" stroke="#ece7df" stroke-width="2.5"/>'
    '<circle cx="11" cy="11" r="1.6" fill="#ece7df" opacity=".5"/>'
    '<circle cx="37" cy="11" r="1.6" fill="#ece7df" opacity=".5"/>'
    '<circle cx="11" cy="37" r="1.6" fill="#ece7df" opacity=".5"/>'
    '<circle cx="37" cy="37" r="1.6" fill="#ece7df" opacity=".5"/>'
    '<path class="vc-tick" pathLength="1" d="M14 25.5 L21.5 33 L35 15.5" stroke="var(--accent)" '
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


def card_html(item, d, idx):
    body, title, _ = body_html(item, d)
    return (f'<div class="vc-card" style="--i:{idx}"><div class="vc-glow"></div>'
            f'{poster_html(d, title)}<div class="vc-body">{body}</div></div>')


def render_grid(cards):
    st.markdown('<div class="vc-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# Styles
# ------------------------------------------------------------
CSS = """<style>
:root {--accent: __ACCENT__; --accent-rgb: __RGB__;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}
.stApp {background: #0b0a0a; overflow-x: hidden;}
.block-container {padding-top: 2.4rem; max-width: 1200px; position: relative; z-index: 2;}

/* aurora background */
.stApp::before {content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
background:
radial-gradient(40% 35% at 20% 25%, rgba(var(--accent-rgb), .22), transparent 70%),
radial-gradient(35% 30% at 80% 20%, rgba(120, 90, 255, .14), transparent 70%),
radial-gradient(45% 40% at 60% 85%, rgba(var(--accent-rgb), .14), transparent 70%);
filter: blur(40px); animation: vc-drift 22s ease-in-out infinite alternate; transition: background 1s;}
@keyframes vc-drift {0% {transform: translate3d(0,0,0) scale(1);} 100% {transform: translate3d(3%,-4%,0) scale(1.12);}}
/* film grain */
.stApp::after {content:""; position:fixed; inset:0; z-index:1; pointer-events:none; opacity:.07;
background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");}

/* hero */
.vc-head {display:flex; align-items:center; gap:16px; margin-bottom:2px;}
.vc-name {font-size:2.7rem; font-weight:800; letter-spacing:-0.035em; margin:0; line-height:1.1;
background: linear-gradient(100deg, #ece7df 0%, var(--accent) 45%, #ece7df 90%); background-size:220% 100%;
-webkit-background-clip:text; background-clip:text; color:transparent; animation: vc-sheen 6s linear infinite;}
@keyframes vc-sheen {to {background-position: -220% 0;}}
.vc-sub {color:#9a948a; margin:0 0 1.6rem 0; font-size:1.05rem;}
.vc-sub b {color: var(--accent); font-weight:600; transition: color .6s;}
.vc-tick {stroke-dasharray:1; stroke-dashoffset:1; animation: vc-draw 1.2s cubic-bezier(.6,0,.2,1) .3s forwards;}
@keyframes vc-draw {to {stroke-dashoffset:0;}}
.vc-mark {filter: drop-shadow(0 0 14px rgba(var(--accent-rgb), .55)); animation: vc-float 5s ease-in-out infinite;}
@keyframes vc-float {50% {transform: translateY(-4px) rotate(-3deg);}}

/* widgets */
label, [data-testid="stWidgetLabel"] p {color:#9a948a !important; font-size:.78rem !important; letter-spacing:.08em; text-transform:uppercase;}
[data-baseweb="select"] > div, .stTextInput input {background: rgba(255,255,255,.035) !important; border:1px solid rgba(255,255,255,.09) !important;
border-radius:14px !important; backdrop-filter: blur(12px); transition: border-color .25s, box-shadow .25s, background .25s;}
[data-baseweb="select"] > div:hover, .stTextInput input:hover {border-color: rgba(var(--accent-rgb), .55) !important;}
.stTextInput input:focus {border-color: var(--accent) !important; box-shadow: 0 0 0 4px rgba(var(--accent-rgb), .16) !important;}
.stButton > button {border-radius:14px; border:1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.04); color:#ece7df;
font-weight:600; padding:.6rem 1rem; transition: transform .2s cubic-bezier(.2,.8,.2,1), box-shadow .25s, border-color .25s;}
.stButton > button:hover {transform: translateY(-2px); border-color: var(--accent); box-shadow: 0 10px 30px -10px rgba(var(--accent-rgb), .6);}
.stButton > button:active {transform: translateY(0) scale(.98);}
.stButton > button[kind="primary"] {background: linear-gradient(120deg, var(--accent), rgba(var(--accent-rgb), .6)); color:#14110d; border:none;
box-shadow: 0 8px 30px -8px rgba(var(--accent-rgb), .8);}
.stButton > button:disabled {opacity:.35;}

/* cards */
.vc-grid {display:grid; grid-template-columns:repeat(auto-fill, minmax(430px, 1fr)); gap:20px; margin-top:18px; perspective:1200px;}
.vc-card {position:relative; display:flex; gap:18px; padding:16px; border-radius:20px; overflow:hidden;
background: linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.015));
border:1px solid rgba(255,255,255,.08); backdrop-filter: blur(16px);
opacity:0; animation: vc-rise .8s cubic-bezier(.2,.8,.2,1) forwards; animation-delay: calc(var(--i, 0) * 90ms);
transition: transform .18s ease-out, border-color .3s, box-shadow .3s; transform-style: preserve-3d; will-change: transform;}
@keyframes vc-rise {from {opacity:0; transform: translateY(28px) scale(.96); filter: blur(6px);} to {opacity:1; transform:none; filter:none;}}
.vc-card:hover {border-color: rgba(var(--accent-rgb), .5); box-shadow: 0 24px 60px -24px rgba(var(--accent-rgb), .55);}
.vc-glow {position:absolute; inset:0; pointer-events:none; opacity:0; transition: opacity .3s;
background: radial-gradient(360px circle at var(--mx, 50%) var(--my, 50%), rgba(var(--accent-rgb), .16), transparent 60%);}
.vc-card:hover .vc-glow {opacity:1;}
.vc-poster {position:relative; flex:0 0 124px; height:184px; border-radius:12px; overflow:hidden; background:#1f1d1b;
box-shadow: 0 14px 30px -12px rgba(0,0,0,.8); transform: translateZ(30px);}
.vc-poster img {position:absolute; inset:0; width:100%; height:100%; object-fit:cover; transition: transform .6s cubic-bezier(.2,.8,.2,1);}
.vc-card:hover .vc-poster img {transform: scale(1.08);}
.vc-ph {position:absolute; inset:0; display:flex; align-items:center; justify-content:center; text-align:center; padding:10px; font-size:.8rem; color:#6d675e;}
.vc-body {position:relative; flex:1; min-width:0;}
.vc-meta {font-size:.76rem; color: var(--accent); letter-spacing:.06em; text-transform:uppercase; margin-bottom:5px; transition: color .6s;}
.vc-title {font-size:1.22rem; font-weight:700; color:#f3eee6; margin:0 0 2px 0; line-height:1.25; letter-spacing:-0.01em;}
.vc-genre {font-size:.8rem; color:#8c867c; margin-bottom:9px;}
.vc-reason {font-size:.92rem; line-height:1.55; color:#cfc9bf; margin:0 0 9px 0;}
.vc-body details {margin-bottom:10px;}
.vc-body summary {cursor:pointer; font-size:.8rem; color:#9a948a; transition: color .2s;}
.vc-body summary:hover {color: var(--accent);}
.vc-body details p {font-size:.88rem; color:#b5afa5; margin:6px 0 0 0; animation: vc-rise .4s ease both;}
.vc-links {display:flex; gap:8px;}
.vc-links a {font-size:.78rem; text-decoration:none; color:#ece7df; border:1px solid rgba(255,255,255,.14); padding:5px 12px; border-radius:999px;
transition: all .2s;}
.vc-links a:hover {border-color: var(--accent); color:#14110d; background: var(--accent);}

/* skeleton */
.vc-skel {opacity:1; animation:none;}
.vc-skel .vc-poster, .vc-bar {background: linear-gradient(90deg, #1b1918, #2a2725, #1b1918); background-size:200% 100%; animation: vc-shimmer 1.3s infinite;}
.vc-bar {height:12px; border-radius:6px; margin-bottom:11px;}
@keyframes vc-shimmer {0% {background-position:200% 0;} 100% {background-position:-200% 0;}}

/* custom cursor (desktop only) */
@media (pointer: fine) {html, body, .stApp, .stApp * {cursor: none !important;}}
.vc-cur-dot, .vc-cur-ring {position:fixed; left:0; top:0; pointer-events:none; z-index:2147483647; opacity:0; will-change: transform;}
.vc-cur-dot {width:8px; height:8px; margin:-4px 0 0 -4px; border-radius:50%; background: var(--accent); box-shadow: 0 0 16px 2px var(--accent);
transition: opacity .25s, background .5s, box-shadow .5s, width .2s, height .2s, margin .2s;}
.vc-cur-ring {width:40px; height:40px; margin:-20px 0 0 -20px; border-radius:50%; border:1.5px solid rgba(var(--accent-rgb), .75);
display:flex; align-items:center; justify-content:center; font:600 10px/1 system-ui, sans-serif; letter-spacing:.06em; color: var(--accent);
transition: opacity .25s, width .3s cubic-bezier(.2,.8,.2,1), height .3s cubic-bezier(.2,.8,.2,1), margin .3s cubic-bezier(.2,.8,.2,1), background .3s, border-color .5s;}
.vc-cur-ring.hot {width:68px; height:68px; margin:-34px 0 0 -34px; background: rgba(var(--accent-rgb), .12);}
.vc-cur-ring.card {width:90px; height:90px; margin:-45px 0 0 -45px; background: rgba(var(--accent-rgb), .07); border-style:dashed;}
.vc-cur-ring.down {width:26px; height:26px; margin:-13px 0 0 -13px; background: rgba(var(--accent-rgb), .3);}
.vc-cur-dot.hot {width:5px; height:5px; margin:-2.5px 0 0 -2.5px;}
.vc-spark, .vc-burst {position:fixed; left:0; top:0; pointer-events:none; z-index:2147483646; border-radius:50%; background: var(--accent);}

@media (max-width: 520px) {.vc-grid {grid-template-columns:1fr;} .vc-poster {flex-basis:96px; height:142px;} .vc-name {font-size:2.1rem;}}
.element-container:has(iframe[height="0"]), [data-testid="stElementContainer"]:has(iframe[height="0"]) {height:0; margin:0; overflow:hidden; position:absolute;}
</style>"""

CURSOR_JS = """
<script>
(function () {
  var p = window.parent, d = p.document;
  if (p.__vcCursor) return;
  if (!p.matchMedia('(pointer: fine)').matches) return;
  p.__vcCursor = true;

  var dot = d.createElement('div'); dot.className = 'vc-cur-dot';
  var ring = d.createElement('div'); ring.className = 'vc-cur-ring';
  d.body.appendChild(dot); d.body.appendChild(ring);

  var x = -100, y = -100, rx = -100, ry = -100, lastSpark = 0, shown = false, activeCard = null;

  d.addEventListener('mousemove', function (e) {
    x = e.clientX; y = e.clientY;
    if (!shown) { shown = true; dot.style.opacity = 1; ring.style.opacity = 1; rx = x; ry = y; }
    dot.style.transform = 'translate(' + x + 'px,' + y + 'px)';

    var t = e.target;
    var hot = t.closest && t.closest('a, button, summary, input, [role="button"], [data-baseweb="select"], [data-baseweb="popover"] li');
    var card = t.closest && t.closest('.vc-card');
    ring.classList.toggle('hot', !!hot);
    dot.classList.toggle('hot', !!hot);
    ring.classList.toggle('card', !!card && !hot);
    ring.textContent = (hot && hot.closest && hot.closest('.vc-links')) ? 'OPEN ↗' : '';

    if (card !== activeCard) {
      if (activeCard) activeCard.style.transform = '';
      activeCard = card;
    }
    if (card) {
      var r = card.getBoundingClientRect();
      var px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
      card.style.setProperty('--mx', (px * 100) + '%');
      card.style.setProperty('--my', (py * 100) + '%');
      card.style.transform = 'perspective(900px) rotateX(' + ((0.5 - py) * 7) + 'deg) rotateY(' + ((px - 0.5) * 9) + 'deg) translateY(-4px)';
    }

    var now = performance.now();
    if (now - lastSpark > 45) {
      lastSpark = now;
      var s = d.createElement('div'); s.className = 'vc-spark';
      var size = 3 + Math.random() * 4;
      s.style.width = s.style.height = size + 'px';
      s.style.transform = 'translate(' + (x - size / 2) + 'px,' + (y - size / 2) + 'px)';
      d.body.appendChild(s);
      var a = s.animate([
        { opacity: .55, transform: s.style.transform + ' scale(1)' },
        { opacity: 0, transform: 'translate(' + (x - size / 2 + (Math.random() - .5) * 16) + 'px,' + (y - size / 2 + 14) + 'px) scale(.2)' }
      ], { duration: 650, easing: 'ease-out' });
      a.onfinish = function () { s.remove(); };
    }
  });

  d.addEventListener('mouseleave', function () {
    dot.style.opacity = 0; ring.style.opacity = 0; shown = false;
    if (activeCard) { activeCard.style.transform = ''; activeCard = null; }
  });
  d.addEventListener('mouseenter', function () { dot.style.opacity = 1; ring.style.opacity = 1; shown = true; });
  d.addEventListener('mousedown', function () { ring.classList.add('down'); });
  d.addEventListener('mouseup', function () { ring.classList.remove('down'); });

  d.addEventListener('click', function (e) {
    for (var i = 0; i < 12; i++) {
      var b = d.createElement('div'); b.className = 'vc-burst';
      var size = 3 + Math.random() * 4;
      b.style.width = b.style.height = size + 'px';
      d.body.appendChild(b);
      var ang = (Math.PI * 2 * i) / 12 + Math.random() * .4, dist = 28 + Math.random() * 34;
      var a = b.animate([
        { transform: 'translate(' + e.clientX + 'px,' + e.clientY + 'px) scale(1)', opacity: 1 },
        { transform: 'translate(' + (e.clientX + Math.cos(ang) * dist) + 'px,' + (e.clientY + Math.sin(ang) * dist) + 'px) scale(0)', opacity: 0 }
      ], { duration: 600, easing: 'cubic-bezier(.2,.8,.2,1)' });
      a.onfinish = (function (el) { return function () { el.remove(); }; })(b);
    }
  });

  (function loop() {
    rx += (x - rx) * 0.16; ry += (y - ry) * 0.16;
    ring.style.transform = 'translate(' + rx + 'px,' + ry + 'px)';
    p.requestAnimationFrame(loop);
  })();
})();
</script>
"""

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
# UI
# ------------------------------------------------------------
st.markdown(
    f'<div class="vc-head">{MARK_SVG}<h1 class="vc-name">VibeCheck</h1></div>'
    '<p class="vc-sub">Tell me your mood. I\'ll tell you what to <b>watch</b>.</p>',
    unsafe_allow_html=True)

c1, c2, c3 = st.columns([2, 1.3, 1.5])
with c1:
    mood = st.selectbox("Mood", MOODS)
with c2:
    category = st.radio("Format", ["Movies", "Shows", "Either"], horizontal=True, index=2)
with c3:
    genre = st.selectbox("Genre", GENRES)

prefs = st.text_input("Anything else? (optional)",
                      placeholder="e.g. nothing too long, loved Parasite, no jump scares")

b1, b2, _ = st.columns([1, 1.2, 4])
go = b1.button("Find my picks", type="primary", use_container_width=True)
more = b2.button("Show different ones", use_container_width=True,
                 disabled=not st.session_state["picks"])

# Theme + cursor, driven by the chosen mood
accent = MOOD_COLORS[mood]
st.markdown(CSS.replace("__ACCENT__", accent).replace("__RGB__", hex_to_rgb(accent)),
            unsafe_allow_html=True)
components.html(CURSOR_JS, height=0)


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

if st.session_state["error"]:
    st.error(st.session_state["error"])
elif st.session_state["picks"]:
    render_grid([card_html(item, d, i) for i, (item, d) in enumerate(st.session_state["picks"])])
    if not OMDB_API_KEY:
        st.caption("No OMDB_API_KEY set, so posters and ratings are off. Picks still work.")
