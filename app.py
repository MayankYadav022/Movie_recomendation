import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # <-- important for local .env in Streamlit

DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_TMDB_BASE = "https://api.themoviedb.org/3"
DEFAULT_TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"  # standard poster size


def _clean_secret(v: str) -> str:
    v = (v or "").strip()
    if len(v) >= 2 and ((v[0] == "'" and v[-1] == "'") or (v[0] == '"' and v[-1] == '"')):
        v = v[1:-1].strip()
    return v


def _get_secret_or_env(name: str, default: str = "") -> str:
    try:
        sv = st.secrets.get(name, "")
    except Exception:
        sv = ""
    ev = os.getenv(name, "")
    return _clean_secret(sv) or _clean_secret(ev) or default


API_BASE = _get_secret_or_env("API_BASE", DEFAULT_API_BASE)
API_FALLBACK = _get_secret_or_env("API_FALLBACK", "")
TMDB_API_KEY = _get_secret_or_env("TMDB_API_KEY", "")
TMDB_BASE = _get_secret_or_env("TMDB_BASE_URL", DEFAULT_TMDB_BASE)
TMDB_IMAGE_BASE = _get_secret_or_env("TMDB_IMAGE_BASE_URL", DEFAULT_TMDB_IMAGE_BASE)

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System")
st.caption("TF-IDF recommendations + TMDB posters and movie details")


@st.cache_data(ttl=30)
def api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[str]]:
    bases = [API_BASE] + ([API_FALLBACK] if API_FALLBACK else [])
    last_err = "Unknown error"

    for base in bases:
        for _ in range(2):
            try:
                r = requests.get(f"{base}{path}", params=params, timeout=(10, 60))
                if r.status_code < 400:
                    return r.json(), None
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except requests.exceptions.RequestException as e:
                last_err = str(e)
    return None, last_err


def _normalize_title_for_tmdb(title: str) -> str:
    # remove year in brackets: "Movie (1995)" -> "Movie"
    t = re.sub(r"\(\d{4}\)", "", title or "").strip()
    # keep letters/numbers/basic punctuation, collapse spaces
    t = re.sub(r"\s+", " ", t)
    return t


@st.cache_data(ttl=3600)
def tmdb_search_movie(title: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not TMDB_API_KEY:
        return None, "TMDB_API_KEY missing"

    q1 = _normalize_title_for_tmdb(title)
    q2 = (title or "").strip()

    for q in [q1, q2]:
        if not q:
            continue
        try:
            r = requests.get(
                f"{TMDB_BASE}/search/movie",
                params={
                    "api_key": TMDB_API_KEY,
                    "query": q,
                    "include_adult": "false",
                    "language": "en-US",
                    "page": 1,
                },
                timeout=20,
            )

            if r.status_code == 401:
                return None, "TMDB auth failed (invalid/revoked API key)"
            if r.status_code >= 400:
                continue

            results = r.json().get("results", [])
            if results:
                return results[0], None

        except requests.exceptions.RequestException:
            continue

    return None, "No TMDB match found"


def tmdb_poster_url(poster_path: Optional[str]) -> Optional[str]:
    if not poster_path:
        return None
    return f"{TMDB_IMAGE_BASE}{poster_path}"


def render_movie_card(title: str, subtitle: str = "") -> None:
    data, tmdb_err = tmdb_search_movie(title)
    poster = tmdb_poster_url(data.get("poster_path")) if data else None
    overview = (data.get("overview") or "Overview not available.") if data else "Overview not available."
    release_date = data.get("release_date", "N/A") if data else "N/A"
    rating = data.get("vote_average", "N/A") if data else "N/A"
    lang = data.get("original_language", "N/A") if data else "N/A"

    with st.container(border=True):
        if subtitle:
            st.caption(subtitle)
        st.markdown(f"**{title}**")
        if poster:
            st.image(poster, width=220)  # fixed width to avoid blurry oversized images
        st.write(overview[:280] + ("..." if len(overview) > 280 else ""))
        st.caption(f"Release: {release_date} • Rating: {rating} • Lang: {lang}")
        if tmdb_err and tmdb_err.startswith("TMDB auth failed"):
            st.warning(tmdb_err)


with st.sidebar:
    st.markdown("### Backend")
    st.code(API_BASE, language="text")
    if st.button("Check API health"):
        data, err = api_get("/health")
        if err:
            st.error(err)
        else:
            st.success(f"Connected. Movies loaded: {data.get('movies_loaded', 0)}")

    st.markdown("### TMDB")
    if TMDB_API_KEY:
        st.success("TMDB key found")
    else:
        st.warning("TMDB_API_KEY missing (posters/details disabled)")


typed = st.text_input("Search movie title", placeholder="e.g. Avatar, Batman, Inception")
top_n = st.slider("Number of recommendations", 5, 20, 10)

suggestions: List[str] = []
if typed.strip():
    suggestions_data, err = api_get("/titles", {"query": typed.strip(), "limit": 30})
    if err:
        st.warning(f"Suggestion error: {err}")
    else:
        suggestions = suggestions_data or []

if not suggestions:
    fallback, _ = api_get("/titles", {"limit": 30})
    suggestions = fallback or []

selected_title = st.selectbox("Pick a movie", options=suggestions if suggestions else [""])

if st.button("Recommend", type="primary"):
    if not selected_title:
        st.warning("Select a movie title first.")
    else:
        data, err = api_get("/recommend", {"title": selected_title, "top_n": top_n})
        if err:
            st.error(err)
        elif not data or not data.get("recommendations"):
            st.info("No recommendations found.")
        else:
            st.subheader("Selected Movie")
            render_movie_card(selected_title)

            st.subheader("Recommended for you")
            recs = data["recommendations"]
            cols = st.columns(3)
            for i, item in enumerate(recs, start=1):
                with cols[(i - 1) % 3]:
                    render_movie_card(item["title"], subtitle=f"Match score: {item['score']:.4f}")