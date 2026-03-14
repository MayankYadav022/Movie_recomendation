import os
import pickle
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")

df: Optional[pd.DataFrame] = None
title_to_idx: Dict[str, int] = {}
tfidf_matrix: Any = None


class RecItem(BaseModel):
    title: str
    score: float


class RecommendResponse(BaseModel):
    query_title: str
    recommendations: List[RecItem]


def _norm(text: str) -> str:
    return str(text).strip().lower()


def _build_title_to_idx(indices_obj: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}

    if isinstance(indices_obj, dict):
        for k, v in indices_obj.items():
            out[_norm(k)] = int(v)
        return out

    # Handles pandas Series-like objects
    if hasattr(indices_obj, "items"):
        for k, v in indices_obj.items():
            out[_norm(k)] = int(v)
        return out

    raise RuntimeError("indices.pkl must be a dict or pandas Series-like object.")


def _recommend_from_title(title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    global df, tfidf_matrix, title_to_idx

    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    idx = title_to_idx.get(_norm(title))
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Title not found: {title}")

    # If TF-IDF vectors are L2 normalized, dot product ~= cosine similarity
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)

    recs: List[Tuple[str, float]] = []
    for i in order:
        i = int(i)
        if i == int(idx):
            continue
        try:
            t = str(df.iloc[i]["title"])
        except Exception:
            continue
        recs.append((t, float(scores[i])))
        if len(recs) >= top_n:
            break

    return recs


@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, tfidf_matrix, title_to_idx

    try:
        with open(DF_PATH, "rb") as f:
            df = pickle.load(f)

        with open(INDICES_PATH, "rb") as f:
            indices_obj = pickle.load(f)

        with open(TFIDF_MATRIX_PATH, "rb") as f:
            tfidf_matrix = pickle.load(f)

        if df is None or "title" not in df.columns:
            raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column.")

        title_to_idx = _build_title_to_idx(indices_obj)
    except Exception as e:
        raise RuntimeError(f"Startup failed while loading .pkl files: {e}") from e

    yield


app = FastAPI(title="Movie Recommendation API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "movies_loaded": int(len(df)) if df is not None else 0,
        "index_size": int(len(title_to_idx)),
    }


@app.get("/titles", response_model=List[str])
def titles(
    query: str = Query("", description="Optional title prefix/contains search"),
    limit: int = Query(20, ge=1, le=100),
):
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")

    all_titles = [str(x) for x in df["title"].dropna().tolist()]

    q = _norm(query)
    if q:
        all_titles = [t for t in all_titles if q in _norm(t)]

    # unique while preserving order
    seen = set()
    out = []
    for t in all_titles:
        k = _norm(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= limit:
            break

    return out


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = _recommend_from_title(title, top_n=top_n)
    return RecommendResponse(
        query_title=title,
        recommendations=[RecItem(title=t, score=s) for t, s in recs],
    )