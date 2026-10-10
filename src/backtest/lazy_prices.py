"""Lazy Prices features: how much did this year's report change versus last year's?

For every 10-K (vs the previous year's 10-K) and every 10-Q (vs the same
quarter a year earlier), per section (Risk Factors, MD&A):

    sim_tfidf_<s>   cosine similarity of TF-IDF vectors
    sim_emb_<s>     cosine similarity of sentence-embedding document vectors (10-K only: CPU budget)
    dsim_tfidf_<s>  change vs the previous comparison of the same form
    len_chg_<s>     log(length now / length a year earlier)

Point-in-time rules:
- a filing's features exist from its SEC filing date;
- the TF-IDF vocabulary and IDF weights are fit ONLY on filings made before
  `IDF_FIT_END` (2014), so later documents never shape the weights used to
  compare earlier ones;
- the embedding model (all-MiniLM-L6-v2) is a fixed, pretrained model -- no
  fitting on this corpus at all.

Monthly alignment (`monthly_features`): at each month-end, the latest filing
(either form) filed on or before it and within `MAX_AGE_DAYS`.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from backtest import protocol

logger = logging.getLogger(__name__)

SECTIONS = ("risk_factors", "mdna")
SHORT = {"risk_factors": "rf", "mdna": "mdna"}
IDF_FIT_END = pd.Timestamp("2014-01-01")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_CHARS = 1000  # ~250 tokens: fits MiniLM's 256-token window, nothing silently truncated
MAX_CHUNKS = 8
EMBED_FORMS = ("10-K",)  # CPU budget: embeddings for annual reports only; 10-Qs get TF-IDF similarity
MAX_AGE_DAYS = 200
FEATURES = [f"{kind}_{SHORT[s]}" for s in SECTIONS for kind in ("sim_tfidf", "sim_emb", "dsim_tfidf", "len_chg")]
EMB_CACHE = protocol.PHASE_DIR / "embeddings.parquet"


# ----------------------------------------------------------------------------
# Pairing each filing with its year-ago comparable
# ----------------------------------------------------------------------------


def pair_filings(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `prior_accession`: the same form one year earlier (10-K: previous 10-K filed 300-430 days
    before; 10-Q: the 10-Q whose period ended 350-380 days earlier, else filed 330-400 days earlier)."""
    f = frame.copy()
    f["filing_date"] = pd.to_datetime(f["filing_date"])
    f["report_date"] = pd.to_datetime(f["report_date"], errors="coerce")
    f["prior_accession"] = None
    for form, group in f.groupby("form"):
        group = group.sort_values("filing_date")
        for idx, row in group.iterrows():
            earlier = group[group["filing_date"] < row["filing_date"]]
            if form == "10-K":
                gap = (row["filing_date"] - earlier["filing_date"]).dt.days
                match = earlier[gap.between(300, 430)]
            else:
                gap = (row["report_date"] - earlier["report_date"]).dt.days if pd.notna(row["report_date"]) else pd.Series(dtype=float)
                match = earlier[gap.between(350, 380)] if len(gap) else earlier.iloc[0:0]
                if match.empty:
                    fgap = (row["filing_date"] - earlier["filing_date"]).dt.days
                    match = earlier[fgap.between(330, 400)]
            if len(match):
                f.at[idx, "prior_accession"] = match.sort_values("filing_date").iloc[-1]["accession"]
    return f


# ----------------------------------------------------------------------------
# Similarity measures
# ----------------------------------------------------------------------------


def fit_tfidf(reference_texts: list[str]):
    """TF-IDF fit on pre-2014 filings only (vocabulary + IDF), unigrams, English stop words removed."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.9, sublinear_tf=True, dtype=np.float32)
    vec.fit(reference_texts)
    return vec


def cosine_pairs(a, b) -> np.ndarray:
    """Row-wise cosine similarity of two equally shaped (sparse or dense) matrices."""
    if hasattr(a, "multiply"):
        num = np.asarray(a.multiply(b).sum(axis=1)).ravel()
        na = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel())
        nb = np.sqrt(np.asarray(b.multiply(b).sum(axis=1)).ravel())
    else:
        num = (a * b).sum(axis=1)
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((na > 0) & (nb > 0), num / (na * nb), np.nan)


def chunks(text: str, size: int = CHUNK_CHARS, limit: int = MAX_CHUNKS) -> list[str]:
    """Evenly spaced chunks across the whole section (not just its beginning)."""
    parts = [text[i:i + size] for i in range(0, len(text), size)]
    if len(parts) <= limit:
        return parts
    picks = np.linspace(0, len(parts) - 1, limit).round().astype(int)
    return [parts[i] for i in picks]


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _MODEL


def embed_documents(keys: list[tuple[str, str]], texts: list[str], batch_size: int = 64) -> dict:
    """Mean-pooled, L2-normalized chunk embeddings per (accession, section); cached on disk."""
    cache = pd.read_parquet(EMB_CACHE) if EMB_CACHE.exists() else pd.DataFrame(columns=["key", "vec"])
    known = dict(zip(cache["key"], cache["vec"]))
    out, todo = {}, []
    for key, text in zip(keys, texts):
        k = f"{key[0]}|{key[1]}"
        if k in known:
            out[key] = np.asarray(known[k], dtype=np.float32)
        elif text:
            todo.append((key, text))
    new_rows = []
    for start in range(0, len(todo), 200):
        block = todo[start:start + 200]
        all_chunks, owners = [], []
        for key, text in block:
            cs = chunks(text)
            all_chunks += cs
            owners += [key] * len(cs)
        vecs = _model().encode(all_chunks, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        frame = pd.DataFrame(vecs)
        frame["owner"] = [f"{o[0]}|{o[1]}" for o in owners]
        for owner, g in frame.groupby("owner", sort=False):
            v = g.drop(columns="owner").to_numpy(dtype=np.float32).mean(axis=0)
            v = v / (np.linalg.norm(v) or 1.0)
            acc, sec = owner.split("|")
            out[(acc, sec)] = v
            new_rows.append({"key": owner, "vec": v.tolist()})
        logger.info("embedded %d/%d documents", min(start + 200, len(todo)), len(todo))
        if new_rows and (start // 200) % 10 == 9:
            _save_embeddings(cache, new_rows)
    if new_rows:
        _save_embeddings(cache, new_rows)
    return out


def _save_embeddings(cache: pd.DataFrame, new_rows: list[dict]) -> None:
    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    merged = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates("key", keep="last")
    merged.to_parquet(EMB_CACHE, index=False)


# ----------------------------------------------------------------------------
# Per-filing features
# ----------------------------------------------------------------------------


def filing_features(sections: pd.DataFrame, vectorizer, use_embeddings: bool = True) -> pd.DataFrame:
    """One row per filing with a year-ago comparable: similarity, change in similarity, length change."""
    paired = pair_filings(sections)
    paired = paired[paired["prior_accession"].notna()].copy()
    by_acc = sections.set_index("accession")
    out = paired[["cik", "accession", "form", "filing_date", "prior_accession"]].copy()
    for section in SECTIONS:
        s = SHORT[section]
        now = paired[section].fillna("").tolist()
        before = [by_acc.at[a, section] if a in by_acc.index and isinstance(by_acc.at[a, section], str) else ""
                  for a in paired["prior_accession"]]
        both = np.array([bool(x) and bool(y) for x, y in zip(now, before)])
        sim = np.full(len(paired), np.nan)
        if both.any():
            a = vectorizer.transform([x for x, ok in zip(now, both) if ok])
            b = vectorizer.transform([y for y, ok in zip(before, both) if ok])
            sim[both] = cosine_pairs(a, b)
        out[f"sim_tfidf_{s}"] = sim
        out[f"len_chg_{s}"] = [math.log(len(x) / len(y)) if x and y else np.nan for x, y in zip(now, before)]
        if use_embeddings:
            embed = paired["form"].isin(EMBED_FORMS).to_numpy()
            keys_now = [(a, section) for a in paired["accession"]]
            keys_before = [(a, section) for a in paired["prior_accession"]]
            pick = [i for i, ok in enumerate(embed) if ok]
            vecs = embed_documents([keys_now[i] for i in pick] + [keys_before[i] for i in pick],
                                   [now[i] for i in pick] + [before[i] for i in pick])
            out[f"sim_emb_{s}"] = [float(np.dot(vecs[k1], vecs[k2])) if ok and k1 in vecs and k2 in vecs else np.nan
                                   for k1, k2, ok in zip(keys_now, keys_before, embed)]
        else:
            out[f"sim_emb_{s}"] = np.nan
    out = out.sort_values(["cik", "form", "filing_date"])
    for section in SECTIONS:
        s = SHORT[section]
        out[f"dsim_tfidf_{s}"] = out.groupby(["cik", "form"])[f"sim_tfidf_{s}"].diff()
    return out.reset_index(drop=True)


def monthly_features(per_filing: pd.DataFrame, keys: pd.DataFrame, cik_of: dict) -> pd.DataFrame:
    """Features known at each (date, ticker): the latest filing on/before the date, <= MAX_AGE_DAYS old."""
    out = keys[["date", "ticker"]].copy()
    out["cik"] = out["ticker"].map(cik_of)
    pf = per_filing.sort_values("filing_date")
    frames = []
    for cik, g in out.dropna(subset=["cik"]).groupby("cik"):
        filings = pf[pf["cik"] == int(cik)]
        if filings.empty:
            continue
        merged = pd.merge_asof(g.sort_values("date").reset_index(), filings[["filing_date"] + FEATURES],
                               left_on="date", right_on="filing_date", direction="backward")
        stale = (merged["date"] - merged["filing_date"]).dt.days > MAX_AGE_DAYS
        merged.loc[stale, FEATURES] = np.nan
        frames.append(merged.set_index("index")[FEATURES])
    feats = pd.concat(frames) if frames else pd.DataFrame(columns=FEATURES)
    return out.join(feats)[FEATURES]
