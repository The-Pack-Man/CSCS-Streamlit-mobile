from __future__ import annotations

"""
study_core.py — CSCS Study Trainer core

Goals of this rewrite:
- Keep the SAME public API your Streamlit app expects (functions/constants/paths).
- Make persistence selectable:
    - "disk": read/write stats.json + progress files (desktop)
    - "session": no file I/O (Streamlit Cloud friendly)
- Harden content loading/validation (better errors, more forgiving schemas).
- Keep behavior stable + predictable while still feeling shuffled.

This file intentionally does NOT import Streamlit.
"""

import json
import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ============================================================
# Paths
# ============================================================
CONTENT_DIR = Path("content")
TERMS_DIR = CONTENT_DIR / "terms"
QUESTIONS_DIR = CONTENT_DIR / "questions"
PRACTICE_TESTS_DIR = CONTENT_DIR / "PracticeTests"
TABLES_DIR = CONTENT_DIR / "tables"
REFERENCES_FILE = CONTENT_DIR / "references" / "references.json"

STATS_FILE = Path("stats.json")

# Progress tracking
PROGRESS_LOG_FILE = Path("progress.jsonl")          # append-only events
PROGRESS_DAILY_FILE = Path("progress_daily.json")   # aggregated by day

# Backups
BACKUP_DIR = Path("backups")


# ============================================================
# Persistence backend
# ============================================================
# "disk"   -> read/write json files above
# "session"-> no disk reads/writes (caller keeps state in memory/session_state)
PERSISTENCE_BACKEND = "disk"


def set_persistence_backend(mode: str) -> None:
    global PERSISTENCE_BACKEND
    m = str(mode or "").strip().lower()
    if m not in ("disk", "session"):
        m = "disk"
    PERSISTENCE_BACKEND = m


# ============================================================
# Constants / Defaults
# ============================================================
AUTOSAVE_EVERY_N_GRADES = 5

MATURE_INTERVAL_DAYS = 14
STRONG_STRENGTH_THRESHOLD = 0.85
STRONG_INTERVAL_THRESHOLD = 10

RATING_MULTIPLIERS = {
    "again": 0.0,
    "hard": 1.2,
    "good": 2.0,
    "easy": 2.8,
}
RATING_EASE_ADJUST = {
    "again": -0.20,
    "hard": -0.05,
    "good": +0.00,
    "easy": +0.07,
}
EASE_MIN = 1.30
EASE_MAX = 2.80

# Leech thresholds
LEECH_LAPSES = 8
LEECH_WRONG_TODAY = 4

# Bury length
BURY_HOURS = 24

# Term prompt selection strategy:
# - "rotate_seen": idx = seen % n (stable, rotates as seen increases)
# - "progressive_strength": idx based on strength (harder as you learn)
# - "seeded_daily": deterministic randomness per day+seen
TERM_DEFINITION_STRATEGY = "rotate_seen"


# ============================================================
# Helpers
# ============================================================
def now_ts() -> int:
    return int(time.time())


def today_ymd() -> str:
    return date.today().isoformat()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_load_json(path: Path) -> Any:
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Failed reading {path}: {e}") from e
    try:
        return json.loads(txt)
    except JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path} (line {e.lineno}, col {e.colno}): {e.msg}") from e


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s\-\(\)]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(s: str) -> Set[str]:
    s = normalize(s)
    return {w for w in s.split() if len(w) >= 3}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def pretty_chapter_name(chapter_id: str) -> str:
    s = (chapter_id or "").replace("_", " ").strip()
    s = re.sub(r"^chapter(\d+)\b", r"Chapter \1", s, flags=re.IGNORECASE)
    s = re.sub(r"^ch(\d+)\b", r"Ch \1", s, flags=re.IGNORECASE)
    return s[:80]


def fmt_ref(ref: Dict[str, Any]) -> str:
    if not ref:
        return ""
    bits: List[str] = []
    if ref.get("title"):
        bits.append(str(ref["title"]))
    if ref.get("year"):
        bits.append(str(ref["year"]))
    if ref.get("note"):
        bits.append(str(ref["note"]))
    return " | ".join(bits)


def sparkline(values: List[int]) -> str:
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    mx = max(values) if max(values) > 0 else 1
    out: List[str] = []
    for v in values:
        idx = int(round((v / mx) * (len(blocks) - 1)))
        idx = max(0, min(len(blocks) - 1, idx))
        out.append(blocks[idx])
    return "".join(out)


def band_str(d: Dict[str, Any]) -> str:
    mn = d.get("min")
    mx = d.get("max")
    if mn is None and mx is None:
        return ""
    if mn is None:
        return f"≤ {mx}"
    if mx is None:
        return f"≥ {mn}"
    return f"{mn}–{mx}"


# ============================================================
# Term prompt selection (multi-definition)
# ============================================================
def _coerce_definitions(item: Dict[str, Any]) -> List[str]:
    """
    Backward compatible:
      - old: {"term": "...", "definition": "...."}
      - new: {"term": "...", "definitions": ["...", "...", ...]}
    Returns a non-empty list[str] or raises ValueError.
    """
    defs: List[str] = []

    raw_defs = item.get("definitions", None)
    if isinstance(raw_defs, list) and raw_defs:
        defs = [str(d).strip() for d in raw_defs if str(d).strip()]

    if not defs:
        raw_def = item.get("definition", None)
        if raw_def is not None and str(raw_def).strip():
            defs = [str(raw_def).strip()]

    if not defs:
        raise ValueError(f"Term '{item.get('term')}' missing definition/definitions.")
    return defs


def pick_term_definition(card: Dict[str, Any], entry: Dict[str, Any]) -> Tuple[str, int]:
    """
    Returns (prompt_text, prompt_index).
    """
    defs = card.get("definitions") or []
    defs = [str(d).strip() for d in defs if str(d).strip()]
    if not defs:
        d = str(card.get("definition", "") or "").strip()
        defs = [d] if d else [""]

    n = len(defs)
    if n <= 1:
        return defs[0], 0

    strategy = str(TERM_DEFINITION_STRATEGY or "rotate_seen").strip().lower()
    seen = int(entry.get("seen", 0))
    strength = float(entry.get("strength", 0.2))

    if strategy == "progressive_strength":
        idx = int(round(clamp(strength, 0.0, 1.0) * (n - 1)))
        idx = max(0, min(n - 1, idx))
        return defs[idx], idx

    if strategy == "seeded_daily":
        seed = hash((card.get("chapter"), card.get("term"), today_ymd(), seen))
        rng = random.Random(seed)
        idx = rng.randrange(0, n)
        return defs[idx], idx

    # default: rotate_seen
    idx = seen % n
    return defs[idx], idx


def bump_def_seen(entry: Dict[str, Any], idx: int) -> None:
    m = entry.setdefault("def_seen", {})
    if not isinstance(m, dict):
        m = {}
        entry["def_seen"] = m
    k = str(int(idx))
    m[k] = int(m.get(k, 0)) + 1


# ============================================================
# Content Loaders
# ============================================================
def load_terms() -> List[Dict[str, Any]]:
    if not TERMS_DIR.exists():
        raise SystemExit("Missing content/terms directory.")
    terms: List[Dict[str, Any]] = []

    for f in sorted(TERMS_DIR.glob("*.json")):
        data = safe_load_json(f)
        if not isinstance(data, list) or not data:
            continue
        chapter_id = f.stem

        for item in data:
            if not isinstance(item, dict):
                continue
            if "term" not in item:
                raise ValueError(f"Bad term entry in {f.name}. Need 'term'.")

            defs = _coerce_definitions(item)

            terms.append(
                {
                    "chapter": chapter_id,
                    "term": item["term"],
                    "definitions": defs,          # always list
                    "definition": defs[0],        # legacy convenience
                    "tags": item.get("tags", []),
                    "why": item.get("why", ""),
                    "example": item.get("example", ""),
                    "cloze": item.get("cloze", []),
                }
            )

    return terms


def _unwrap_question_container(data: Any) -> Any:
    # Allow:
    # 1) [ {...}, {...} ]
    # 2) { "questions": [ ... ] }
    # 3) { "items": [ ... ] }
    if isinstance(data, dict):
        if isinstance(data.get("questions"), list):
            return data["questions"]
        if isinstance(data.get("items"), list):
            return data["items"]
    return data


def load_questions() -> List[Dict[str, Any]]:
    """
    Loads questions from:
      - content/questions/*.json   => chapter_id = file stem
      - content/PracticeTests/*.json => chapter_id = "pt_" + file stem
    Supports qtype "mcq" or "wrapper".

    For MCQ:
      {id, question, choices, answer_index(int|null), explanation?, tags?}

    For WRAPPER:
      {id, type:"wrapper", question, choices, contexts:[{prompt, answer_index:int}, ...], explanation?, tags?}
    """
    questions: List[Dict[str, Any]] = []

    dirs: List[Tuple[str, Path]] = []
    if QUESTIONS_DIR.exists():
        dirs.append(("questions", QUESTIONS_DIR))
    if PRACTICE_TESTS_DIR.exists():
        dirs.append(("practice", PRACTICE_TESTS_DIR))

    for kind, qdir in dirs:
        for f in sorted(qdir.glob("*.json")):
            data = safe_load_json(f)
            data = _unwrap_question_container(data)
            if not isinstance(data, list) or not data:
                continue

            base = f.stem
            chapter_id = base if kind == "questions" else f"pt_{base}"

            for q in data:
                if not isinstance(q, dict):
                    continue

                qtype = str(q.get("type", "mcq") or "mcq").strip().lower()

                # common required
                req = {"id", "question", "choices"}
                if not req.issubset(q.keys()):
                    raise ValueError(f"Bad question entry in {f.name}. Need {req}.")

                choices = q.get("choices")
                if not isinstance(choices, list) or len(choices) < 2:
                    raise ValueError(f"Question {q.get('id')} choices must be list with 2+ entries.")

                if qtype != "wrapper":
                    if "answer_index" not in q:
                        raise ValueError(f"Bad MCQ entry in {f.name}. Need answer_index.")
                    ai = q.get("answer_index", None)
                    if ai is not None:
                        try:
                            ai = int(ai)
                        except Exception:
                            raise ValueError(f"Question {q.get('id')} answer_index must be int or null.")

                    questions.append(
                        {
                            "chapter": chapter_id,
                            "id": str(q["id"]),
                            "type": "mcq",
                            "question": str(q["question"]),
                            "choices": list(choices),
                            "answer_index": ai,        # int or None
                            "contexts": [],
                            "explanation": str(q.get("explanation", "") or ""),
                            "tags": q.get("tags", []),
                        }
                    )
                    continue

                # WRAPPER
                if "contexts" not in q:
                    raise ValueError(f"Bad WRAPPER entry in {f.name}. Need contexts.")
                ctxs = q.get("contexts", [])
                if not isinstance(ctxs, list) or len(ctxs) < 2:
                    raise ValueError(f"Wrapper {q.get('id')} contexts must be a list with 2+ entries.")

                norm_ctxs: List[Dict[str, Any]] = []
                for i, c in enumerate(ctxs):
                    if not isinstance(c, dict):
                        raise ValueError(f"Wrapper {q.get('id')} context[{i}] must be an object.")
                    if "prompt" not in c or "answer_index" not in c:
                        raise ValueError(f"Wrapper {q.get('id')} context[{i}] needs prompt + answer_index.")
                    try:
                        cai = int(c.get("answer_index"))
                    except Exception:
                        raise ValueError(f"Wrapper {q.get('id')} context[{i}] answer_index must be int.")
                    norm_ctxs.append(
                        {
                            "prompt": str(c.get("prompt", "") or "").strip(),
                            "answer_index": cai,
                        }
                    )

                questions.append(
                    {
                        "chapter": chapter_id,
                        "id": str(q["id"]),
                        "type": "wrapper",
                        "question": str(q["question"]),
                        "choices": list(choices),
                        "answer_index": None,
                        "contexts": norm_ctxs,
                        "explanation": str(q.get("explanation", "") or ""),
                        "tags": q.get("tags", []),
                    }
                )

    return questions


def load_tables() -> List[Dict[str, Any]]:
    if not TABLES_DIR.exists():
        return []
    tables: List[Dict[str, Any]] = []

    for f in sorted(TABLES_DIR.glob("*.json")):
        data = safe_load_json(f)
        if isinstance(data, dict) and "tables" in data:
            data = data["tables"]
        if not isinstance(data, list) or not data:
            continue

        file_chapter = f.stem
        for t in data:
            if not isinstance(t, dict):
                continue
            if "id" not in t or "type" not in t:
                raise ValueError(f"Bad table entry in {f.name}. Need at least id + type.")
            chapter_id = t.get("chapter", file_chapter)
            t2 = dict(t)
            t2["chapter"] = chapter_id
            t2["tags"] = t2.get("tags", [])
            tables.append(t2)

    return tables


def load_references() -> Dict[str, Dict[str, Any]]:
    if not REFERENCES_FILE.exists():
        return {}
    data = safe_load_json(REFERENCES_FILE)
    if not isinstance(data, dict) or "refs" not in data or not isinstance(data["refs"], list):
        raise ValueError('references.json must be {"refs": [ ... ]}')
    out: Dict[str, Dict[str, Any]] = {}
    for r in data["refs"]:
        if isinstance(r, dict) and r.get("id"):
            out[str(r["id"])] = r
    return out


# ============================================================
# Practice test import helper
# ============================================================
def import_questions_file(
    src_json_path: Path,
    chapter_id: str,
    *,
    tags: Optional[List[str]] = None,
    overwrite: bool = False,
) -> Path:
    """
    Imports a JSON array of MCQ questions into content/questions/<chapter_id>.json.

    Expected schema per item:
      { "id": str, "question": str, "choices": [..], "answer_index": int|null, "explanation": str? }
    """
    if not src_json_path.exists():
        raise ValueError(f"Missing source file: {src_json_path}")

    data = safe_load_json(src_json_path)
    data = _unwrap_question_container(data)
    if not isinstance(data, list) or not data:
        raise ValueError("Source questions JSON must be a non-empty list.")

    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = QUESTIONS_DIR / f"{chapter_id}.json"

    if out_path.exists() and not overwrite:
        raise ValueError(f"{out_path} already exists. Pass overwrite=True to replace it.")

    add_tags = tags or []
    normalized: List[Dict[str, Any]] = []

    for q in data:
        if not isinstance(q, dict):
            continue

        required = {"id", "question", "choices", "answer_index"}
        if not required.issubset(q.keys()):
            raise ValueError(f"Bad question entry: need {required}. Got keys={sorted(q.keys())}")

        if not isinstance(q["choices"], list) or len(q["choices"]) < 2:
            raise ValueError(f"Question {q.get('id')} choices must be list with 2+ entries.")

        ai = q.get("answer_index", None)
        if ai is not None:
            try:
                ai = int(ai)
            except Exception:
                raise ValueError(f"Question {q.get('id')} answer_index must be int or null.")

        qt = {
            "id": str(q["id"]),
            "question": str(q["question"]),
            "choices": list(q["choices"]),
            "answer_index": ai,
            "explanation": str(q.get("explanation", "") or ""),
            "tags": list(dict.fromkeys((q.get("tags", []) or []) + add_tags)),
        }
        normalized.append(qt)

    atomic_write_json(out_path, normalized, indent=2)
    return out_path


# ============================================================
# Keys
# ============================================================
def term_key(card: Dict[str, Any]) -> str:
    return f"term::{card['chapter']}::{card['term']}"


def q_key(q: Dict[str, Any]) -> str:
    return f"q::{q['chapter']}::{q['id']}"


def table_key(t: Dict[str, Any]) -> str:
    return f"tbl::{t.get('chapter','')}::{t.get('id','')}"


# ============================================================
# Stats model
# ============================================================
def default_stats_entry() -> Dict[str, Any]:
    ts = now_ts()
    return {
        "strength": 0.20,
        "seen": 0,
        "correct": 0,
        "wrong": 0,
        "streak": 0,
        "ease": 2.10,
        "interval_days": 0.0,
        "due": ts,
        "lapses": 0,
        "last_seen": 0,
        "last_correct": 0,
        "mature": False,
        "first_seen": 0,
        "wrong_today": 0,
        "last_day": "",

        # controls
        "suspended": False,
        "buried_until": 0,
        "notes": "",
        "last_mistake_note": "",

        # multi-definition exposure tracking for term prompts
        "def_seen": {},
    }


def migrate_stats_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    base = default_stats_entry()
    if isinstance(e, dict):
        base.update(e)

    # ensure + normalize types
    base["ease"] = clamp(float(base.get("ease", 2.1)), EASE_MIN, EASE_MAX)
    base["interval_days"] = float(base.get("interval_days", 0.0))
    base["due"] = int(base.get("due", now_ts()))
    base["seen"] = int(base.get("seen", 0))
    base["correct"] = int(base.get("correct", 0))
    base["wrong"] = int(base.get("wrong", 0))
    base["lapses"] = int(base.get("lapses", 0))
    base["last_seen"] = int(base.get("last_seen", 0))
    base["last_correct"] = int(base.get("last_correct", 0))
    base["streak"] = int(base.get("streak", 0))
    base["strength"] = clamp(float(base.get("strength", 0.2)), 0.0, 1.0)
    base["mature"] = bool(base.get("mature", False))
    base["first_seen"] = int(base.get("first_seen", 0))
    base["wrong_today"] = int(base.get("wrong_today", 0))
    base["last_day"] = str(base.get("last_day", ""))

    base["suspended"] = bool(base.get("suspended", False))
    base["buried_until"] = int(base.get("buried_until", 0))
    base["notes"] = str(base.get("notes", "") or "")
    base["last_mistake_note"] = str(base.get("last_mistake_note", "") or "")

    ds = base.get("def_seen", {})
    if not isinstance(ds, dict):
        ds = {}
    norm_ds: Dict[str, int] = {}
    for kk, vv in ds.items():
        try:
            norm_ds[str(int(kk))] = int(vv)
        except Exception:
            continue
    base["def_seen"] = norm_ds

    return base


def load_stats(
    terms: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
) -> Dict[str, Any]:
    raw: Dict[str, Any] = {}

    if PERSISTENCE_BACKEND == "disk" and STATS_FILE.exists():
        loaded = safe_load_json(STATS_FILE)
        if isinstance(loaded, dict):
            raw = loaded

    stats: Dict[str, Any] = {k: migrate_stats_entry(v) for k, v in raw.items()}

    def ensure(k: str) -> None:
        if k not in stats:
            stats[k] = default_stats_entry()

    for c in terms:
        ensure(term_key(c))
    for q in questions:
        ensure(q_key(q))
    for t in tables:
        ensure(table_key(t))

    return stats


def save_stats(stats: Dict[str, Any]) -> None:
    if PERSISTENCE_BACKEND != "disk":
        return
    atomic_write_json(STATS_FILE, stats, indent=2)


# ============================================================
# Progress tracking
# ============================================================
def load_progress_daily() -> Dict[str, Any]:
    if PERSISTENCE_BACKEND != "disk":
        return {"days": {}}

    if not PROGRESS_DAILY_FILE.exists():
        return {"days": {}}
    try:
        data = safe_load_json(PROGRESS_DAILY_FILE)
        if not isinstance(data, dict) or "days" not in data or not isinstance(data["days"], dict):
            return {"days": {}}
        return data
    except Exception:
        return {"days": {}}


def save_progress_daily(data: Dict[str, Any]) -> None:
    if PERSISTENCE_BACKEND != "disk":
        return
    atomic_write_json(PROGRESS_DAILY_FILE, data, indent=2)


def append_progress_event(event: Dict[str, Any]) -> None:
    if PERSISTENCE_BACKEND != "disk":
        return
    try:
        line = json.dumps(event, ensure_ascii=False)
        with PROGRESS_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def update_progress_daily_from_event(daily: Dict[str, Any], event: Dict[str, Any]) -> None:
    day = str(event.get("day", ""))
    if not day:
        return
    days = daily.setdefault("days", {})
    d = days.setdefault(
        day,
        {"reviews": 0, "again": 0, "correct_like": 0, "seconds": 0.0, "by_kind": {}},
    )

    d["reviews"] = int(d.get("reviews", 0)) + 1
    if event.get("rating") == "again":
        d["again"] = int(d.get("again", 0)) + 1
    else:
        d["correct_like"] = int(d.get("correct_like", 0)) + 1

    rt = float(event.get("response_sec", 0.0))
    d["seconds"] = float(d.get("seconds", 0.0)) + max(0.0, rt)

    kind = str(event.get("kind", ""))
    if kind:
        bk = d.setdefault("by_kind", {})
        bk[kind] = int(bk.get(kind, 0)) + 1


# ============================================================
# Daily reset + SRS
# ============================================================
def reset_daily_fields(entry: Dict[str, Any]) -> None:
    td = today_ymd()
    if entry.get("last_day") != td:
        entry["wrong_today"] = 0
        entry["last_day"] = td


def is_suspended_or_buried(entry: Dict[str, Any]) -> bool:
    if bool(entry.get("suspended", False)):
        return True
    buried_until = int(entry.get("buried_until", 0))
    if buried_until and now_ts() < buried_until:
        return True
    return False


def is_due(entry: Dict[str, Any]) -> bool:
    reset_daily_fields(entry)
    if is_suspended_or_buried(entry):
        return False
    return int(entry.get("due", 0)) <= now_ts()


def mark_seen(entry: Dict[str, Any]) -> None:
    ts = now_ts()
    entry["last_seen"] = ts
    if int(entry.get("first_seen", 0)) == 0:
        entry["first_seen"] = ts


def update_strength_legacy(entry: Dict[str, Any], got_right: bool) -> None:
    entry["seen"] = int(entry.get("seen", 0)) + 1
    if got_right:
        entry["correct"] = int(entry.get("correct", 0)) + 1
        entry["streak"] = int(entry.get("streak", 0)) + 1
        s = float(entry.get("strength", 0.2))
        entry["strength"] = min(1.0, s + (1.0 - s) * 0.18)
    else:
        entry["wrong"] = int(entry.get("wrong", 0)) + 1
        entry["streak"] = 0
        s = float(entry.get("strength", 0.2))
        entry["strength"] = max(0.0, s * 0.70)


def schedule_with_rating(entry: Dict[str, Any], rating: str) -> None:
    rating = str(rating or "good").lower().strip()
    if rating not in RATING_MULTIPLIERS:
        rating = "good"

    reset_daily_fields(entry)

    ease = float(entry.get("ease", 2.1))
    ease += float(RATING_EASE_ADJUST.get(rating, 0.0))
    ease = clamp(ease, EASE_MIN, EASE_MAX)
    entry["ease"] = ease

    interval = float(entry.get("interval_days", 0.0))

    if rating == "again":
        if interval >= 3:
            entry["lapses"] = int(entry.get("lapses", 0)) + 1
        entry["wrong_today"] = int(entry.get("wrong_today", 0)) + 1
        entry["interval_days"] = 0.0
        entry["due"] = now_ts() + 60 * 10
        entry["mature"] = False
        entry["streak"] = 0
        return

    # hard/good/easy
    if interval <= 0.0:
        interval = 1.0 if rating in ("hard", "good") else 2.0
    else:
        mult = float(RATING_MULTIPLIERS[rating])
        interval = max(1.0, interval * mult * ease / 2.1)

    interval = min(interval, 365.0 * 3)
    entry["interval_days"] = interval
    entry["due"] = now_ts() + int(interval * 24 * 3600)

    if rating in ("good", "easy"):
        entry["last_correct"] = now_ts()
        entry["streak"] = int(entry.get("streak", 0)) + 1
    else:
        entry["streak"] = 0

    entry["mature"] = interval >= MATURE_INTERVAL_DAYS


# ============================================================
# Wrapper questions helpers
# ============================================================
def is_wrapper_question(q: Dict[str, Any]) -> bool:
    return str(q.get("type", "mcq") or "mcq").strip().lower() == "wrapper"


def question_total_parts(q: Dict[str, Any]) -> int:
    if not is_wrapper_question(q):
        return 1
    ctxs = q.get("contexts", [])
    return len(ctxs) if isinstance(ctxs, list) else 0


def question_part_prompt(q: Dict[str, Any], part_idx: int = 0) -> str:
    base = str(q.get("question", "") or "").strip()
    if not is_wrapper_question(q):
        return base

    ctxs = q.get("contexts", [])
    if not isinstance(ctxs, list) or not ctxs:
        return base

    part_idx = max(0, min(int(part_idx), len(ctxs) - 1))
    c = ctxs[part_idx]
    cp = str(c.get("prompt", "") or "").strip()
    return f"{base}\n\nPart {part_idx + 1}/{len(ctxs)}: {cp}"


def grade_question_response(q: Dict[str, Any], response: Any) -> Dict[str, Any]:
    """
    For MCQ:
      response = int (selected index)

    For WRAPPER:
      response = List[int] length == len(contexts)
      OR Dict[str,int] mapping part_idx -> selected index
    """
    if not is_wrapper_question(q):
        correct_ai = q.get("answer_index", None)
        try:
            correct_ai = int(correct_ai) if correct_ai is not None else None
        except Exception:
            correct_ai = None

        sel = None
        try:
            sel = int(response)
        except Exception:
            sel = None

        is_correct = (correct_ai is not None and sel == correct_ai)
        return {
            "is_correct": is_correct,
            "correct_parts": 1 if is_correct else 0,
            "total_parts": 1,
            "correct_answer_indices": [correct_ai],
            "response_indices": [sel],
        }

    ctxs = q.get("contexts", [])
    if not isinstance(ctxs, list):
        ctxs = []
    total = len(ctxs)

    correct: List[Optional[int]] = []
    for c in ctxs:
        try:
            correct.append(int(c.get("answer_index")))
        except Exception:
            correct.append(None)

    resp_list: List[Optional[int]] = [None] * total
    if isinstance(response, list):
        for i in range(min(total, len(response))):
            try:
                resp_list[i] = int(response[i])
            except Exception:
                resp_list[i] = None
    elif isinstance(response, dict):
        for k, v in response.items():
            try:
                i = int(k)
                if 0 <= i < total:
                    resp_list[i] = int(v)
            except Exception:
                continue
    else:
        try:
            resp_list[0] = int(response)
        except Exception:
            pass

    correct_parts = 0
    for i in range(total):
        if correct[i] is not None and resp_list[i] == correct[i]:
            correct_parts += 1

    return {
        "is_correct": (correct_parts == total and total > 0),
        "correct_parts": correct_parts,
        "total_parts": total,
        "correct_answer_indices": correct,
        "response_indices": resp_list,
    }


# ============================================================
# Tag helpers
# ============================================================
def get_item_tags(item: Dict[str, Any]) -> Set[str]:
    tags = item.get("tags", [])
    if isinstance(tags, list):
        return {str(t).strip() for t in tags if str(t).strip()}
    return set()


def pass_filters(item: Dict[str, Any], chapter_filter: Optional[Set[str]], tag_filter: Optional[Set[str]]) -> bool:
    if chapter_filter and item.get("chapter") not in chapter_filter:
        return False
    if tag_filter:
        its = get_item_tags(item)
        if not its or its.isdisjoint(tag_filter):
            return False
    return True


# ============================================================
# Leech detection
# ============================================================
def is_leech(entry: Dict[str, Any]) -> bool:
    reset_daily_fields(entry)
    lapses = int(entry.get("lapses", 0))
    wrong_today = int(entry.get("wrong_today", 0))
    return lapses >= LEECH_LAPSES or wrong_today >= LEECH_WRONG_TODAY


def leech_score(entry: Dict[str, Any]) -> float:
    reset_daily_fields(entry)
    lapses = int(entry.get("lapses", 0))
    wrong_today = int(entry.get("wrong_today", 0))
    strength = float(entry.get("strength", 0.2))
    return lapses * 2.0 + wrong_today * 2.5 + (1.0 - strength) * 2.0


# ============================================================
# Study selection
# ============================================================
def weighted_choice_due(
    items: List[Dict[str, Any]],
    stats: Dict[str, Any],
    make_key_fn,
    chapter_filter: Optional[Set[str]] = None,
    tag_filter: Optional[Set[str]] = None,
    mode: str = "spaced",
) -> Optional[Dict[str, Any]]:
    """
    Returns one item from `items` based on `mode` + SRS fields.
    - Adds slight randomness to prevent deterministic “same card loop”.
    """
    pool: List[Dict[str, Any]] = []
    weights: List[float] = []

    for it in items:
        if not pass_filters(it, chapter_filter, tag_filter):
            continue
        k = make_key_fn(it)
        e = stats.get(k)
        if not isinstance(e, dict):
            continue

        reset_daily_fields(e)

        if is_suspended_or_buried(e):
            continue

        if mode == "test_all":
            pool.append(it)
            weights.append(1.0)
            continue

        due_flag = is_due(e)
        seen = int(e.get("seen", 0))
        strength = float(e.get("strength", 0.2))
        lapses = int(e.get("lapses", 0))
        wrong_today = int(e.get("wrong_today", 0))
        interval = float(e.get("interval_days", 0.0))

        if mode == "new_only":
            if seen > 0:
                continue
            pool.append(it)
            weights.append(1.0)
            continue

        if mode == "missed_today":
            if wrong_today <= 0:
                continue
            pool.append(it)
            weights.append(1.0 + wrong_today * 0.8 + (1.0 - strength))
            continue

        if mode == "leeches":
            if not is_leech(e):
                continue
            pool.append(it)
            weights.append(1.0 + leech_score(e))
            continue

        if mode == "cram_weak":
            need = (1.0 - strength)
            w = 0.60 + need * 1.8 + lapses * 0.8
            if interval < MATURE_INTERVAL_DAYS:
                w += 0.2
            pool.append(it)
            weights.append(w)
            continue

        # spaced
        if due_flag:
            need = (1.0 - strength)
            w = 1.0 + need * 1.3 + lapses * 0.5
            pool.append(it)
            weights.append(w)
        else:
            # keep a trickle of unseen items even if not "due"
            if seen == 0:
                pool.append(it)
                weights.append(0.20)

    if not pool:
        return None

    # break stable ordering + small tie jitter
    idxs = list(range(len(pool)))
    random.shuffle(idxs)

    pool2 = [pool[i] for i in idxs]
    weights2: List[float] = []
    for i in idxs:
        w = float(weights[i])
        w *= (1.0 + random.random() * 0.02)  # 0-2% jitter
        weights2.append(w)

    return random.choices(pool2, weights=weights2, k=1)[0]


def smart_mix_pick(
    terms: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    stats: Dict[str, Any],
    chapter_filter: Optional[Set[str]],
    tag_filter: Optional[Set[str]],
    study_mode: str,
    ratio: Tuple[int, int, int] = (60, 30, 10),
) -> Tuple[Optional[Dict[str, Any]], str]:
    # ratio = (terms, questions, tables)
    buckets: List[str] = []
    weights: List[int] = []

    if terms:
        buckets.append("term")
        weights.append(max(0, int(ratio[0])))
    if questions:
        buckets.append("question")
        weights.append(max(0, int(ratio[1])))
    if tables:
        buckets.append("table")
        weights.append(max(0, int(ratio[2])))

    if not buckets or sum(weights) <= 0:
        return None, ""

    # try multiple times to avoid empty picks in a bucket
    for _ in range(6):
        kind = random.choices(buckets, weights=weights, k=1)[0]
        if kind == "term":
            it = weighted_choice_due(terms, stats, term_key, chapter_filter, tag_filter, mode=study_mode)
        elif kind == "question":
            it = weighted_choice_due(questions, stats, q_key, chapter_filter, tag_filter, mode=study_mode)
        else:
            it = weighted_choice_due(tables, stats, table_key, chapter_filter, tag_filter, mode=study_mode)
        if it:
            return it, kind

    # fallback: try each
    for kind in ("term", "question", "table"):
        if kind == "term":
            it = weighted_choice_due(terms, stats, term_key, chapter_filter, tag_filter, mode=study_mode)
        elif kind == "question":
            it = weighted_choice_due(questions, stats, q_key, chapter_filter, tag_filter, mode=study_mode)
        else:
            it = weighted_choice_due(tables, stats, table_key, chapter_filter, tag_filter, mode=study_mode)
        if it:
            return it, kind

    return None, ""


# ============================================================
# Validation / health report
# ============================================================
def validate_content(
    terms: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
) -> List[str]:
    issues: List[str] = []
    seen_q_ids: Set[str] = set()

    for q in questions:
        qtype = str(q.get("type", "mcq") or "mcq").strip().lower()
        choices = q.get("choices", [])
        if not isinstance(choices, list) or len(choices) < 2:
            issues.append(f"Question {q.get('id')} has invalid choices.")
            continue

        if qtype != "wrapper":
            ai = q.get("answer_index", None)
            if ai is not None:
                try:
                    ai = int(ai)
                except Exception:
                    issues.append(f"Question {q.get('id')} answer_index not int/null.")
                    continue
                if not (0 <= ai < len(choices)):
                    issues.append(f"Question {q.get('id')} has answer_index out of range.")
        else:
            ctxs = q.get("contexts", [])
            if not isinstance(ctxs, list) or len(ctxs) < 2:
                issues.append(f"Wrapper {q.get('id')} contexts must be list with 2+ entries.")
            else:
                for i, c in enumerate(ctxs):
                    if not isinstance(c, dict):
                        issues.append(f"Wrapper {q.get('id')} context[{i}] must be object.")
                        continue
                    if "prompt" not in c or "answer_index" not in c:
                        issues.append(f"Wrapper {q.get('id')} context[{i}] missing prompt/answer_index.")
                        continue
                    try:
                        cai = int(c.get("answer_index"))
                    except Exception:
                        issues.append(f"Wrapper {q.get('id')} context[{i}] answer_index not int.")
                        continue
                    if not (0 <= cai < len(choices)):
                        issues.append(f"Wrapper {q.get('id')} context[{i}] answer_index out of range.")

        key = f"{q.get('chapter')}::{q.get('id')}"
        if key in seen_q_ids:
            issues.append(f"Duplicate question id within chapter: {key}")
        seen_q_ids.add(key)

    seen_terms: Set[str] = set()
    for t in terms:
        key = f"{t.get('chapter')}::{normalize(t.get('term',''))}"
        if key in seen_terms:
            issues.append(f"Duplicate term in chapter: {t.get('chapter')} :: {t.get('term')}")
        seen_terms.add(key)

        defs = t.get("definitions", [])
        if not isinstance(defs, list) or not defs:
            issues.append(f"Term missing definitions list: {t.get('chapter')} :: {t.get('term')}")
        else:
            defs2 = [str(d).strip() for d in defs if str(d).strip()]
            if not defs2:
                issues.append(f"Term has empty definitions: {t.get('chapter')} :: {t.get('term')}")

    for tb in tables:
        ttype = tb.get("type")
        if ttype == "points_table":
            if "sex_sections" not in tb or "points" not in tb:
                issues.append(f"Table {tb.get('id')} missing sex_sections/points.")
        elif ttype == "classification_ranges":
            if "ranges" not in tb:
                issues.append(f"Table {tb.get('id')} missing ranges.")

    return issues


def content_health_report(
    terms: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
) -> str:
    issues = validate_content(terms, questions, tables)
    tags_all: Set[str] = set()
    for x in terms + questions + tables:
        tags_all |= get_item_tags(x)

    chapters = sorted(
        {t["chapter"] for t in terms}
        | {q["chapter"] for q in questions}
        | {tb.get("chapter", "") for tb in tables if tb.get("chapter")}
    )

    lines: List[str] = []
    lines.append(f"Content health report — {today_ymd()}")
    lines.append(f"Chapters: {len(chapters)} | Tags: {len(tags_all)}")
    lines.append(f"Terms: {len(terms)} | Questions: {len(questions)} | Tables: {len(tables)}")

    if issues:
        lines.append("")
        lines.append("Warnings:")
        for s in issues[:40]:
            lines.append(f"- {s}")
        if len(issues) > 40:
            lines.append(f"... and {len(issues)-40} more")
    else:
        lines.append("")
        lines.append("No validation issues detected.")
    return "\n".join(lines)


# ============================================================
# Queue inspection
# ============================================================
def compute_queue_stats(stats: Dict[str, Any]) -> Dict[str, int]:
    due_now = 0
    new = 0
    missed_today = 0
    leeches = 0
    suspended = 0
    buried = 0

    for _, e in stats.items():
        if not isinstance(e, dict):
            continue
        reset_daily_fields(e)

        if bool(e.get("suspended", False)):
            suspended += 1

        bu = int(e.get("buried_until", 0))
        if bu and now_ts() < bu:
            buried += 1

        if int(e.get("seen", 0)) == 0:
            new += 1

        if int(e.get("wrong_today", 0)) > 0:
            missed_today += 1

        if is_leech(e):
            leeches += 1

        if is_due(e):
            due_now += 1

    return {
        "due_now": due_now,
        "new": new,
        "missed_today": missed_today,
        "leeches": leeches,
        "suspended": suspended,
        "buried": buried,
    }


# ============================================================
# Backup
# ============================================================
def write_backup(stats: Dict[str, Any], progress_daily: Dict[str, Any]) -> Path:
    if PERSISTENCE_BACKEND != "disk":
        return Path("(session-only: no backup written)")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = BACKUP_DIR / f"backup_{stamp}.json"
    payload = {
        "ts": now_ts(),
        "day": today_ymd(),
        "stats": stats,
        "progress_daily": progress_daily,
    }
    atomic_write_json(out, payload, indent=2)
    return out


# ============================================================
# IMPORTANT integration note (safe to delete)
# ============================================================
"""
Wherever you render term cards, swap:

  prompt = card["definition"]

for:

  k = term_key(card)
  entry = stats[k]
  prompt, idx = pick_term_definition(card, entry)
  bump_def_seen(entry, idx)

Then display `prompt` to the user.
"""
