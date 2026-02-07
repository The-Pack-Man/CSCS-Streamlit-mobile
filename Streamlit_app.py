from __future__ import annotations

import random
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import streamlit as st

import study_core as core


# ============================================================
# Similarity
# ============================================================
def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, core.normalize(a), core.normalize(b)).ratio()


def mixed_similarity(def_a: str, def_b: str) -> float:
    seq = SequenceMatcher(None, core.normalize(def_a), core.normalize(def_b)).ratio()
    tok = core.jaccard(core.tokenize(def_a), core.tokenize(def_b))
    return 0.65 * seq + 0.35 * tok


# ============================================================
# Term distractors (smart)
# ============================================================
def pick_term_distractors_smart(
    all_terms: List[Dict[str, Any]],
    correct_card: Dict[str, Any],
    target_def: str,
    k: int = 3,
) -> List[str]:
    correct_term = correct_card["term"]
    chapter = correct_card.get("chapter")
    tags = core.get_item_tags(correct_card)

    scored: List[Tuple[float, str]] = []
    for c in all_terms:
        if c.get("term") == correct_term:
            continue

        cand_defs = c.get("definitions") or ([c.get("definition", "")] if c.get("definition") else [])
        best_score = -1.0
        for d in cand_defs:
            ds = mixed_similarity(target_def, str(d or ""))
            if ds > best_score:
                best_score = ds
        s_def = best_score if best_score >= 0 else mixed_similarity(target_def, "")

        s_term = SequenceMatcher(None, core.normalize(correct_term), core.normalize(c.get("term", ""))).ratio()
        score = (0.85 * s_def) + (0.15 * s_term)

        if c.get("chapter") == chapter:
            score += 0.10

        ct = core.get_item_tags(c)
        if tags and ct and not tags.isdisjoint(ct):
            score += 0.08

        overlap = core.jaccard(core.tokenize(correct_term), core.tokenize(c.get("term", "")))
        if overlap > 0.45:
            score -= 0.10

        scored.append((score, c.get("term", "")))

    scored.sort(reverse=True, key=lambda x: x[0])

    out: List[str] = []
    seen = set()
    for _, t in scored:
        nt = core.normalize(t)
        if not t or nt in seen:
            continue
        out.append(t)
        seen.add(nt)
        if len(out) == k:
            break
    return out


# ============================================================
# Practice helpers
# ============================================================
def is_practice_chapter(ch: str) -> bool:
    c = (ch or "").lower()
    return c.startswith(("pt_", "practice_", "mock_", "test_"))


def sort_qid_natural(qid: str) -> Tuple[int, str]:
    s = str(qid or "")
    digits = "".join([c for c in s if c.isdigit()])
    if digits:
        try:
            return (int(digits), s)
        except Exception:
            pass
    return (10**9, s)

def auto_select_chapters(chapters: List[str]) -> Set[str]:
    return {
        c for c in chapters
        if "chapter" in (c or "").lower()
    }

# ============================================================
# Loading (cached content; mutable stats not cached)
# ============================================================
@st.cache_data(show_spinner=False)
def load_content_cached():
    terms = core.load_terms()
    questions = core.load_questions()
    tables = core.load_tables()
    refs = core.load_references()
    issues = core.validate_content(terms, questions, tables)

    term_by_key = {core.term_key(t): t for t in terms}
    q_by_key = {core.q_key(q): q for q in questions}
    tbl_by_key = {core.table_key(t): t for t in tables}

    return terms, questions, tables, refs, issues, term_by_key, q_by_key, tbl_by_key


# ============================================================
# State
# ============================================================
def ensure_state():
    if st.session_state.get("initialized"):
        return

    terms, questions, tables, refs, issues, term_by_key, q_by_key, tbl_by_key = load_content_cached()
    stats = core.load_stats(terms, questions, tables)
    progress_daily = core.load_progress_daily()

    st.session_state.initialized = True

    # content
    st.session_state.terms = terms
    st.session_state.questions = questions
    st.session_state.tables = tables
    st.session_state.refs = refs
    st.session_state.issues = issues

    st.session_state.term_by_key = term_by_key
    st.session_state.q_by_key = q_by_key
    st.session_state.tbl_by_key = tbl_by_key

    # mutable
    st.session_state.stats = stats
    st.session_state.progress_daily = progress_daily

    # filters
    st.session_state.chapter_filter = set()  # type: Set[str]
    st.session_state.tag_filter = set()      # type: Set[str]
    # filters
    st.session_state.chapter_filter = set()  # type: Set[str]
    st.session_state.tag_filter = set()      # type: Set[str]
    
    # auto-select "chapter*" entries on first load only
    all_chaps = sorted(
        {c["chapter"] for c in terms}
        | {q["chapter"] for q in questions}
        | {t.get("chapter") for t in tables if t.get("chapter")}
    )
    
    st.session_state.chapter_filter = auto_select_chapters(all_chaps)

    # modes
    st.session_state.content_mode = "smart_mix"  # smart_mix / terms / questions / practice_test / tables_*
    st.session_state.study_mode = "spaced"       # spaced / cram_weak / new_only / missed_today / leeches
    st.session_state.term_style = "auto"         # auto / mcq / typing / cloze
    st.session_state.mcq_difficulty = "hard"     # easy / medium / hard / evil

    # smart mix ratio
    st.session_state.mix_terms = 60
    st.session_state.mix_questions = 30
    st.session_state.mix_tables = 10

    # current item
    st.session_state.current_item = None
    st.session_state.current_kind = ""
    st.session_state.current_key = ""

    # answer/feedback
    st.session_state.answered = False
    st.session_state.pending_rate = False
    st.session_state.feedback_banner = ""
    st.session_state.last_correct_text = ""
    st.session_state.last_explanation = ""
    st.session_state.last_was_correct = None  # Optional[bool]
    st.session_state.answer_start_ts = core.now_ts()
    st.session_state.guessed = False

    # wrapper state (lock part per question key)
    st.session_state.wrapper_part_by_key = {}  # type: Dict[str, int]

    # term multi-def prompt
    st.session_state.term_prompt_def_idx = 0
    st.session_state.term_prompt_text = ""

    # practice test
    st.session_state.practice_set = ""
    st.session_state.practice_shuffle = False
    st.session_state.practice_shuffle_seed = 0
    st.session_state.practice_show_expl_on_submit = True
    st.session_state.practice_keys = []
    st.session_state.practice_i = 0
    st.session_state.practice_correct = 0
    st.session_state.practice_done = 0
    st.session_state.practice_waiting_next = False  # NEW: stop auto-advance so feedback can be seen

    # session tracking
    st.session_state.session_done = 0
    st.session_state.session_correct_like = 0
    st.session_state.session_goal = 50
    st.session_state.session_timer_start = core.now_ts()
    st.session_state.auto_reset_on_goal = False

    pick_next_item()


def save_all():
    core.save_stats(st.session_state.stats)
    core.save_progress_daily(st.session_state.progress_daily)


def all_tags() -> Set[str]:
    tags: Set[str] = set()
    for t in st.session_state.terms:
        tags |= core.get_item_tags(t)
    for q in st.session_state.questions:
        tags |= core.get_item_tags(q)
    for tb in st.session_state.tables:
        tags |= core.get_item_tags(tb)
    return {x for x in tags if x}


def practice_active() -> bool:
    return st.session_state.content_mode == "practice_test" and bool(st.session_state.practice_keys)


def current_key_for(it: Dict[str, Any], kind: str) -> str:
    if kind == "term":
        return core.term_key(it)
    if kind == "question":
        return core.q_key(it)
    return core.table_key(it)


def reset_answer_state():
    st.session_state.answered = False
    st.session_state.pending_rate = False
    st.session_state.feedback = ""
    st.session_state.last_correct_text = ""
    st.session_state.last_explanation = ""
    st.session_state.answer_start_ts = core.now_ts()
    st.session_state.guessed = False
    st.session_state.current_choices = []
    st.session_state.wrapper_part_idx = 0
    st.session_state.wrapper_part_total = 0
    st.session_state.term_prompt_def_idx = 0
    st.session_state.term_prompt_text = ""

    # --- ADD THIS: clear per-question frozen MCQ state ---
    # remove any prior mcq choice caches
    for k in list(st.session_state.keys()):
        if k.startswith("mcq_choices::"):
            del st.session_state[k]
        if k.startswith("term_mcq_pick::"):
            del st.session_state[k]
        if k.startswith("q_pick::"):
            del st.session_state[k]


def pick_next_item(force: Optional[Tuple[Dict[str, Any], str]] = None):
    reset_answer_state()

    chap = st.session_state.chapter_filter or None
    tags = st.session_state.tag_filter or None
    smode = st.session_state.study_mode

    # forced jump
    if force:
        it, kind = force
        st.session_state.current_item = it
        st.session_state.current_kind = kind
        st.session_state.current_key = current_key_for(it, kind)
        return

    # practice sequencing
    if practice_active():
        # if we're waiting for user to press next, don't advance
        if st.session_state.practice_waiting_next:
            return

        if st.session_state.practice_i >= len(st.session_state.practice_keys):
            total = len(st.session_state.practice_keys)
            done = st.session_state.practice_done
            correct = st.session_state.practice_correct
            acc = (correct / done * 100.0) if done else 0.0
            st.session_state.feedback_banner = f"✅ Practice completed: {correct}/{total} ({acc:.0f}%)"
            st.session_state.practice_keys = []
            st.session_state.content_mode = "questions"
            return

        k = st.session_state.practice_keys[st.session_state.practice_i]
        it = st.session_state.q_by_key.get(k)
        if not it:
            st.session_state.practice_i += 1
            pick_next_item()
            return

        st.session_state.current_item = it
        st.session_state.current_kind = "question"
        st.session_state.current_key = k
        return

    cm = st.session_state.content_mode

    if cm == "smart_mix":
        ratio = (st.session_state.mix_terms, st.session_state.mix_questions, st.session_state.mix_tables)
        it, kind = core.smart_mix_pick(
            st.session_state.terms,
            st.session_state.questions,
            st.session_state.tables,
            st.session_state.stats,
            chap,
            tags,
            smode,
            ratio=ratio,
        )
        if not it:
            st.session_state.current_item = None
            st.session_state.current_kind = ""
            st.session_state.current_key = ""
            st.session_state.feedback_banner = "No matching items. Adjust filters or study mode."
            return
        st.session_state.current_item = it
        st.session_state.current_kind = kind
        st.session_state.current_key = current_key_for(it, kind)
        return

    if cm == "terms":
        it = core.weighted_choice_due(st.session_state.terms, st.session_state.stats, core.term_key, chap, tags, mode=smode)
        if not it:
            st.session_state.current_item = None
            st.session_state.feedback_banner = "No matching terms."
            return
        st.session_state.current_item = it
        st.session_state.current_kind = "term"
        st.session_state.current_key = core.term_key(it)
        return

    if cm in ("questions", "practice_test"):
        it = core.weighted_choice_due(st.session_state.questions, st.session_state.stats, core.q_key, chap, tags, mode=smode)
        if not it:
            st.session_state.current_item = None
            st.session_state.feedback_banner = "No matching questions."
            return
        st.session_state.current_item = it
        st.session_state.current_kind = "question"
        st.session_state.current_key = core.q_key(it)
        return

    # tables modes
    it = core.weighted_choice_due(st.session_state.tables, st.session_state.stats, core.table_key, chap, tags, mode=smode)
    if not it:
        st.session_state.current_item = None
        st.session_state.feedback_banner = "No matching tables."
        return
    st.session_state.current_item = it
    st.session_state.current_kind = "table"
    st.session_state.current_key = core.table_key(it)


# ============================================================
# Grading + Feedback
# ============================================================
def apply_grade(rating: str, got_right: bool, response_sec: float, guessed: bool, prompt_note: bool = True):
    k = st.session_state.current_key
    if not k:
        return

    e = st.session_state.stats.get(k)
    if not e:
        st.session_state.stats[k] = core.default_stats_entry()
        e = st.session_state.stats[k]

    core.mark_seen(e)
    core.update_strength_legacy(e, got_right)
    core.schedule_with_rating(e, rating)

    if prompt_note and rating == "again" and (not practice_active()):
        st.session_state._need_mistake_note = True
    else:
        st.session_state._need_mistake_note = False

    event = {
        "ts": core.now_ts(),
        "day": core.today_ymd(),
        "key": k,
        "kind": st.session_state.current_kind,
        "rating": rating,
        "correct": bool(got_right),
        "guessed": bool(guessed),
        "response_sec": float(response_sec),
    }
    core.append_progress_event(event)
    core.update_progress_daily_from_event(st.session_state.progress_daily, event)

    if (st.session_state.session_done % core.AUTOSAVE_EVERY_N_GRADES) == 0:
        save_all()


def after_answer(got_right: bool, correct_text: str, explanation: str):
    st.session_state.answered = True
    st.session_state.last_was_correct = bool(got_right)
    st.session_state.last_correct_text = correct_text
    st.session_state.last_explanation = explanation

    rt = max(0.2, float(core.now_ts() - st.session_state.answer_start_ts))
    rating_auto = "good" if got_right else "again"
    apply_grade(rating_auto, got_right=got_right, response_sec=rt, guessed=st.session_state.guessed, prompt_note=False)

    # session stats
    st.session_state.session_done += 1
    if got_right:
        st.session_state.session_correct_like += 1

    # PRACTICE MODE: do NOT auto-advance if show-on-submit is enabled.
    if practice_active():
        st.session_state.practice_done += 1
        if got_right:
            st.session_state.practice_correct += 1

        if st.session_state.practice_show_expl_on_submit:
            st.session_state.practice_waiting_next = True
            save_all()
            return

        # if not showing explanation, auto-advance (fast mode)
        st.session_state.practice_i += 1
        save_all()
        pick_next_item()
        return

    # non-practice: allow manual rating buttons
    st.session_state.pending_rate = True


# ============================================================
# Practice test start/end
# ============================================================
def start_practice_test(chap: str, shuffle: bool):
    qs = [q for q in st.session_state.questions if q.get("chapter") == chap]
    if not qs:
        st.session_state.feedback_banner = f"No questions found for set: {chap}"
        return

    qs_sorted = sorted(qs, key=lambda x: sort_qid_natural(str(x.get("id", ""))))
    keys = [core.q_key(q) for q in qs_sorted if core.q_key(q) in st.session_state.q_by_key]

    st.session_state.practice_shuffle_seed = int(core.now_ts()) ^ random.randint(1, 1_000_000)

    if shuffle:
        rng = random.Random(st.session_state.practice_shuffle_seed)
        rng.shuffle(keys)

    st.session_state.practice_keys = keys
    st.session_state.practice_i = 0
    st.session_state.practice_correct = 0
    st.session_state.practice_done = 0
    st.session_state.practice_waiting_next = False
    st.session_state.content_mode = "practice_test"
    pick_next_item()


def end_practice_test():
    total = len(st.session_state.practice_keys)
    done = st.session_state.practice_done
    correct = st.session_state.practice_correct
    acc = (correct / done * 100.0) if done else 0.0

    st.session_state.feedback_banner = f"Practice test ended. Done {done}/{total} | Correct {correct} | Acc {acc:.0f}%"
    st.session_state.practice_keys = []
    st.session_state.practice_i = 0
    st.session_state.practice_correct = 0
    st.session_state.practice_done = 0
    st.session_state.practice_waiting_next = False
    st.session_state.content_mode = "questions"
    pick_next_item()


# ============================================================
# UI helpers
# ============================================================
def meta_line(chapter_id: str, st_entry: Dict[str, Any], extra: str = "", tags: Optional[Set[str]] = None) -> str:
    due_ts = int(st_entry.get("due", core.now_ts()))
    due_str = datetime.fromtimestamp(due_ts).strftime("%Y-%m-%d %H:%M")
    interval = float(st_entry.get("interval_days", 0.0))
    ease = float(st_entry.get("ease", 2.1))
    seen = int(st_entry.get("seen", 0))
    lapses = int(st_entry.get("lapses", 0))
    strength = float(st_entry.get("strength", 0.2))
    suspended = bool(st_entry.get("suspended", False))
    buried_until = int(st_entry.get("buried_until", 0))
    buried = (buried_until and core.now_ts() < buried_until)
    leech = core.is_leech(st_entry)

    parts = [
        core.pretty_chapter_name(chapter_id),
        f"seen={seen}",
        f"str={strength:.2f}",
        f"int={interval:.1f}d",
        f"ease={ease:.2f}",
        f"due={due_str}",
    ]
    if lapses:
        parts.append(f"lapses={lapses}")
    if suspended:
        parts.append("SUSPENDED")
    if buried:
        parts.append("BURIED")
    if leech:
        parts.append("LEECH")
    if extra:
        parts.append(extra)
    if tags:
        parts.append("tags=" + ",".join(sorted(tags))[:80])

    if practice_active():
        total = len(st.session_state.practice_keys)
        parts.append(
            f"PT {st.session_state.practice_i+1}/{max(1,total)} "
            f"score={st.session_state.practice_correct}/{max(1, st.session_state.practice_done)} "
            f"seed={st.session_state.practice_shuffle_seed if st.session_state.practice_shuffle else 'NA'}"
        )

    return " | ".join(parts)


def show_feedback_block():
    """Shows correct/incorrect + correct answer + explanation AFTER submit/reveal."""
    if not st.session_state.answered:
        return

    ok = st.session_state.last_was_correct
    if ok is True:
        st.success("✅ Correct")
    elif ok is False:
        st.error("❌ Incorrect")
    else:
        st.info("Revealed")

    if st.session_state.last_correct_text:
        st.markdown(f"**{st.session_state.last_correct_text}**")

    if st.session_state.last_explanation:
        # respect practice toggle: if user wants explanation on submit, it's already here;
        # if they turned it off, we keep it blank upstream.
        with st.expander("Explanation", expanded=True):
            st.write(st.session_state.last_explanation)

    # Practice: provide a clear next-step button when we are holding on feedback
    if practice_active() and st.session_state.practice_waiting_next:
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Next practice item", type="primary", use_container_width=True):
                st.session_state.practice_waiting_next = False
                st.session_state.practice_i += 1
                pick_next_item()
                st.rerun()
        with c2:
            st.caption("Practice test is paused so you can see the correct answer/explanation.")


# ============================================================
# Terms
# ============================================================
def term_should_escalate(card: Dict[str, Any]) -> bool:
    e = st.session_state.stats.get(core.term_key(card), core.default_stats_entry())
    interval = float(e.get("interval_days", 0.0))
    strength = float(e.get("strength", 0.2))
    return (strength >= core.STRONG_STRENGTH_THRESHOLD) or (interval >= core.STRONG_INTERVAL_THRESHOLD)


def term_prompt(card: Dict[str, Any]) -> str:
    entry = st.session_state.stats.get(core.term_key(card), core.default_stats_entry())
    prompt, def_idx = core.pick_term_definition(card, entry)
    core.bump_def_seen(entry, def_idx)
    st.session_state.term_prompt_def_idx = int(def_idx)
    st.session_state.term_prompt_text = str(prompt)
    return str(prompt)


def render_term(card: Dict[str, Any]):
    style = st.session_state.term_style
    if style == "mcq":
        term_mcq(card)
        return
    if style == "typing":
        term_typing(card)
        return
    if style == "cloze":
        if term_cloze(card):
            return
        term_typing(card)
        return

    # auto
    if term_should_escalate(card):
        if term_cloze(card):
            return
        term_typing(card)
    else:
        term_mcq(card)


def term_mcq(card: Dict[str, Any]):
    st.subheader("Terms (MCQ)")

    # prompt (and def idx) is set once per card
    if not st.session_state.get("term_prompt_text") or st.session_state.get("term_prompt_key") != st.session_state.current_key:
        prompt = term_prompt(card)
        st.session_state.term_prompt_key = st.session_state.current_key
    else:
        prompt = st.session_state.term_prompt_text

    prompt_out = prompt
    if card.get("why") and random.random() < 0.15:
        prompt_out += f"\n\nWhy it matters:\n{card['why']}"
    if card.get("example") and random.random() < 0.15:
        prompt_out += f"\n\nExample:\n{card['example']}"
    st.write(prompt_out)

    correct = str(card.get("term", "") or "")
    diff = st.session_state.mcq_difficulty

    # ------------------------------------------------------------
    # IMPORTANT: freeze choices per card key so Streamlit reruns
    # don't reshuffle them after you click a radio option.
    # ------------------------------------------------------------
    choices_key = f"mcq_choices::{st.session_state.current_key}"
    if choices_key not in st.session_state:
        distractors: List[str] = []

        if diff == "easy":
            pool = [c["term"] for c in st.session_state.terms if c.get("term") != correct]
            random.shuffle(pool)
            distractors = pool[:3]

        elif diff == "medium":
            sim = pick_term_distractors_smart(
                st.session_state.terms, card, target_def=st.session_state.term_prompt_text, k=8
            )
            distractors = sim[:2]
            pool = [
                c["term"] for c in st.session_state.terms
                if c.get("term") != correct and c.get("term") not in distractors
            ]
            random.shuffle(pool)
            distractors += pool[:1]

        elif diff == "evil":
            sim = pick_term_distractors_smart(
                st.session_state.terms,
                card,
                target_def=st.session_state.term_prompt_text,
                k=min(14, max(3, len(st.session_state.terms) - 1)),
            )
            top = sim[:10] if len(sim) >= 10 else sim
            distractors = random.sample(top, k=min(3, len(top))) if top else []

        else:  # hard
            distractors = pick_term_distractors_smart(
                st.session_state.terms, card, target_def=st.session_state.term_prompt_text, k=3
            )

        # pad if needed
        if len(distractors) < 3:
            pool = [
                c["term"] for c in st.session_state.terms
                if c.get("term") != correct and c.get("term") not in distractors
            ]
            random.shuffle(pool)
            while len(distractors) < 3 and pool:
                distractors.append(pool.pop(0))

        # final options list (strings) + shuffle ONCE
        choices = list(dict.fromkeys(distractors + [correct]))
        random.shuffle(choices)

        st.session_state[choices_key] = choices

    choices = st.session_state[choices_key]

    # Use STRINGS as options (not indices)
    pick = st.radio(
        "Pick the correct term:",
        options=choices,
        index=None,
        key=f"term_mcq_pick::{st.session_state.current_key}",
        disabled=st.session_state.answered,
    )

    if st.button("Submit", disabled=st.session_state.answered):
        if pick is None:
            st.warning("Pick an option first.")
            return

        got_right = core.normalize(str(pick)) == core.normalize(correct)

        defs = card.get("definitions") or []
        di = int(st.session_state.term_prompt_def_idx)
        expl = str(defs[di] or "") if isinstance(defs, list) and 0 <= di < len(defs) else (card.get("definition", "") or "")

        after_answer(got_right, correct_text=f"Correct: {correct}", explanation=expl)
        st.rerun()

    show_feedback_block()



def term_typing(card: Dict[str, Any]):
    st.subheader("Terms (Typing)")
    prompt = term_prompt(card)
    st.write(prompt)

    typed = st.text_input("Type the term:", key=f"term_typed::{st.session_state.current_key}", disabled=st.session_state.answered)

    if st.button("Submit", disabled=st.session_state.answered):
        correct = (card.get("term") or "").strip()
        got_right = core.normalize(typed.strip()) == core.normalize(correct)

        defs = card.get("definitions") or []
        di = int(st.session_state.term_prompt_def_idx)
        expl = str(defs[di] or "") if isinstance(defs, list) and 0 <= di < len(defs) else (card.get("definition", "") or "")

        after_answer(got_right, correct_text=f"Correct: {correct}", explanation=expl)
        st.rerun()

    show_feedback_block()


def term_cloze(card: Dict[str, Any]) -> bool:
    cloze = card.get("cloze", [])
    if not isinstance(cloze, list) or not cloze:
        return False

    candidates = [c for c in cloze if isinstance(c, dict) and c.get("prompt") and c.get("answer")]
    if not candidates:
        return False
    item = random.choice(candidates)

    st.subheader("Terms (Cloze)")
    prompt = str(item["prompt"])
    if "____" not in prompt:
        prompt += "\n\n(Answer the missing part.)"
    st.write(prompt)

    typed = st.text_input("Fill in the blank:", key=f"cloze_typed::{st.session_state.current_key}", disabled=st.session_state.answered)

    if st.button("Submit", disabled=st.session_state.answered):
        correct = str(item["answer"]).strip()
        got_right = core.normalize(typed.strip()) == core.normalize(correct)

        defs = card.get("definitions") or []
        di = int(st.session_state.term_prompt_def_idx)
        expl = str(defs[di] or "") if isinstance(defs, list) and 0 <= di < len(defs) else (card.get("definition", "") or "")

        after_answer(got_right, correct_text=f"Correct: {correct}", explanation=expl)
        st.rerun()

    show_feedback_block()
    return True


# ============================================================
# Questions
# ============================================================
def render_question(q: Dict[str, Any]):
    st.subheader("Practice Test (MCQ)" if practice_active() else "Questions (MCQ)")

    # wrapper support: lock part per question key
    qkey = st.session_state.current_key
    is_wrap = getattr(core, "is_wrapper_question", None) and core.is_wrapper_question(q)

    if is_wrap:
        ctxs = q.get("contexts", []) or []
        total = len(ctxs)

        if total > 0:
            if qkey not in st.session_state.wrapper_part_by_key:
                st.session_state.wrapper_part_by_key[qkey] = random.randrange(0, total)

            pi = st.session_state.wrapper_part_by_key[qkey]
            pi = max(0, min(int(pi), total - 1))
            scen = str(ctxs[pi].get("prompt", "") or "").strip()

            st.write(q.get("question", ""))
            if scen:
                st.write(f"**Scenario:** {scen}")
            st.caption(f"Part {pi+1}/{total}")
        else:
            st.write(q.get("question", ""))
    else:
        st.write(q.get("question", ""))

    choices = q.get("choices", []) or []
    pick = st.radio(
        "Choose the best answer:",
        options=list(range(len(choices))),
        format_func=lambda i: choices[i],
        index=None,
        key=f"q_pick::{qkey}",
        disabled=st.session_state.answered,
    )

    if st.button("Submit", disabled=st.session_state.answered):
        if pick is None:
            st.warning("Pick an option first.")
            return

        expl = str(q.get("explanation", "") or "")

        if is_wrap:
            ctxs = q.get("contexts", []) or []
            if not ctxs:
                got_right = False
                correct_text = "Correct: (wrapper has no contexts)"
            else:
                pi = st.session_state.wrapper_part_by_key.get(qkey, 0)
                pi = max(0, min(int(pi), len(ctxs) - 1))
                cai = ctxs[pi].get("answer_index", None)
                try:
                    cai = int(cai) if cai is not None else None
                except Exception:
                    cai = None

                if cai is None:
                    got_right = False
                    correct_text = "Correct: (Not keyed yet — answer_index is null)"
                else:
                    got_right = (int(pick) == cai)
                    correct_text = f"Correct: {choices[cai]}"
        else:
            ai = q.get("answer_index", None)
            try:
                ai = int(ai) if ai is not None else None
            except Exception:
                ai = None

            if ai is None:
                got_right = False
                correct_text = "Correct: (Not keyed yet — answer_index is null)"
            else:
                got_right = (int(pick) == ai)
                correct_text = f"Correct: {choices[ai]}"

        # practice: optionally suppress explanation even on submit
        if practice_active() and not st.session_state.practice_show_expl_on_submit:
            expl = ""

        after_answer(got_right, correct_text=correct_text, explanation=expl)
        st.rerun()

    show_feedback_block()


# ============================================================
# Tables (minimal)
# ============================================================
def render_table(tb: Dict[str, Any]):
    st.subheader(f"Tables ({st.session_state.content_mode})")
    st.write(tb.get("title", "") or tb.get("id", ""))
    st.caption(tb.get("notes", "") or "")
    st.json(tb)


# ============================================================
# Reveal / Manual rate
# ============================================================
def reveal_current():
    if not st.session_state.current_item:
        return

    st.session_state.answered = True
    st.session_state.pending_rate = True
    st.session_state.last_was_correct = None

    it = st.session_state.current_item
    kind = st.session_state.current_kind

    if kind == "term":
        correct = it.get("term", "")
        defs = it.get("definitions") or []
        di = int(st.session_state.term_prompt_def_idx)
        definition = str(defs[di] or "") if isinstance(defs, list) and 0 <= di < len(defs) else (it.get("definition", "") or "")
        st.session_state.last_correct_text = f"Reveal: {correct}"
        st.session_state.last_explanation = definition

    elif kind == "question":
        choices = it.get("choices", []) or []
        is_wrap = getattr(core, "is_wrapper_question", None) and core.is_wrapper_question(it)
        if is_wrap:
            ctxs = it.get("contexts", []) or []
            if ctxs:
                pi = st.session_state.wrapper_part_by_key.get(st.session_state.current_key, 0)
                pi = max(0, min(int(pi), len(ctxs) - 1))
                cai = ctxs[pi].get("answer_index", None)
                try:
                    cai = int(cai) if cai is not None else None
                except Exception:
                    cai = None
                st.session_state.last_correct_text = "Reveal: (no key)" if cai is None else f"Reveal: {choices[cai]}"
            else:
                st.session_state.last_correct_text = "Reveal: (wrapper has no contexts)"
        else:
            ai = it.get("answer_index", None)
            try:
                ai = int(ai) if ai is not None else None
            except Exception:
                ai = None
            st.session_state.last_correct_text = "Reveal: (Not keyed yet — answer_index is null)" if ai is None else f"Reveal: {choices[ai]}"

        st.session_state.last_explanation = str(it.get("explanation", "") or "")

    else:
        st.session_state.last_correct_text = "Reveal."
        st.session_state.last_explanation = ""

    st.rerun()


def rate(rating: str):
    if not st.session_state.pending_rate or practice_active():
        return

    got_right = rating in ("good", "easy")
    rt = max(0.2, float(core.now_ts() - st.session_state.answer_start_ts))
    apply_grade(rating, got_right=got_right, response_sec=rt, guessed=st.session_state.guessed, prompt_note=True)

    st.session_state.pending_rate = False
    pick_next_item()

    if st.session_state.auto_reset_on_goal and st.session_state.session_done >= st.session_state.session_goal:
        st.session_state.session_done = 0
        st.session_state.session_correct_like = 0
        st.session_state.session_timer_start = core.now_ts()

    st.rerun()


# ============================================================
# Main
# ============================================================
def main():
    st.set_page_config(page_title="CSCS Study Trainer (Streamlit)", layout="wide")
    ensure_state()

    st.title("CSCS Study Trainer (SRS+ Ultra) — Streamlit UI")

    if st.session_state.issues:
        with st.expander("⚠️ Content validation warnings", expanded=False):
            for s in st.session_state.issues[:30]:
                st.write("-", s)
            if len(st.session_state.issues) > 30:
                st.write(f"... and {len(st.session_state.issues)-30} more")

    # Top controls
    left, right = st.columns([3, 2], vertical_alignment="top")

    with left:
        c1, c2, c3 = st.columns([2, 2, 2], vertical_alignment="top")

        with c1:
            st.session_state.content_mode = st.radio(
                "Content",
                ["smart_mix", "terms", "questions", "practice_test", "tables_view", "tables_review", "tables_lookup", "tables_row_quiz"],
                index=["smart_mix","terms","questions","practice_test","tables_view","tables_review","tables_lookup","tables_row_quiz"].index(st.session_state.content_mode),
                horizontal=True,
            )

        with c2:
            st.session_state.study_mode = st.selectbox(
                "Study mode",
                ["spaced", "cram_weak", "new_only", "missed_today", "leeches"],
                index=["spaced", "cram_weak", "new_only", "missed_today", "leeches"].index(st.session_state.study_mode),
            )
            st.session_state.term_style = st.selectbox(
                "Term style",
                ["auto", "mcq", "typing", "cloze"],
                index=["auto","mcq","typing","cloze"].index(st.session_state.term_style),
            )
            st.session_state.mcq_difficulty = st.selectbox(
                "MCQ difficulty",
                ["easy", "medium", "hard", "evil"],
                index=["easy","medium","hard","evil"].index(st.session_state.mcq_difficulty),
            )

        with c3:
            st.write("Smart Mix ratio (T / Q / Tb)")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.session_state.mix_terms = st.number_input("Terms", 0, 100, st.session_state.mix_terms, 1)
            with r2:
                st.session_state.mix_questions = st.number_input("Questions", 0, 100, st.session_state.mix_questions, 1)
            with r3:
                st.session_state.mix_tables = st.number_input("Tables", 0, 100, st.session_state.mix_tables, 1)

    with right:
        elapsed = max(1, core.now_ts() - st.session_state.session_timer_start)
        minutes = elapsed // 60
        done = st.session_state.session_done
        acc = (st.session_state.session_correct_like / done * 100.0) if done else 0.0

        st.session_state.session_goal = st.number_input("Session goal", 10, 500, st.session_state.session_goal, 1)
        st.session_state.auto_reset_on_goal = st.checkbox("Auto-reset on goal", value=st.session_state.auto_reset_on_goal)

        st.write(f"**Session:** Done {done}/{st.session_state.session_goal} | Acc {acc:.0f}% | Time {minutes}m")

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("Reset session", use_container_width=True):
                st.session_state.session_done = 0
                st.session_state.session_correct_like = 0
                st.session_state.session_timer_start = core.now_ts()
                st.session_state.feedback_banner = "🔄 Session reset."
        with b2:
            if st.button("Save", use_container_width=True):
                save_all()
                st.session_state.feedback_banner = "💾 Saved."
        with b3:
            if st.button("Backup", use_container_width=True):
                path = core.write_backup(st.session_state.stats, st.session_state.progress_daily)
                st.session_state.feedback_banner = f"✅ Backup written: {path}"
        with b4:
            # In practice mode, if we're paused on feedback, "Next" should advance
            if st.button("Next", use_container_width=True, type="primary"):
                if practice_active() and st.session_state.practice_waiting_next:
                    st.session_state.practice_waiting_next = False
                    st.session_state.practice_i += 1
                pick_next_item()
                st.rerun()

    # Today bar
    d = st.session_state.progress_daily.get("days", {}).get(core.today_ymd(), {})
    reviews = int(d.get("reviews", 0))
    again = int(d.get("again", 0))
    correct_like = int(d.get("correct_like", 0))
    sec = float(d.get("seconds", 0.0))
    minutes_today = sec / 60.0 if sec else 0.0
    acc_today = (correct_like / reviews * 100.0) if reviews else 0.0

    # streak: consecutive days with >= 10 reviews
    days = st.session_state.progress_daily.get("days", {})
    streak = 0
    cur = datetime.now().date()
    for _ in range(365):
        key = cur.isoformat()
        if int(days.get(key, {}).get("reviews", 0)) >= 10:
            streak += 1
            cur = cur - timedelta(days=1)
        else:
            break

    queues = core.compute_queue_stats(st.session_state.stats)
    st.info(
        f"Today: {reviews} reviews | Acc {acc_today:.0f}% | Again {again} | Time {minutes_today:.1f}m "
        f"| Streak(10+/day): {streak} | Due now: {queues['due_now']} | New: {queues['new']} | Leeches: {queues['leeches']}"
    )

    # Sidebar
    with st.sidebar:
        st.header("Filters")

        chapters = sorted(
            {c["chapter"] for c in st.session_state.terms}
            | {q["chapter"] for q in st.session_state.questions}
            | {t.get("chapter") for t in st.session_state.tables if t.get("chapter")}
        )
        chap_display = [core.pretty_chapter_name(c) for c in chapters]
        chap_map = {core.pretty_chapter_name(c): c for c in chapters}

        selected_chap_disp = st.multiselect(
            "Chapters",
            chap_display,
            default=[core.pretty_chapter_name(c) for c in sorted(st.session_state.chapter_filter)] if st.session_state.chapter_filter else [],
        )
        st.session_state.chapter_filter = {chap_map[d] for d in selected_chap_disp if d in chap_map}

        tags = sorted(all_tags())
        selected_tags = st.multiselect(
            "Tags",
            tags,
            default=sorted(st.session_state.tag_filter) if st.session_state.tag_filter else [],
        )
        st.session_state.tag_filter = set(selected_tags)

        st.divider()

        st.header("Practice test")
        q_chaps = sorted({q.get("chapter", "") for q in st.session_state.questions if q.get("chapter")})
        preferred = [c for c in q_chaps if is_practice_chapter(c)]
        other = [c for c in q_chaps if c not in preferred]
        sets = preferred + other

        st.session_state.practice_set = st.selectbox("Set", sets, index=0 if sets else 0)
        st.session_state.practice_shuffle = st.checkbox("Shuffle", value=st.session_state.practice_shuffle)
        st.session_state.practice_show_expl_on_submit = st.checkbox(
            "Show explanation on submit",
            value=st.session_state.practice_show_expl_on_submit,
        )

        a1, a2 = st.columns(2)
        with a1:
            if st.button("Start / Restart", use_container_width=True):
                start_practice_test(st.session_state.practice_set, st.session_state.practice_shuffle)
                st.rerun()
        with a2:
            if st.button("End", use_container_width=True):
                end_practice_test()
                st.rerun()

    # Banner feedback
    if st.session_state.feedback_banner:
        st.success(st.session_state.feedback_banner)

    # Main render
    it = st.session_state.current_item
    if not it:
        st.warning("No item selected. Adjust filters/mode and press Next.")
        return

    kind = st.session_state.current_kind
    k = st.session_state.current_key
    st_entry = st.session_state.stats.get(k, core.default_stats_entry())

    chapter_id = it.get("chapter", "")
    tags = core.get_item_tags(it)
    extra = f"ID: {it.get('id','')}" if kind == "question" else ""
    st.caption(meta_line(chapter_id, st_entry, extra=extra, tags=tags))

    if kind == "term":
        render_term(it)
    elif kind == "question":
        render_question(it)
    else:
        render_table(it)

    # Bottom actions (manual ratings disabled in practice)
    st.divider()
    b1, b2, b3, b4, b5, b6 = st.columns([1, 1, 1, 1, 1, 1])

    with b1:
        st.session_state.guessed = st.checkbox("I guessed", value=st.session_state.guessed, disabled=practice_active())

    with b2:
        if st.button("Reveal"):
            reveal_current()

    with b3:
        if st.button("Again", disabled=practice_active()):
            rate("again")
    with b4:
        if st.button("Hard", disabled=practice_active()):
            rate("hard")
    with b5:
        if st.button("Good", disabled=practice_active()):
            rate("good")
    with b6:
        if st.button("Easy", disabled=practice_active()):
            rate("easy")


if __name__ == "__main__":
    main()
