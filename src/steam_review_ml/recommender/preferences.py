from __future__ import annotations

from dataclasses import dataclass
import re


_SPLIT_RE = re.compile(r"[,\n;.!?]+")
_SPACE_RE = re.compile(r"\s+")
_POSITIVE_CUES = ["love", "like", "enjoy", "want", "more"]
_NEGATIVE_CUES = ["hate", "dislike", "avoid", "too", "less", "don't like"]
_NEGATIONS = ["not", "never", "no", "dont", "don't"]


def _clean_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = _SPACE_RE.sub(" ", t)
    return t


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        x = _clean_text(item)
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _contrastive_clauses(text: str) -> list[str]:
    """Split on ', but ...' so 'I loved X, but I hate Y' yields two clauses."""
    t = _clean_text(text)
    if not t:
        return []
    parts = re.split(r",\s*but\s+", t, maxsplit=1)
    if len(parts) == 1:
        return [t]
    a, b = parts[0].strip(" ,."), parts[1].strip(" ,.")
    out = [x for x in (a, b) if x]
    return out if out else [t]


def _dislike_redundant_vs_avoid(avoid_items: list[str], dislike: str) -> bool:
    """True if every significant token of dislike already appears in one avoid line."""
    dc = _clean_text(dislike)
    if not dc:
        return True
    if any(dc in ac or (ac and ac in dc) for ac in avoid_items):
        return True
    words = [w for w in dc.split() if len(w) > 2]
    if not words:
        return False
    for ac in avoid_items:
        a = _clean_text(ac)
        if a and all(w in a for w in words):
            return True
    return False


def _merge_avoid_with_dislikes(avoid: list[str], dislikes: list[str]) -> list[str]:
    """
    Append dislikes to avoid for embedding, but skip when an avoid line already
    covers the same idea (substring or all significant words present).
    """
    avoid_clean = [_clean_text(a) for a in avoid if a]
    merged = list(avoid)
    for d in dislikes:
        if _dislike_redundant_vs_avoid(avoid_clean, d):
            continue
        merged.append(d)
        avoid_clean.append(_clean_text(d))
    return _dedupe_keep_order(merged)


def _clip_love_window_before_contrast(window: str) -> str:
    """So '...that i love, but i hated Y' does not put 'but i hated Y' in likes."""
    m = re.search(r",\s*but\s+", window)
    if m:
        window = window[: m.start()].strip(" ,.")
    return window


def _extract_phrase_windows(
    text: str,
    cue_patterns: list[str],
    *,
    clip_contrast: bool = False,
) -> list[str]:
    """
    Heuristic extractor:
    takes phrase windows around cues (e.g. "too grindy", "i love ...").
    """
    t = _clean_text(text)
    if not t:
        return []
    hits: list[str] = []
    for cue in cue_patterns:
        for m in re.finditer(cue, t):
            start = m.end()
            window = t[start : min(len(t), start + 120)].strip(" -:,.")
            if clip_contrast:
                window = _clip_love_window_before_contrast(window)
            if not window:
                continue
            # clip phrase to first punctuation-like split
            first = _SPLIT_RE.split(window)[0].strip()
            if first:
                hits.append(first)
    return _dedupe_keep_order(hits)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]


def _contains_any(text: str, terms: list[str]) -> bool:
    t = _clean_text(text)
    return any(term in t for term in terms)


def _polarity_for_sentence(sentence: str) -> str:
    """
    Very lightweight polarity for preference extraction.
    Returns one of: "positive", "negative", "neutral".
    """
    s = _clean_text(sentence)
    has_pos = _contains_any(s, _POSITIVE_CUES)
    has_neg = _contains_any(s, _NEGATIVE_CUES)
    has_negation = _contains_any(s, _NEGATIONS)

    if has_neg and not has_pos:
        return "negative"
    if has_pos and not has_neg:
        # "not fun" should be negative despite "fun" possibly appearing in positive terms.
        if has_negation:
            return "negative"
        return "positive"
    if has_pos and has_neg:
        # Mixed sentence; prefer negative for safety in avoid logic.
        return "negative"
    return "neutral"


def _extract_explicit_dislikes(text: str) -> list[str]:
    """After 'I dislike / hated / hate / don't like', take rest of sentence; split on commas."""
    t = _clean_text(text)
    items: list[str] = []
    for m in re.finditer(r"\bi (?:dislike|disliked|hate|hated|don'?t like)\b", t):
        rest = t[m.end() :]
        end = re.search(r"[.!?\n]", rest)
        frag = rest[: end.start() if end else len(rest)].strip(" ,.")
        if not frag:
            continue
        for part in frag.split(","):
            p = part.strip(" ,.")
            if len(p) > 1:
                items.append(p)
    return _dedupe_keep_order(items)


_POSITIVE_CUE_PATTERNS = [
    r"\bi like\b",
    r"\bi liked\b",
    r"\bi love\b",
    r"\bi loved\b",
    r"\bi enjoy\b",
    r"\bi enjoyed\b",
    r"\bi want\b",
    r"\bi wanted\b",
    r"\blooking for\b",
    r"\bmore of\b",
]
_NEGATIVE_OTHER_PATTERNS = [
    r"\btoo\b",
    r"\bavoid\b",
    r"\bless of\b",
]


def _extract_base_lists(text: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Likes/dislikes from full text, but positive cue windows never swallow a ', but ...' tail.
    With two contrastive clauses, only pull likes from non-negative clauses and dislikes from
    negative clauses (avoids 'that i love' matching then capturing 'but i hated ...').
    """
    t = _clean_text(text)
    clauses = _contrastive_clauses(t)
    likes: list[str] = []
    dislikes_explicit: list[str] = []
    dislikes_other: list[str] = []

    if len(clauses) >= 2:
        for clause in clauses:
            pol = _polarity_for_sentence(clause)
            if pol == "positive":
                likes.extend(
                    _extract_phrase_windows(
                        clause, _POSITIVE_CUE_PATTERNS, clip_contrast=True
                    )
                )
            elif pol == "negative":
                dislikes_explicit.extend(_extract_explicit_dislikes(clause))
                dislikes_other.extend(
                    _extract_phrase_windows(clause, _NEGATIVE_OTHER_PATTERNS)
                )
            else:
                likes.extend(
                    _extract_phrase_windows(
                        clause, _POSITIVE_CUE_PATTERNS, clip_contrast=True
                    )
                )
                dislikes_explicit.extend(_extract_explicit_dislikes(clause))
                dislikes_other.extend(
                    _extract_phrase_windows(clause, _NEGATIVE_OTHER_PATTERNS)
                )
    else:
        likes = _extract_phrase_windows(
            t, _POSITIVE_CUE_PATTERNS, clip_contrast=True
        )
        dislikes_explicit = _extract_explicit_dislikes(t)
        dislikes_other = _extract_phrase_windows(t, _NEGATIVE_OTHER_PATTERNS)

    dislikes = _dedupe_keep_order(dislikes_explicit + dislikes_other)
    must_have = _extract_phrase_windows(
        t,
        cue_patterns=[r"\bmust have\b", r"\bneed\b", r"\brequire\b"],
    )
    style_keywords = [
        "serious",
        "tactical",
        "sandbox",
        "absurd",
        "cozy",
        "competitive",
        "relaxing",
        "immersive",
    ]
    style_tone = [kw for kw in style_keywords if kw in t]
    return likes, dislikes, must_have, style_tone


def _extract_progression_signals(text: str) -> tuple[list[str], list[str], list[str]]:
    """
    Polarity-aware handling for long/grindy/repetitive terms.
    """
    constraints: list[str] = []
    avoid: list[str] = []
    likes: list[str] = []
    keywords = ["long", "grind", "grindy", "repetitive", "forever", "endless"]

    chunks: list[str] = []
    for clause in _contrastive_clauses(text):
        chunks.extend(_sentences(clause))
    if not chunks:
        chunks = _sentences(text)

    for sent in chunks:
        if not any(k in sent for k in keywords):
            continue
        polarity = _polarity_for_sentence(sent)
        if "long" in sent:
            if polarity == "negative" or "too long" in sent:
                constraints.append("shorter pacing preferred")
                avoid.append("overly long or bloated progression")
            elif polarity == "positive":
                likes.append("long-form progression and extended campaigns")
        if any(k in sent for k in ["grind", "grindy", "repetitive"]):
            if polarity == "negative":
                constraints.append("low grind preferred")
                avoid.append("grindy or repetitive progression loops")
            elif polarity == "positive":
                likes.append("grindy/repetitive progression loops")
    return constraints, avoid, likes


def _extract_multiplayer_signals(text: str) -> tuple[list[str], list[str], list[str]]:
    constraints: list[str] = []
    avoid: list[str] = []
    likes: list[str] = []

    multiplayer_terms = ["multiplayer", "multi-player"]
    coop_terms = ["co-op", "coop", "cooperative"]
    pvp_terms = ["pvp", "versus"]
    solo_terms = ["single player", "single-player", "solo"]

    chunks_m: list[str] = []
    for clause in _contrastive_clauses(text):
        chunks_m.extend(_sentences(clause))
    if not chunks_m:
        chunks_m = _sentences(text)

    for sent in chunks_m:
        s = _clean_text(sent)
        has_multi = _contains_any(s, multiplayer_terms + coop_terms + pvp_terms + solo_terms)
        if not has_multi:
            continue
        polarity = _polarity_for_sentence(s)
        if _contains_any(s, pvp_terms) and polarity == "negative":
            constraints.append("single-player/co-op preferred over pvp")
            avoid.append("heavy pvp focus")
        if _contains_any(s, multiplayer_terms + coop_terms):
            if polarity == "positive":
                likes.append("multiplayer/co-op interaction")
            elif polarity == "negative":
                avoid.append("forced multiplayer focus")
        if _contains_any(s, solo_terms):
            if polarity == "positive":
                constraints.append("single-player preference")
            elif polarity == "negative":
                avoid.append("single-player-only experiences")
    return constraints, avoid, likes


def _compute_confidence(
    text: str,
    likes: list[str],
    dislikes: list[str],
    must_have: list[str],
    avoid: list[str],
    style_tone: list[str],
    constraints: list[str],
) -> str:
    total_signals = (
        len(likes)
        + len(dislikes)
        + len(must_have)
        + len(avoid)
        + len(style_tone)
        + len(constraints)
    )
    return "normal" if total_signals >= 2 and len(text) >= 20 else "low"


@dataclass
class PreferenceProfile:
    likes: list[str]
    dislikes: list[str]
    must_have: list[str]
    avoid: list[str]
    style_tone: list[str]
    constraints: list[str]
    confidence: str
    raw_query: str

    def to_dict(self) -> dict[str, object]:
        return {
            "likes": self.likes,
            "dislikes": self.dislikes,
            "must_have": self.must_have,
            "avoid": self.avoid,
            "style_tone": self.style_tone,
            "constraints": self.constraints,
            "confidence": self.confidence,
            "raw_query": self.raw_query,
        }


def extract_preferences(raw_query: str) -> dict[str, object]:
    """
    MVP heuristic extractor (LLM-free):
    - Pulls positive cues into likes/must_have/style_tone.
    - Pulls negative cues into dislikes/avoid.
    - Adds rough constraints for common "too long"/"grindy" complaints.
    """
    text = _clean_text(raw_query)
    if not text:
        profile = PreferenceProfile(
            likes=[],
            dislikes=[],
            must_have=[],
            avoid=[],
            style_tone=[],
            constraints=[],
            confidence="low",
            raw_query="",
        )
        return profile.to_dict()

    likes, dislikes, must_have, style_tone = _extract_base_lists(text)
    constraints_prog, avoid_prog, likes_prog = _extract_progression_signals(text)
    constraints_multi, avoid_multi, likes_multi = _extract_multiplayer_signals(text)

    constraints: list[str] = constraints_prog + constraints_multi
    avoid: list[str] = avoid_prog + avoid_multi
    likes.extend(likes_prog)
    likes.extend(likes_multi)

    likes = _dedupe_keep_order(likes)
    dislikes = _dedupe_keep_order(dislikes)
    must_have = _dedupe_keep_order(must_have)
    avoid = _merge_avoid_with_dislikes(_dedupe_keep_order(avoid), dislikes)
    style_tone = _dedupe_keep_order(style_tone)
    constraints = _dedupe_keep_order(constraints)

    confidence = _compute_confidence(
        text=text,
        likes=likes,
        dislikes=dislikes,
        must_have=must_have,
        avoid=avoid,
        style_tone=style_tone,
        constraints=constraints,
    )

    profile = PreferenceProfile(
        likes=likes,
        dislikes=dislikes,
        must_have=must_have,
        avoid=avoid,
        style_tone=style_tone,
        constraints=constraints,
        confidence=confidence,
        raw_query=raw_query.strip(),
    )
    return profile.to_dict()


def _optional_context_snippet(
    raw_query: str,
    prefs: dict[str, object],
    *,
    max_context_chars: int,
) -> str:
    """
    When the user wrote 'praise, but criticism', repeating the full raw line in
    OPTIONAL_CONTEXT re-injects praised entities (e.g. Agent 47) into the same
    embedding as DISLIKES. If we extracted dislikes and there is a contrastive
    tail clause, keep only that tail for context.
    """
    context_src = raw_query.strip()
    context = _clean_text(context_src)
    if not context:
        return ""
    dislikes = prefs.get("dislikes") or []
    if dislikes and len(_contrastive_clauses(context)) >= 2:
        tail = _contrastive_clauses(context)[-1].strip()
        if tail:
            context = _clean_text(tail)
    if max_context_chars > 0 and len(context) > max_context_chars:
        return context[:max_context_chars].rstrip()
    return context


def build_embedding_input(
    prefs: dict[str, object],
    raw_query: str | None = None,
    *,
    max_context_chars: int = 220,
) -> str:
    """
    Deterministic, normalized preference paragraph for embedding.
    """
    likes = _dedupe_keep_order([str(x) for x in prefs.get("likes", [])])
    dislikes = _dedupe_keep_order([str(x) for x in prefs.get("dislikes", [])])
    must_have = _dedupe_keep_order([str(x) for x in prefs.get("must_have", [])])
    avoid = _dedupe_keep_order([str(x) for x in prefs.get("avoid", [])])
    style_tone = _dedupe_keep_order([str(x) for x in prefs.get("style_tone", [])])
    constraints = _dedupe_keep_order([str(x) for x in prefs.get("constraints", [])])
    confidence = str(prefs.get("confidence", "unknown")).strip().lower() or "unknown"

    context_src = (raw_query or str(prefs.get("raw_query", ""))).strip()
    context = _optional_context_snippet(
        context_src,
        prefs,
        max_context_chars=max_context_chars,
    )

    def _list_or_none(items: list[str]) -> str:
        return ", ".join(items) if items else "none"

    lines = [
        f"WANTS: {_list_or_none(likes)}.",
        f"AVOIDS: {_list_or_none(avoid)}.",
        f"MUST_HAVE: {_list_or_none(must_have)}.",
        f"DISLIKES: {_list_or_none(dislikes)}.",
        f"STYLE_TONE: {_list_or_none(style_tone)}.",
        f"CONSTRAINTS: {_list_or_none(constraints)}.",
        f"CONFIDENCE: {confidence}.",
    ]
    if context:
        lines.append(f"OPTIONAL_CONTEXT: {context}.")

    return " ".join(lines)
