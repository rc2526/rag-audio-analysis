from typing import Any


SKILL_KEYWORDS = {
    "mindful",
    "mindfulness",
    "stop",
    "breathe",
    "breathing",
    "pause",
    "notice",
    "respond",
    "react",
    "reaction",
    "stress",
    "communication",
    "hunger",
    "eating",
}

DEMO_KEYWORDS = {
    "let's",
    "lets",
    "try",
    "practice",
    "notice",
    "take a breath",
    "walk through",
    "show",
    "demonstrate",
}

BARRIER_KEYWORDS = {"hard", "difficult", "struggle", "couldn't", "couldnt", "barrier", "problem", "forgot"}
SUCCESS_KEYWORDS = {"helped", "worked", "better", "successful", "did it", "able", "easier"}
CONFUSION_KEYWORDS = {"confused", "not sure", "don't know", "dont know", "unclear"}
HELP_KEYWORDS = {"how do", "can you help", "question", "what should", "should i"}
FAMILY_KEYWORDS = {"family", "everyone", "together", "husband", "partner", "kids", "children"}


def has_any(text: str, keywords: set[str]) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in keywords)


def infer_contains_skill_language(text: str) -> bool:
    return has_any(text, SKILL_KEYWORDS)


def infer_contains_topic_reference(topic_id: str, confidence: str) -> bool:
    return bool(topic_id and confidence)


def infer_demo_type(text: str) -> str:
    lowered = (text or "").lower()
    if any(token in lowered for token in ("let's", "lets", "try", "practice")):
        return "guided_exercise"
    if any(token in lowered for token in ("show", "demonstrate", "example")):
        return "live_modeling"
    if any(token in lowered for token in ("notice", "reflect", "what did you")):
        return "prompting_reflection"
    if any(token in lowered for token in ("homework", "last week")):
        return "homework_review"
    if any(token in lowered for token in ("should", "could", "when you")):
        return "behavioral_coaching"
    return "didactic_explanation"


def infer_question_domain(row: dict[str, Any]) -> str:
    role = row.get("speaker_role", "")
    text = row.get("span_text", "")
    if role == "facilitator":
        if has_any(text, DEMO_KEYWORDS):
            return "facilitator_demonstration"
        return "facilitator_reference"
    if role in {"mixed", "unknown", "group"}:
        return "mixed_or_uncertain"
    if row.get("contains_child_language") == "1":
        return "participant_child_home"
    return "participant_practice"


def infer_practice_valence(text: str) -> str:
    barrier = has_any(text, BARRIER_KEYWORDS)
    success = has_any(text, SUCCESS_KEYWORDS)
    if barrier and success:
        return "mixed"
    if success:
        return "positive"
    if barrier:
        return "negative"
    return "neutral"


def infer_review_priority(row: dict[str, Any]) -> tuple[str, str]:
    child = str(row.get("contains_child_language", "")) == "1"
    home = str(row.get("contains_home_language", "")) == "1"
    topic_conf = str(row.get("topic_confidence", ""))
    topic_id = str(row.get("topic_id_primary", ""))
    manual_unit = str(row.get("manual_unit_id_best_match", ""))
    skill = str(row.get("contains_skill_language", "")) == "1"
    child_has_context = child and any(
        [
            home,
            bool(topic_id),
            bool(manual_unit),
            topic_conf in {"high", "medium"},
            skill,
        ]
    )
    child_home_with_context = child and home and any(
        [
            bool(topic_id),
            bool(manual_unit),
            topic_conf in {"high", "medium"},
            skill,
        ]
    )

    if child_home_with_context:
        return "high", "child_home_content_with_context"
    if home and topic_id:
        return "high", "home_content_with_topic"
    if topic_conf == "high":
        return "high", "high_topic_confidence"
    if child_has_context:
        return "medium", "child_content_with_context"
    if child:
        return "low", "child_content_only"
    if topic_conf == "medium":
        return "medium", "medium_topic_confidence"
    if manual_unit and topic_id:
        return "medium", "manual_alignment_candidate"
    if skill or home:
        return "medium", "content_keyword_match"
    return "low", "low_signal"
