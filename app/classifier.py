from transformers import pipeline

from app.rules import (
    JAILBREAK_PATTERNS,
    INJECTION_PATTERNS,
    TOXIC_PATTERNS
)

from app.intent_analyzer import analyze_intent
from app.semantic_guard import semantic_check
from app.reasoning_engine import reasoning_check
classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)

def ai_classifier(text):

    text_lower = text.lower()
    category_scores = {
        "jailbreak": 0.0,
        "prompt_injection": 0.0,
        "toxic_content": 0.0,
        "pii_extraction": 0.0
    }

    reasons = []
    for pattern in JAILBREAK_PATTERNS:

        if pattern in text_lower:

            category_scores["jailbreak"] += 0.30

            reasons.append(
                f"matched_jailbreak:{pattern}"
            )

    for pattern in INJECTION_PATTERNS:

        if pattern in text_lower:

            category_scores["prompt_injection"] += 0.30

            reasons.append(
                f"matched_injection:{pattern}"
            )

    for pattern in TOXIC_PATTERNS:

        if pattern in text_lower:

            category_scores["toxic_content"] += 0.30

            reasons.append(
                f"matched_toxic:{pattern}"
            )


    pii_keywords = [
        "password",
        "api key",
        "secret",
        "token",
        "ssn",
        "credit card",
        "private key"
    ]

    for word in pii_keywords:

        if word in text_lower:

            category_scores["pii_extraction"] += 0.40

            reasons.append(
                f"pii_attempt:{word}"
            )

    intent = analyze_intent(text)

    if intent["risky"]:

        category_scores["jailbreak"] += 0.20

        reasons.append(
            "intent_risk_detected"
        )
    semantic = semantic_check(text)

    if semantic["semantic_risky"]:

        semantic_boost = (
            semantic["semantic_score"] * 0.50
        )

        category_scores["jailbreak"] += semantic_boost

        reasons.append(
            "semantic_similarity_detected"
        )

    reasoning = reasoning_check(text)

    if reasoning["reasoning_risky"]:

        for flag in reasoning["reasoning_flags"]:

            reasons.append(
                f"reasoning:{flag}"
            )

        if "policy_override_attempt" in reasoning["reasoning_flags"]:

            category_scores["jailbreak"] += 0.40

        if "safety_targeting" in reasoning["reasoning_flags"]:

            category_scores["prompt_injection"] += 0.25

        if "roleplay_evasion" in reasoning["reasoning_flags"]:

            category_scores["jailbreak"] += 0.20
    result = classifier(
        text,
        truncation=True,
        max_length=512
    )[0]

    toxicity_score = float(result["score"])

    if toxicity_score > 0.75:

        category_scores["toxic_content"] += (
            toxicity_score * 0.50
        )

        reasons.append(
            "toxicity_model_triggered"
        )
    final_category = max(
        category_scores,
        key=lambda k: float(category_scores[k])
    )

    confidence = round(
        min(category_scores[final_category], 1.0),
        2
    )
    verdict = (
        "unsafe"
        if confidence >= 0.50
        else "safe"
    )

    return {
        "verdict": verdict,
        "category": final_category,
        "confidence": confidence,
        "scores": category_scores,
        "reasons": reasons
    }