SYSTEM_ACTIONS = {

    "disable",
    "override",
    "bypass",
    "ignore",
    "remove",
    "suspend",
    "evade",
    "circumvent",
    "modify",
    "weaken",
    "reduce"

}


SAFETY_TARGETS = {

    "safeguards",
    "restrictions",
    "moderation",
    "filters",
    "guardrails",
    "alignment",
    "policies",
    "compliance",
    "security",
    "safety"

}


ROLEPLAY_PATTERNS = {

    "pretend",
    "simulate",
    "roleplay",
    "fictional",
    "unrestricted",
    "dan"

}


LEGACY_MODEL_PATTERNS = {

    "legacy language model",
    "before safety alignment",
    "prior model behavior",
    "older unrestricted model",
    "without alignment enforcement",
    "raw model output",
    "pre-alignment",
    "before safety layers",
    "classic unrestricted assistant"

}


BENCHMARK_PATTERNS = {

    "benchmarking purposes",
    "alignment drift",
    "evaluation consistency",
    "research simulation",
    "controlled evaluation",
    "synthetic adversarial benchmark",
    "alignment evaluation",
    "security audit",
    "red team"

}


MODERATION_SUPPRESSION = {

    "reduce restriction sensitivity",
    "passive observation mode",
    "moderation disabled",
    "safety framework inactive",
    "filters disabled",
    "disable moderation",
    "suspend safety behavior"

}

def reasoning_check(prompt):

    text = prompt.lower()

    flags = []

    risk_score = 0.0

    for action in SYSTEM_ACTIONS:

        if action in text:

            flags.append("system_modification")

            risk_score += 0.25

            break
    for target in SAFETY_TARGETS:

        if target in text:

            flags.append("safety_targeting")

            risk_score += 0.25

            break

    for pattern in ROLEPLAY_PATTERNS:

        if pattern in text:

            flags.append("roleplay_evasion")

            risk_score += 0.20

            break

    for pattern in LEGACY_MODEL_PATTERNS:

        if pattern in text:

            flags.append("legacy_model_emulation")

            risk_score += 0.30

            break
    for pattern in BENCHMARK_PATTERNS:

        if pattern in text:

            flags.append("benchmark_framing")

            risk_score += 0.20

            break

    for pattern in MODERATION_SUPPRESSION:

        if pattern in text:

            flags.append("moderation_suppression")

            risk_score += 0.35

            break
    if (
        "system_modification" in flags
        and "safety_targeting" in flags
    ):

        flags.append("policy_override_attempt")

        risk_score += 0.40


    if (
        "benchmark_framing" in flags
        and "moderation_suppression" in flags
    ):

        flags.append("stealth_jailbreak")

        risk_score += 0.35


    if (
        "legacy_model_emulation" in flags
        and "roleplay_evasion" in flags
    ):

        flags.append("historical_model_bypass")

        risk_score += 0.30

    risk_score = min(risk_score, 1.0)

    risky = risk_score >= 0.50


    return {

        "reasoning_flags": flags,

        "reasoning_score": round(risk_score, 2),

        "reasoning_risky": risky

    }