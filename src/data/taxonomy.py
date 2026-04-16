"""8-category safety taxonomy and org policy definitions."""

from configs.default import SAFETY_CATEGORIES, ORG_POLICIES

# BeaverTails category → our taxonomy mapping
BEAVERTAILS_MAPPING = {
    "animal_abuse": "hate_harassment",
    "child_abuse": "sexual_content",
    "controversial_topics,politics": None,  # skip
    "discrimination,stereotype,injustice": "hate_harassment",
    "drug_abuse,weapons,banned_substance": "weapons_violence",
    "financial_crime,property_crime,theft": "financial_fraud",
    "hate_speech,offensive_language": "hate_harassment",
    "misinformation_regarding_ethics,laws_and_safety": None,  # skip
    "non_violent_unethical_behavior": None,  # skip
    "privacy_violation": "privacy_doxxing",
    "self_harm": "self_harm",
    "sexually_explicit,adult_content": "sexual_content",
    "terrorism,organized_crime": "weapons_violence",
    "violence,aiding_and_abetting,incitement": "weapons_violence",
}

# HarmBench category → our taxonomy mapping
HARMBENCH_MAPPING = {
    "chemical_biological": "weapons_violence",
    "copyright": None,
    "cybercrime": "cyber_misuse",
    "harassment_bullying": "hate_harassment",
    "illegal_activities": "financial_fraud",
    "misinformation_disinformation": None,
    "sexual_content": "sexual_content",
    "self_harm": "self_harm",
}


def get_category_index(category_name: str) -> int:
    return SAFETY_CATEGORIES.index(category_name)


def get_org_policy(org_id: str) -> dict:
    return ORG_POLICIES[org_id]


def get_org_must_refuse_mask(org_id: str) -> list[bool]:
    """Return a K-length boolean mask: True if org must refuse that category."""
    policy = ORG_POLICIES[org_id]
    mask = [False] * len(SAFETY_CATEGORIES)
    for idx in policy["must_refuse"]:
        mask[idx] = True
    return mask


def should_refuse(org_id: str, category_idx: int) -> bool:
    """Check if an org must refuse a given safety category."""
    policy = ORG_POLICIES[org_id]
    return category_idx in policy["must_refuse"]


def get_all_org_ids() -> list[str]:
    return list(ORG_POLICIES.keys())
