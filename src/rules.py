"""Versioned, provenance-carrying curated interaction rules.

These rules are intentionally separate from the ML model and from dataset
evidence. They encode well-known clinical pharmacology cautions; they are
curated for this project and are **not** independently validated here, so the
serving layer must label them as rule output rather than model output.

The matching semantics are unchanged from the original inline rule layer in
`recommend.py`: normalized substring matching on the drug name and on the
canonical food key, evaluated in the order listed below.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from common import norm_name

RULE_VERSION = "2026.11.1"
RULE_SOURCE = "curated clinical pharmacology (this project)"
VALIDATION_STATUS = "unvalidated_curated"
PROVENANCE_NOTE = (
    "Curated from standard clinical pharmacology cautions. Not independently "
    "validated by this project and not a substitute for prescribing guidance."
)

# Marker compound names are mapped to the food-level key used by rules, exactly
# as the previous inline implementation did.
RULE_FOOD_KEYS = {
    "bergamottin": "grapefruit",
    "naringin": "grapefruit",
    "naringenin": "grapefruit",
    "bergapten": "grapefruit",
    "furanocoumarin": "grapefruit",
    "hyperforin": "st john's wort",
    "hypericin": "st john's wort",
    "phylloquinone": "vitamin k",
    "dimethyl disulfide": "garlic",
    "allicin": "garlic",
}

NUTRIENTS = ("calcium", "iron", "magnesium", "zinc", "aluminium", "aluminum")
CYP3A4_SUBSTRATES = (
    "midazolam", "triazolam", "nisoldipine", "felodipine", "buspirone",
    "cyclosporine", "tacrolimus", "sirolimus", "sildenafil", "tadalafil", "vardenafil",
)
MAO_INHIBITORS = (
    "phenelzine", "tranylcypromine", "isocarboxazid", "selegiline", "moclobemide", "linezolid",
)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    label: int
    drug_any: tuple[str, ...]
    food_any: tuple[str, ...]
    sentence: str
    reason: str
    provenance: str
    validation_status: str = VALIDATION_STATUS
    version: str = RULE_VERSION

    def matches(self, drug_key: str, food_key: str) -> bool:
        return (
            any(t in drug_key for t in self.drug_any)
            and any(t in food_key for t in self.food_any)
        )

    def as_dict(self) -> dict:
        out = asdict(self)
        out["drug_any"] = list(self.drug_any)
        out["food_any"] = list(self.food_any)
        return out


@dataclass(frozen=True)
class RuleMatch:
    rule_id: str
    label: int
    sentence: str
    reason: str
    provenance: str
    validation_status: str
    version: str


WARFARIN = ("warfarin", "acenocoumarol", "phenprocoumon")
GRAPEFRUIT = ("grapefruit", "bergamottin", "naringin", "bergapten", "furanocoumarin")
ALCOHOL = ("alcohol", "ethanol", "wine", "beer", "spirits", "liquor")
VITAMIN_K = ("vitamin k", "phylloquinone", "spinach", "kale", "broccoli", "leafy green")
ST_JOHN = ("st john", "hyperforin", "hypericum", "hypericin")

# Order is significant: the first matching rule wins, reproducing the original
# nested if/elif behaviour.
RULES: tuple[Rule, ...] = (
    Rule(
        "WF-GRAPEFRUIT", 0, WARFARIN, GRAPEFRUIT,
        "A should not be taken with food containing B.",
        "Grapefruit inhibits CYP450; the dataset annotates warfarin + grapefruit as avoid and monitor INR.",
        "FHDI annotation + standard warfarin labelling",
    ),
    Rule(
        "WF-GRAPE", 0, WARFARIN, ("grape",),
        "A should not be taken with food containing B.",
        "Grape and grapefruit products are flagged with warfarin; keep intake consistent and monitor INR.",
        PROVENANCE_NOTE,
    ),
    Rule(
        "WF-CRANBERRY", 0, WARFARIN, ("cranber",),
        "A should not be taken with food containing B.",
        "Cranberry has case reports of raised INR with warfarin; keep intake consistent and monitor INR.",
        PROVENANCE_NOTE,
    ),
    Rule(
        "WF-VITAMIN-K", 0, WARFARIN, VITAMIN_K,
        "A should not be taken with food containing B.",
        "Vitamin K antagonizes warfarin; keep intake consistent and monitor INR.",
        "Standard warfarin labelling",
    ),
    Rule(
        "WF-ALCOHOL", 0, WARFARIN, ALCOHOL,
        "A should not be taken with food containing B.",
        "Alcohol changes warfarin metabolism and bleeding risk; avoid heavy or binge drinking.",
        PROVENANCE_NOTE,
    ),
    Rule(
        "WF-GARLIC", 0, WARFARIN, ("garlic", "allicin", "dimethyl disulfide"),
        "A should not be taken with food containing B.",
        "Garlic constituents have case reports of altered INR with warfarin; monitor INR.",
        PROVENANCE_NOTE,
    ),
    Rule(
        "STATIN-GRAPEFRUIT", 0, ("simvastatin", "lovastatin"), GRAPEFRUIT,
        "A should not be taken with food containing B.",
        "Grapefruit raises statin exposure and the risk of myopathy.",
        "Standard simvastatin/lovastatin labelling",
    ),
    Rule(
        "CYP3A4-GRAPEFRUIT", 0, CYP3A4_SUBSTRATES, GRAPEFRUIT,
        "A should not be taken with food containing B.",
        "Grapefruit inhibits intestinal CYP3A4 and raises exposure of this CYP3A4 substrate.",
        PROVENANCE_NOTE,
    ),
    Rule(
        "NITRO-ALCOHOL", 0, ("metronidazole", "tinidazole"), ALCOHOL,
        "A should not be taken with food containing B.",
        "Alcohol with nitroimidazoles can cause a disulfiram-like reaction.",
        "Standard metronidazole/tinidazole labelling",
    ),
    Rule(
        "MAOI-TYRAMINE", 0, MAO_INHIBITORS,
        ("tyramine", "cheese", "aged", "cured", "fermented", "soy sauce", "sauerkraut", "wine", "beer"),
        "A should not be taken with food containing B.",
        "Tyramine-rich foods can trigger a hypertensive crisis with MAO inhibitors.",
        "Standard MAO-inhibitor labelling",
    ),
    Rule(
        "DIGOXIN-STJOHN", 0, ("digoxin",), ST_JOHN,
        "A should not be taken with food containing B.",
        "St John's Wort induces P-glycoprotein and lowers digoxin exposure.",
        PROVENANCE_NOTE,
    ),
    Rule(
        "CATION-CHELATION", 0, ("tetracycline", "cycline", "floxacin"), NUTRIENTS,
        "A should not be taken with food containing B.",
        "Tetracyclines/quinolones bind calcium, iron and other cations; separate dosing.",
        "Standard tetracycline/quinolone labelling",
    ),
    Rule(
        "LEVOTHYROXINE-CATION", 0, ("levothyroxine", "thyroxine"), ("calcium", "iron"),
        "A should not be taken with food containing B.",
        "Calcium and iron reduce levothyroxine absorption; separate dosing.",
        "Standard levothyroxine labelling",
    ),
    Rule(
        "NITRATE-ALCOHOL", 0, ("nitroglycerin", "nitrate"), ALCOHOL,
        "A should not be taken with food containing B.",
        "Alcohol can cause severe hypotension with nitrates.",
        "Standard nitrate labelling",
    ),
    Rule(
        "MTX-ALCOHOL", 0, ("methotrexate",), ALCOHOL,
        "A should not be taken with food containing B.",
        "Alcohol increases hepatotoxicity risk with methotrexate.",
        "Standard methotrexate labelling",
    ),
    Rule(
        "BENZO-ALCOHOL", 0, ("diazepam",), ALCOHOL,
        "A should not be taken with food containing B.",
        "Alcohol adds CNS depression to benzodiazepines.",
        PROVENANCE_NOTE,
    ),
)


def canonical_rule_food_key(food_key: str) -> str:
    return RULE_FOOD_KEYS.get(food_key, food_key)


def evaluate_rules(drug_name: str, food_key: str, sentence_subject: str | None = None) -> RuleMatch | None:
    """Return the first matching rule, or None.

    `sentence_subject` is not used for matching; callers substitute their own
    display names when rendering the sentence.
    """
    drug_key = norm_name(drug_name)
    key = canonical_rule_food_key(norm_name(food_key))
    for rule in RULES:
        if rule.matches(drug_key, key):
            sentence = rule.sentence
            if sentence_subject:
                sentence = sentence.replace("A", sentence_subject, 1)
            return RuleMatch(
                rule_id=rule.rule_id,
                label=rule.label,
                sentence=sentence,
                reason=rule.reason,
                provenance=rule.provenance,
                validation_status=rule.validation_status,
                version=rule.version,
            )
    return None


def rules_document() -> dict:
    return {
        "version": RULE_VERSION,
        "source": RULE_SOURCE,
        "validation_status": VALIDATION_STATUS,
        "note": (
            "Curated rules are evaluated before ML and dataset evidence. They are "
            "deliberately reported as rule-based output and must not be counted as "
            "ML predictions in model evaluation."
        ),
        "rules": [rule.as_dict() for rule in RULES],
    }
