"""Frozen canonical numeric grader from notebook version 1.0.2."""
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from .config import GRADER_VERSION, FROZEN_GRADER_SHA256

# 2.1 Frozen grader specification + hash

GRADER_SPEC = {
    "version": GRADER_VERSION,
    "purpose": (
        "Extract a final numeric answer from saved GSM8K model output "
        "without solving the problem."
    ),
    "priority": [
        "last #### numeric answer",
        "last boxed numeric answer",
        "last explicit final-answer marker; first numeric value after marker",
        (
            "last conclusion-cue sentence/line beginning with "
            "So/Therefore/Thus/Hence; first numeric value after cue"
        ),
        "fallback last numeric literal",
    ],
    "answer_markers": [
        "final answer",
        "answer",
        "the answer is",
        "therefore, the answer is",
        "therefore the answer is",
        "so the answer is",
        "thus the answer is",
    ],
    "conclusion_cues": [
        "so",
        "therefore",
        "thus",
        "hence",
    ],
    "conclusion_rule": (
        "use the last cue-started sentence/line in the response and "
        "extract the first numeric token after the cue"
    ),
    "normalization": [
        "strip currency symbols",
        "strip thousands separators",
        "normalize unicode minus",
        "compare decimals exactly using Fraction(Decimal(...))",
        "support simple fractions such as 3/2",
    ],
    "non_goals": [
        "do not solve the GSM8K question",
        "do not infer an unstated intended answer",
        "do not use test index, routing mode, NFE, AvgK or pair count",
        "do not use the gold answer during extraction",
    ],
}

GRADER_SPEC_JSON = json.dumps(
    GRADER_SPEC,
    sort_keys=True,
    ensure_ascii=False,
)

GRADER_SPEC_SHA256 = hashlib.sha256(
    GRADER_SPEC_JSON.encode("utf-8")
).hexdigest()

print(
    json.dumps(
        GRADER_SPEC,
        indent=2,
        ensure_ascii=False,
    )
)
print("\nSHA256:", GRADER_SPEC_SHA256)


# 3.1 Numeric token grammar

NUMBER_TOKEN_PATTERN = r"""
(?<![\w.])
(?P<num>
    [+\-−]?
    \$?
    (?:
        \d{1,3}(?:,\d{3})+(?:\.\d+)?
        |
        \d+(?:\.\d+)?
        |
        \.\d+
    )
    (?:\s*/\s*[+\-−]?\d+)?
)
(?!\w)
"""

NUMBER_RE = re.compile(
    NUMBER_TOKEN_PATTERN,
    flags=re.VERBOSE,
)

# Literal # must be escaped under re.VERBOSE.
HASH_ANSWER_RE = re.compile(
    r"\#\#\#\#\s*" + NUMBER_TOKEN_PATTERN,
    flags=re.VERBOSE | re.IGNORECASE,
)

BOXED_RE = re.compile(
    r"\\boxed\s*\{\s*"
    + NUMBER_TOKEN_PATTERN
    + r"\s*\}",
    flags=re.VERBOSE | re.IGNORECASE,
)

MARKER_RE = re.compile(
    r"""
    (?:
        final\s+answer
        |
        therefore\s*,?\s*the\s+answer\s+is
        |
        so\s+the\s+answer\s+is
        |
        thus\s+the\s+answer\s+is
        |
        the\s+answer\s+is
        |
        answer
    )
    \s*
    (?:
        is\s*
    )?
    [:=]*\s*
    """,
    flags=re.VERBOSE | re.IGNORECASE,
)

# A conclusion cue must begin a sentence-like segment:
# start of text/line OR after sentence punctuation, followed by optional
# markdown/punctuation and then So/Therefore/Thus/Hence.
CONCLUSION_CUE_RE = re.compile(
    r"""
    (?:
        ^
        |
        (?<=[\n.!?])
    )
    [\s>*_`#\-–—]*
    (?P<cue>
        so
        |
        therefore
        |
        thus
        |
        hence
    )
    \b
    [\s,:;\-–—]*
    """,
    flags=re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

print("Regexes compiled.")


# 3.2 Exact numeric normalization

def normalize_numeric_token(raw):
    if raw is None:
        return None

    s = str(raw).strip()
    s = s.replace("−", "-")
    s = s.replace("$", "")
    s = s.replace(",", "")
    s = s.strip()
    s = s.strip("*_` ")
    s = s.rstrip(".,;:!?")

    if not s:
        return None

    if "/" in s:
        parts = s.split("/")

        if len(parts) != 2:
            return None

        left = parts[0].strip()
        right = parts[1].strip()

        try:
            numerator = Fraction(Decimal(left))
            denominator = Fraction(Decimal(right))
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None

        if denominator == 0:
            return None

        return numerator / denominator

    try:
        return Fraction(Decimal(s))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def fraction_to_display(value):
    if value is None:
        return None

    if value.denominator == 1:
        return str(value.numerator)

    return f"{value.numerator}/{value.denominator}"


# 3.3 Extraction hierarchy

def _record(
    raw,
    rule,
    marker=None,
    span=None,
):
    value = normalize_numeric_token(
        raw
    )

    return {
        "raw": raw,
        "value": value,
        "normalized": fraction_to_display(
            value
        ),
        "rule": rule,
        "marker": marker,
        "span": span,
    }


def canonical_numeric_extract(text):
    """
    Extract a final numeric answer using frozen canonical_numeric_v1.0.2.

    This function receives generated text only.
    """
    if text is None:
        return _record(
            None,
            "no_text",
        )

    text = str(text)

    # 1. Last #### answer.
    hash_matches = list(
        HASH_ANSWER_RE.finditer(
            text
        )
    )

    if hash_matches:
        m = hash_matches[-1]

        return _record(
            m.group("num"),
            "hash_answer",
            marker="####",
            span=m.span(),
        )

    # 2. Last boxed answer.
    boxed_matches = list(
        BOXED_RE.finditer(
            text
        )
    )

    if boxed_matches:
        m = boxed_matches[-1]

        return _record(
            m.group("num"),
            "boxed_answer",
            marker="boxed",
            span=m.span(),
        )

    # 3. Last explicit answer marker; first number after marker.
    marker_matches = list(
        MARKER_RE.finditer(
            text
        )
    )

    if marker_matches:
        marker = marker_matches[-1]
        suffix_start = marker.end()

        suffix = text[
            suffix_start:
            suffix_start + 160
        ]

        number_match = NUMBER_RE.search(
            suffix
        )

        if number_match:
            abs_span = (
                suffix_start
                + number_match.start(),
                suffix_start
                + number_match.end(),
            )

            return _record(
                number_match.group("num"),
                "explicit_answer_marker",
                marker=marker.group(0),
                span=abs_span,
            )

    # 4. Last conclusion-cue segment; first number after cue.
    cue_matches = list(
        CONCLUSION_CUE_RE.finditer(
            text
        )
    )

    if cue_matches:
        cue = cue_matches[-1]
        suffix_start = cue.end()

        # Bound to the local conclusion segment rather than searching the
        # entire remaining response.
        remainder = text[
            suffix_start:
            suffix_start + 240
        ]

        # Stop at the next hard sentence/line boundary when possible.
        boundary = re.search(
            r"(?:\n\s*\n|[.!?](?:\s|$))",
            remainder,
        )

        if boundary:
            conclusion_segment = remainder[
                :boundary.end()
            ]
        else:
            conclusion_segment = remainder

        number_match = NUMBER_RE.search(
            conclusion_segment
        )

        if number_match:
            abs_span = (
                suffix_start
                + number_match.start(),
                suffix_start
                + number_match.end(),
            )

            return _record(
                number_match.group("num"),
                "conclusion_cue",
                marker=cue.group("cue"),
                span=abs_span,
            )

    # 5. Fallback: last numeric token.
    number_matches = list(
        NUMBER_RE.finditer(
            text
        )
    )

    if number_matches:
        m = number_matches[-1]

        return _record(
            m.group("num"),
            "fallback_last_numeric",
            marker=None,
            span=m.span(),
        )

    return _record(
        None,
        "no_numeric_answer",
    )


# 3.8 Canonical grading helper

def gold_numeric_from_doc_answer(
    answer_text,
):
    matches = list(
        HASH_ANSWER_RE.finditer(
            str(
                answer_text
            )
        )
    )

    if not matches:
        raise RuntimeError(
            "Gold answer missing #### numeric answer."
        )

    return normalize_numeric_token(
        matches[
            -1
        ].group(
            "num"
        )
    )


def canonical_grade_text(
    generated_text,
    gold_answer_text,
):
    pred = canonical_numeric_extract(
        generated_text
    )

    gold = gold_numeric_from_doc_answer(
        gold_answer_text
    )

    return {
        "canonical_prediction": pred[
            "normalized"
        ],
        "canonical_rule": pred[
            "rule"
        ],
        "canonical_marker": pred[
            "marker"
        ],
        "canonical_correct": int(
            pred[
                "value"
            ]
            == gold
        ),
        "gold_canonical": fraction_to_display(
            gold
        ),
    }


