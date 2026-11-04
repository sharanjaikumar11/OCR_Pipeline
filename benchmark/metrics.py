"""Reference-weighted edit distances. No ground-truth-assisted OCR matching."""

import unicodedata
from collections import Counter

from .extraction import LINE_FIELDS


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def distance(a, b):
    from rapidfuzz.distance import Levenshtein

    return Levenshtein.distance(a, b)


def text_counts(reference, prediction):
    reference, prediction = normalize(reference), normalize(prediction)
    words, other = reference.split(), prediction.split()
    return {
        "char_edits": distance(reference, prediction),
        "reference_chars": len(reference),
        "word_edits": distance(words, other),
        "reference_words": len(words),
        "token_tp": sum((Counter(words) & Counter(other)).values()),
        "reference_tokens": len(words),
        "prediction_tokens": len(other),
    }


def flatten(record):
    return {k: str(v) for k, v in record.items() if k != "poLineArray"}


def compare_record(reference, prediction):
    rows = []

    def add(path, expected, actual, extra=False):
        # Empty reference values are unknown, not negative labels.
        if not extra and not str(expected).strip():
            return
        rows.append(
            {
                "field": path,
                "ground_truth": str(expected),
                "prediction": str(actual),
                "exact": normalize(expected) == normalize(actual),
                **text_counts(expected, actual),
            }
        )

    for key, value in flatten(reference).items():
        add(key, value, prediction.get(key, ""))
    expected = reference.get("poLineArray", [])
    actual = prediction.get("poLineArray", [])
    pairs = []
    if expected and actual:
        from scipy.optimize import linear_sum_assignment

        # One-to-one assignment prevents repeated item rows from being reused.
        def cost(a, b):
            keys = [k for k in LINE_FIELDS if str(a.get(k, "")).strip()]
            return sum(
                distance(normalize(a.get(k, "")), normalize(b.get(k, "")))
                / max(1, len(normalize(a.get(k, ""))))
                for k in keys
            ) / max(1, len(keys))

        matrix = [[cost(a, b) for b in actual] for a in expected]
        left, right = linear_sum_assignment(matrix)
        pairs = list(zip(map(int, left), map(int, right)))
    mapping = dict(pairs)
    exact_items = 0
    for index, item in enumerate(expected):
        other = actual[mapping[index]] if index in mapping else {}
        keys = [k for k in LINE_FIELDS if str(item.get(k, "")).strip()]
        if keys and all(normalize(item[k]) == normalize(other.get(k, "")) for k in keys):
            exact_items += 1
        for key in LINE_FIELDS:
            add(f"poLineArray[{index}].{key}", item.get(key, ""), other.get(key, ""))
    matched = {j for i, j in pairs}
    for index, item in enumerate(actual):
        if index not in matched:
            for key, value in item.items():
                if str(value).strip():
                    add(f"extraItem[{index}].{key}", "", value, extra=True)
    return {
        "fields": rows,
        "reference_items": len(expected),
        "prediction_items": len(actual),
        "exact_items": exact_items,
        "correct_fields": sum(r["exact"] for r in rows),
        "scored_fields": len(rows),
        "structured_counts": {k: sum(r[k] for r in rows) for k in text_counts("", "")},
    }


def rates(counts):
    return {
        "cer": counts["char_edits"] / counts["reference_chars"]
        if counts["reference_chars"]
        else None,
        "wer": counts["word_edits"] / counts["reference_words"]
        if counts["reference_words"]
        else None,
        "token_f1": 2
        * counts["token_tp"]
        / max(1, counts["reference_tokens"] + counts["prediction_tokens"]),
    }
