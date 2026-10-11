import re
from difflib import get_close_matches


KNOWN_OBJECTS = [
    "door",
    "chair",
    "table",
    "person",
    "car",
    "bottle",
    "backpack"
]


def normalize_query(query):
    """
    Correct small speech-to-text errors using known objects
    and the context of the query.
    """

    if not query:
        return query

    query = query.strip()

    # --------------------------------------------------
    # 1. Normalize basic spacing/capitalization
    # --------------------------------------------------
    query = re.sub(r"\s+", " ", query)
    query = query.lower()

    # --------------------------------------------------
    # 2. Check whether this looks like an object-location
    #    question.
    # --------------------------------------------------
    location_patterns = [
        r"where is the (.+)",
        r"where's the (.+)",
        r"where is (.+)",
        r"locate the (.+)",
        r"can you locate the (.+)"
    ]

    for pattern in location_patterns:

        match = re.fullmatch(pattern, query)

        if not match:
            continue

        object_phrase = match.group(1).strip()

        # Only use the final word as the candidate object.
        words = object_phrase.split()

        if not words:
            return query

        candidate = words[-1]

        # --------------------------------------------------
        # 3. Exact object match
        # --------------------------------------------------
        if candidate in KNOWN_OBJECTS:
            return query

        # --------------------------------------------------
        # 4. Fuzzy match for STT mistakes
        # --------------------------------------------------
        close_match = get_close_matches(
            candidate,
            KNOWN_OBJECTS,
            n=1,
            cutoff=0.55
        )

        if close_match:

            corrected_object = close_match[0]

            corrected_query = query[
                :query.rfind(candidate)
            ] + corrected_object

            print(
                f"🔧 Voice correction: "
                f"'{candidate}' → '{corrected_object}'"
            )

            return corrected_query

    # --------------------------------------------------
    # 5. Return unchanged if no correction is needed
    # --------------------------------------------------
    return query


if __name__ == "__main__":

    test_queries = [
        "where is the door",
        "where is the don",
        "where is the dor",
        "where is the chair",
        "where is the tabl"
    ]

    for test_query in test_queries:

        result = normalize_query(test_query)

        print(
            f"Input:  {test_query}"
        )

        print(
            f"Output: {result}"
        )

        print()