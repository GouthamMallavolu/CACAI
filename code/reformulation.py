import re

PRONOUNS = {"it", "its", "they", "their", "them", "this", "that", "these", "those", "he", "his", "she", "her"}

def extract_topic_entity(query: str) -> str | None:
    """
    Extract topic entity from definitional queries like:
    - 'what is X'
    - 'who is X'
    - 'tell me about X'
    - 'define X'
    - 'explain X'
    """
    q = query.strip().lower().rstrip("?.!")

    patterns = [
        r"^what is (.+)$",
        r"^who is (.+)$",
        r"^tell me about (.+)$",
        r"^define (.+)$",
        r"^explain (.+)$",
    ]

    for pat in patterns:
        m = re.match(pat, q)
        if m:
            cand = m.group(1).strip()
            tokens = set(cand.split())
            if tokens & PRONOUNS:
                return None
            return cand

    return None


def reformulate_with_entity(query: str, topic_entity: str | None) -> str:
    """
    Replace pronouns with the topic entity.
    Example:
      topic_entity = 'global warming'
      'what are its effects' -> 'what are global warming effects'
      'how does it affect agriculture' -> 'how does global warming affect agriculture'
    """
    if not topic_entity:
        return query.strip()

    tokens = query.strip().split()
    out = []
    for t in tokens:
        tl = t.lower().strip("?.!,")
        if tl in PRONOUNS:
            out.append(topic_entity)
        else:
            out.append(t)

    # Clean extra spaces
    return " ".join(out).replace("  ", " ").strip()