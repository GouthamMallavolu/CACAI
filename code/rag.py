from __future__ import annotations
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

MODEL_NAME = "google/flan-t5-small"

@lru_cache(maxsize=1)
def _load():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    mdl = mdl.to(device)
    return tok, mdl, device

def _best_sentence_fallback(query: str, contexts: list[str], must_contain: str | None = None) -> str:
    q_tokens = set(re.findall(r"[a-zA-Z]+", query.lower()))
    if must_contain:
        must_contain = must_contain.lower()

    candidates = []
    for ctx in contexts[:3]:
        # split into sentences
        sents = re.split(r"(?<=[.!?])\s+", ctx.strip())
        for s in sents:
            s_clean = s.strip()
            if len(s_clean) < 30:
                continue
            if must_contain and must_contain not in s_clean.lower():
                continue
            s_tokens = set(re.findall(r"[a-zA-Z]+", s_clean.lower()))
            overlap = len(q_tokens & s_tokens)
            candidates.append((overlap, s_clean))

    candidates.sort(key=lambda x: x[0], reverse=True)
    if candidates and candidates[0][0] >= 2:
        return candidates[0][1]

    return "Not enough evidence in retrieved documents."

def generate_answer(query: str, contexts: list[str], topic_entity: str | None = None) -> str:
    tok, mdl, device = _load()

    trimmed_contexts = [c[:450] for c in contexts[:3]]
    evidence = "\n\n".join(trimmed_contexts)[:1200]

    prompt = (
        "You are a retrieval-grounded assistant.\n"
        "Answer the QUESTION using ONLY the EVIDENCE.\n"
        "If the evidence does not contain the answer, say: Not enough evidence in retrieved documents.\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        f"QUESTION: {query}\n"
        "ANSWER (1-2 sentences):"
    )

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = mdl.generate(**inputs, max_new_tokens=60, do_sample=False)
    ans = tok.decode(out[0], skip_special_tokens=True).strip()

    if topic_entity:
        if topic_entity.lower() not in ans.lower():
            return _best_sentence_fallback(query, trimmed_contexts, must_contain=topic_entity)

    if len(ans) < 5 or "EVIDENCE:" in ans:
        return _best_sentence_fallback(query, trimmed_contexts, must_contain=topic_entity)

    return ans