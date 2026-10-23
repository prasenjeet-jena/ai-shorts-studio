import re
from typing import List, Dict, Any, Optional, Tuple

HOOK_PATTERNS = [
    r'\b(why|how|what if|did you know|imagine|the secret|the truth|stop doing|never|always)\b',
    r'\b(insane|crazy|incredible|unbelievable|shocking|worst|best|massive|huge|mind-blowing)\b',
    r'\b(mistake|problem|solution|hack|trick|warning|secret|rules?)\b',
    r'\b(first time|listen to this|watch this|here is the thing|the reason is|i don\'t care)\b'
]


class CandidateSelector:
    """
    Deterministic candidate window generator, story arc evaluator, and boundary validator.
    Ensures clips start on strong hooks, preserve narrative setup, and conclude on clean punchlines/payoffs.
    """

    def propose_candidates(
        self,
        summary_blocks: List[Dict[str, Any]],
        max_duration: float = 30.0,
        min_duration: float = 12.0,
        count: int = 5,
        scene_cuts: Optional[List[float]] = None,
        min_separation: float = 0.0
    ) -> List[Dict[str, Any]]:
        candidates = []
        n = len(summary_blocks)
        cuts = scene_cuts or []

        for i in range(n):
            start_block = summary_blocks[i]
            start_t = start_block["start"]

            # Scene-aware start snapping: if a clean camera cut exists within 0.35s before start, snap to it
            for sc in cuts:
                if 0 < (start_t - sc) <= 0.35:
                    start_t = sc
                    break

            first_text = start_block["text"].strip()
            hook_score, hook_type = self._score_hook_arc(first_text)
            accumulated_text = [first_text]

            for j in range(i, n):
                end_block = summary_blocks[j]
                end_t = end_block["end"]
                dur = round(end_t - start_t, 2)

                if dur > max_duration:
                    break

                if j > i:
                    accumulated_text.append(end_block["text"].strip())

                if dur >= min_duration:
                    last_text = end_block["text"].strip()
                    ends_cleanly = last_text.endswith((".", "!", "?"))

                    if ends_cleanly or (dur >= min_duration and not last_text.endswith((",", "and", "or", "but"))):
                        full_passage = " ".join(accumulated_text)

                        # Story arc bonus: rewards complete thought with punchline / payoff
                        arc_score = 0.5 if ends_cleanly else 0.2
                        has_payoff_keyword = any(term in last_text.lower() for term in [
                            "wrong", "facebook", "shared", "truth", "secret", "never", "inflation", "joke", "win", "dollar", "money"
                        ])
                        if has_payoff_keyword:
                            arc_score += 0.15

                        score = int((hook_score * 0.45 + arc_score * 0.55) * 100)
                        hook_snippet = first_text[:90] + ("..." if len(first_text) > 90 else "")

                        # Construct transparent selection rationale
                        rationale_parts = [f"Hook category: {hook_type}."]
                        if ends_cleanly:
                            rationale_parts.append("Concludes on a complete, syntactically clean sentence.")
                        if has_payoff_keyword:
                            rationale_parts.append("Features a high-impact narrative payoff keyword.")
                        rationale = " ".join(rationale_parts)

                        candidates.append({
                            "id": f"cand_{i}_{j}",
                            "start": start_t,
                            "end": end_t,
                            "duration": dur,
                            "hook": hook_snippet,
                            "hook_type": hook_type,
                            "summary": full_passage[:160] + ("..." if len(full_passage) > 160 else ""),
                            "score": min(100, score),
                            "has_clean_payoff": ends_cleanly,
                            "reason": rationale
                        })

        # Sort by score descending
        candidates.sort(key=lambda c: c["score"], reverse=True)

        # Overlap suppression & near-duplicate diversity filtering
        selected: List[Dict[str, Any]] = []

        for cand in candidates:
            if len(selected) >= count:
                break

            # Quality gate: do not select candidates with poor hook and incomplete arc
            if cand.get("score", 0) < 40:
                continue

            # Time collision check: disallow overlapping intervals with configurable separation
            collides = False
            for s in selected:
                # Two intervals do not overlap if one finishes before the other begins (plus min_separation)
                is_separated = (cand["end"] + min_separation <= s["start"] or cand["start"] >= s["end"] + min_separation)
                if not is_separated:
                    collides = True
                    break

            if collides:
                continue

            # Text diversity check: reject near-duplicate coverage of the same point
            c_words = set(w.lower() for w in cand["summary"].split() if len(w) > 3)
            is_redundant = False
            for s in selected:
                s_words = set(w.lower() for w in s["summary"].split() if len(w) > 3)
                if c_words and s_words:
                    sim = len(c_words & s_words) / len(c_words | s_words)
                    if sim > 0.35:
                        is_redundant = True
                        break

            if is_redundant:
                continue

            cand_copy = dict(cand)
            cand_copy["id"] = f"short_{len(selected) + 1}"
            selected.append(cand_copy)

        return selected[:count]

    def _score_hook_arc(self, text: str) -> Tuple[float, str]:
        text_lower = text.lower()
        score = 0.5
        hook_type = "declarative"

        if any(w in text_lower for w in ["i don't care", "wrong", "stop", "never", "hate", "quit", "attention spans"]):
            score += 0.40
            hook_type = "contrarian_thesis"
        elif any(w in text_lower for w in ["million dollars", "entire grocery store", "win", "$1,000,000", "40,000", "impossible"]):
            score += 0.40
            hook_type = "high_stakes_premise"
        elif any(re.search(pat, text_lower) for pat in HOOK_PATTERNS):
            score += 0.35
            hook_type = "curiosity_revelation"
        elif "?" in text:
            score += 0.20
            hook_type = "rhetorical_question"

        return min(1.0, score), hook_type
