from collections import Counter
from typing import Dict, List, Sequence


def tokens_to_term_freqs(tokens: Sequence[str]) -> Dict[str, int]:
    return dict(Counter(tokens))


def tokens_to_pos_idxs(tokens: Sequence[str]) -> Dict[str, List[int]]:
    pos_idxs = {}

    for i, term in enumerate(tokens):
        if term not in pos_idxs:
            pos_idxs[term] = [i]
        else:
            pos_idxs[term].append(i)

    return pos_idxs
