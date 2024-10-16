from collections import Counter
from typing import Dict, Sequence


def tokens_to_term_freqs(tokens: Sequence[str]) -> Dict[str, int]:
    return dict(Counter(tokens))
