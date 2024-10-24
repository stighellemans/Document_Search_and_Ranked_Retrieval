from collections import Counter, defaultdict
from typing import Dict, List, Sequence, TypeVar

import numpy as np
import pandas as pd

QueryID = TypeVar("QueryID")
DocID = TypeVar("DocID")


def tokens_to_term_freqs(tokens: Sequence[str]) -> Dict[str, int]:
    return dict(Counter(tokens))


def tokens_to_pos_idxs(tokens: Sequence[str]) -> Dict[str, List[int]]:
    pos_idxs = defaultdict(list)
    for i, term in enumerate(tokens):
        pos_idxs[term].append(i)
    return pos_idxs


def process_query_results(
    queries: Dict[QueryID, str],
    query_results: pd.DataFrame,
    doc_col_name: str = "doc_number",
) -> Dict[QueryID, List[DocID]]:
    new_query_results = {}
    for q_id in queries:
        if q_id in query_results.index:
            relevant_docs = [query_results.loc[q_id][doc_col_name]]
            # Handle multiple relevant docs scenario
            if not isinstance(relevant_docs[0], np.int64):
                relevant_docs = list(relevant_docs[0])  # If multiple relevant docs
            new_query_results[q_id] = relevant_docs
    return new_query_results
