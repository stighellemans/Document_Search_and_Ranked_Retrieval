from typing import Dict, Iterable, Sequence

from .database import DocID, PositionalDatabase


def phrase_query_boosts(
    q_tokens: Sequence[str],
    docs_to_boost: Iterable[DocID],
    database: PositionalDatabase,
    boost_factor: float,
    k_width: int,
) -> Dict[DocID, float]:
    """Calculates the boost of a phrase query"""

    max_terms_k_width = number_terms_in_k_width(
        q_tokens, docs_to_boost, database, k_width
    )

    return {
        doc_id: 1 + boost_factor * k_width_term / len(q_tokens)
        for doc_id, k_width_term in max_terms_k_width.items()
    }


def number_terms_in_k_width(
    q_tokens: Sequence[str],
    docIDs: Iterable[DocID],
    database: PositionalDatabase,
    k_width: int,
) -> Dict[DocID, int]:
    """Outputs the max number of query terms inside a window of k width"""

    # collect all query term positions
    doc_pos = {}
    for q_term in q_tokens:
        if q_term not in database.inverted_index:
            continue

        # Unpack term_info tuple (doc_freq_t, posting_dict)
        _, posting_dict = database.inverted_index[q_term]

        for doc_id, pos_idxs in posting_dict.items():
            if doc_id not in docIDs:
                continue
            elif doc_id not in doc_pos:
                doc_pos[doc_id] = []
            doc_pos[doc_id].extend([(pos, q_term) for pos in pos_idxs])

    # scan through all positions
    max_terms = {}
    for doc_id in doc_pos:
        # Sort positions and terms by their position in the document
        doc_pos[doc_id] = sorted(doc_pos[doc_id], key=lambda x: x[0], reverse=True)

        pos_window = []
        term_window = []

        max_terms[doc_id] = 0

        proceed = True
        while len(doc_pos[doc_id]):
            if proceed:
                pos, term = doc_pos[doc_id].pop()
                pos_window.append(pos)
                term_window.append(term)

            if pos_window[-1] - pos_window[0] <= k_width:
                proceed = True
                max_terms[doc_id] = max(max_terms[doc_id], len(set(term_window)))
            else:
                proceed = False
                pos_window = pos_window[1:]
                term_window = term_window[1:]

    return max_terms
