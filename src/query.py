from math import log10, sqrt
from typing import List, Tuple

from .database import Database, DocID, PositionalDatabase
from .helpers import tokens_to_term_freqs
from .phrases import phrase_query_boosts


def query_database(database: Database, query: str) -> List[Tuple[DocID, float]]:
    q_tokens = database.tokenize(query)
    # remove tokens not in vocab
    q_tokens = [term for term in q_tokens if term in database.inverted_index]

    q_term_freqs = tokens_to_term_freqs(q_tokens)

    q_norm_accum = 0
    doc_term_sim = {}
    for q_term, tf_tq in q_term_freqs.items():
        # Retrieve term_info as tuple (doc_freq_t, posting_dict)
        term_info = database.inverted_index[q_term]
        df_t, posting_dict = term_info

        w_tq = (1 + log10(tf_tq)) * log10(database.db_size / df_t)
        q_norm_accum += w_tq**2

        for doc_id, tf_td in posting_dict.items():
            if doc_id not in doc_term_sim:
                doc_term_sim[doc_id] = 0

            w_td = (1 + log10(tf_td)) * log10(database.db_size / df_t)
            doc_term_sim[doc_id] += w_td * w_tq

    q_norm = sqrt(q_norm_accum)

    similarities = []
    for doc_id, raw_sim in doc_term_sim.items():
        d_norm = database.doc_normalizations[doc_id]
        similarity = raw_sim / (d_norm * q_norm)
        similarities.append((doc_id, similarity))

    return sorted(similarities, key=lambda x: x[1], reverse=True)


def query_pos_database(
    database: PositionalDatabase,
    query: str,
    boost_factor: float = 1.0,
    k_width: int = 10,
    q_fraction: float = 1.0,
) -> List[Tuple[DocID, float]]:
    q_tokens = database.tokenize(query)
    # remove tokens not in vocab
    q_tokens = [term for term in q_tokens if term in database.inverted_index]

    q_term_freqs = tokens_to_term_freqs(q_tokens)

    q_norm_accum = 0
    doc_term_sim = {}
    num_q_terms = {}
    for q_term, tf_tq in q_term_freqs.items():
        # Retrieve term_info as tuple (doc_freq_t, posting_dict)
        term_info = database.inverted_index[q_term]
        df_t, posting_dict = term_info

        w_tq = (1 + log10(tf_tq)) * log10(database.db_size / df_t)
        q_norm_accum += w_tq**2

        for doc_id, pos_idxs in posting_dict.items():
            tf_td = len(pos_idxs)
            if doc_id not in doc_term_sim:
                doc_term_sim[doc_id] = 0
                num_q_terms[doc_id] = 0

            num_q_terms[doc_id] += 1

            w_td = (1 + log10(tf_td)) * log10(database.db_size / df_t)
            doc_term_sim[doc_id] += w_td * w_tq

    q_norm = sqrt(q_norm_accum)

    # only boost when certain fraction of q_tokens
    docs_to_boost = [
        doc_id
        for doc_id in doc_term_sim
        if num_q_terms[doc_id] / len(q_tokens) >= q_fraction
    ]
    phrase_q_boosts = phrase_query_boosts(
        q_tokens, docs_to_boost, database, boost_factor, k_width
    )

    similarities = []
    for doc_id, raw_sim in doc_term_sim.items():
        d_norm = database.doc_normalizations[doc_id]
        if doc_id in phrase_q_boosts:
            boost = phrase_q_boosts[doc_id]
        else:
            boost = 1
        similarity = raw_sim / (d_norm * q_norm) * boost
        similarities.append((doc_id, similarity))

    return sorted(similarities, key=lambda x: x[1], reverse=True)
