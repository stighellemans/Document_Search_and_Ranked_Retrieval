from math import exp, log10, sqrt

from .database import Database
from .distances import calc_min_distances
from .helpers import tokens_to_term_freqs


def query_database(database: Database, query: str) -> dict:
    q_tokens = database.tokenize(query)
    # remove tokens not in vocab
    q_tokens = [term for term in q_tokens if term in database.inverted_index]

    q_term_freqs = tokens_to_term_freqs(q_tokens)

    q_norm_accum = 0
    doc_accum = {}
    for q_term, tf_tq in q_term_freqs.items():
        term_info = database.inverted_index[q_term]
        df_t = term_info.doc_freq_t

        w_tq = (1 + log10(tf_tq)) * log10(database.db_size() / df_t)
        q_norm_accum += w_tq**2

        for post in term_info.posting_list:
            doc_id = post.doc_id
            tf_td = len(post.pos_idxs)
            if doc_id not in doc_accum:
                doc_accum[doc_id] = {"similarity": 0, "d_norm": 0}

            w_td = (1 + log10(tf_td)) * log10(database.db_size() / df_t)
            doc_accum[doc_id]["similarity"] += w_td * w_tq

            # d normalization
            doc_accum[doc_id]["d_norm"] += w_td**2

    q_norm = sqrt(q_norm_accum)

    similarities = []
    for doc_id, accum in doc_accum.items():
        d_norm = sqrt(accum["d_norm"])
        similarity = accum["similarity"] / (d_norm * q_norm)
        similarities.append((doc_id, similarity))

    return sorted(similarities, key=lambda x: x[1], reverse=True)


def pos_query_database(database: Database, query: str, alpha: float) -> dict:
    q_tokens = database.tokenize(query)
    # remove tokens not in vocab
    q_tokens = [term for term in q_tokens if term in database.inverted_index]

    q_term_freqs = tokens_to_term_freqs(q_tokens)

    min_dists = calc_min_distances(q_tokens, database)

    q_norm_accum = 0
    doc_accum = {}
    for q_term, tf_tq in q_term_freqs.items():
        term_info = database.inverted_index[q_term]
        df_t = term_info.doc_freq_t

        w_tq = (1 + log10(tf_tq)) * log10(database.db_size() / df_t)
        q_norm_accum += w_tq**2

        for post in term_info.posting_list:
            doc_id = post.doc_id
            tf_td = len(post.pos_idxs)
            if doc_id not in doc_accum:
                doc_accum[doc_id] = {"similarity": 0, "d_norm": 0}

            pos_w = log10(alpha + exp(-min_dists[doc_id]))

            w_td = (1 + log10(tf_td)) * log10(database.db_size() / df_t)
            doc_accum[doc_id]["similarity"] += w_td * w_tq + pos_w

            # d normalization
            doc_accum[doc_id]["d_norm"] += w_td**2

    q_norm = sqrt(q_norm_accum)

    similarities = []
    for doc_id, accum in doc_accum.items():
        d_norm = sqrt(accum["d_norm"])
        similarity = accum["similarity"] / (d_norm * q_norm)
        similarities.append((doc_id, similarity))

    return sorted(similarities, key=lambda x: x[1], reverse=True)
