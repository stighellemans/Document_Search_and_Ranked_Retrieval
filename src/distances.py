from typing import Optional, Sequence

from .database import Database


def calc_min_distances(q_tokens: Sequence[str], database: Database):
    # remove tokens not in vocab
    q_tokens = [term for term in q_tokens if term in database.inverted_index]

    # get pos indices to calculate a distance
    doc_pos = {}
    for q_term in q_tokens:
        term_info = database.inverted_index[q_term]

        for post in term_info.posting_list:
            doc_id = post.doc_id
            if doc_id not in doc_pos:
                doc_pos[doc_id] = []
            doc_pos[doc_id].append(post.pos_idxs)

    return {doc_id: min_distance(doc_pos[doc_id]) for doc_id in database.doc_ids}


def min_distance(d_pos_idxs: Sequence[Sequence[int]]) -> Optional[float]:
    min_dist = float("inf")
    for i in range(len(d_pos_idxs)):
        for j in range(i + 1, len(d_pos_idxs)):
            for pos1 in d_pos_idxs[i]:
                for pos2 in d_pos_idxs[j]:
                    dist = abs(pos1 - pos2)
                    if dist < min_dist:
                        min_dist = dist
    return min_dist if min_dist != float("inf") else None
