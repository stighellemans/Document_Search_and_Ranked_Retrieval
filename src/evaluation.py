from typing import Callable, Dict, Sequence, TypeVar, Union

import numpy as np
from tqdm import tqdm

from .database import Database, PositionalDatabase

QueryID = TypeVar("QueryID")
DocID = TypeVar("DocID")


def precision_at_k(relevant_docs: Sequence, retrieved_docs: Sequence, k: int) -> float:
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = len(set(relevant_docs).intersection(set(retrieved_k)))
    return relevant_retrieved / k


def recall_at_k(relevant_docs: Sequence, retrieved_docs: Sequence, k: int) -> float:
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = len(set(relevant_docs).intersection(set(retrieved_k)))
    total_relevant = len(relevant_docs)
    if total_relevant == 0:
        return 0.0
    return relevant_retrieved / total_relevant


def map_at_k(
    queries: Dict[QueryID, str],
    query_results: Dict[QueryID, Sequence[DocID]],
    db: Union[Database, PositionalDatabase],
    query_function: Callable,
    k: int,
) -> float:
    """mean average precision at k search results"""
    precisions = []
    for q_id, q in tqdm(queries.items(), desc=f"Processing queries for MAP@K={k}"):
        if q_id in query_results:
            relevant_docs = query_results[q_id]
            retrieved_docs = [doc_id for doc_id, _ in query_function(db, q)[:k]]
            precision = precision_at_k(relevant_docs, retrieved_docs, k)
            precisions.append(precision)

    return float(np.mean(precisions))


def mar_at_k(
    queries: Dict[QueryID, str],
    query_results: Dict[QueryID, Sequence[DocID]],
    db: Union[Database, PositionalDatabase],
    query_function: Callable,
    k: int,
) -> float:
    """mean average recall at k search results"""
    recalls = []
    for q_id, q in tqdm(queries.items(), desc=f"Processing queries for MAP@K={k}"):
        if q_id in query_results:
            relevant_docs = query_results[q_id]
            retrieved_docs = [doc_id for doc_id, _ in query_function(db, q)[:k]]
            recall = recall_at_k(relevant_docs, retrieved_docs, k)
            recalls.append(recall)

    return float(np.mean(recalls))


# def map_at_k(queries, query_results, db, k):
#     """mean average precision at k search results"""
#     precisions = []
#     for q_id, row in queries.iterrows():
#         relevant_docs = [query_results.loc[q_id]["doc_number"]]
#         q = row["Query"]

#         if q_id in query_results.index:
#             retrieved_docs = [doc_id for doc_id, _ in query_database(db, q)[:k]]

#             # Handle multiple relevant docs scenario
#             if not isinstance(relevant_docs[0], np.int64):
#                 relevant_docs = relevant_docs[0]  # If multiple relevant docs

#             precision = precision_at_k(relevant_docs, retrieved_docs, k)
#             precisions.append(precision)

#     return np.mean(precisions)


# def mar_at_k(queries, query_results, db, k):
#     """mean average recall at k search results"""
#     recalls = []
#     for q_id, row in queries.iterrows():
#         relevant_docs = [query_results.loc[q_id]["doc_number"]]
#         q = row["Query"]

#         if q_id in query_results.index:
#             retrieved_docs = [doc_id for doc_id, _ in query_database(db, q)[:k]]

#             # Handle multiple relevant docs scenario
#             if not isinstance(relevant_docs[0], np.int64):
#                 relevant_docs = relevant_docs[0]  # If multiple relevant docs

#             recall = recall_at_k(relevant_docs, retrieved_docs, k)
#             recalls.append(recall)

#     return np.mean(recalls)
