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


import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# Global variables for worker processes
global_db = None
global_query_results = None
global_query_function = None
global_k = None


def init_worker(db_pickle_path, query_results_param, query_function_param, k_param):
    global global_db, global_query_results, global_query_function, global_k
    # Load the database from the pickled file in each worker process
    with open(db_pickle_path, "rb") as f:
        global_db = pickle.load(f)
    global_query_results = query_results_param
    global_query_function = query_function_param
    global_k = k_param


def worker(q_id_q):
    q_id, q = q_id_q
    if q_id in global_query_results:
        relevant_docs = global_query_results[q_id]
        retrieved_docs = [
            doc_id for doc_id, _ in global_query_function(global_db, q)[:global_k]
        ]
        precision = precision_at_k(relevant_docs, retrieved_docs, global_k)
        return precision
    else:
        return None


def map_at_k(
    queries: Dict[QueryID, str],
    query_results: Dict[QueryID, Sequence[DocID]],
    db: Union[Database, PositionalDatabase],
    query_function: Callable,
    k: int,
) -> float:
    # Save the database to a pickle file
    db_pickle_path = "db.pickle"
    with open(db_pickle_path, "wb") as f:
        pickle.dump(db, f)

    # Use ProcessPoolExecutor with initializer
    with ProcessPoolExecutor(
        initializer=init_worker,
        initargs=(db_pickle_path, query_results, query_function, k),
    ) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(worker, item) for item in queries.items()]
        precisions = []
        # Use tqdm to monitor the progress as futures complete
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Processing queries for MAP@K={k}",
        ):
            result = future.result()
            if result is not None:
                precisions.append(result)
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
