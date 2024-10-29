import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Sequence, TypeVar, Union

import numpy as np
from tqdm import tqdm

from .database import Database, PositionalDatabase
from .index import DocID

QueryID = TypeVar("QueryID")


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


# Global variables for worker processes
global_db = None
global_query_results = None
global_query_function = None
global_k = None


def init_worker(db_path, query_results_param, query_function_param, k_param):
    global global_db, global_query_results, global_query_function, global_k
    # Load the database from the pickled file in each worker process
    with open(db_path, "rb") as f:
        global_db = pickle.load(f)
    global_query_results = query_results_param
    global_query_function = query_function_param
    global_k = k_param


def worker(q_id_q, metric="precision"):
    q_id, q = q_id_q
    if q_id in global_query_results:
        relevant_docs = global_query_results[q_id]
        retrieved_docs = [
            doc_id for doc_id, _ in global_query_function(global_db, q)[:global_k]
        ]
        if metric == "precision":
            return precision_at_k(relevant_docs, retrieved_docs, global_k)
        elif metric == "recall":
            return recall_at_k(relevant_docs, retrieved_docs, global_k)
    return None


def map_at_k(
    queries: Dict[QueryID, str],
    query_results: Dict[QueryID, Sequence[DocID]],
    db: Union[Database, PositionalDatabase],
    query_function: Callable,
    k: int,
    max_workers: int = 4,
) -> float:
    db_path = Path(db.index_path).with_suffix(".pkl")
    with open(db_path, "wb") as f:
        pickle.dump(db, f)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(db_path, query_results, query_function, k),
    ) as executor:
        futures = [
            executor.submit(worker, item, metric="precision")
            for item in queries.items()
        ]
        precisions = []
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
    max_workers: int = 4,
) -> float:
    db_path = Path(db.index_path).with_suffix(".pkl")
    with open(db_path, "wb") as f:
        pickle.dump(db, f)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(db_path, query_results, query_function, k),
    ) as executor:
        futures = [
            executor.submit(worker, item, metric="recall") for item in queries.items()
        ]
        recalls = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Processing queries for MAR@K={k}",
        ):
            result = future.result()
            if result is not None:
                recalls.append(result)
    return float(np.mean(recalls))


def retrieval_worker(q_id_q):
    q_id, query = q_id_q
    retrieved_docs = [
        doc_id for doc_id, _ in global_query_function(global_db, query)[:global_k]
    ]
    return q_id, retrieved_docs


def retrieve_top_k_docs(
    queries: Dict[QueryID, str],
    db: Union[Database, PositionalDatabase],
    query_function: Callable,
    k: int,
    max_workers: int = 4,
) -> Dict[QueryID, Sequence[DocID]]:
    """
    Retrieve the top k documents per query from the database in parallel.

    Args:
        queries (Dict[QueryID, str]): A dictionary of queries with their IDs.
        db (Union[Database, PositionalDatabase]): The database instance to query.
        query_function (Callable): A function to retrieve documents from the database.
        k (int): Number of top documents to retrieve per query.
        max_workers (int): The maximum number of worker processes.

    Returns:
        Dict[QueryID, Sequence[DocID]]: A dictionary mapping each query ID to its top k retrieved documents.
    """
    # Serialize the database to avoid reloading in each process
    db_path = Path(db.index_path).with_suffix(".pkl")
    with open(db_path, "wb") as f:
        pickle.dump(db, f)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(db_path, {}, query_function, k),
    ) as executor:
        futures = [executor.submit(retrieval_worker, item) for item in queries.items()]
        retrieved_docs = {}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Retrieving top-k documents"
        ):
            q_id, docs = future.result()
            retrieved_docs[q_id] = docs
    return retrieved_docs
