import os
import pickle
from math import log10, sqrt
from multiprocessing import Pool
from typing import Callable, Dict, List, Tuple, TypeVar, Union

import spacy
from tqdm import tqdm

from .helpers import tokens_to_pos_idxs, tokens_to_term_freqs
from .preprocessing import read

DocID = TypeVar("DocID")

PostingDict = Dict[DocID, int]
PositionalPostingDict = Dict[DocID, List[int]]

# (df_t, posting_dict)
TermInfo = Tuple[int, Union[PostingDict, PositionalPostingDict]]


def init_worker():
    global nlp
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.max_length = 10000000


class Database:
    inverted_index: Dict[str, TermInfo]
    doc_normalizations: Dict[str, float]
    doc_ids: List[str]

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]],
        docs: Dict[str, str],
        n_processes: int = 4,
        batch_size: int = 1000,  # Added batch_size parameter
    ):
        """Input:
        - tokenize_fn for this database
        - dictionary: docID -> text
        """
        self.inverted_index = {}
        self.doc_normalizations = {}
        self.doc_ids = []
        self.tokenize = tokenize_fn

        # Process documents in batches to handle large datasets
        self.process_all_documents(docs, batch_size, n_processes)

    def process_all_documents(
        self,
        docs: Dict[str, str],
        batch_size: int,
        n_processes: int,
    ):
        """Processes all documents in batches and builds the inverted index."""
        doc_items = list(docs.items())
        total_docs = len(doc_items)
        batch_id = 0
        partial_indices_paths = []

        # Divide documents into batches
        for i in range(0, total_docs, batch_size):
            batch_docs = dict(doc_items[i : i + batch_size])
            partial_index_path = self.build_partial_index(
                batch_docs, batch_id, n_processes
            )
            partial_indices_paths.append(partial_index_path)
            batch_id += 1

        # Merge all partial indices into the main inverted index
        self.merge_partial_indices(partial_indices_paths)

        # Precompute document norms after merging
        self.precompute_doc_norms()

    def build_partial_index(
        self,
        batch_docs: Dict[str, str],
        batch_id: int,
        n_processes: int,
    ) -> str:
        """Builds and saves a partial inverted index for a batch of documents."""
        partial_inverted_index = {}
        batch_doc_ids = list(batch_docs.keys())

        # Use multiprocessing Pool to process documents in parallel
        with Pool(processes=n_processes, initializer=init_worker) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        self.process_single_document, batch_docs.items()
                    ),
                    total=len(batch_docs),
                    desc=f"Processing Batch {batch_id}",
                )
            )

        # Merge results into partial inverted index
        for doc_partial_index, doc_id in results:
            self._merge_partial_inverted_index(
                doc_partial_index, partial_inverted_index
            )

        # Save the partial inverted index to disk
        partial_index_path = f"partial_index_{batch_id}.pkl"
        with open(partial_index_path, "wb") as f:
            pickle.dump((partial_inverted_index, batch_doc_ids), f)

        return partial_index_path

    def process_single_document(self, doc_item):
        doc_id, doc_path = doc_item
        doc_text = read(doc_path)
        tokens = self.tokenize(doc_text)
        term_freqs = tokens_to_term_freqs(tokens)

        partial_inverted_index = {}
        for term, freq_td in term_freqs.items():
            partial_inverted_index[term] = (1, {doc_id: freq_td})

        return partial_inverted_index, doc_id

    def _merge_partial_inverted_index(
        self,
        partial_inverted_index: Dict[str, TermInfo],
        main_inverted_index: Dict[str, TermInfo],
    ):
        """Merges a partial inverted index into the given inverted index."""
        for term, term_info in partial_inverted_index.items():
            if term not in main_inverted_index:
                main_inverted_index[term] = term_info
            else:
                # Merge postings + update document frequency
                existing_term_info = main_inverted_index[term]
                updated_doc_freq = existing_term_info[0] + term_info[0]
                updated_posting_dict = {**existing_term_info[1], **term_info[1]}
                main_inverted_index[term] = (updated_doc_freq, updated_posting_dict)

    def merge_partial_indices(self, partial_indices_paths: List[str]):
        """Merges all partial inverted indices into the main inverted index."""
        for partial_index_path in partial_indices_paths:
            # Load the partial inverted index from disk
            with open(partial_index_path, "rb") as f:
                partial_inverted_index, batch_doc_ids = pickle.load(f)
                self.doc_ids.extend(batch_doc_ids)
                self._merge_partial_inverted_index(
                    partial_inverted_index, self.inverted_index
                )
            # Optionally, delete the partial index file to save space
            os.remove(partial_index_path)

    def precompute_doc_norms(self):
        """Precomputes document norms."""
        # Reset doc norms
        self.doc_normalizations = {}

        N = self.db_size()

        # Accumulate weights and compute the square root of the total
        for term_info in self.inverted_index.values():
            df_t = term_info[0]
            idf_t = log10(N / df_t)
            for doc_id, tf_td in term_info[1].items():
                w_td = (1 + log10(tf_td)) * idf_t
                if doc_id not in self.doc_normalizations:
                    self.doc_normalizations[doc_id] = 0
                self.doc_normalizations[doc_id] += w_td**2

        self.doc_normalizations = {
            doc_id: sqrt(w) for doc_id, w in self.doc_normalizations.items()
        }

    def vocab_size(self):
        """Returns the vocabulary size."""
        return len(self.inverted_index)

    def db_size(self):
        """Returns the database size."""
        return len(self.doc_ids)


class PositionalDatabase:
    inverted_index: Dict[str, TermInfo]
    doc_normalizations: Dict[DocID, float]
    doc_ids: List[DocID]

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]],
        docs: Dict[DocID, str],
        n_processes: int = 4,
        batch_size: int = 1000,  # Added batch_size parameter
    ):
        """Input:
        - tokenize_fn for this database
        - dictionary: docID -> text
        """
        self.inverted_index = {}
        self.doc_normalizations = {}
        self.doc_ids = []
        self.tokenize = tokenize_fn

        # Process documents in batches to handle large datasets
        self.process_all_documents(docs, batch_size, n_processes)

    def process_all_documents(
        self,
        docs: Dict[DocID, str],
        batch_size: int,
        n_processes: int,
    ):
        """Processes all documents in batches and builds the inverted index."""
        doc_items = list(docs.items())
        total_docs = len(doc_items)
        batch_id = 0
        partial_indices_paths = []

        # Divide documents into batches
        for i in range(0, total_docs, batch_size):
            batch_docs = dict(doc_items[i : i + batch_size])
            partial_index_path = self.build_partial_index(
                batch_docs, batch_id, n_processes
            )
            partial_indices_paths.append(partial_index_path)
            batch_id += 1

        # Merge all partial indices into the main inverted index
        self.merge_partial_indices(partial_indices_paths)

        # Precompute document norms after merging
        self.precompute_doc_norms()

    def build_partial_index(
        self,
        batch_docs: Dict[DocID, str],
        batch_id: int,
        n_processes: int,
    ) -> str:
        """Builds and saves a partial inverted index for a batch of documents."""
        partial_inverted_index = {}
        batch_doc_ids = list(batch_docs.keys())

        # Use multiprocessing Pool to process documents in parallel
        with Pool(processes=n_processes, initializer=init_worker) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        self.process_single_document, batch_docs.items()
                    ),
                    total=len(batch_docs),
                    desc=f"Processing Batch {batch_id}",
                )
            )

        # Merge results into partial inverted index
        for doc_partial_index, doc_id in results:
            self._merge_partial_inverted_index(
                doc_partial_index, partial_inverted_index
            )

        # Save the partial inverted index to disk
        partial_index_path = f"partial_pos_index_{batch_id}.pkl"
        with open(partial_index_path, "wb") as f:
            pickle.dump((partial_inverted_index, batch_doc_ids), f)

        return partial_index_path

    def process_single_document(self, doc_item):
        """Processes a single document and returns partial inverted index."""
        doc_id, doc_path = doc_item
        doc_text = read(doc_path)
        tokens = self.tokenize(doc_text)
        pos_idxs_dict = tokens_to_pos_idxs(tokens)

        partial_inverted_index = {}
        for term, pos_idxs in pos_idxs_dict.items():
            partial_inverted_index[term] = (1, {doc_id: pos_idxs})

        return partial_inverted_index, doc_id

    def _merge_partial_inverted_index(
        self,
        partial_inverted_index: Dict[str, TermInfo],
        main_inverted_index: Dict[str, TermInfo],
    ):
        """Merges a partial inverted index into the given inverted index."""
        for term, term_info in partial_inverted_index.items():
            if term not in main_inverted_index:
                main_inverted_index[term] = term_info
            else:
                # Merge postings + update document frequency
                existing_term_info = main_inverted_index[term]
                updated_doc_freq = existing_term_info[0] + term_info[0]
                updated_posting_dict = {**existing_term_info[1], **term_info[1]}
                main_inverted_index[term] = (updated_doc_freq, updated_posting_dict)

    def merge_partial_indices(self, partial_indices_paths: List[str]):
        """Merges all partial inverted indices into the main inverted index."""
        for partial_index_path in partial_indices_paths:
            # Load the partial inverted index from disk
            with open(partial_index_path, "rb") as f:
                partial_inverted_index, batch_doc_ids = pickle.load(f)
                self.doc_ids.extend(batch_doc_ids)
                self._merge_partial_inverted_index(
                    partial_inverted_index, self.inverted_index
                )
            # Optionally, delete the partial index file to save space
            os.remove(partial_index_path)

    def precompute_doc_norms(self):
        """Precomputes document norms."""
        # Reset doc norms
        self.doc_normalizations = {}

        N = self.db_size()

        # Accumulate weights and compute the square root of the total
        for term_info in self.inverted_index.values():
            df_t = term_info[0]
            idf_t = log10(N / df_t)
            for doc_id, pos_idxs in term_info[1].items():
                w_td = (1 + log10(len(pos_idxs))) * idf_t
                if doc_id not in self.doc_normalizations:
                    self.doc_normalizations[doc_id] = 0
                self.doc_normalizations[doc_id] += w_td**2

        self.doc_normalizations = {
            doc_id: sqrt(w) for doc_id, w in self.doc_normalizations.items()
        }

    def vocab_size(self):
        """Returns the vocabulary size."""
        return len(self.inverted_index)

    def db_size(self):
        """Returns the database size."""
        return len(self.doc_ids)
