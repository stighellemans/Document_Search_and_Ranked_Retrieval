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
    nlp = spacy.load("en_core_web_sm")


class Database:
    inverted_index: Dict[str, TermInfo]
    doc_normalizations: Dict[str, int]
    doc_ids: List[str]

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]],
        docs: Dict[str, str],
        n_processes: int = 4,
    ):
        """Input:
        - tokenize_fn for this database (not used since we use global 'preprocess')
        - dictionary: docID -> text
        """
        self.inverted_index = {}
        self.doc_normalizations = {}
        self.doc_ids = []
        self.tokenize = tokenize_fn

        # Create a list of document items (doc_id, doc_text)
        doc_items = list(docs.items())
        total_docs = len(doc_items)

        # Use multiprocessing Pool to process documents in parallel
        with Pool(processes=n_processes, initializer=init_worker) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self.process_single_document, doc_items),
                    total=total_docs,
                    desc="Processing Documents",
                )
            )

        # Merge results into inverted index and doc ids
        for partial_inverted_index, doc_id in results:
            self._merge_partial_inverted_index(partial_inverted_index)
            self.doc_ids.append(doc_id)

        self.precompute_doc_norms()

    def process_single_document(self, doc_item):
        doc_id, doc_path = doc_item
        doc_text = read(doc_path)
        tokens = self.tokenize(doc_text)
        term_freqs = tokens_to_term_freqs(tokens)

        partial_inverted_index = {}
        for term, freq_td in term_freqs.items():
            partial_inverted_index[term] = (1, {doc_id: freq_td})

        return partial_inverted_index, doc_id

    def _merge_partial_inverted_index(self, partial_inverted_index):
        """Merges a partial inverted index into the global inverted index."""
        for term, term_info in partial_inverted_index.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = term_info
            else:
                # Merge postings + update document frequency
                existing_term_info = self.inverted_index[term]
                updated_doc_freq = existing_term_info[0] + term_info[0]
                updated_posting_dict = {**existing_term_info[1], **term_info[1]}
                self.inverted_index[term] = (updated_doc_freq, updated_posting_dict)

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
        """returns the vocabulary size"""
        return len(self.inverted_index)

    def db_size(self):
        """returns database size"""
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
    ):
        """Input:
        - tokenize_fn for this database
        - dictionary: docID -> text
        """
        self.inverted_index = {}
        self.doc_normalizations = {}
        self.doc_ids = []
        self.tokenize = tokenize_fn

        # Create a list of document items (doc_id, doc_text)
        doc_items = list(docs.items())
        total_docs = len(doc_items)

        # Use multiprocessing Pool to process documents in parallel
        with Pool(processes=n_processes) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self.process_single_document, doc_items),
                    total=total_docs,
                    desc="Processing Documents",
                )
            )

        # Merge results into inverted index and doc ids
        for partial_inverted_index, doc_id in results:
            self._merge_partial_inverted_index(partial_inverted_index)
            self.doc_ids.append(doc_id)

        self.precompute_doc_norms()

    def process_single_document(self, doc_item):
        """Processes a single document and returns partial inverted index."""
        doc_id, doc_path = doc_item
        doc_text = read(doc_path)
        tokens = self.tokenize(doc_text)
        pos_idxs_dict = tokens_to_pos_idxs(tokens)

        partial_inverted_index = {}
        for term, pos_idxs in pos_idxs_dict.items():
            if term not in partial_inverted_index:
                partial_inverted_index[term] = (1, {doc_id: pos_idxs})
            else:
                # Increment doc frequency and update postings
                doc_freq, posting_dict = partial_inverted_index[term]
                posting_dict[doc_id] = pos_idxs
                partial_inverted_index[term] = (doc_freq + 1, posting_dict)

        return partial_inverted_index, doc_id

    def _merge_partial_inverted_index(self, partial_inverted_index):
        """Merges a partial inverted index into the global inverted index."""
        for term, term_info in partial_inverted_index.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = term_info
            else:
                # Merge postings + update document frequency
                existing_term_info = self.inverted_index[term]
                updated_doc_freq = existing_term_info[0] + term_info[0]
                updated_posting_dict = {**existing_term_info[1], **term_info[1]}
                self.inverted_index[term] = (updated_doc_freq, updated_posting_dict)

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
        """returns the vocabulary size"""
        return len(self.inverted_index)

    def db_size(self):
        """returns the database size"""
        return len(self.doc_ids)
