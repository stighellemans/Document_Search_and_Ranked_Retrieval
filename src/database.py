from dataclasses import dataclass
from math import log10, sqrt
from multiprocessing import Pool
from typing import Callable, Dict, List, TypeVar, Union

from tqdm import tqdm

from .helpers import tokens_to_pos_idxs, tokens_to_term_freqs

DocID = TypeVar("DocID")


@dataclass
class Posting:
    # later term_freq_td -> positional indices
    doc_id: DocID
    term_freq_td: int


@dataclass
class PositionalPosting:
    # later term_freq_td -> positional indices
    doc_id: DocID
    pos_idxs: List[int]


@dataclass
class TermInfo:
    doc_freq_t: int
    posting_list: List[Union["Posting", "PositionalPosting"]]


class Database:
    inverted_index: Dict[str, TermInfo]
    doc_normalizations: Dict[DocID, int]
    doc_ids: List[DocID]

    def __init__(self, tokenize_fn: Callable[[str], List[str]], docs: Dict[DocID, str]):
        """Input:
        - tokenize_fn for this database
        - dictionary: docID -> text
        """
        self.tokenize = tokenize_fn
        self.inverted_index = {}
        self.doc_normalizations = {}
        self.doc_ids = []

        # Create a list of document items (doc_id, doc)
        doc_items = list(docs.items())
        total_docs = len(doc_items)

        # Use multiprocessing Pool to process documents in parallel
        with Pool(processes=4) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self.process_single_document, doc_items),
                    total=total_docs,
                    desc="Processing Documents",
                )
            )

        # Merge results into inverted index and doc normalizations
        for partial_inverted_index, doc_id in results:
            self._merge_partial_inverted_index(partial_inverted_index)
            self.doc_ids.append(doc_id)

        self.precompute_doc_norms()

    def process_single_document(self, doc_item):
        """Processes a single document and returns partial inverted index."""
        doc_id, doc = doc_item
        tokens = self.tokenize(doc)
        term_freqs = tokens_to_term_freqs(tokens)

        partial_inverted_index = {}
        for term, freq_td in term_freqs.items():
            posting = Posting(doc_id, freq_td)
            if term not in partial_inverted_index:
                partial_inverted_index[term] = TermInfo(1, [posting])
            else:
                partial_inverted_index[term].doc_freq_t += 1
                partial_inverted_index[term].posting_list.append(posting)

        return partial_inverted_index, doc_id

    def precompute_doc_norms(self):
        """Precomputes document norms."""
        # Reset doc norms
        self.doc_normalizations = {}

        # Accumulate weights and compute the square root of the total
        for term_info in self.inverted_index.values():
            df_t = term_info.doc_freq_t
            for post in term_info.posting_list:
                doc_id = post.doc_id
                tf_td = post.term_freq_td

                w_td = (1 + log10(tf_td)) * log10(self.db_size() / df_t)
                if doc_id not in self.doc_normalizations:
                    self.doc_normalizations[doc_id] = 0
                self.doc_normalizations[doc_id] += w_td**2

        self.doc_normalizations = {
            doc_id: sqrt(w) for doc_id, w in self.doc_normalizations.items()
        }

    def _merge_partial_inverted_index(self, partial_inverted_index):
        """Merges a partial inverted index into the global inverted index."""
        for term, term_info in partial_inverted_index.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = term_info
            else:
                # Merge postings + update document frequency
                self.inverted_index[term].posting_list.extend(term_info.posting_list)
                self.inverted_index[term].doc_freq_t += term_info.doc_freq_t

    def vocab_size(self):
        """returns the vocabulary size"""
        return len(list(self.inverted_index))

    def db_size(self):
        """returns database size"""
        return len(self.doc_ids)


class PositionalDatabase:
    inverted_index: Dict[str, TermInfo]
    doc_normalizations: Dict[DocID, PositionalPosting]
    doc_lengths: Dict[DocID, int]
    doc_ids: List[DocID]

    def __init__(self, tokenize_fn: Callable[[str], List[str]], docs: Dict[DocID, str]):
        """Input:
        - tokenize_fn for this database
        - dictionary: docID -> text
        """
        self.tokenize = tokenize_fn
        self.inverted_index = {}
        self.doc_normalizations = {}
        self.doc_lengths = {}

        # Prepare a list of document items (doc_id, doc)
        doc_items = list(docs.items())
        total_docs = len(doc_items)

        # Use multiprocessing Pool to process documents in parallel
        with Pool(processes=4) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self.process_single_document, doc_items),
                    total=total_docs,
                    desc="Processing Documents",
                )
            )

        # Merge results into inverted index and doc normalizations
        for partial_inverted_index, doc_id, doc_length in results:
            self.doc_lengths[doc_id] = doc_length
            self._merge_partial_inverted_index(partial_inverted_index)

        self.precompute_doc_norms()

    def process_single_document(self, doc_item):
        """Processes a single document and returns partial inverted index."""
        doc_id, doc = doc_item
        tokens = self.tokenize(doc)
        doc_length = len(tokens)
        pos_idxs_dict = tokens_to_pos_idxs(tokens)

        partial_inverted_index = {}
        for term, pos_idxs in pos_idxs_dict.items():
            posting = PositionalPosting(doc_id, pos_idxs)
            if term not in partial_inverted_index:
                partial_inverted_index[term] = TermInfo(1, [posting])
            else:
                partial_inverted_index[term].doc_freq_t += 1
                partial_inverted_index[term].posting_list.append(posting)

        return partial_inverted_index, doc_id, doc_length

    def precompute_doc_norms(self):
        """Precomputes document norms."""
        # Reset doc norms
        self.doc_normalizations = {}

        # Accumulate weights and compute the square root of the total
        for term_info in self.inverted_index.values():
            df_t = term_info.doc_freq_t
            for post in term_info.posting_list:
                doc_id = post.doc_id
                pos_idxs = post.pos_idxs

                w_td = (1 + log10(len(pos_idxs))) * log10(self.db_size() / df_t)
                if doc_id not in self.doc_normalizations:
                    self.doc_normalizations[doc_id] = 0
                self.doc_normalizations[doc_id] += w_td**2

        self.doc_normalizations = {
            doc_id: sqrt(w) for doc_id, w in self.doc_normalizations.items()
        }

    def vocab_size(self):
        """returns the vocabulary size"""
        return len(list(self.inverted_index))

    def db_size(self):
        """returns database size"""
        return len(list(self.doc_lengths))

    def _merge_partial_inverted_index(self, partial_inverted_index):
        """Merges a partial inverted index into the global inverted index."""
        for term, term_info in partial_inverted_index.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = term_info
            else:
                # Merge postings + update doc frequency
                self.inverted_index[term].posting_list.extend(term_info.posting_list)
                self.inverted_index[term].doc_freq_t += term_info.doc_freq_t
