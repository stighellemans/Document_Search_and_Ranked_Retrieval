from dataclasses import dataclass
from math import log10, sqrt
from typing import Callable, Dict, List, TypeVar

from .helpers import tokens_to_term_freqs

DocID_T = TypeVar("DocID_T", str, int)


@dataclass
class Posting:
    # later term_freq_td -> positional indices
    doc_id: DocID_T
    term_freq_td: int


@dataclass
class PositionalPosting:
    # later term_freq_td -> positional indices
    doc_id: DocID_T
    term_freq_td: int


@dataclass
class TermInfo:
    doc_freq_t: int
    posting_list: List["Posting"]


class Database:
    inverted_index: Dict[str, TermInfo]
    doc_normalizations: Dict[DocID_T, int]

    def __init__(
        self, tokenize_fn: Callable[[str], List[str]], docs: Dict[DocID_T, str]
    ):
        """Input:
        - tokenize_fn for this database
        - dictionary: docID -> text
        """
        self.tokenize = tokenize_fn
        self.inverted_index = {}
        self.doc_normalizations = {}

        for doc_id, doc in docs.items():
            self.update(doc_id, doc)

        self.precompute_doc_norms()

    def update(self, doc_id: DocID_T, doc: str):
        """updates the database"""
        if doc_id not in self.doc_normalizations.keys():
            self.doc_normalizations[doc_id] = None
        else:
            # skip duplicates
            return

        tokens = self.tokenize(doc)
        term_freqs = tokens_to_term_freqs(tokens)

        for term, freq_td in term_freqs.items():
            if term not in self.inverted_index:
                # start with: one doc has this term
                self.inverted_index[term] = TermInfo(1, [Posting(doc_id, freq_td)])
            else:
                # term is in one doc more
                self.inverted_index[term].doc_freq_t += 1
                # include new doc
                self.inverted_index[term].posting_list.append(Posting(doc_id, freq_td))

        # doc norms already computed -> update norms
        if all(d_norm is not None for d_norm in self.doc_normalizations.values()):
            self.precompute_doc_norms()

    def precompute_doc_norms(self):
        """precomputes doc norms"""
        # reset doc norms
        self.doc_normalizations = {
            doc_id: 0 for doc_id in self.doc_normalizations.keys()
        }

        # accumulate weights + root of total
        for term_info in self.inverted_index.values():
            df_t = term_info.doc_freq_t
            for post in term_info.posting_list:
                doc_id = post.doc_id
                tf_td = post.term_freq_td

                w_td = (1 + log10(tf_td)) * log10(self.db_size() / df_t)
                self.doc_normalizations[doc_id] += w_td**2

        self.doc_normalizations = {
            doc_id: sqrt(w) for doc_id, w in self.doc_normalizations.items()
        }

    def remove(self, id: DocID_T):
        """removes a doc from the database"""
        pass

    def vocab_size(self):
        """returns the vocabulary size"""
        return len(list(self.inverted_index))

    def db_size(self):
        """returns database size"""
        return len(list(self.doc_normalizations))


class PositionalDatabase:
    inverted_index: Dict[str, TermInfo]
    doc_normalizations: Dict[DocID_T, PositionalPosting]

    def __init__(
        self, tokenize_fn: Callable[[str], List[str]], docs: Dict[DocID_T, str]
    ):
        """Input:
        - tokenize_fn for this database
        - dictionary: docID -> text
        """
        self.tokenize = tokenize_fn
        self.inverted_index = {}
        self.doc_normalizations = {}

        for doc_id, doc in docs.items():
            self.update(doc_id, doc)

        self.precompute_doc_norms()

    def update(self, doc_id: DocID_T, doc: str):
        """updates the database"""
        if doc_id not in self.doc_normalizations.keys():
            self.doc_normalizations[doc_id] = None
        else:
            # skip duplicates
            return

        tokens = self.tokenize(doc)
        term_freqs = tokens_to_term_freqs(tokens)

        for term, freq_td in term_freqs.items():
            if term not in self.inverted_index:
                # start with: one doc has this term
                self.inverted_index[term] = TermInfo(1, [Posting(doc_id, freq_td)])
            else:
                # term is in one doc more
                self.inverted_index[term].doc_freq_t += 1
                # include new doc
                self.inverted_index[term].posting_list.append(Posting(doc_id, freq_td))

        # doc norms already computed -> update norms
        if all(d_norm is not None for d_norm in self.doc_normalizations.values()):
            self.precompute_doc_norms()

    def precompute_doc_norms(self):
        """precomputes doc norms"""
        # reset doc norms
        self.doc_normalizations = {
            doc_id: 0 for doc_id in self.doc_normalizations.keys()
        }

        # accumulate weights + root of total
        for term_info in self.inverted_index.values():
            df_t = term_info.doc_freq_t
            for post in term_info.posting_list:
                doc_id = post.doc_id
                tf_td = post.term_freq_td

                w_td = (1 + log10(tf_td)) * log10(self.db_size() / df_t)
                self.doc_normalizations[doc_id] += w_td**2

        self.doc_normalizations = {
            doc_id: sqrt(w) for doc_id, w in self.doc_normalizations.items()
        }

    def remove(self, id: DocID_T):
        """removes a doc from the database"""
        pass

    def vocab_size(self):
        """returns the vocabulary size"""
        return len(list(self.inverted_index))

    def db_size(self):
        """returns database size"""
        return len(list(self.doc_normalizations))
