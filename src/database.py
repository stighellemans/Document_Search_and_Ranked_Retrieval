from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, TypeVar

from .helpers import tokens_to_term_freqs

DocID_T = TypeVar("DocID_T", str, int)


@dataclass
class Posting:
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
    doc_ids: List[DocID_T]

    def __init__(
        self, tokenize_fn: Callable[[str], List[str]], docs: Dict[DocID_T, str]
    ):
        """Input:
        - tokenize_fn for this database
        - dictionary: docID -> text
        """
        self.tokenize = tokenize_fn
        self.inverted_index = {}
        self.doc_ids = []

        for doc_id, doc in docs.items():
            self.update(doc_id, doc)

    def update(self, doc_id: DocID_T, doc: str):
        """updates the database"""
        if doc_id not in self.doc_ids:
            self.doc_ids.append(doc_id)
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

    def remove(self, id: DocID_T):
        """removes a doc from the database"""
        pass

    def vocab_size(self):
        """returns the vocabulary size"""
        return len(list(self.inverted_index))

    def db_size(self):
        """returns database size"""
        return len(self.doc_ids)
