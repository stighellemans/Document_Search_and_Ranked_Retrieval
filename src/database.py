from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, TypeVar

from .helpers import tokens_to_term_freqs

DocID_T = TypeVar("DocID_T", str, int)


class Database:

    @dataclass
    class PostingList:
        docID: DocID_T
        term_freqs_td: Sequence[int] = ()

    inverted_index: Dict[str, PostingList]
    doc_normalizations: Dict[DocID_T, int]

    def __init__(
        self, tokenize_fn: Callable[[str], List[str]], docs: Dict[DocID_T, str]
    ):
        """Input:
        - tokenize_fn for this database
        - dictionary: docID -> text
        """
        self.tokenize = tokenize_fn

        for id, doc in docs.items():
            self.update(id, doc)

    def update(self, id: DocID_T, doc: str):
        """updates the database"""

        pass

    def remove(self, id: DocID_T):
        """removes a doc from the database"""
        pass

    def vocab_size(self):
        """returns the vocabulary size"""
        return len(list(self.inverted_index))
