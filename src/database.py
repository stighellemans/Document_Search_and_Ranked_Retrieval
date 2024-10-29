from math import log10, sqrt
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, List, Union

import spacy
from tqdm import tqdm

from .helpers import remove_folder, tokens_to_pos_idxs, tokens_to_term_freqs
from .index import DocID, InvertedIndex, PositionalInvertedIndex, PosTermInfo, TermInfo
from .preprocessing import read


def init_worker():
    """Initialize a SpaCy NLP pipeline in each worker, with named entity recognition
    and parsing disabled to speed up processing."""
    global nlp
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.max_length = 10000000


class Database:
    """A class to manage document indexing, normalization, and retrieval
    using an inverted index for efficient information retrieval."""

    doc_normalizations: Dict[str, float]
    doc_ids: List[DocID]

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]],
        index_path: Union[str, Path],
    ):
        """
        Initializes the database with a tokenization function and an index path.

        Args:
            tokenize_fn (Callable): Tokenization function for text processing.
            index_path (Union[str, Path]): Path where the index will be stored.

        Loads or creates an inverted index at the specified path and
        precomputes document norms.
        """
        self.tokenize = tokenize_fn
        self.index_path = Path(index_path)
        self.doc_normalizations = {}

        # Load or initialize an InvertedIndex at the specified path
        self.inverted_index = InvertedIndex(self.index_path)
        self.doc_ids = list(self.inverted_index.doc_ids)

        # Precompute document norms
        self.precompute_doc_norms()

    def build_index(
        self,
        docs: Dict[DocID, str],
        batch_size: int = 5000,
        n_processes: int = 4,
    ):
        """
        Builds the inverted index in batches for the given documents.

        Args:
            docs (Dict[DocID, str]): A dictionary of document IDs and file paths.
            batch_size (int): Number of documents to process per batch.
            n_processes (int): Number of processes for parallel processing.

        Batches the documents, builds partial indices, and merges them into the main index.
        """
        doc_items = list(docs.items())
        total_docs = len(doc_items)
        self.doc_ids = list(docs.keys())
        batch_id = 0
        partial_indices_paths = []

        # Create and clean up a tmp folder
        tmp_dir = Path(self.index_path).parent / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            for i in range(0, total_docs, batch_size):
                batch_docs = dict(doc_items[i : i + batch_size])
                partial_index_path = self.build_partial_index(
                    batch_docs, batch_id, n_processes, tmp_dir
                )
                partial_indices_paths.append(partial_index_path)
                batch_id += 1

            # Merge all partial indices into the main inverted index file
            InvertedIndex.merge_partial_indices(partial_indices_paths, self.index_path)

            # Reload the inverted index with the merged index
            self.inverted_index = InvertedIndex(self.index_path)
            self.doc_ids = list(self.inverted_index.doc_ids)

            # Precompute document norms
            self.precompute_doc_norms()
        finally:
            # Remove tmp_dir once done
            remove_folder(tmp_dir)

    def build_partial_index(
        self,
        batch_docs: Dict[DocID, str],
        batch_id: int,
        n_processes: int,
        tmp_dir: Path,
    ) -> Path:
        """
        Builds a partial inverted index for a batch of documents.

        Args:
            batch_docs (Dict[DocID, str]): A dictionary of document IDs and file paths for the batch.
            batch_id (int): Identifier for the batch.
            n_processes (int): Number of processes for parallel processing.
            tmp_dir (Path): Path to temporary directory.

        Returns:
            Path: Path to the saved partial index file.
        """
        partial_inverted_index = {}
        doc_items = list(batch_docs.items())

        # Process documents in parallel
        with Pool(processes=n_processes, initializer=init_worker) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self.process_single_document, doc_items),
                    total=len(batch_docs),
                    desc=f"Processing Batch {batch_id}",
                )
            )

        # Merge results into the partial inverted index
        for doc_partial_index, doc_id in results:
            self._merge_partial_inverted_index(
                doc_partial_index, partial_inverted_index
            )

        # Save the partial inverted index to disk in the tmp folder
        partial_index_path = tmp_dir / f"partial_index_{batch_id}.idx"
        InvertedIndex.write_partial_index(partial_inverted_index, partial_index_path)

        return partial_index_path

    def process_single_document(self, doc_item):
        """
        Processes a single document, generating term frequencies for indexing.

        Args:
            doc_item (tuple): A tuple of (document ID, document path).

        Returns:
            tuple: A tuple of partial inverted index and document ID.
        """
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
        """
        Merges a partial inverted index into the main inverted index.

        Args:
            partial_inverted_index (Dict[str, TermInfo]): Partial index for a batch.
            main_inverted_index (Dict[str, TermInfo]): The main inverted index.
        """
        for term, term_info in partial_inverted_index.items():
            if term not in main_inverted_index:
                main_inverted_index[term] = term_info
            else:
                # Merge postings and update document frequency
                existing_term_info = main_inverted_index[term]
                updated_doc_freq = existing_term_info[0] + term_info[0]
                updated_posting_dict = {**existing_term_info[1], **term_info[1]}
                main_inverted_index[term] = (updated_doc_freq, updated_posting_dict)

    def precompute_doc_norms(self):
        """
        Precomputes document norms for efficient ranking and retrieval.

        Computes and stores the L2 norms for each document, accounting for term
        weights based on term frequency and inverse document frequency.
        """
        self.doc_normalizations = {}
        N = self.db_size

        # Outer tqdm for terms
        for term in tqdm(
            self.inverted_index.index_pointers.keys(), desc="Precomputing doc norms"
        ):
            term_info = self.inverted_index[term]
            df_t = term_info[0]
            idf_t = log10(N / df_t)

            for doc_id, tf_td in term_info[1].items():
                w_td = (1 + log10(tf_td)) * idf_t
                if doc_id not in self.doc_normalizations:
                    self.doc_normalizations[doc_id] = 0
                self.doc_normalizations[doc_id] += w_td**2

        # Normalize document weights
        self.doc_normalizations = {
            doc_id: sqrt(w) for doc_id, w in self.doc_normalizations.items()
        }

    @property
    def vocab_size(self):
        """
        Returns the vocabulary size of the index.

        Returns:
            int: Number of unique terms in the vocabulary.
        """
        return self.inverted_index.vocab_size

    @property
    def db_size(self):
        """
        Returns the total number of documents in the database.

        Returns:
            int: Number of documents.
        """
        return len(self.doc_ids)


class PositionalDatabase:
    """A class to manage document indexing with positional information."""

    doc_normalizations: Dict[DocID, float]
    doc_ids: List[DocID]

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]],
        index_path: Union[str, Path],
    ):
        """
        Initializes the positional database with a tokenization function and an index path.

        Args:
            tokenize_fn (Callable): Tokenization function for text processing.
            index_path (Union[str, Path]): Path where the index will be stored.

        Loads or creates an inverted index at the specified path and
        precomputes document norms.
        """
        self.tokenize = tokenize_fn
        self.index_path = Path(index_path)
        self.doc_normalizations = {}

        # Load or initialize an InvertedIndex at the specified path
        self.inverted_index = PositionalInvertedIndex(self.index_path)
        self.doc_ids = list(self.inverted_index.doc_ids)

        # Precompute document norms
        self.precompute_doc_norms()

    def build_index(
        self,
        docs: Dict[DocID, str],
        batch_size: int = 5000,
        n_processes: int = 4,
    ):
        """
        Builds the inverted index in batches for the given documents.

        Args:
            docs (Dict[DocID, str]): A dictionary of document IDs and file paths.
            batch_size (int): Number of documents to process per batch.
            n_processes (int): Number of processes for parallel processing.

        Batches the documents, builds partial indices, and merges them into the main index.
        """
        doc_items = list(docs.items())
        total_docs = len(doc_items)
        self.doc_ids = list(docs.keys())
        batch_id = 0
        partial_indices_paths = []

        # Create and clean up a tmp folder
        tmp_dir = Path(self.index_path).parent / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            for i in range(0, total_docs, batch_size):
                batch_docs = dict(doc_items[i : i + batch_size])
                partial_index_path = self.build_partial_index(
                    batch_docs, batch_id, n_processes, tmp_dir
                )
                partial_indices_paths.append(partial_index_path)
                batch_id += 1

            # Merge all partial indices into the main inverted index file
            PositionalInvertedIndex.merge_partial_indices(
                partial_indices_paths, self.index_path
            )

            # Reload the inverted index with the merged index
            self.inverted_index = PositionalInvertedIndex(self.index_path)
            self.doc_ids = list(self.inverted_index.doc_ids)

            # Precompute document norms
            self.precompute_doc_norms()
        finally:
            # Remove tmp_dir once done
            remove_folder(tmp_dir)

    def build_partial_index(
        self,
        batch_docs: Dict[DocID, str],
        batch_id: int,
        n_processes: int,
        tmp_dir: Path,
    ) -> Path:
        """
        Builds a partial inverted index for a batch of documents.

        Args:
            batch_docs (Dict[DocID, str]): A dictionary of document IDs and file paths for the batch.
            batch_id (int): Identifier for the batch.
            n_processes (int): Number of processes for parallel processing.
            tmp_dir (Path): Path to temporary directory.

        Returns:
            Path: Path to the saved partial index file.
        """
        partial_inverted_index = {}

        # Prepare data for multiprocessing
        doc_items = list(batch_docs.items())

        # Process documents in parallel
        with Pool(processes=n_processes, initializer=init_worker) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self.process_single_document, doc_items),
                    total=len(batch_docs),
                    desc=f"Processing Batch {batch_id}",
                )
            )

        # Merge results into the partial inverted index
        for doc_partial_index, doc_id in results:
            self._merge_partial_inverted_index(
                doc_partial_index, partial_inverted_index
            )

        # Save the partial inverted index to disk in the tmp folder
        partial_index_path = tmp_dir / f"partial_index_{batch_id}.idx"
        PositionalInvertedIndex.write_partial_index(
            partial_inverted_index, partial_index_path
        )

        return partial_index_path

    def process_single_document(self, doc_item):
        """
        Processes a single document, generating positional indexes for indexing.

        Args:
            doc_item (tuple): A tuple of (document ID, document path).

        Returns:
            tuple: A tuple of partial inverted index and document ID.
        """
        doc_id, doc_path = doc_item
        doc_text = read(doc_path)
        tokens = self.tokenize(doc_text)
        pos_idxs_dict = tokens_to_pos_idxs(tokens)

        partial_inverted_index = {}
        for term, pos_idxs in pos_idxs_dict.items():
            partial_inverted_index[term] = (1, {doc_id: pos_idxs})

        return (partial_inverted_index, doc_id)

    def _merge_partial_inverted_index(
        self,
        partial_inverted_index: Dict[str, PosTermInfo],
        main_inverted_index: Dict[str, PosTermInfo],
    ):
        """
        Merges a partial inverted index into the main inverted index.

        Args:
            partial_inverted_index (Dict[str, PosTermInfo]): Partial index for a batch.
            main_inverted_index (Dict[str, PosTermInfo]): The main inverted index.
        """
        for term, term_info in partial_inverted_index.items():
            if term not in main_inverted_index:
                main_inverted_index[term] = term_info
            else:
                # Merge postings and update document frequency
                existing_term_info = main_inverted_index[term]
                updated_doc_freq = existing_term_info[0] + term_info[0]
                # Merge postings: combine postings for term in both partial and main index
                existing_postings = existing_term_info[1]
                new_postings = term_info[1]
                # Update postings by adding new postings; for positional index, postings are lists
                for doc_id, positions in new_postings.items():
                    if doc_id in existing_postings:
                        existing_postings[doc_id].extend(positions)
                    else:
                        existing_postings[doc_id] = positions
                main_inverted_index[term] = (updated_doc_freq, existing_postings)

    def precompute_doc_norms(self):
        """
        Precomputes document norms for efficient ranking and retrieval.

        Computes and stores the L2 norms for each document, accounting for term
        weights based on term frequency and inverse document frequency.
        """
        self.doc_normalizations = {}

        N = self.db_size

        for term in self.inverted_index.index_pointers.keys():
            term_info = self.inverted_index[term]
            df_t = term_info[0]
            idf_t = log10(N / df_t) if df_t != 0 else 0
            for doc_id, pos_idxs in term_info[1].items():
                tf_td = len(pos_idxs)
                w_td = (1 + log10(tf_td)) * idf_t
                if doc_id not in self.doc_normalizations:
                    self.doc_normalizations[doc_id] = 0
                self.doc_normalizations[doc_id] += w_td**2

        self.doc_normalizations = {
            doc_id: sqrt(w) for doc_id, w in self.doc_normalizations.items()
        }

    @property
    def vocab_size(self):
        """
        Returns the vocabulary size of the index.

        Returns:
            int: Number of unique terms in the vocabulary.
        """
        return self.inverted_index.vocab_size

    @property
    def db_size(self):
        """
        Returns the total number of documents in the database.

        Returns:
            int: Number of documents.
        """
        return len(self.doc_ids)
