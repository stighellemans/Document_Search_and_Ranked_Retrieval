import heapq
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

DocID = int

PostingDict = Dict[DocID, int]
PositionalPostingDict = Dict[DocID, List[int]]

# (df_t, posting_dict)
TermInfo = Tuple[int, PostingDict]
PosTermInfo = Tuple[int, PositionalPostingDict]


class InvertedIndex:
    index_pointers: Dict[str, int]
    _num_terms_to_show: int = 10
    _num_postings_to_show: int = 5

    def __init__(
        self, index_path: Union[str, Path], index: Optional[Dict[str, TermInfo]] = None
    ):
        self.index_path = Path(index_path)

        if not self.index_path.exists():
            self.index_path.touch()
        if index:
            self.write_partial_index(index, index_path)

        self.index_path = Path(index_path)
        self.index_pointers = InvertedIndex.load(self.index_path)

    def __getitem__(self, term: str) -> TermInfo:
        """
        Fetches the TermInfo (df_t, posting_dict) for a given term using index pointers.
        """
        offset = self.index_pointers.get(term)
        if offset is not None:
            with open(self.index_path, "rb") as f:
                record = InvertedIndex.read_record(f, binary_position=offset)
                if record:
                    return record[1:]
        raise KeyError(f"Term '{term}' not found in index.")

    @property
    def vocab_size(self) -> int:
        return len(self.index_pointers)

    @property
    def doc_ids(self) -> Set[DocID]:
        """
        Collects all unique DocIDs in the index.
        Returns:
            A set of all DocIDs found in the postings of all terms.
        """
        all_doc_ids = set()
        for term in self.index_pointers.keys():
            try:
                _, postings = self[term]
                all_doc_ids.update(postings.keys())
            except KeyError:
                continue  # Skip terms that may not exist in the index

        return all_doc_ids

    def __str__(self) -> str:
        """
        Returns a string representation of the first 10 terms in the index with their TermInfo.
        If there are more than 10 terms, shows an ellipsis to indicate there are more terms.
        Also displays the total vocabulary size and the file size at the end.
        """

        def format_size(size_in_bytes: int) -> str:
            """Convert bytes to a readable format (kB, MB, GB)."""
            for unit in ["Bytes", "kB", "MB", "GB", "TB"]:
                if size_in_bytes < 1024:
                    return f"{size_in_bytes:.2f} {unit}"
                size_in_bytes /= 1024
            return f"{size_in_bytes:.2f} PB"

        terms = sorted(self.index_pointers.keys())
        output = []

        for i, term in enumerate(terms[: self._num_terms_to_show]):
            term_info = self[term]
            doc_freq = term_info[0]
            postings = term_info[1]

            limited_postings = list(postings.items())[: self._num_postings_to_show]
            postings_str = ", ".join(
                f"{doc_id}: {freq}" for doc_id, freq in limited_postings
            )

            if len(postings) > self._num_postings_to_show:
                postings_str += ", ..."

            output.append(
                f"{i+1}. {term}: df={doc_freq}, postings={{ {postings_str} }}"
            )

        if len(terms) > self._num_terms_to_show:
            output.append("...")

        file_size = os.path.getsize(self.index_path)
        readable_file_size = format_size(file_size)

        output.append(f"Vocabulary size: {len(terms)}, File size: {readable_file_size}")

        return "\n".join(output)

    def __repr__(self) -> str:
        """
        Returns the same as __str__
        """
        return self.__str__()

    def __contains__(self, term: str) -> bool:
        """
        Checks if a term exists in the index by verifying if it has an entry in index_pointers.
        """
        return term in self.index_pointers

    @staticmethod
    def write_partial_index(
        partial_index: Dict[str, TermInfo], file_path: Union[str, Path]
    ) -> None:
        """Writes a partial index to a binary file in sorted order."""
        with open(file_path, "wb") as f:
            for term in sorted(partial_index.keys()):
                term_info = partial_index[term]
                doc_freq = term_info[0]
                postings = term_info[1]

                InvertedIndex.write_record(f, term, doc_freq, postings)

    @staticmethod
    def read_record(
        file, binary_position: Optional[int] = None
    ) -> Optional[Tuple[str, int, PostingDict]]:
        """
        Reads a term and postings record from a binary file at a specific position.
        """
        if binary_position is not None:
            file.seek(binary_position)

        term_length_bytes = file.read(2)
        if not term_length_bytes:
            return None  # End of file
        term_length = struct.unpack("H", term_length_bytes)[0]

        term_bytes = file.read(term_length)
        term = term_bytes.decode("utf-8")

        doc_freq_bytes = file.read(4)
        num_postings_bytes = file.read(4)
        doc_freq = struct.unpack("I", doc_freq_bytes)[0]
        num_postings = struct.unpack("I", num_postings_bytes)[0]

        postings = {}

        # Read postings (DocID as integer)
        for _ in range(num_postings):
            doc_id_bytes = file.read(4)
            doc_id = struct.unpack("I", doc_id_bytes)[0]

            # Read term frequency
            freq_bytes = file.read(4)
            freq = struct.unpack("I", freq_bytes)[0]

            postings[doc_id] = freq

        return term, doc_freq, postings

    @staticmethod
    def write_record(
        file,
        term: str,
        doc_freq: int,
        postings: PostingDict,
        binary_displacement: Optional[int] = None,
    ) -> None:
        """
        Writes a term and its postings to a binary file,
        optionally displaced by binary_displacement.
        """
        if binary_displacement is not None:
            file.seek(binary_displacement)

        term_bytes = term.encode("utf-8")
        term_length = len(term_bytes)

        file.write(struct.pack("H", term_length))
        file.write(term_bytes)

        num_postings = len(postings)
        file.write(struct.pack("I", doc_freq))
        file.write(struct.pack("I", num_postings))

        # Write postings (DocID as integer)
        for doc_id, freq in postings.items():
            file.write(struct.pack("I", doc_id))
            file.write(struct.pack("I", freq))

    @staticmethod
    def merge_partial_indices(
        partial_indices_paths: List[Union[str, Path]], output_path: Union[str, Path]
    ) -> None:
        """
        Merges all partial inverted indices into a single index
        using external merging.
        """
        files = [open(path, "rb") for path in partial_indices_paths]
        iterators = []
        heap = []

        for file_idx, f in enumerate(files):
            record = InvertedIndex.read_record(f)
            if record:
                term, doc_freq, postings = record
                heapq.heappush(heap, (term, file_idx, doc_freq, postings))
                iterators.append(f)
            else:
                f.close()

        with open(output_path, "wb") as output_file:
            current_term = None
            current_doc_freq = 0
            current_postings = {}

            while heap:
                term, file_idx, doc_freq, postings = heapq.heappop(heap)

                if term != current_term and current_term is not None:
                    InvertedIndex.write_record(
                        output_file, current_term, current_doc_freq, current_postings
                    )
                    current_doc_freq = 0
                    current_postings = {}

                current_term = term
                current_doc_freq += doc_freq
                for doc_id, freq in postings.items():
                    current_postings[doc_id] = current_postings.get(doc_id, 0) + freq

                next_record = InvertedIndex.read_record(files[file_idx])
                if next_record:
                    next_term, next_doc_freq, next_postings = next_record
                    heapq.heappush(
                        heap, (next_term, file_idx, next_doc_freq, next_postings)
                    )
                else:
                    files[file_idx].close()

            if current_term is not None:
                InvertedIndex.write_record(
                    output_file, current_term, current_doc_freq, current_postings
                )

        for f in files:
            if not f.closed:
                f.close()

        for path in partial_indices_paths:
            os.remove(path)

    @staticmethod
    def load(path: Union[Path, str]) -> Dict[str, int]:
        """Builds a dictionary of term offsets for quick access by using read_record."""
        index_pointers = {}

        with open(path, "rb") as f:
            while True:
                offset = f.tell()
                record = InvertedIndex.read_record(f)
                if record is None:
                    break

                term, _, _ = record
                index_pointers[term] = offset

        return index_pointers


class PositionalInvertedIndex:
    index_pointers: Dict[str, int]
    _num_terms_to_show: int = 10
    _num_postings_to_show: int = 5

    def __init__(
        self,
        index_path: Union[str, Path],
        index: Optional[Dict[str, PosTermInfo]] = None,
    ):
        self.index_path = Path(index_path)

        if not self.index_path.exists():
            self.index_path.touch()
        if index:
            self.write_partial_index(index, self.index_path)

        self.index_pointers = self.load(self.index_path)

    def __getitem__(self, term: str) -> PosTermInfo:
        """
        Fetches the PosTermInfo (df_t, positional_posting_dict) for a given term using index pointers.
        """
        offset = self.index_pointers.get(term)
        if offset is not None:
            with open(self.index_path, "rb") as f:
                record = self.read_record(f, binary_position=offset)
                if record:
                    return record[1:]
        raise KeyError(f"Term '{term}' not found in index.")

    def __str__(self) -> str:
        """
        Returns a string representation of the first 10 terms in the positional index with their PosTermInfo.
        If there are more than 10 terms, shows an ellipsis to indicate there are more terms.
        Also displays the total vocabulary size and the file size at the end.
        """

        def format_size(size_in_bytes: int) -> str:
            """Convert bytes to a readable format (kB, MB, GB)."""
            for unit in ["Bytes", "kB", "MB", "GB", "TB"]:
                if size_in_bytes < 1024:
                    return f"{size_in_bytes:.2f} {unit}"
                size_in_bytes /= 1024
            return f"{size_in_bytes:.2f} PB"

        terms = sorted(self.index_pointers.keys())
        output = []

        for i, term in enumerate(terms[: self._num_terms_to_show]):
            term_info = self[term]
            doc_freq = term_info[0]
            postings = term_info[1]

            limited_postings = list(postings.items())[: self._num_postings_to_show]
            postings_str = ", ".join(
                f"{doc_id}: {positions}" for doc_id, positions in limited_postings
            )

            if len(postings) > self._num_postings_to_show:
                postings_str += ", ..."

            output.append(
                f"{i+1}. {term}: df={doc_freq}, positional_postings={{ {postings_str} }}"
            )

        if len(terms) > self._num_terms_to_show:
            output.append("...")

        file_size = os.path.getsize(self.index_path)
        readable_file_size = format_size(file_size)

        output.append(f"Vocabulary size: {len(terms)}, File size: {readable_file_size}")

        return "\n".join(output)

    def __repr__(self) -> str:
        """
        Returns the same as __str__
        """
        return self.__str__()

    @property
    def vocab_size(self) -> int:
        return len(self.index_pointers)

    @property
    def doc_ids(self) -> Set[DocID]:
        """
        Collects all unique DocIDs in the index.
        """
        all_doc_ids = set()
        with open(self.index_path, "rb") as f:
            while True:
                record = self.read_record(f)
                if record is None:
                    break
                _, _, postings = record
                all_doc_ids.update(postings.keys())

        return all_doc_ids

    def __contains__(self, term: str) -> bool:
        return term in self.index_pointers

    @staticmethod
    def write_partial_index(
        partial_index: Dict[str, PosTermInfo], file_path: Union[str, Path]
    ) -> None:
        """Writes a partial positional index to a binary file."""
        with open(file_path, "wb") as f:
            for term in sorted(partial_index.keys()):
                term_info = partial_index[term]
                doc_freq = term_info[0]
                postings = term_info[1]

                PositionalInvertedIndex.write_record(f, term, doc_freq, postings)

    @staticmethod
    def read_record(
        file, binary_position: Optional[int] = None
    ) -> Optional[Tuple[str, int, PositionalPostingDict]]:
        """Reads a term and positional postings record from a binary file at a specific position."""
        if binary_position is not None:
            file.seek(binary_position)

        term_length_bytes = file.read(2)
        if not term_length_bytes:
            return None  # End of file
        term_length = struct.unpack("H", term_length_bytes)[0]
        term_bytes = file.read(term_length)
        term = term_bytes.decode("utf-8")

        doc_freq_bytes = file.read(4)
        if not doc_freq_bytes:
            return None
        num_postings_bytes = file.read(4)
        doc_freq = struct.unpack("I", doc_freq_bytes)[0]
        num_postings = struct.unpack("I", num_postings_bytes)[0]

        postings = {}
        for _ in range(num_postings):
            doc_id_bytes = file.read(4)
            if not doc_id_bytes:
                break
            doc_id = struct.unpack("I", doc_id_bytes)[0]

            pos_list_length_bytes = file.read(4)
            pos_list_length = struct.unpack("I", pos_list_length_bytes)[0]
            pos_idxs = [
                struct.unpack("I", file.read(4))[0] for _ in range(pos_list_length)
            ]

            postings[doc_id] = pos_idxs

        return term, doc_freq, postings

    @staticmethod
    def write_record(
        file,
        term: str,
        doc_freq: int,
        postings: PositionalPostingDict,
        binary_displacement: Optional[int] = None,
    ) -> None:
        """Writes a term and its positional postings to a binary file."""
        if binary_displacement is not None:
            file.seek(binary_displacement)

        term_bytes = term.encode("utf-8")
        term_length = len(term_bytes)
        if term_length > 65535:
            raise ValueError(f"Term '{term}' is too long to be encoded.")

        file.write(struct.pack("H", term_length))
        file.write(term_bytes)

        num_postings = len(postings)
        file.write(struct.pack("I", doc_freq))
        file.write(struct.pack("I", num_postings))

        for doc_id, pos_idxs in postings.items():
            file.write(struct.pack("I", doc_id))
            pos_list_length = len(pos_idxs)
            file.write(struct.pack("I", pos_list_length))
            for pos in pos_idxs:
                file.write(struct.pack("I", pos))

    @staticmethod
    def merge_partial_indices(
        partial_indices_paths: List[Union[str, Path]], output_path: Union[str, Path]
    ) -> None:
        """Merges all partial positional inverted indices into a single index."""
        files = [open(path, "rb") for path in partial_indices_paths]
        heap = []

        for file_idx, f in enumerate(files):
            record = PositionalInvertedIndex.read_record(f)
            if record:
                term, doc_freq, postings = record
                heapq.heappush(heap, (term, file_idx, doc_freq, postings))
            else:
                f.close()

        with open(output_path, "wb") as output_file:
            current_term = None
            current_doc_freq = 0
            current_postings = {}

            while heap:
                term, file_idx, doc_freq, postings = heapq.heappop(heap)

                if term != current_term and current_term is not None:
                    PositionalInvertedIndex.write_record(
                        output_file, current_term, current_doc_freq, current_postings
                    )
                    current_doc_freq = 0
                    current_postings = {}

                current_term = term
                current_doc_freq += doc_freq
                for doc_id, pos_idxs in postings.items():
                    if doc_id in current_postings:
                        current_postings[doc_id].extend(pos_idxs)
                    else:
                        current_postings[doc_id] = pos_idxs

                next_record = PositionalInvertedIndex.read_record(files[file_idx])
                if next_record:
                    next_term, next_doc_freq, next_postings = next_record
                    heapq.heappush(
                        heap, (next_term, file_idx, next_doc_freq, next_postings)
                    )
                else:
                    files[file_idx].close()

            if current_term is not None:
                PositionalInvertedIndex.write_record(
                    output_file, current_term, current_doc_freq, current_postings
                )

        for f in files:
            if not f.closed:
                f.close()

        for path in partial_indices_paths:
            os.remove(path)

    @staticmethod
    def load(path: Union[Path, str]) -> Dict[str, int]:
        """Builds a dictionary of term offsets for quick access by reading the positional index."""
        index_pointers = {}

        with open(path, "rb") as f:
            while True:
                offset = f.tell()
                record = PositionalInvertedIndex.read_record(f)
                if record is None:
                    break

                term, _, _ = record
                index_pointers[term] = offset

        return index_pointers
