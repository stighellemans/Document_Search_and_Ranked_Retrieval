import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, TypeVar, Union

import numpy as np
import pandas as pd

QueryID = TypeVar("QueryID")
DocID = TypeVar("DocID")


def tokens_to_term_freqs(tokens: Sequence[str]) -> Dict[str, int]:
    return dict(Counter(tokens))


def tokens_to_pos_idxs(tokens: Sequence[str]) -> Dict[str, List[int]]:
    pos_idxs = defaultdict(list)
    for i, term in enumerate(tokens):
        pos_idxs[term].append(i)
    return pos_idxs


def process_query_results(
    queries: Dict[QueryID, str],
    query_results: pd.DataFrame,
    doc_col_name: str = "doc_number",
) -> Dict[QueryID, List[DocID]]:
    new_query_results = {}
    for q_id in queries:
        if q_id in query_results.index:
            relevant_docs = [query_results.loc[q_id][doc_col_name]]
            # Handle multiple relevant docs scenario
            if not isinstance(relevant_docs[0], np.int64):
                relevant_docs = list(relevant_docs[0])  # If multiple relevant docs
            new_query_results[q_id] = relevant_docs
    return new_query_results


def remove_folder(dir: Union[Path, str]):
    dir = Path(dir)
    if dir.exists() and dir.is_dir():
        shutil.rmtree(dir)


def vbyte_encode(number: int) -> bytes:
    """Variable-byte encode a single integer."""
    bytes_list = []
    while True:
        bytes_list.insert(0, number & 0x7F)  # Take the last 7 bits of the number
        if number < 128:
            break
        number >>= 7  # Shift right by 7 bits to prepare for the next 7 bits
    bytes_list[-1] |= 0x80  # Set the last byte’s highest bit to 1 (continuation bit)
    return bytes(bytes_list)


def vbyte_decode(byte_data: bytes) -> int:
    """Variable-byte decode a sequence of bytes into an integer."""
    number = 0
    for byte in byte_data:
        number = (number << 7) | (byte & 0x7F)
        if byte & 0x80:
            break
    return number


def delta_encode(numbers: List[int]) -> List[int]:
    """Apply delta encoding to a list of integers."""
    return [numbers[0]] + [numbers[i] - numbers[i - 1] for i in range(1, len(numbers))]


def delta_decode(deltas: List[int]) -> List[int]:
    """Apply delta decoding to a list of deltas."""
    numbers = [deltas[0]]
    for delta in deltas[1:]:
        numbers.append(numbers[-1] + delta)
    return numbers
