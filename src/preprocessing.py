import re
from pathlib import Path
from typing import List, Union


def read(path: Union[str, Path]) -> str:
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"The file {path} does not exist.")


def tokenize(text: str) -> List[str]:
    # lowercase the text for case-insensitivity
    text = text.lower()

    # remove punctuation and split by whitespace
    tokens = re.findall(r"\b\w+\b", text)

    return tokens
    ##raise NotImplementedError()


# stuff that can be added here: dealing with numbers or lemmatization
# however, this will serve as a basic template for reading the files and tokenization
