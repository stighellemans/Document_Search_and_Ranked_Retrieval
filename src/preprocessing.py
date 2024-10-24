import re
from pathlib import Path
from typing import List, Union

import spacy

# load the English model from spacy for lemmatization and stopwords
# disable for performance optimizations
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


def read(path: Union[str, Path]) -> str:
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"The file {path} does not exist.")


def tokenize(text: str) -> List[str]:
    text = text.lower()
    # adjusted regex to handle both words and numbers (decimals and exponents included)
    tokens = re.findall(r"\b\d+(?:\.\d+)?(?:\^[-+]?\d+)?|\b\w+\b", text)

    return tokens


def preprocess(text: str) -> List[str]:
    global nlp
    doc = nlp(text.lower())
    # tokenize, remove stopwords, lemmatize, and retain numbers
    filtered_tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and (token.is_alpha or token.like_num)
    ]
    return filtered_tokens
