import re
from pathlib import Path
from typing import List, Union

import spacy

# load the English model from spacy for lemmatization and stopwords
# disable for performance optimizations
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.max_length = 10000000


def read(path: Union[str, Path]) -> str:
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"The file {path} does not exist.")


def tokenize(text: str) -> List[str]:
    text = text.lower()

    # only letters words
    tokens = re.findall(r"\b[a-zA-Z]+\b", text)

    return tokens


def preprocess(text: str) -> List[str]:
    global nlp
    clean_text = re.sub(r"[^a-zA-Z]+", " ", text)
    doc = nlp(clean_text.lower())
    # tokenize, remove stopwords, lemmatize
    filtered_tokens = [
        token.lemma_ for token in doc if not token.is_stop and token.is_alpha
    ]
    return filtered_tokens
