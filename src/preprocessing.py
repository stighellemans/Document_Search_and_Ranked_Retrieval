from pathlib import Path
from typing import List, Union
import re
import nltk

# Initialize stemmer and lemmatizer
stemmer = nltk.PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()

# Load stopwords from nltk
stop_words = nltk.corpus.stopwords.words("english")


def read(path: Union[str, Path]) -> str:
    file_path = Path(path)
    try:
        with file_path.open('r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"The file {path} does not exist.")

def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def preprocess(text: str, use_stemming: bool = False, use_lemmatization: bool = True) -> List[str]:
    # tokenize the text using the tokenize function
    tokens = tokenize(text)

    # remove stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # use stemming or lemmatization based on the parameter supplied by the user
    if use_stemming:
        filtered_tokens = [stemmer.stem(word) for word in filtered_tokens]
    elif use_lemmatization:
        filtered_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return filtered_tokens