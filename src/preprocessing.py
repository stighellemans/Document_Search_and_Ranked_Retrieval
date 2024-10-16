from pathlib import Path
from typing import List, Union


def read(path: Union[str, Path]) -> str:
    raise NotImplementedError()


def tokenize(text: str) -> List[str]:
    raise NotImplementedError()
