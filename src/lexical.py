import json
import math
import re
from collections import Counter
from pathlib import Path
from src.models import MinimalSource
import pickle
from .models import MinimalSource
from .errors import Patherror
from typing import cast

STOPWORDS_EN = [
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
]

class Tokens:
    def __init__(self, ids, vocab, tf, idf, docs_length):
        self.ids: list[list[int]] = ids
        self.docs_length: list[int] = docs_length
        self.vocab: dict[str, int] = vocab
        self.tf: dict[int, dict[int, int]] = tf
        self.idf: list[float] = idf

    @classmethod
    def tokenize(cls, save_path: Path) -> Tokens:
        chunks: list[MinimalSource] = [
            MinimalSource(**data) for data in json.loads(save_path.read_text())
        ]
        documents = [
            Path(chunk.file_path).read_text()[
                chunk.first_character_index : chunk.last_character_index
            ]
            for chunk in chunks
        ]

        vocab: dict[str, int] = {}
        ids: list[list[int]] = []
        docs_length: list[int] = []
        counter = 0
        tf = {}
        idf: list[float] = []

        for index, document in enumerate(documents):
            document = document.lower()
            splited = re.findall(r"(?u)\b\w\w+\b", document)
            splited = [part for part in splited if part not in STOPWORDS_EN]
            count = Counter(splited)
            length = 0
            document_ids = []
            for item in count:
                token_id = -1
                if item not in vocab:
                    token_id = counter
                    vocab[item] = token_id
                    counter += 1
                else:
                    token_id = vocab[item]
                tf.setdefault(token_id, {})
                tf[token_id][index] = count[item]
                length += count[item]
                document_ids += [token_id]
            ids.append(document_ids)
            docs_length.append(length)
        N = len(ids)
        for _ , token_id in vocab.items():
            df = len(tf[token_id])
            idf.append(math.log(1 + (N - df + 0.5) / (df + 0.5)))
        return cls(ids, vocab, tf, idf, docs_length)

    @staticmethod
    def get_tokens(query: str, tokenz: Tokens) -> list[int]:
        query = query.lower()
        splited = re.findall(r"(?u)\b\w\w+\b", query)
        splited = [part for part in splited if part not in STOPWORDS_EN]
        return [tokenz.vocab[term] for term in splited if term in tokenz.vocab]


class BM25:
    def __init__(
        self,
        processed_path: Path,
        chunks_path: Path,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self._index = {}
        self.save_path = processed_path / "lexical"
        self.chunks_path = chunks_path
        self.k1 = k1
        self.b = b
        self.loaded = False

    def index(self) -> None:
        self.tokens = Tokens.tokenize(self.chunks_path)
        self.avgdl = self._calculate_avgdl()      
        for index, doc in enumerate(self.tokens.ids):
            self._index.setdefault(index, {})
            for token_id in doc:
                score = self._calculate_term_score(token_id, index)
                self._index[index][token_id] = score
        self.save()

    def save(self):
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        if self.save_path.is_file():
            raise Patherror("the save dir for lexical is dir")
        with open(self.save_path / "bm52.pkl","wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    def load(self) -> BM25:
        with open(self.save_path / "bm52.pkl", "rb") as f:
            bm25 = pickle.load(f)
        bm25 = cast(BM25, bm25)
        bm25.loaded = True
        return bm25
    def _calculate_term_score(self, token_id: int, doc: int) -> float:
        score = 0
        D = self.tokens.docs_length[doc]
        tf_qi = self.tokens.tf[token_id].get(doc, 0)
        bast = tf_qi * (self.k1 + 1)
        maqam = tf_qi + self.k1 * (1 - self.b + (self.b * (D / self.avgdl)))
        score = self.tokens.idf[token_id] * (bast / maqam)
        return score
    def _calculate_avgdl(self) -> float:
        return sum(self.tokens.docs_length) / len(self.tokens.ids)

    def search(self, query: str, k):
        tokens = Tokens.get_tokens(query, self.tokens)
        scores = []
        for doc in self._index:
            score = 0
            for token_id in tokens:
                if token_id not in self._index[doc]:
                    continue
                score += self._index[doc][token_id]
            scores += [[doc, score]]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[:k]
        return [idx for idx, _ in scores]
