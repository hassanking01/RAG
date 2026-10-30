import json
import math
import re
from collections import Counter
from pathlib import Path
from src.models import MinimalSource
import pickle
from .errors import Patherror
from typing import cast
from .enums import Filestatus
from rich import print
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
    def __init__(self, ids, vocab, tf, idf, docs_length, counter, files_history):
        self.ids: dict[str, list[int]] = ids
        self.docs_length: dict[str, int] = docs_length
        self.vocab: dict[str, int] = vocab
        self.tf: dict[int, dict[str, int]] = tf
        self.idf: dict[int, float] = idf
        self.counter = counter
        self.files_history: dict[str, list[str]] = files_history

    @classmethod
    def tokenize(cls, save_path: Path) -> Tokens:
        from typing import Any
        chunks: list[Any] = json.loads(save_path.read_text())
        documents = [
            Path(chunk["file_path"]).read_text()[
                chunk["first_character_index"] : chunk["last_character_index"]
            ]
            for chunk in chunks
        ]
        files_history: dict[str, list[str]] = {}
        ids: dict[str, list[int]] = {}
        docs_length: dict[str, int] = {}
        vocab: dict[str, int] = {}
        counter = 0
        tf: dict[str, dict[str, int]] = {}
        idf: dict[int, float] = {}

        for index, document in enumerate(documents):
            file_name = Path(chunks[index]["file_path"]).name
            document = (file_name + " " + document)
            splited = re.findall(r"(?u)\b\w\w+\b", document)
            splited = [part for part in splited if part not in STOPWORDS_EN]
            to_index = []
            #TODO add snake_case and camelCase logic
            for token in splited:
                to_index += []
            count = Counter(splited)
            length = 0
            token_ids = []
            document_id = chunks[index]["id"]
            for item in count:
                token_id = -1
                if item not in vocab:
                    token_id = counter
                    vocab[item] = token_id
                    counter += 1
                else:
                    token_id = vocab[item]
                tf.setdefault(token_id, {})
                tf[token_id][document_id] = count[item]
                length += count[item]
                token_ids += [token_id]
            file_name = chunks[index]["file_path"]
            files_history.setdefault(file_name, [])
            ids[document_id] = token_ids
            docs_length[document_id] = length
            files_history[file_name] += [document_id]
        N = len(ids)
        for _ , token_id in vocab.items():
            df = len(tf[token_id])
            idf[token_id] = math.log(1 + (N - df + 0.5) / (df + 0.5))
        return cls(ids, vocab, tf, idf, docs_length, counter, files_history)

    def delete_docs(self, file: str) -> set[int]:
        docs = self.files_history.pop(file, None)
        if not docs:
            return
        affected_tokens = set()
        for doc in docs:
            for token_id in self.ids[doc]:
                _ = self.tf[token_id].pop(doc, None)
                if len(self.tf[token_id]) == 0:
                    _ = self.tf.pop(token_id, None)
                    _ = self.idf.pop(token_id)
                else:
                    affected_tokens.add(token_id)
    def modify_tokens(self, file: str, chunks_path: Path) -> set[int]:
        chunks = json.loads(chunks_path.read_text())
        this_file_chunks = { chunk["id"]:chunk for chunk in chunks if chunk["file_path"] == file}        
        old_docs = set(self.files_history[file])
        new_docs = set(chunk_id for chunk_id in this_file_chunks)
        to_skip = old_docs & new_docs
        to_remove = old_docs - new_docs
        affected_tokens = set()
        to_add = new_docs - old_docs
        for doc in old_docs:
            if doc in to_skip:
                continue
            if doc in to_remove:
                for token_id in self.ids[doc]:
                    _ = self.tf[token_id].pop(doc, None)
                    if len(self.tf[token_id]) == 0:
                        _ = self.tf.pop(token_id, None)
                        _ = self.idf.pop(token_id, None)
                    else:
                        affected_tokens.add(token_id)
                self.files_history[file].remove(doc)
        for doc in to_add:
            self.files_history[file] += [doc]
            first_index = this_file_chunks[doc]["first_character_index"]
            last_index = this_file_chunks[doc]["last_character_index"]
            tokens = Tokens.get_tokens(Path(file).read_text()[first_index:last_index])
            counter = Counter(tokens)
            self.ids.setdefault(doc, [])
            for token in counter:
                if token in self.vocab:
                    token_id = self.vocab[token]
                else:
                    self.vocab[token] = self.counter
                    self.counter += 1
                    token_id = self.vocab[token]
                affected_tokens.add(token_id)
                self.tf.setdefault(token_id, {})
                self.tf[token_id][doc] = counter[token]
                self.ids[doc] += [token_id]
            self.files_history[file] += [doc]
        return affected_tokens
                
    def add_tokens(self, file: str, chunks_path: Path) -> None:
        chunks = json.loads(chunks_path.read_text())
        this_file_chunks = {chunk["id"]:chunk for chunk in chunks if chunk["file_path"] == file}
        affected_tokens = set()
        for doc in this_file_chunks:
            self.files_history[file] += [doc]
            first_index = this_file_chunks[doc]["first_character_index"]
            last_index = this_file_chunks[doc]["last_character_index"]
            tokens = Tokens.get_tokens(Path(file).read_text()[first_index:last_index])
            self.ids.setdefault(doc, [])
            counter = Counter(tokens)
            for token in counter:
                if token in self.vocab:
                    token_id = self.vocab[token]
                else:
                    self.vocab[token] = self.counter
                    self.counter += 1
                    token_id = self.vocab[token]
                affected_tokens.add(token_id)
                self.tf.setdefault(token_id, {})
                self.tf[token_id][doc] = counter[token]
                self.ids[doc] += [token_id]
            self.files_history[file] += [doc]
        return affected_tokens

    def recalculat_idf(self, affected_tokens: set[int], is_full_idf: bool) -> None:
        N = len(self.ids)
        if is_full_idf:
            for _ , token_id in self.vocab.items():
                df = len(self.tf[token_id])
                self.idf[token_id] = math.log(1 + (N - df + 0.5) / (df + 0.5))
        else:
            for token_id in affected_tokens:
                df = len(self.tf[token_id])
                self.idf[token_id] = math.log(1 + (N - df + 0.5) / (df + 0.5))

    @staticmethod
    def get_tokens(query: str) -> list[int]:
        query = query.lower()
        splited = re.findall(r"(?u)\b\w\w+\b", query)
        splited = [part for part in splited if part not in STOPWORDS_EN]
        return splited

class BM25:
    def __init__(
        self,
        processed_path: Path,
        chunks_path: Path,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.save_path = processed_path / "lexical"
        self.chunks_path = chunks_path
        self.k1 = k1
        self.b = b
        self.loaded = False
        self.avgdl: float = 0.0
    def index(self) -> None:
        self.tokens: Tokens = Tokens.tokenize(self.chunks_path)
        self._calculate_avgdl()
        self.save()

    def re_index(self, rec_result: tuple[bool, dict[str, Filestatus]]):
        affected_tokens = set()
        full_idf , edited = rec_result
        for file, state in edited.items():
            match state:
                case Filestatus.DELETED:
                    affected_tokens.union(self.tokens.delete_docs(file))
                case Filestatus.MODIFIED:
                    affected_tokens.union(self.tokens.modify_tokens(file, self.chunks_path))
                case Filestatus.ADDED:
                    affected_tokens.union(self.tokens.add_tokens(file, self.chunks_path))
        self.tokens.recalculat_idf(affected_tokens=affected_tokens, is_full_idf=full_idf)
        if full_idf:
            self._calculate_avgdl()
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
    def _calculate_term_score(self, token_id: int, doc: str) -> float:
        score = 0
        D = self.tokens.docs_length[doc]
        tf_qi = self.tokens.tf[token_id].get(doc, 0)
        bast = tf_qi * (self.k1 + 1)
        maqam = tf_qi + self.k1 * (1 - self.b + (self.b * (D / self.avgdl)))
        score = self.tokens.idf[token_id] * (bast / maqam)
        return score
    def _calculate_avgdl(self) -> None:
        doc_lengths = self.tokens.docs_length
        self.avgdl = sum([doc_lengths[doc] for doc in doc_lengths ]) / len(self.tokens.ids)

    def search(self, query: str, k):
        tokens = Tokens.get_tokens(query)
        chunks = json.loads(self.chunks_path.read_text())
        chunks_index = {chunk["id"]:index for index, chunk in enumerate(chunks)}
        scores = []
        for doc in self.tokens.ids:
            score = 0
            for token in tokens:
                token_id = self.tokens.vocab.get(token, -1)
                if token_id == -1:
                    continue
                if token_id not in self.tokens.ids[doc]:
                    continue
                score += self._calculate_term_score(token_id, doc)
            scores += [[doc, score]]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[:k]
        return [chunks_index[idx] for idx, _ in scores]
