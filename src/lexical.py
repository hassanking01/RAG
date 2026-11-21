import json, math, re, pickle, tqdm
import token
from collections import Counter
from pathlib import Path
from .errors import Patherror
from typing import cast, Any
from .enums import Filestatus
from functools import lru_cache
from heapq import nlargest
from tqdm import tqdm

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

    @staticmethod
    def _handel_cases(token: str) -> list[str]:
              
        result = []
        result += [token.lower()]
        if "_" in token:
            for sub_token in token.split("_"):
                result += [sub_token.lower()]
        if re.search(r"([a-z]{2,})([A-Z][a-z]+)", token):
            tokens = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token)
            for item in tokens.lower().split():
                result += [item]
        return list(filter(lambda x:x not in STOPWORDS_EN and len(x) > 1 ,result))


    @classmethod
    def tokenize(cls, save_path: Path) -> Tokens:

        @lru_cache(maxsize=1)
        def get_file_content(file_name: str) -> str:
            return Path(file_name).read_text()

        chunks: list[Any] = json.loads(save_path.read_text())

        files_history: dict[str, list[str]] = {}
        ids: dict[str, list[int]] = {}
        docs_length: dict[str, int] = {}
        vocab: dict[str, int] = {}
        counter = 0
        tf: dict[str, dict[str, int]] = {}
        idf: dict[int, float] = {}

        for chunk in tqdm(
            chunks,
            desc="indexing tokens",
            colour="#0250f7",
            ascii=" ▒"):

            file_name = chunk["file_path"]
            document_id = chunk["id"]
            files_history.setdefault(file_name, [])
            chunk_content = get_file_content(file_name)[
                chunk["first_character_index"]:
                chunk["last_character_index"]
            ]
            document = (Path(file_name).name + " " + chunk_content)
        
            to_index = Tokens.get_tokens(document)
            count = Counter(to_index)
            length = 0
            token_ids = []
            for item in count:
                if item not in vocab:
                    vocab[item] = counter
                    counter += 1

                token_id = vocab[item]
                tf.setdefault(token_id, {})
                tf[token_id][document_id] = count[item]
                length += count[item]
                token_ids += [token_id]

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
            self.docs_length.setdefault(doc, 0)
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
                self.docs_length[doc] += counter[token]
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
    def get_tokens(query: str) -> list[str]:
        result = []
        splited: list[str] = re.findall(r"\w\w+", query)
        for token in splited:
            if token.lower() in STOPWORDS_EN or token.isdigit():
                continue
            result += Tokens._handel_cases(token)
        return result

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
        self.tokens= Tokens.tokenize(self.chunks_path)
        self._calculate_avgdl()
        self.save()

    def re_index(self, rec_result: tuple[bool, dict[str, Filestatus]]):
        self.load()
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
        self._calculate_avgdl()
        self.save()

    def save(self):
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        if self.save_path.is_file():
            raise Patherror("the save dir for lexical is dir")
        with open(self.save_path / "bm52.pkl","wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> None:
        with open(self.save_path / "bm52.pkl", "rb") as f:
            bm25 = pickle.load(f)
        bm25 = cast(BM25, bm25)
        self.tokens = bm25.tokens
        self.avgdl = bm25.avgdl
        self.loaded = True

    @lru_cache
    def _calculate_term_score(self, token_id: int, doc: str) -> float:
        score = 0
        D = self.tokens.docs_length[doc]
        tf_qi = self.tokens.tf[token_id].get(doc)
        bast = tf_qi * (self.k1 + 1)
        maqam = tf_qi + self.k1 * (1 - self.b + self.b * (D / self.avgdl))
        score = self.tokens.idf[token_id] * (bast / maqam)
        return score

    def _calculate_avgdl(self) -> None:
        doc_lengths = self.tokens.docs_length
        self.avgdl = sum([doc_lengths[doc] for doc in doc_lengths ]) / len(self.tokens.ids)

    @lru_cache
    def get_chunks_index(self) -> list[dict[str, Any]]:
        chunks = json.loads(self.chunks_path.read_text())
        return {chunk["id"]:index for index, chunk in enumerate(chunks)}

    def search(self, query: str, k) -> list[int]:
        tokens = Tokens.get_tokens(query)
        chunks_index = self.get_chunks_index()
        scores = {}
        for token in tokens:
            token_id = self.tokens.vocab.get(token, -1)
            if token_id == -1:
                continue
            for doc in self.tokens.tf[token_id]:
                scores.setdefault(doc, 0.0)
                scores[doc] += self._calculate_term_score(token_id, doc)
        scores = nlargest(k, scores, key=lambda x: scores[x])
        return [chunks_index[idx] for idx in scores]
