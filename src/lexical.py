import json
from pathlib import Path
from collections import Counter
import re
from models import MinimalSource







STOPWORDS_EN = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with']

import bm25s
class BM25:
    def __init__(
            self,
            processed_path: Path = Path("data/processed"),
            chunks_path: Path = Path("data/processed/chunks/chunks.json")
        ):
        self.chunks_path = chunks_path
        self.save_path = processed_path / "lexical"
        self.retriever =  bm25s.BM25()
    def index(self, chunks_path: Path):
        chunks: list[MinimalSource] = [
            MinimalSource(**data) for data in json.loads(chunks_path.read_text())
        ]
        contents = [
            Path(data.file_path).read_text()
            [data.first_character_index:data.last_character_index]
            for data in chunks
        ]
        corpus_tokens = bm25s.tokenize(contents)
        self.retriever.index(corpus_tokens)
        print(self.retriever.idf_method)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.retriever.save(self.save_path)
    def search(self, query: str, k: int):
        self.retriever = bm25s.BM25.load(self.save_path)
        print(dir(self.retriever))
        return
        chunks: list[MinimalSource] = [
            MinimalSource(**data)
            for data in json.loads(self.chunks_path.read_text())
        ]
        results, scors = self.retriever.retrieve(
            bm25s.tokenize(query),
            k=k,
        )
        for result in results:
            print(result)

























class Tokens:
    def __init__(self, ids, vocab, vocab_ids, tf, idf):
        self.ids: dict[int, list] = ids
        self.vocab: dict[str, int] = vocab
        self.vocab_ids: dict[int, str] = vocab_ids
        self.tf: dict[str, dict[int, int]]= tf
        self.idf: dict[str, float] = idf

    @classmethod
    def tokenize(cls, documents: str | list[str]) :
        import math
        vocab: dict[str, int] = {}
        ids: dict[int, list] = {}
        vocab_ids: dict[int, str] = {}
        counter = 0
        tf = {}
        idf = {}
        if isinstance(documents, list):
            for index, document in enumerate(documents):
                document = document.lower()
                splited = re.findall(r"(?u)\b\w\w+\b", document)
                splited = [part for  part in splited if part not in STOPWORDS_EN]
                count = Counter(splited)
                document_ids = []
                for item in count:
                    if not vocab.get(item):
                        vocab_ids[item] = counter
                        counter += 1
                    tf.setdefault(item, {})
                    tf[item][index] = count[item]
                    vocab.setdefault(item, 0)
                    vocab[item] += count[item]
                    document_ids += [vocab_ids[item]]
                ids[index] = document_ids
            N = len(ids)
            for term in vocab:
                df = len(tf[term])
                idf[term] = math.log(1 + (N - df + 0.5) / (df + 0.5))
            return cls(ids, vocab, vocab_ids,tf, idf)
    @staticmethod
    def get_tokens(query: str) -> list[str]:
        splited = re.findall(r"(?u)\b\w\w+\b", query)
        splited = [part for  part in splited if part not in STOPWORDS_EN]
        return splited

class MYBM25:
    def __init__(self, tokens: Tokens, chunks: list[MinimalSource]):
        self.tokens = tokens
        self.chunks = chunks
        self.index = {}

    def search(self, query: str, k):
        print(self.tokens.idf)
        exit()
        k1 = 1.5
        b = 0.75
        query = Tokens.get_tokens(query=query)
        scores = []
        AVGDL = sum(len(self.tokens.ids[doc]) for doc in self.tokens.ids) / len(self.tokens.ids)
        for index in self.tokens.ids:
            score = []
            for token in query:
                if token not in self.tokens.vocab :
                    continue
                tf_qi = self.tokens.tf[token].get(index, 0)
                bast = (tf_qi * (k1 + 1))
                D = len(self.tokens.ids[index])
                maqam = tf_qi + k1 * (1 - b + b * (D / AVGDL)) 
                score += [self.tokens.idf[token] * (bast / maqam)]
            scores += [[index, sum(score)]]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[:k]
        return scores                



if __name__ == "__main__":
    chunks: list[MinimalSource] = [ MinimalSource(**data) for data in json.loads(Path("data/processed/chunks/chunks.json").read_text())]
    tokens = Tokens.tokenize([Path(file.file_path).read_text()[file.first_character_index:file.last_character_index] for file in chunks])
    bm_ = BM25()
    bm = MYBM25(tokens, chunks)
    question = "What determines the values in cudagraph_inputs_embeds when capturing CUDA graph shapes in vLLM's ModelRunner?"
    bm_.search(question, 1)
    # print("-" * 100)
    # bm.search(question, 1)













