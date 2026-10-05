from annotated_types import doc

import bm25s, json
from pathlib import Path
from models import MinimalSource
from collections import Counter
import re
class Tokens:
    def __init__(self, ids, vocab, vocab_ids, tf, idf):
        self.ids: dict[int, list] = ids
        self.vocab: dict[str, int] = vocab
        self.vocab_ids: dict[int, str] = vocab_ids
        self.tf: dict[str, dict[int, int]]= tf
        self.idf: dict[str, float] = idf
    @staticmethod
    def tokenize(documents: str | list[str]) -> Tokens:
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
                splited = [part for  part in splited if part not in bm25s.stopwords.STOPWORDS_EN]
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
            import math
            N = len(vocab)
            for term in vocab:
                df = len(tf[term])
                idf[term] = math.log(1 + (N - df + 0.5) / (df + 0.5))
            return Tokens(ids, vocab, vocab_ids,tf, idf)

if __name__ == "__main__":
    chunks: list[MinimalSource] = [MinimalSource(**data) for data in json.loads(Path("data/processed/chunks/chunks.json").read_text())]
    chunks = [chunk for chunk in chunks if chunk.file_path.endswith(".py")]
    chunks = [
        Path(data.file_path).read_text()[data.first_character_index:data.last_character_index]
        for data in chunks
    ]
    # with open("bm25.json", "w") as file:
    #     json.dump(
    #         bm25s.tokenize(chunks).vocab,
    #         file,
    #         indent=4
    #     )
    Tokens.tokenize(chunks)
    exit()


































































class BM25:
    def __init__(
            self,
            processed_path: Path,
            chunks_path: Path
        ):
        self.chunks_path = chunks_path
        self.save_path = processed_path / "lexical"
        self.retriever = bm25s.BM25()
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
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.retriever.save(self.save_path)
    def search(self, query: str, k: int):
        print(bm25s.tokenize(query))
        exit()
        self.retriever = bm25s.BM25.load(self.save_path)
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
 
