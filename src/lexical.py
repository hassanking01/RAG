from annotated_types import doc

import bm25s, json
from pathlib import Path
from models import MinimalSource
from collections import Counter
class Tokens:
    def __init__(self, ids, vocab):
        ids: dict[int, list] = {}
        vocab: dict[str, int] = {}
    @staticmethod
    def tokenize(documents: str | list[str]) -> Tokens:
        vocab: dict[str, int] = {}
        ids: dict[int, list] = {}
        vocab_ids: dict[str, int] = {}
        counter = 0
        if isinstance(documents, list):
            for index, document in enumerate(documents):
                document = document.lower()
                splited = document.split()
                count = Counter(splited)
                document_ids = []
                for item in count:
                    vocab.setdefault(item, 0)
                    vocab[item] += count[item]
                    if not vocab_ids.get(item):
                        vocab_ids[item] = counter
                        counter += 1
                    document_ids += [vocab_ids[item]]
                ids[index] = document_ids
        with open("r.json" , "w") as file:
            json.dump(
                vocab_ids,
                file,
                indent=4
            )
if __name__ == "__main__":
    chunks: list[MinimalSource] = [MinimalSource(**data) for data in json.loads(Path("data/processed/chunks/chunks.json").read_text())]
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
 
