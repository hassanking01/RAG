import bm25s, json
from pathlib import Path
from .objects import MinimalSource
class BM25:
    def __init__(
            self,
            processed_path: Path,
        ):
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

