import json
from pathlib import Path
import numpy
from sentence_transformers import SentenceTransformer

from .models import MinimalSource


class VectorDb:
    def __init__(
        self,
        processed_path: Path,
        chunks_path: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.chunks_path = chunks_path
        self.save_path = processed_path / "semantic/embeddings.npy"
        self._is_loaded = False
    def index(self):

        chunks: list[MinimalSource] = [
            MinimalSource(**data) for data in json.loads(self.chunks_path.read_text())
        ]
        documents = [
            Path(data.file_path).read_text()[
                data.first_character_index : data.last_character_index
            ]
            for data in chunks
        ]
        result = self.model.encode(documents, show_progress_bar=True)
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True)
        numpy.save(self.save_path, result)


    def search(self, query: str, k: int) -> list[int]:
        query_embedding = self.model.encode(query, show_progress_bar=True)
        if not self._is_loaded:
            self._embeddings = numpy.load(self.save_path)
            self._is_loaded = True
        scores = []

        for index, vector in enumerate(self._embeddings):
            score = numpy.sum((query_embedding - vector) ** 2)
            scores += [(score, index)]
        scores = sorted(scores, key=lambda x: x[0])

        scores = scores[:k]

        return [i[1] for i in scores]
