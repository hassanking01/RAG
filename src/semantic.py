from sentence_transformers import SentenceTransformer
from pathlib import Path
from .objects import MinimalSource
import numpy, json
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
    def index(
            self
        ):

        chunks: list[MinimalSource] = [
            MinimalSource(**data)
            for data in json.loads(self.chunks_path.read_text())
        ]
        data: list[MinimalSource] = chunks.copy()
        import csv
        with open("metadata.tsv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            
            writer.writerow(["file", "start", "end"])
            writer.writerows([[f"{Path(chunk.file_path).name}[{chunk.first_character_index}:{chunk.last_character_index}]", chunk.first_character_index, chunk.last_character_index] for chunk in data])   
        exit()
        chunks = [
            Path(data.file_path).read_text()
            [data.first_character_index:data.last_character_index]
            for data in chunks
        ]
        result = self.model.encode(chunks, show_progress_bar=True)
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True)
        numpy.save(self.save_path, result)
        numpy.savetxt(self.save_path.parent / "embeddings.tsv", result, delimiter="\t", fmt="%.6f")
    def search(self,query:str,  k: int):
        query_embedding = self.model.encode(query, show_progress_bar=True)
        chunks: list[MinimalSource] = json.loads(self.chunks_path.read_text())
        embeddings = numpy.load(self.save_path)
        embeddings = numpy.vstack((embeddings, query_embedding))
        numpy.savetxt(self.save_path.parent / "embeddings.tsv", embeddings, delimiter="\t", fmt="%.6f")
        import csv
        with open("metadata.tsv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query", 0, 0])
        exit()
        scores = []
        for index, vector in enumerate(embeddings):
            score = numpy.sum((query_embedding - vector) ** 2)
            scores += [(score, index)]
        scores = sorted(scores, key=lambda x:x[0])
        
        scores = scores[:k]
        
        chunks = [chunks[i[1]] for i in scores]
