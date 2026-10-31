import json
from pathlib import Path
import numpy
from sentence_transformers import SentenceTransformer
from typing import Any
from .enums import Filestatus
from functools import lru_cache
class VectorDb:
    def __init__(
        self,
        processed_path: Path,
        chunks_path: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.chunks_path = chunks_path
        self.save_path = processed_path / "semantic/embeddings.npy"
        self._is_loaded = False
        self.metadata = []
        self.files_history: dict[str, list[str]] = {}
    def index(self):

        chunks: list[dict[str, Any]] = [
            data
            for data in json.loads(self.chunks_path.read_text())
        ]
        documents = []
        for data in chunks:
            documents += [
                Path(data["file_path"]).read_text()[
                    data["first_character_index"] : data["last_character_index"]
                ]
            ]
            self.metadata += [
                [data["id"],data["file_path"]] 
            ]
            self.files_history.setdefault(data["file_path"], [])
            self.files_history[data["file_path"]] += [data["id"]]
        result = self.model.encode(documents, show_progress_bar=True)
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True)
        numpy.save(self.save_path, result)

    @lru_cache
    def _get_chunks(self, file="") -> list[dict[str, Any]]:
        chunks = json.loads(self.chunks_path.read_text())
        if not file:
            return { chunk["id"]:chunk for chunk in chunks}
        else:
            return { chunk["id"]:chunk for chunk in chunks if chunk["file_path"] == file}

    def _get_chunk_data(self, chunk_id: str):
        chunks = self._get_chunks()
        return chunks[chunk_id]

    def _get_modified(self, file) -> tuple[list, list]:
        this_file_chunks = self._get_chunks(file)        
        old_docs = set(self.files_history[file])
        new_docs = set(chunk_id for chunk_id in this_file_chunks)
        to_remove = old_docs - new_docs
        to_add = new_docs - old_docs
        return list(to_add) , list(to_remove)
    
    def _get_added(self, file) -> list[str]:
        this_file_chunks = self._get_chunks(file)        
        return [chunk_id for chunk_id in this_file_chunks]

    def re_index(self, rec_result: tuple[bool, dict[str, Filestatus]]) :
        _ , to_reindex = rec_result
        to_remove = []
        to_added = []
        for file, status in to_reindex.items():
            match status:
                case Filestatus.DELETED:
                    to_remove += self.files_history[file]
                    _ = self.files_history.pop(file, None)
                case Filestatus.MODIFIED:
                    added, removed = self._get_modified( file)
                    for doc in set(self.files_history[file]):
                        if doc in removed:
                            self.files_history[file].remove(doc)
                    to_remove += removed
                    to_added += added
                case Filestatus.ADDED:
                    added = self._get_added(file)
                    to_added += added.copy()
                    self.files_history[file] = added
        
        to_keep = [ 
            index
            for index , data in enumerate(self.metadata)
            if data[0] not in to_remove
        ]
        self._embeddings = self._embeddings[to_keep]
        new_docs = []
        for chunk_id in to_added:
            chunk = self._get_chunk_data(chunk_id)
            new_docs += [
                Path(chunk["file_path"]).read_text()[
                    chunk["first_character_index"] : chunk["last_character_index"]
                ]
            ]

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
