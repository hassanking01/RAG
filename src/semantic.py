import json
from pathlib import Path
import numpy
from sentence_transformers import SentenceTransformer
from typing import Any
from .enums import Filestatus
from functools import lru_cache
import pickle
from .errors import Patherror
from typing import cast
import os

class VectorDb:
    def __init__(
        self,
        processed_path: Path,
        chunks_path: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the semantic vector database."""
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.chunks_path = chunks_path
        self.save_path = processed_path / "semantic"
        self._is_loaded = False
        self.metadata = []
        self.files_history: dict[str, list[str]] = {}

    @lru_cache(maxsize=1)
    def get_file_content(self, file: str):
        """Read and return a file's text."""
        return Path(file).read_text()

    def index(self):
        """Create and save embeddings for all chunks."""

        chunks: list[dict[str, Any]] = json.loads(self.chunks_path.read_text())
        documents = []
        for data in chunks:
            file_content = self.get_file_content(data["file_path"])
            documents += [
                file_content[
                    data["first_character_index"]:
                    data["last_character_index"]
                ]
            ]
            self.metadata += [
                [data["id"],data["file_path"]] 
            ]
            self.files_history.setdefault(data["file_path"], [])
            self.files_history[data["file_path"]] += [data["id"]]
        self._embeddings = self.model.encode(documents, show_progress_bar=True)
        self.save()

    def check_save_files(self):
        """Check that semantic index files can be saved."""
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        if self.save_path.is_file():
            raise Patherror(
                "the save directory for semantic"
                " is alreay existing file"
                )
        if (self.save_path / "metadata.pkl").exists():
            if not os.access(self.save_path / "metadata.pkl" , os.R_OK | os.W_OK):
                raise Patherror(
                    f"the save file for metadata [{self.save_path / "metadata.pkl"}]"
                    " does not have Read and Write"
                    " permissions delete it and try again"
                    )
        if (self.save_path / "embeddings.npy").exists():
            if not os.access(self.save_path / "embeddings.npy" , os.R_OK | os.W_OK):
                raise Patherror(
                    f"the save file for embeddings [{self.save_path / "embeddings.npy"}]"
                    " does not have Read and Write"
                    " permissions delete it and try again"
                    )

    def save(self):
        """Save semantic metadata and embeddings to disk."""
        self.check_save_files()
        with open(self.save_path / "metadata.pkl","wb") as f:
                pickle.dump((self.files_history, self.metadata), f, protocol=pickle.HIGHEST_PROTOCOL)
        numpy.save(self.save_path / "embeddings.npy", self._embeddings)

    def load(self) -> None:
        """Load semantic metadata and embeddings from disk."""
        with open(self.save_path / "metadata.pkl", "rb") as f:
            t_metadata = pickle.load(f)
        t_metadata = cast(tuple[dict[str, list[str]], list[list[str]]], t_metadata)
        self.files_history, self.metadata = t_metadata
        self._embeddings = numpy.load(self.save_path / "embeddings.npy")
        self._is_loaded = True

    @lru_cache
    def _get_chunks(self, file="") -> list[dict[str, Any]]:
        """Return saved chunks, optionally filtered by file."""
        chunks = json.loads(self.chunks_path.read_text())
        if not file:
            return { chunk["id"]:chunk for chunk in chunks}
        else:
            return { chunk["id"]:chunk for chunk in chunks if chunk["file_path"] == file}

    def _get_chunk_data(self, chunk_id: str):
        """Return chunk data for a chunk ID."""
        chunks = self._get_chunks()
        return chunks[chunk_id]

    def _get_modified(self, file) -> tuple[list, list]:
        """Return added and removed chunks for a modified file."""
        this_file_chunks = self._get_chunks(file)        
        old_docs = set(self.files_history[file])
        new_docs = set(chunk_id for chunk_id in this_file_chunks)
        to_remove = old_docs - new_docs
        to_add = new_docs - old_docs
        return list(to_add) , list(to_remove)
    
    def _get_added(self, file) -> list[str]:
        """Return chunk IDs belonging to a new file."""
        this_file_chunks = self._get_chunks(file)        
        return [chunk_id for chunk_id in this_file_chunks]

    def re_index(self, rec_result: tuple[bool, dict[str, Filestatus]]) :
        """Update embeddings after files change."""
        self.load()
        _ , to_reindex = rec_result
        to_remove = []
        to_add = []
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
                    self.files_history[file] += added
                    to_remove += removed
                    to_add += added
                case Filestatus.ADDED:
                    added = self._get_added(file)
                    to_add += added.copy()
                    self.files_history[file] = added
        
        to_keep = [ 
            index
            for index , data in enumerate(self.metadata)
            if data[0] not in to_remove
        ]
        self._embeddings = self._embeddings[to_keep]

        cleaned_metadata = []
        for index in to_keep:
            cleaned_metadata += [self.metadata[index]]
        self.metadata = cleaned_metadata
        if len(to_add) > 0:
            new_chunks = [
                self._get_chunk_data(chunk_id)
                for chunk_id in to_add
            ]
            new_metadata = []
            new_docs = []
            for chunk in new_chunks:
                new_metadata += [[chunk["id"], chunk["file_path"]]]
                new_docs += [
                    Path(chunk["file_path"]).read_text()[
                        chunk["first_character_index"]:
                        chunk["last_character_index"]
                    ]
                    ]
            self.metadata.extend(new_metadata)
            new_embiddings = self.model.encode(new_docs, show_progress_bar=True)
            result = numpy.vstack((self._embeddings, new_embiddings))
            self._embeddings = result
        self.save()

    def calculate_cos_sim(self, vector_a, vector_q) -> float:
        """Calculate cosine similarity between two vectors."""
        mag_a = numpy.linalg.norm(vector_a)
        mag_q = numpy.linalg.norm(vector_q)
        if not mag_a or not mag_q:
            return 0.0
        bast = numpy.dot(vector_a, vector_q)
        maqam = mag_a * mag_q
        return float(bast / maqam)

    def search(self, query: str, k: int) -> list[int]:
        """Return positions of the most similar chunks."""
        query_embedding = self.model.encode(query)
        if not self._is_loaded:
            self._embeddings = numpy.load(self.save_path)
            self._is_loaded = True
        scores = []
        for index, vector in enumerate(self._embeddings):
            scores += [(self.calculate_cos_sim(vector, query_embedding), index)]
        scores = sorted(scores, key=lambda x: x[0], reverse=True)
        scores = scores[:k]
        return [i[1] for i in scores]
