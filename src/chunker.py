import json
from pathlib import Path
from rich import print
import tqdm
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

import hashlib
from .enums import Filestatus

class Chunker:
    def __init__(
        self,
        processed_path: Path,
    ):
        self.save_path = processed_path / "chunks/chunks.json"
        self.file_map = {
            "py": Language.PYTHON,
            "txt": Language.MARKDOWN,
            "md": Language.MARKDOWN,
            "cpp": Language.CPP,
            "c": Language.C,
        }

    def _chunk_file(self, file: Path, max_chunk_size: int):
        file_chunks = []
        language = self.get_file_lang(file)
        spliter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=max_chunk_size,
            chunk_overlap=int(max_chunk_size * 0.05),
        )
        file_text = file.read_text()               
        split_result = spliter.split_text(file_text)
        for string in split_result:
            first_index = file_text.index(string)
            last_index = first_index + len(string);
            hash_id = hashlib.sha256(
                f"{str(file)}[{first_index}:{last_index}]|{file_text[first_index:last_index]}".encode()
                ).hexdigest()

            file_chunks += [
                {
                    "id": hash_id,
                    "mtime": file.lstat().st_mtime,
                    "file_path":str(file),
                    "first_character_index":first_index,
                    "last_character_index":last_index,
                }
                
            ]
        return file_chunks
    def chunk(self, data_path: Path, max_chunk_size: int):
        files_progress = tqdm.tqdm(data_path.rglob("*"))
        all_chunks = []
        for file in files_progress:
            if not self.is_valid_file(file) or "/tests/" in str(file):
                continue
            files_progress.set_description(f"Chunking {file.name}:  ")
            all_chunks += self._chunk_file(file, max_chunk_size=max_chunk_size)
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True)
        with open(self.save_path, "w") as file:
            json.dump(all_chunks, file, indent=4)

    def re_chunk(self, data_path: Path, max_chunk_size: int):
        with open(self.save_path) as file:
            chunks = json.load(file)
        old_files = {chunk["file_path"]:chunk["mtime"] for chunk in chunks}
        files_progrecess = tqdm.tqdm(data_path.rglob("*"))
        edited = {}
        for file in files_progrecess:
            if not self.is_valid_file(file) or "/tests/" in str(file) or len(file.read_text()) == 0:
                continue
            if str(file) not in old_files :
                edited[str(file)] = Filestatus.ADDED
            elif old_files.get(str(file)) and old_files.get(str(file)) != file.lstat().st_mtime:
                edited[str(file)] = Filestatus.MODIFIED
            if old_files.get(str(file)):
                del old_files[str(file)]
        for file in old_files:
            edited[str(file)] = Filestatus.DELETED
        new_chunks = []
        for file in edited:
            if edited[file] == Filestatus.ADDED or edited[file] == Filestatus.MODIFIED:
                new_chunks += self._chunk_file(Path(file), max_chunk_size=max_chunk_size)
        filted_chunks = [chunk for chunk in chunks if chunk["file_path"] not in edited]
        filted_chunks += new_chunks
        with open(self.save_path, "w") as file:
            json.dump(
                filted_chunks,
                file,
                indent=4
            )
        return len(filted_chunks) != len(chunks) , edited
    
    def is_valid_file(self, file: Path):
        file_extension = file.name.split(".")[-1]
        return file_extension in self.file_map

    def get_file_lang(self, file: Path) -> Language:
        file_extension = file.name.split(".")[-1]
        return self.file_map[file_extension]
