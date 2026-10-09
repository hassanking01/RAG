import json
from pathlib import Path

import tqdm
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .models import MinimalSource


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

    def chunk(self, data_path: Path, max_chunk_size: int):
        files_progress = tqdm.tqdm(data_path.rglob("*"))
        chunks = []
        for file in files_progress:
            if not self.is_valid_file(file) or "/tests/" in str(file):
                continue
            files_progress.set_description(f"Chunking {file.name}:  ")
            language = self.get_file_lang(file)
            spliter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=max_chunk_size,
                chunk_overlap=int(max_chunk_size * 0.05),
            )
            file_text = file.read_text()
            split_result = spliter.split_text(file_text)
            for string in split_result:
                index = file_text.index(string)
                chunks += [
                    MinimalSource(
                        file_path=str(file),
                        first_character_index=index,
                        last_character_index=index + len(string),
                    ).model_dump(mode="json")
                ]
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True)
        with open(self.save_path, "w") as file:
            json.dump(chunks, file, indent=4)

    def is_valid_file(self, file: Path):
        file_extension = file.name.split(".")[-1]
        return file_extension in self.file_map

    def get_file_lang(self, file: Path) -> Language:
        file_extension = file.name.split(".")[-1]
        return self.file_map[file_extension]
