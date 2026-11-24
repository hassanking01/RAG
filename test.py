from sys import argv
from typing import Any, Counter
from hashlib import sha256
from functools import lru_cache
from itertools import accumulate
import ast
from pathlib import Path
from enum import Enum
from tqdm import tqdm
import json 
import re
from rich import print

class FileType(Enum):
    PYTHON = "py"
    MARKDOWN = "md"
    TEXT = "txt"

class chunker:
    def __init__(self, processed: Path):
        self.save_path = Path("data/processed") / "chunks"
        self.map_files = {
            "py": FileType.PYTHON,
            "md": FileType.MARKDOWN,
            "txt": FileType.TEXT
        }
        self.lines = []
        self.max_chunk_size = 200
        self.buffer: list[ast.stmt] = []
        self.current_file: Path
        self.content: str


    def _is_valid_file(self, file: Path):
        extension = file.name.split(".")[-1]
        return extension in self.map_files

    def _file_type(self, file: Path) -> FileType:
        extension = file.name.split(".")[-1]
        return self.map_files[extension]        

    def _node_length(self, node: ast.stmt) -> bool:
        start = self.lines[node.lineno - 1] + node.col_offset
        end = self.lines[node.end_lineno - 1] + node.end_col_offset
        return end - start 

    def _get_first_buffer(self):
        if not self.buffer:
            return -1
        line_number = self.buffer[0].lineno - 1
        return self.lines[line_number] + self.buffer[0].col_offset 

    @lru_cache(maxsize=1)
    def _get_chunk_contet(self, file: Path) -> str:
        return file.read_text()

    def _calculate_hash_id(self, start: int, end: int) -> str:
        content = self._get_chunk_contet(self.current_file)[start:end]
        chunk_id = f"{self.current_file}[{start}:{end}]|{content}"
        return sha256(chunk_id.encode()).hexdigest()

    def _flush_buffer(self) -> list[dict[str, Any]]:
        if not self.buffer:
            return []
        result = []
        start_line = self.buffer[0].lineno - 1
        start_offsit = self.buffer[0].col_offset
        end_line = self.buffer[-1].end_lineno - 1
        end_offsit = self.buffer[-1].end_col_offset - 1
        start = self.lines[start_line] + start_offsit
        end = self.lines[end_line] + end_offsit 
        result.extend(self._flush(start, end))
        self.buffer.clear()
        return result
    
    def _flush(self, start, end):
        chunk = {}
        chunk_id = self._calculate_hash_id(start, end)
        chunk["id"] = chunk_id
        chunk["file_path"] = str(self.current_file)
        chunk["first_character_index"] = start
        chunk["last_character_index"] = end
        chunk["mtime"] = self.current_file.lstat().st_mtime
        chunk["len"] = end - start
        return [chunk]

    def _procces_one_node(self, node:ast.stmt) -> list[dict[str, Any]]:
        chunks = []
        chunks.extend(self._flush_buffer())
        over_lap = int(self.max_chunk_size * 0.05)
        end = self.lines[node.end_lineno - 1] + node.end_col_offset
        start = self.lines[node.lineno - 1] + node.col_offset
        new_end = start
        data = self.content[start:].splitlines(keepends=True)
        chunk = ""

        while start < end and data:
            line = data.pop(0)
            if len(line) > self.max_chunk_size:
                if len(chunk) > 0:
                    chunks.extend(self._flush( start,new_end))
                    start = new_end
                
                for i in range(0, len(line), self.max_chunk_size):
                    chunk = line[i:i+self.max_chunk_size]
                    new_end += len(chunk)
                    chunks.extend(self._flush( start, new_end))
                    start = new_end 
                
                continue
            
            if len(chunk) + len(line) > self.max_chunk_size:
                new_end += len(chunk)
                chunks.extend(self._flush(start, new_end))
                start = max(0, new_end - over_lap)
                data = self.content[start:].splitlines(True)
                new_end = start
                chunk = line
            else:
                chunk += line
                
        return chunks

    def _handel_python(self, file: Path) -> list[dict[str, Any]]:
        self.content = file.read_text()
        if not self.content.strip():
            return []
        tree = ast.parse(self.content, file)
        chunks = []
        lines = [0]
        for l in self.content.splitlines(keepends=True):
            lines.append(lines[-1] + len(l))
        self.lines = list(
            accumulate(
                [len(line)
                 for line in self.content.splitlines(keepends=True)
                 ],
                 initial=0))

        for node in tree.body:
            node_length = self._node_length(node)
            if node_length >= self.max_chunk_size:
                chunks.extend(self._procces_one_node(node))
                continue

            start = self._get_first_buffer()
        
            if start == -1 :
                self.buffer.append(node)
                continue
        
            end = self.lines[node.end_lineno - 1] + node.end_col_offset
            if end - start > self.max_chunk_size:
                chunks.extend(self._flush_buffer())
                
            self.buffer.append(node)
        if len(self.buffer) > 0:
            chunks.extend(self._flush_buffer())
        return chunks


    def _handel_markdown(self, file: Path) -> list[dict[str, Any]]:
        result = []
        header_pattern = r"^ {0,3}(#{1,6})\s+(.*)$"
        content = self.current_file.read_text()
        content_lines = content.splitlines(True)
        is_header = [re.search(header_pattern, line) is not None for line in content_lines]
        index = 0
        chunk = ""
        start = 0
        end = 0
        while not is_header[index]:
            line = content_lines[index]
            index += 1
            if len(line) > self.max_chunk_size:
                if chunk:
                    end = start + len(chunk) - 1
                    result.extend(self._flush(start, end))
                    start = end + 1
                    chunk = ""
                for i in range(0, len(line), self.max_chunk_size):
                    end = start + len(line[i:i+self.max_chunk_size])
                    result.extend(self._flush(start, end))
                    start = end + 1
                continue
            if len(chunk) + len(line) > self.max_chunk_size:
                end = start + len(chunk) - 1
                result.extend(self._flush(start, end))
                start = end + 1
                chunk = ""
            chunk += line

        if chunk:
            end = start + len(chunk) - 1
            result.extend(self._flush(start, end))
            start = end + 1
            chunk = ""

        while index < len(is_header):
            chunk += content_lines[index]
            index += 1
            while index < len(is_header):
                if is_header[index]:
                    break
                line = content_lines[index]
                if len(chunk) + len(line) > self.max_chunk_size:
                    end = start + len(chunk) - 1
                    result.extend(self._flush(start, end))
                    start = end + 1
                    chunk = ""
                chunk += line
                index += 1

        if chunk:
            end = start + len(chunk) - 1
            result.extend(self._flush(start, end))
            start = end + 1
            chunk = ""

        return result

    def _handel_text(self, file: Path) -> list[dict[str, Any]]:
        return []

    def chunk_data(self, data_path: Path, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size
        files = [
            file for file in data_path.rglob("*")
            if self._is_valid_file(file)
        ]
        number_files = len(files)
        chunks = []
        for file in tqdm(
            files,
            total=number_files,
            desc="chunking files",
            colour="#0250f7",
            ascii=" ▒"):
            file_type = self._file_type(file)
            self.current_file = file
            match file_type:
                case FileType.PYTHON:
                    chunks.extend(self._handel_python(file))
                case FileType.MARKDOWN:
                    chunks.extend(self._handel_markdown(file))
                case FileType.TEXT:
                    chunks.extend(self._handel_text(file))
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

        with open(self.save_path / "chunks.json", "w") as file:
            json.dump(chunks, file, indent=4)

c = chunker(Path("test"))
c.chunk_data(Path(argv[1]))