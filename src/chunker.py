import ast
import json
import re
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm

from .enums import Filestatus, FileType
from .errors import EmptyData


class Chunker:
    """Split supported files into indexed chunks with source offsets."""
    def __init__(
        self,
        processed_path: Path,
    ):
        """Initialize the chunker and its output paths."""
        self.save_path = processed_path / "chunks/chunks.json"
        self.file_map = {
            "py": FileType.PYTHON,
            "txt": FileType.TEXT,
            "md": FileType.MARKDOWN
        }
        self.lines: list[int] = []
        self.max_chunk_size = 200
        self.buffer: list[ast.stmt] = []
        self.current_file: Path
        self.content: str

    def _file_type(self) -> FileType:
        """Return the type of the current file."""
        extension = self.current_file.name.split(".")[-1]
        return self.file_map[extension]

    def _node_length(self, node: ast.stmt) -> int:
        """Return the character length of an AST node."""
        start = self.lines[cast(int, node.lineno) - 1] + cast(
            int, node.col_offset
        )
        end = self.lines[cast(int, node.end_lineno) - 1] + cast(
            int, node.end_col_offset
        )
        return end - start

    def _get_first_buffer(self):
        """Return the starting position of the first buffered node."""
        if not self.buffer:
            return -1
        line_number = cast(int, self.buffer[0].lineno) - 1
        return self.lines[line_number] + cast(
            int, self.buffer[0].col_offset
        )

    @lru_cache(maxsize=1)
    def _get_chunk_contet(self, file: Path) -> str:
        """Read and return a file's content."""
        return file.read_text()

    def _calculate_hash_id(self, start: int, end: int) -> str:
        """Create an ID for a file section."""
        content = self._get_chunk_contet(self.current_file)[start:end]
        chunk_id = f"{self.current_file}[{start}:{end}]|{content}"
        return sha256(chunk_id.encode()).hexdigest()

    def _flush_buffer(self) -> list[dict[str, Any]]:
        """Convert the buffered nodes into one chunk."""
        if not self.buffer:
            return []
        result = []
        start_line = cast(int, self.buffer[0].lineno) - 1
        start_offsit = cast(int, self.buffer[0].col_offset)
        end_line = cast(int, self.buffer[-1].end_lineno) - 1
        end_offsit = cast(int, self.buffer[-1].end_col_offset) - 1
        start = self.lines[start_line] + start_offsit
        end = self.lines[end_line] + end_offsit
        result.extend(self._flush(start, end))
        self.buffer.clear()
        return result

    def _flush(self, start: int, end: int) -> list[dict[str, Any]]:
        """Create chunk metadata for a character range."""
        chunk: dict[str, Any] = {}
        chunk_id = self._calculate_hash_id(start, end)
        chunk["id"] = chunk_id
        chunk["file_path"] = str(self.current_file)
        chunk["first_character_index"] = start
        chunk["last_character_index"] = end
        chunk["mtime"] = self.current_file.lstat().st_mtime
        return [chunk]

    def _procces_one_node(self, node: ast.stmt) -> list[dict[str, Any]]:
        """Split a large Python node into chunks."""
        chunks = []
        chunks.extend(self._flush_buffer())
        over_lap = int(self.max_chunk_size * 0.05)
        end = self.lines[cast(int, node.end_lineno) - 1] + cast(
            int, node.end_col_offset
        )
        start = self.lines[cast(int, node.lineno) - 1] + cast(
            int, node.col_offset
        )
        new_end = start
        data = self.content[start:].splitlines(keepends=True)
        chunk = ""

        while start < end and data:
            line = data.pop(0)
            if len(line) > self.max_chunk_size:
                if len(chunk) > 0:
                    new_end = start + len(chunk) - 1
                    chunks.extend(self._flush(start, new_end))
                    start = new_end + 1

                for i in range(0, len(line), self.max_chunk_size):
                    chunk = line[i:i + self.max_chunk_size]
                    new_end += len(chunk)
                    chunks.extend(self._flush(start, new_end))
                    start = new_end

                continue
            if len(chunk) + len(line) > self.max_chunk_size:
                new_end = start + len(chunk) - 1
                chunks.extend(self._flush(start, new_end))
                tail = self._get_overlap(chunk, over_lap)
                if len(tail) + len(line) > self.max_chunk_size:
                    tail = ""
                start = new_end + 1 - len(tail)
                chunk = tail
                data = self.content[start:].splitlines(True)
                chunk = line
            else:
                chunk += line

        if len(chunk) > 0:
            new_end = start + len(chunk) - 1
            chunks.extend(self._flush(start, new_end))
            start = new_end + 1
        return chunks

    def _handel_python(self) -> list[dict[str, Any]]:
        """Chunk the current Python file."""
        self.content = self.current_file.read_text()
        if not self.content.strip():
            return []
        tree = ast.parse(self.content)
        chunks = []
        self.lines = [0]
        for line in self.content.splitlines(keepends=True):
            self.lines.append(self.lines[-1] + len(line))

        for node in tree.body:
            node_length = self._node_length(node)
            if node_length >= self.max_chunk_size:
                chunks.extend(self._procces_one_node(node))
                continue

            start = self._get_first_buffer()

            if start == -1:
                self.buffer.append(node)
                continue

            end = self.lines[cast(int, node.end_lineno) - 1] + cast(
                int, node.end_col_offset
            )
            if end - start > self.max_chunk_size:
                chunks.extend(self._flush_buffer())

            self.buffer.append(node)
        if len(self.buffer) > 0:
            chunks.extend(self._flush_buffer())
        return chunks

    def _handel_markdown(self):
        """Chunk the current Markdown file."""
        result = []
        content = self.current_file.read_text()
        content_lines = content.splitlines(True)
        header_pattern = r"^ {0,3}#{1,6}[ \t]+\S"
        code_pattern = r"^ {0,3}(```|~~~)"
        min_chunk_size = 200
        overlap = int(self.max_chunk_size * 0.05)

        is_header = []
        in_code = False
        for line in content_lines:
            if re.match(code_pattern, line):
                in_code = not in_code
                is_header.append(False)
            else:
                is_header.append(
                    not in_code and re.match(header_pattern, line) is not None
                )

        chunk = ""
        start = 0
        for index, line in enumerate(content_lines):

            if is_header[index] and len(chunk) >= min_chunk_size:
                end = start + len(chunk) - 1
                result.extend(self._flush(start, end))
                start = end + 1
                chunk = ""

            if len(line) > self.max_chunk_size:
                if chunk:
                    end = start + len(chunk) - 1
                    result.extend(self._flush(start, end))
                    start = end + 1
                    chunk = ""
                for i in range(0, len(line), self.max_chunk_size):
                    piece = line[i:i + self.max_chunk_size]
                    end = start + len(piece) - 1
                    result.extend(self._flush(start, end))
                    start = end + 1
                continue

            if len(chunk) + len(line) > self.max_chunk_size:
                end = start + len(chunk) - 1
                result.extend(self._flush(start, end))
                tail = self._get_overlap(chunk, overlap)
                if len(tail) + len(line) > self.max_chunk_size:
                    tail = ""
                start = end + 1 - len(tail)
                chunk = tail

            chunk += line

        if chunk:
            end = start + len(chunk) - 1
            result.extend(self._flush(start, end))

        return result

    def _get_overlap(self, chunk: str, overlap: int) -> str:
        """Return the last lines up to the overlap size."""
        tail = ""
        lines = chunk.splitlines(True)[::-1]
        for line in lines:
            if len(tail) + len(line) > overlap:
                break
            tail = line + tail
        return tail

    def _handel_text(self) -> list[dict[str, Any]]:
        """Chunk the current text file."""
        try:
            content = self.current_file.read_text()
        except BaseException:
            return []
        if not content.strip():
            return []
        content_lines = content.splitlines(True)
        chunks = []

        chunk = ""
        start = 0
        end = 0
        for line in content_lines:
            if len(line) > self.max_chunk_size:
                if chunk:
                    end = start + len(chunk) - 1
                    chunks.extend(self._flush(start, end))
                    start = end + 1
                    chunk = ""
                for i in range(0, len(line), self.max_chunk_size):
                    end = start + len(line[i:i + self.max_chunk_size])
                    chunks.extend(self._flush(start, end))
                    start = end + 1
                continue
            if len(chunk) + len(line) > self.max_chunk_size:
                end = start + len(chunk) - 1
                chunks.extend(self._flush(start, end))
                start = end + 1
                chunk = ""
            chunk += line
        if chunk:
            end = start + len(chunk) - 1
            chunks.extend(self._flush(start, end))
            start = end + 1
            chunk = ""

        return chunks

    def _chunk_file(self, file: Path):
        """Chunk one supported file."""
        try:
            file.read_text()
        except BaseException:
            return []
        chunks = []
        self.current_file = file
        file_type = self._file_type()
        match file_type:
            case FileType.PYTHON:
                chunks.extend(self._handel_python())
            case FileType.MARKDOWN:
                chunks.extend(self._handel_markdown())
            case FileType.TEXT:
                chunks.extend(self._handel_text())
        return chunks

    def _is_same_chunks_size(self, max_chunk_size: int) -> bool:
        """Check whether the saved chunk size matches the given size."""
        file = self.save_path.parent / ".max_chunk_size"
        try:
            old_max_chunk_size = int(file.read_text())
        except (ValueError, FileNotFoundError):
            return False
        return old_max_chunk_size == max_chunk_size

    def check_empy_files(self, files: list[Path]) -> None:
        """Raise an error when all files are empty."""
        result = []

        for file in files:
            result.append(len(file.read_text().strip()) == 0)

        if all(result):
            raise EmptyData(
                "all files in raw data folder are empty"
            )

    def _save_max_chunk_size(self, max_chunk_size: int) -> None:
        """Save the current maximum chunk size."""
        file = self.save_path.parent / ".max_chunk_size"
        file.write_text(str(max_chunk_size))

    def chunk(self, data_path: Path, max_chunk_size: int):
        """Chunk all supported files in a data directory."""
        self.max_chunk_size = max_chunk_size
        files = [
            file for file in data_path.rglob("*")
            if file.is_file()
            and self._is_valid_file(file)
        ]
        self.check_empy_files(files)

        all_chunks = []
        for file in tqdm(
            files,
            desc="chunking files",
            colour="#0250f7",
            ascii=" ▒",
        ):
            all_chunks += self._chunk_file(file)

        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True)
        with open(self.save_path, "w") as file:
            json.dump(all_chunks, file, indent=4)
        self._save_max_chunk_size(max_chunk_size)

    def re_chunk(self, data_path: Path, max_chunk_size: int):
        """Find changed files and rebuild their chunks."""
        files = [
            file
            for file in data_path.rglob("*")
            if file.is_file() and self._is_valid_file(file)
        ]
        self.check_empy_files(files)
        with open(self.save_path) as file:
            chunks = json.load(file)
        old_files = {
            chunk["file_path"]: chunk["mtime"] for chunk in chunks
        }
        edited = {}
        for file in tqdm(
            files,
            desc="searching for updated files",
            colour="#0250f7",
            ascii=" ▒",
        ):
            if len(file.read_text().strip()) == 0:
                continue
            self.current_file = file
            if self._file_type() == FileType.PYTHON:
                if len(ast.parse(file.read_text()).body) == 0:
                    continue
            if str(file) not in old_files:
                edited[str(file)] = Filestatus.ADDED
            elif (
                str(file) in old_files
                and old_files.get(str(file)) != file.lstat().st_mtime
            ):
                edited[str(file)] = Filestatus.MODIFIED
            if old_files.get(str(file)):
                del old_files[str(file)]
        for file in old_files:
            edited[str(file)] = Filestatus.DELETED
        if len(edited) == 0:
            print("noting to be re_indexed all set :)")
            return (False, {})
        print("found changes re_chunking now!")
        new_chunks = []
        for file in tqdm(
            edited,
            desc="re_chunking edited files",
            colour="#0250f7",
            ascii=" ▒",
        ):
            if (
                edited[file] == Filestatus.ADDED
                or edited[file] == Filestatus.MODIFIED
            ):
                new_chunks += self._chunk_file(Path(file))
        filted_chunks = [
            chunk for chunk in chunks if chunk["file_path"] not in edited
        ]
        filted_chunks += new_chunks
        with open(self.save_path, "w") as file:
            json.dump(
                filted_chunks,
                file,
                indent=4
            )
        self._save_max_chunk_size(max_chunk_size)
        return len(filted_chunks) != len(chunks), edited

    def _is_valid_file(self, file: Path) -> bool:
        """Return whether a file extension is supported."""
        file_extension = file.name.split(".")[-1]
        return file_extension in self.file_map
