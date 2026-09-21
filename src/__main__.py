import fire
from rich.traceback import install
import bm25s
# from rich import print
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from enum import Enum
from .objects import MinimalSource, MinimalSearchResults, MinimalSource, StudentSearchResults, RagDataset, StudentSearchResultsAndAnswer
import json
from tqdm import tqdm
from .model import llm_model
class Chunk(MinimalSource):
    content: str
install()

class RAG:
    def __init__(self):
        self.raw_path = Path("/home/hahchtar/Desktop/student/RAG/data/raw/vllm-0.10.1")
        self.bm25 = bm25s.BM25()
        self.is_print_search= True
        self.file_map = {
            "py": Language.PYTHON,
            "txt": Language.MARKDOWN,
            "md": Language.MARKDOWN,
            "cpp": Language.CPP,
            "c": Language.C
        }
        self.is_print_answer = True
        self.retriever = bm25s.BM25()
        self.processed_path = self.raw_path.parent.parent / "processed"
        self.bm25_index_folder = self.processed_path / "bm25_index"
        self.chunks_json = self.processed_path / "chunks.json"
    def is_valid_file(self, file: Path):
        file_extension = file.name.split(".")[-1]
        return file_extension in self.file_map
    def get_file_lang(self, file: Path) -> Language:
        file_extension = file.name.split(".")[-1]
        return self.file_map[file_extension]        
    def index(self, max_chunk_size: int = 2000):
        chunks: list[Chunk] = []
        files = []
        
        def get_files(folder: Path):
            nonlocal chunks
            for file in folder.glob("*"):
                if file.is_file() and self.is_valid_file(file):
                    language = self.get_file_lang(file)
                    spliter = RecursiveCharacterTextSplitter.from_language(
                        language=language,
                        chunk_size=max_chunk_size,
                        chunk_overlap=int(max_chunk_size * 0.05)
                    )
                    file_text = file.read_text()
                    strings = spliter.split_text(file_text)
                    for string in strings:
                        index = file_text.index(string)
                        chunks += [
                            Chunk(
                                file_path=str(file),
                                content=string,
                                first_character_index=index,
                                last_character_index=index + len(string)
                            )
                        ]
                elif file.is_dir():
                    get_files(file)

        get_files(self.raw_path)
        with open(self.chunks_json, "w") as file:
            json.dump(
                [chunk.model_dump(mode="json") for chunk in chunks],
                file,
                indent=4
            )
        corpus_tokens = bm25s.tokenize([chunk.content for chunk in chunks])
        self.retriever.index(corpus_tokens)
        if not self.processed_path.exists():
            self.processed_path.mkdir()
        self.retriever.save(self.bm25_index_folder)
            

    def search(self, query: str, k: int = 10):
        self.retriever = bm25s.BM25.load(self.bm25_index_folder)
        chunks = json.loads(self.chunks_json.read_text())
        chunks: list[Chunk] = [ Chunk(**data) for data in chunks]
        results, scores = self.retriever.retrieve(
            bm25s.tokenize(query),
            k=k,
        )
        minimal_source_list: list[MinimalSource] = []
        for idx in results[0]:
            if self.is_print_search:
                print(idx, chunks[idx].file_path, chunks[idx].first_character_index, chunks[idx].last_character_index)
            minimal_source_list += [
                MinimalSource(
                    file_path=chunks[idx].file_path,
                    first_character_index=chunks[idx].first_character_index,
                    last_character_index=chunks[idx].last_character_index
                )
            ]
        return MinimalSearchResults(question=query, retrieved_sources=minimal_source_list)
    def search_dataset(self, dataset_path: str, k: int ,save_directory: str):
        path = Path(dataset_path)
        rag_dataset = RagDataset(**json.loads(path.read_text()))
        self.is_print_search = False
        result: list[MinimalSearchResults] = []
            
        for Unansweredquestion in tqdm(rag_dataset.rag_questions, desc="Processing data set"):
            result += [self.search(Unansweredquestion.question, k)]
        student_search = StudentSearchResults(k=k, search_results=result)
        with open(save_directory, "w") as file:
            json.dump(
                student_search.model_dump(mode="json"),
                file,
                indent=4
            )
    def answer(self, query: str, k: int = 10):
        model = llm_model()
        self.is_print_search = False
        minimal_source: MinimalSearchResults = self.search(query, k)
        context = ""
        for source in minimal_source.retrieved_sources:
            file = Path(source.file_path)
            file_text = file.read_text()
            context += f"- {file_text[source.first_character_index:source.last_character_index]}\n"
        answer = model.Generate_answer(query, context)
        if self.is_print_answer:
            print(answer)
        return answer

    def answer_dataset(self, student_search_results_path: str, save_directory: str):
        file = Path(student_search_results_path)
        ragdataset = RagDataset(**json.loads(file.read_text()))
        # for question in ragdataset.rag_questions:
        #     answer = self.answer

if "__main__" == __name__:
    result = fire.Fire(RAG)
