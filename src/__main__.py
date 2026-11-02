import json
from pathlib import Path

import fire
from rich.traceback import install
from tqdm import tqdm

from .chunker import Chunker
from .lexical import BM25
from .LLM import llm_model
from .models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .semantic import VectorDb
from functools import lru_cache

# install()


class RAG:
    def __init__(self):
        self.raw_path = Path("data/raw/vllm-0.10.1")
        self.processed_path = self.raw_path.parent.parent / "processed"
        self.chunker = Chunker(self.processed_path)
        self.lexical = BM25(self.processed_path, self.chunker.save_path)
        self.semantic = VectorDb(self.processed_path, self.chunker.save_path)

    def is_full_indexing(self) -> True:
        lexical_files = not self.lexical.save_path.exists()
        semantic_files = not self.semantic.save_path.exists()
        chunker_files = not self.chunker.save_path.exists()
        if lexical_files or semantic_files or chunker_files:
            return True
        

    def index(self, max_chunk_size: int = 2000):
        if self.is_full_indexing():
            self.chunker.chunk(self.raw_path, max_chunk_size)
            self.lexical.index()
            self.semantic.index()
        else:
            edited = self.chunker.re_chunk(self.raw_path, max_chunk_size=max_chunk_size)
            self.lexical.load()
            self.lexical.re_index(edited)
            self.semantic.re_index(edited)
    def _rrf(self, results: list[list[int]], k=60):
        scores = {}

        for result_list in results:
            for rank, doc_id in enumerate(result_list, start=1):
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        return sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

    @lru_cache
    def _get_chunks(self):
        return[
            MinimalSource(**data)
            for data in json.loads(self.chunker.save_path.read_text())
        ]

    def search(self, query: str, k: int = 10):
        chunks: list[MinimalSource] = self._get_chunks()
        if not self.lexical.loaded:
            self.lexical.load()
        if not self.semantic._is_loaded:
            self.semantic.load()
        lexical_results = self.lexical.search(query, k)
        semantic_results = self.semantic.search(query, k)
        final_result = [data[0] for data in self._rrf([lexical_results, semantic_results])][:k]
        minimal_source_list: list[MinimalSource] = []
        for idx in final_result:
            minimal_source_list.append(chunks[idx])
        return MinimalSearchResults(
            question=query, retrieved_sources=minimal_source_list
        )

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        _dataset_path = Path(dataset_path)
        rag_dataset = RagDataset(**json.loads(_dataset_path.read_text()))
        result: list[MinimalSearchResults] = []
        for Unansweredquestion in tqdm(
            rag_dataset.rag_questions, desc="Processing data set"
        ):
            r = self.search(Unansweredquestion.question, k)
            r.question_id = Unansweredquestion.question_id
            result += [r]
        student_search = StudentSearchResults(k=k, search_results=result)
        save_directory: Path = Path(save_directory)
        if not save_directory.exists():
            save_directory.mkdir(parents=True, exist_ok=True)
        save_result = save_directory / _dataset_path.name
        with open(save_result, "w") as file:
            json.dump(student_search.model_dump(mode="json"), file, indent=4)

    def answer(self, query: str, k: int = 10):
        model = llm_model()
        minimal_source: MinimalSearchResults = self.search(query, k)
        context = ""
        for source in minimal_source.retrieved_sources:
            file = Path(source.file_path)
            file_text = file.read_text()
            context += f"- {file_text[source.first_character_index : source.last_character_index]}\n"
        answer = model.Generate_answer(query, context)

        return MinimalAnswer(
            question=minimal_source.question,
            retrieved_sources=minimal_source.retrieved_sources,
            answer=answer,
        )
    def _answer(self,question_id:str, query: str ,minimal_source: list[MinimalSource], model: llm_model):
        context = ""
        for source in minimal_source:
            file = Path(source.file_path)
            file_text = file.read_text()
            context += f"- {file_text[source.first_character_index : source.last_character_index]}\n"
        answer = model.Generate_answer(query, context)

        return MinimalAnswer(
            question_id=question_id,
            question=query,
            retrieved_sources=minimal_source,
            answer=answer,
        )
        

    def answer_dataset(self, student_search_results_path: str, save_directory: str):
        model = llm_model()
        search_results_file = Path(student_search_results_path)
        student_search_result = StudentSearchResults(**json.loads(search_results_file.read_text()))
        k = student_search_result.k
        results = []
        for Unanswerd_question in tqdm(student_search_result.search_results):
            question_answer = self._answer(
                Unanswerd_question.question_id,
                Unanswerd_question.question,
                Unanswerd_question.retrieved_sources,
                model
            )
            results.append(question_answer)
        student_answer_resutl = StudentSearchResultsAndAnswer(search_results=results, k=k)
        _save_directory = Path(save_directory)
        if not _save_directory.exists():
            _save_directory.mkdir(parents=True, exist_ok=True)
        save_result = _save_directory / search_results_file.name
        with open(save_result, "w") as file:
            json.dump(student_answer_resutl.model_dump(mode="json"), file, indent=4)
        
if "__main__" == __name__:
    try:
        result = fire.Fire(RAG)
    except KeyboardInterrupt:
        print("bye")
    except BaseException:
        print("something really bad happened")
