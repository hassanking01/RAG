from pydantic import ValidationError
from sympy import im
from tqdm import tqdm

from .chunker import Chunker
from .models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .errors import (
    MaxChunkSizeError,
    EvaluationError,
    Patherror,
    ModelError
)

from functools import lru_cache
from rich import print
from pathlib import Path
import json
from rich.traceback import install
install()


class RAG:
    def __init__(self):
        from .lexical import BM25

        self.raw_path = Path("data/raw")
        self.processed_path =  Path("data/processed")
        if not self.processed_path.exists():
            self.processed_path.mkdir(parents=True, exist_ok=True)
        self.chunker = Chunker(self.processed_path)
        self.lexical = BM25(self.processed_path, self.chunker.save_path)
        self.is_semantic_loaded = False

    def is_full_indexing(self, max_chunk_size: int) -> True:
        lexical_files = not self.lexical.save_path.exists()
        semantic_files = not self.semantic.save_path.exists()
        chunker_files = not self.chunker.save_path.exists()
        same_chunk_size = not self.chunker._is_same_chunks_size(max_chunk_size)
        if lexical_files or semantic_files or chunker_files or same_chunk_size:
            return True
        

    def index(self, max_chunk_size: int = 2000):
        from .semantic import VectorDb
        self.semantic = VectorDb(self.processed_path, self.chunker.save_path)
        
        if not 200 < max_chunk_size <= 2000:
            raise MaxChunkSizeError("max_chunk_size should be between 200 and 2000")

        if self.is_full_indexing(max_chunk_size):
            self.chunker.chunk(self.raw_path, max_chunk_size)
            self.lexical.index()
            self.semantic.index()
        else:
            edited = self.chunker.re_chunk(self.raw_path, max_chunk_size=max_chunk_size)
            if len(edited[1]) > 0:
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
        if not self.is_semantic_loaded:
            self._load_semantic()

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
        self.is_semantic_loaded = self._load_semantic()
        _dataset_path = Path(dataset_path)
        rag_dataset = RagDataset(**json.loads(_dataset_path.read_text()))
        result: list[MinimalSearchResults] = []
        for Unansweredquestion in tqdm(
            rag_dataset.rag_questions,
            desc="Processing data set",
            colour="#0250f7",
            ascii=" ▒"
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

    @lru_cache
    def model(self):
        from .LLM import llm_model
        return llm_model()

    
    @lru_cache
    def answer(self, query: str, k: int = 10):
        model = self.model()
        minimal_source: MinimalSearchResults = self.search(query, k)
        context = ""
        for source in minimal_source.retrieved_sources:
            file = Path(source.file_path)
            file_text = file.read_text()
            context += f"- {file_text[source.first_character_index : source.last_character_index]}\n"
        answer = model.Generate_answer(query, context)

        print(answer)

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

    @lru_cache
    def _load_semantic(self):
        from .semantic import VectorDb
        self.semantic = VectorDb(self.processed_path, self.chunker.save_path)
        return True

    def answer_dataset(self, student_search_results_path: str, save_directory: str):
        model = self.model()
        search_results_file = Path(student_search_results_path)
        student_search_result = StudentSearchResults(**json.loads(search_results_file.read_text()))
        k = student_search_result.k
        results = []
        for Unanswerd_question in tqdm(
            student_search_result.search_results,
            desc="answering data set",
            colour="#0250f7",
            ascii=" ▒"
            ):
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

    # TODO 
    # if question is in student result and not in answer.json raise error 
    # what i do now is wrong fix that
    # if question is in data set but not in student result its okay 
    def load_evaluation_models(
            self
            ,_dataset_path: Path,
            _student_search_results_path: Path ) -> tuple[RagDataset, StudentSearchResults]:
        try:
            dataset = RagDataset.model_validate_json(_dataset_path.read_text())
        except ValidationError:
            raise ModelError(
                f"{_dataset_path.name} is not valid RagDaraset model"
            )
        try:
            student_search_results = StudentSearchResults.model_validate_json(_student_search_results_path.read_text())
        except ValidationError:
            raise ModelError(
                f"{_student_search_results_path.name}"
                " is not valid StudentSearchResults model"
            )
        return dataset, student_search_results

    def test(self):
        self.lexical.load()
        from pympler import asizeof
        size = asizeof.asizeof(self.lexical)

        print(f"{size:,} bytes")
        print(f"{size / 1024:.2f} KB")
        print(f"{size / 1024**2:.2f} MB")

    def evaluate(self, student_search_results_path: str,  dataset_path: str):
        _student_search_results_path = Path(student_search_results_path)
        _dataset_path = Path(dataset_path)
        for file in [_student_search_results_path, _dataset_path]:
            if not file.exists():
                raise Patherror(
                    f"{file.name} dose not exist"
                )
            if file.is_dir():
                raise Patherror(
                    f"{file.name} is a directory"
                )

        questions_map:dict[str, list[MinimalSource]] = {}

        dataset, student_search_results = self.load_evaluation_models(
            _dataset_path,
            _student_search_results_path
        )
        original_questions = set( _.question_id for _ in dataset.rag_questions)
        
        for data in student_search_results.search_results:
            questions_map[data.question_id] = data.retrieved_sources

        for question in questions_map:
            if question not in original_questions:
                raise EvaluationError(
                    f"question with id: {question}"
                    " not found in the dataset"
                )

        recall_k = {1:0, 3:0, 5:0, 10:0}

        for qustion in dataset.rag_questions:
            if qustion.question_id not in questions_map:
                continue

            for k in recall_k:
                result = 0
                for source in qustion.sources:
                    for user_source in questions_map[qustion.question_id][:k]:
                        if source.file_path == user_source.file_path:
                            first = max(source.first_character_index, user_source.first_character_index)
                            last = min(source.last_character_index, user_source.last_character_index)
                            intersection = max(
                                0,
                                last - first
                            )
                            union = len(user_source) + len(source) - intersection

                            IoO = intersection / union
                            if IoO >= 0.05:
                                result += 1
                                break
                if len(qustion.sources) > 0:
                    recall_k[k] += result / len(qustion.sources) 
        print("🎯 Evaluation Results")

        print(f"📊 Questions evaluated: {len(questions_map)}")
        print("=" * 30, "\n")
        for k in recall_k:
            print(f"📈 Recall@{k}: {recall_k[k]}")
        print({f"recall@{k}":recall_k[k] for k in recall_k})