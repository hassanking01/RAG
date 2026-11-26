from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, cast
import json

from pydantic import ValidationError
from rich import print
from rich.traceback import install
from tqdm import tqdm

from .chunker import Chunker
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    AnsweredQuestion
)
from .errors import (
    MaxChunkSizeError,
    EvaluationError,
    Patherror,
    ModelError,
    EmptyData
)


if TYPE_CHECKING:
    from .LLM import llm_model

install()


class RAG:
    """Coordinate chunking, retrieval, evaluation, and answer generation."""
    def __init__(self):
        """Initialize the RAG pipeline and its indexes."""
        from .lexical import BM25

        self.raw_path = Path("data/raw")
        self.processed_path = Path("data/processed")
        if not self.processed_path.exists():
            self.processed_path.mkdir(parents=True, exist_ok=True)
        self.chunker = Chunker(self.processed_path)
        self.lexical = BM25(self.processed_path, self.chunker.save_path)
        self.is_semantic_loaded = False

    def _is_full_indexing(self, max_chunk_size: int) -> bool | None:
        """Return whether a full index must be built."""
        if self.lexical.save_path.is_dir():
            raise Patherror("lexical save path is a directory")
        if self.chunker.save_path.is_dir():
            raise Patherror("chunk save path is a directory")
        lexical_files = not self.lexical.save_path.exists()
        semantic_files = not self.semantic.is_save_path()
        chunker_files = not self.chunker.save_path.exists()
        same_chunk_size = not self.chunker._is_same_chunks_size(max_chunk_size)
        if lexical_files or semantic_files or chunker_files or same_chunk_size:
            return True
        return False

    def index(self, max_chunk_size: int = 2000):
        """Build a new index or update the existing one."""
        from .semantic import VectorDb
        self.semantic = VectorDb(
            self.processed_path, self.chunker.save_path
        )

        if not 200 <= max_chunk_size <= 2000:
            raise MaxChunkSizeError(
                "max_chunk_size should be between 200 and 2000"
            )

        if self._is_full_indexing(max_chunk_size):
            self.chunker.chunk(self.raw_path, max_chunk_size)
            self.lexical.index()
            self.semantic.index()
        else:
            edited = self.chunker.re_chunk(
                self.raw_path, max_chunk_size=max_chunk_size
            )
            if len(edited[1]) > 0:
                self.lexical.re_index(edited)
                self.semantic.re_index(edited)

    def _rrf(self, results: list[list[int]], k=60):
        """Combine ranked results using reciprocal rank fusion."""
        scores: dict[int, float] = {}

        for result_list in results:
            for rank, doc_id in enumerate(result_list, start=1):
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        return sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

    @lru_cache
    def _get_chunks(self):
        """Load all saved chunks as source models."""
        return [
            MinimalSource(**data)
            for data in json.loads(self.chunker.save_path.read_text())
        ]

    def search(self, query: str, k: int = 10):
        """Search indexed files and return the best sources."""
        query = str(query)
        if not query.strip():
            raise EmptyData("the query is an empty string")
        if k <= 0:
            raise ValueError("the top k should be positive number")
        if not self.is_semantic_loaded:
            self._load_semantic()

        chunks: list[MinimalSource] = self._get_chunks()
        if not self.lexical.loaded:
            self.lexical.load()
        if not self.semantic._is_loaded:
            self.semantic.load()
        lexical_results = self.lexical.search(query, k)
        semantic_results = self.semantic.search(query, k)
        final_result = [
            data[0]
            for data in self._rrf([lexical_results, semantic_results])
        ][:k]
        minimal_source_list: list[MinimalSource] = []
        for idx in final_result:
            minimal_source_list.append(chunks[idx])
        return MinimalSearchResults(
            question=query, retrieved_sources=minimal_source_list
        )

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        """Search every question in a dataset and save the results."""
        self.is_semantic_loaded = self._load_semantic()
        save_directory_path: Path = Path(save_directory)
        _dataset_path = Path(dataset_path)
        if not _dataset_path.exists():
            raise Patherror(
                "dataset you are providing:"
                f" {_dataset_path} dose not exist"
                )
        if _dataset_path.is_dir():
            raise Patherror(
                "dataset path you provided is a directory"
            )
        if save_directory_path.is_file():
            raise Patherror(
                "the save dir you provided is an existing"
                " file"
            )
        rag_dataset = RagDataset(**json.loads(_dataset_path.read_text()))
        result: list[MinimalSearchResults] = []
        for Unansweredquestion in tqdm(
            rag_dataset.rag_questions,
            desc="Processing data set",
            colour="#0250f7",
            ascii=" ▒",
        ):
            r = self.search(Unansweredquestion.question, k)
            r.question_id = Unansweredquestion.question_id
            result += [r]
        student_search = StudentSearchResults(k=k, search_results=result)
        if not save_directory_path.exists():
            save_directory_path.mkdir(parents=True, exist_ok=True)
        save_result = save_directory_path / _dataset_path.name
        try:
            with open(save_result, "w") as file:
                json.dump(
                    student_search.model_dump(mode="json"),
                    file,
                    indent=4,
                )
        except Exception:
            raise Patherror("could not save search results")

    @lru_cache
    def model(self):
        """Create and cache the language model."""
        from .LLM import llm_model

        return llm_model()

    @lru_cache
    def answer(self, query: str, k: int = 10):
        """Generate an answer using search results as context."""
        model = self.model()
        minimal_source: MinimalSearchResults = self.search(query, k)
        context = ""
        for source in minimal_source.retrieved_sources:
            file = Path(source.file_path)
            try:
                file_text = file.read_text()
                source_text = file_text[
                    source.first_character_index:source.last_character_index
                ]
            except BaseException:
                continue
            context += f"- {source_text}\n"
        if not context:
            raise ValueError(
                "all the files are corrapt or the are binary now"
            )
        answer = model.Generate_answer(query, context)
        result: MinimalAnswer = MinimalAnswer(
            question_id=minimal_source.question_id,
            question=minimal_source.question,
            retrieved_sources=minimal_source.retrieved_sources,
            answer=answer
        )
        print(result.model_dump(mode="json"))
        return result

    def _answer(
        self,
        question_id: str,
        query: str,
        minimal_source: list[MinimalSource],
        model: "llm_model",
    ):
        """Generate a modeled answer for one question."""
        context = ""
        for source in minimal_source:
            file = Path(source.file_path)
            try:
                file_text = file.read_text()
                source_text = file_text[
                    source.first_character_index:source.last_character_index
                ]
                context += f"- {source_text}\n"
            except BaseException:
                raise ValueError(
                    f"all the files in {question_id} sources"
                    " are corrapt or they are binary files"
                )
        answer = model.Generate_answer(query, context)

        return MinimalAnswer(
            question_id=question_id,
            question=query,
            retrieved_sources=minimal_source,
            answer=answer,
        )

    @lru_cache
    def _load_semantic(self):
        """Create and cache the semantic index object."""
        from .semantic import VectorDb

        self.semantic = VectorDb(self.processed_path, self.chunker.save_path)
        return True

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ):
        """Answer saved search results and write the answers."""
        model = self.model()
        search_results_file = Path(student_search_results_path)
        try:
            string_data = search_results_file.read_text()
            data = json.loads(string_data)
        except BaseException:
            raise ValueError(
                f"the file {student_search_results_path} is "
                "corrapted or its binary now"
            )
        student_search_result = StudentSearchResults(
            **data
        )
        k = student_search_result.k
        results = []
        for Unanswerd_question in tqdm(
            student_search_result.search_results,
            desc="answering data set",
            colour="#0250f7",
            ascii=" ▒",
        ):
            question_answer = self._answer(
                Unanswerd_question.question_id,
                Unanswerd_question.question,
                Unanswerd_question.retrieved_sources,
                model,
            )
            results.append(question_answer)
        student_answer_resutl = StudentSearchResultsAndAnswer(
            search_results=results, k=k
        )
        _save_directory = Path(save_directory)
        if not _save_directory.exists():
            _save_directory.mkdir(parents=True, exist_ok=True)
        save_result = _save_directory / search_results_file.name
        try:
            with open(save_result, "w") as file:
                json.dump(
                    student_answer_resutl.model_dump(mode="json"),
                    file,
                    indent=4,
                )
        except Exception:
            raise Patherror("could not save answered search results")

    def load_evaluation_models(
        self,
        _dataset_path: Path,
        _student_search_results_path: Path,
    ) -> tuple[RagDataset, StudentSearchResults]:
        """Load and validate the dataset and search result models."""
        try:
            dataset = RagDataset.model_validate_json(_dataset_path.read_text())
        except ValidationError:
            raise ModelError(
                f"{_dataset_path.name} is not valid RagDaraset model"
            )
        try:
            student_search_results = StudentSearchResults.model_validate_json(
                _student_search_results_path.read_text()
            )
        except ValidationError:
            raise ModelError(
                f"{_student_search_results_path.name}"
                " is not valid StudentSearchResults model"
            )
        return dataset, student_search_results

    def evaluate(self, student_search_results_path: str, dataset_path: str):
        """Calculate recall for saved search results."""
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

        questions_map: dict[str, list[MinimalSource]] = {}

        dataset, student_search_results = self.load_evaluation_models(
            _dataset_path,
            _student_search_results_path
        )
        original_questions = {
            question.question_id for question in dataset.rag_questions
        }

        for data in student_search_results.search_results:
            questions_map[data.question_id] = data.retrieved_sources

        for question in questions_map:
            if question not in original_questions:
                raise EvaluationError(
                    f"question with id: {question}"
                    " not found in the dataset"
                )

        recall_k: dict[int, float] = {1: 0, 3: 0, 5: 0, 10: 0}

        for qustion in dataset.rag_questions:
            if qustion.question_id not in questions_map:
                continue

            for k in recall_k:
                result = 0
                sources = cast(AnsweredQuestion, qustion).sources
                for source in sources:
                    for user_source in questions_map[qustion.question_id][:k]:
                        if source.file_path == user_source.file_path:
                            first = max(
                                source.first_character_index,
                                user_source.first_character_index,
                            )
                            last = min(
                                source.last_character_index,
                                user_source.last_character_index,
                            )
                            intersection = max(0, last - first)
                            union = (
                                len(user_source) + len(source) - intersection
                            )

                            IoO = intersection / union
                            if IoO >= 0.05:
                                result += 1
                                break
                if len(sources) > 0:
                    recall_k[k] += result / len(sources)

        print("🎯 Evaluation Results")

        print(f"📊 Questions evaluated: {len(questions_map)}")
        print("=" * 30, "\n")
        for k in recall_k:
            print(f"📈 Recall@{k}: {recall_k[k]}")
        print({f"recall@{k}": recall_k[k] for k in recall_k})
