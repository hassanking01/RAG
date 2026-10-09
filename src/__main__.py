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

install()


class RAG:
    def __init__(self):
        self.raw_path = Path("data/raw/vllm-0.10.1")
        self.processed_path = self.raw_path.parent.parent / "processed"
        self.chunker = Chunker(self.processed_path)
        self.lexical = BM25(self.processed_path, self.chunker.save_path)
        self.semantic = VectorDb(self.processed_path, self.chunker.save_path)

    def index(self, max_chunk_size: int = 2000):
        self.chunker.chunk(self.raw_path, max_chunk_size)
        self.lexical.index()
        self.semantic.index()

    def search(self, query: str, k: int = 10):
        chunks: list[MinimalSource] = [
            MinimalSource(**data)
            for data in json.loads(self.chunker.save_path.read_text())
        ]
        if not self.lexical.loaded:
            self.lexical = self.lexical.load()
        lexical_results = self.lexical.search(query, k)
        minimal_source_list: list[MinimalSource] = []
        for idx in lexical_results:
            minimal_source_list.append(chunks[idx])
        return MinimalSearchResults(
            question=query, retrieved_sources=minimal_source_list
        )

    def search_dataset(self, dataset_path: str, k: int, save_directory: str):
        path = Path(dataset_path)
        rag_dataset = RagDataset(**json.loads(path.read_text()))
        result: list[MinimalSearchResults] = []
        for Unansweredquestion in tqdm(
            rag_dataset.rag_questions, desc="Processing data set"
        ):
            r = self.search(Unansweredquestion.question, k)
            r.question_id = Unansweredquestion.question_id
            result += [r]
        student_search = StudentSearchResults(k=1, search_results=result)
        with open(save_directory, "w") as file:
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

    def answer_dataset(self, student_search_results_path: str, save_directory: str):
        file = Path(student_search_results_path)
        ragdataset_qustions = RagDataset(**json.loads(file.read_text()))
        ragadataset_answers = RagDataset(rag_questions=[])
        for Unanswerd_question in ragdataset_qustions.rag_questions:
            result = self.answer(Unanswerd_question.question, 5)
            answered = AnsweredQuestion(
                question=Unanswerd_question.question,
                sources=result.retrieved_sources,
                answer=result.answer,
            )
            ragadataset_answers.rag_questions.append(answered)
        with open(save_directory, "w") as file:
            json.dump(ragadataset_answers.model_dump(mode="json"), file, indent=4)


if "__main__" == __name__:
    result = fire.Fire(RAG)
