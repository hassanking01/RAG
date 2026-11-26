import uuid
from typing import Any, List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represent the location of a retrieved source chunk."""
    file_path: str
    first_character_index: int
    last_character_index: int

    def __len__(self) -> int:
        """Return the source range length."""
        return self.last_character_index - self.first_character_index


class UnansweredQuestion(BaseModel):
    """Represent a question before an answer is generated."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question together with its reference answer and sources."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a dataset containing RAG questions and answers."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represent the source chunks retrieved for a question."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    retrieved_sources: List[MinimalSource]

    def __str__(self) -> str:
        """Return the retrieved source locations as text."""
        result = ""
        for search in self.retrieved_sources:
            result += (
                f"{search.file_path} [{search.first_character_index}:"
                f"{search.last_character_index}]\n"
            )
        return result

    def __rich__(self) -> str | dict[str, Any]:
        """Return the retrieved source locations as text .
        for print function
        """
        result = f"{self}"
        return result


class MinimalAnswer(MinimalSearchResults):
    """Represent retrieved sources together with a generated answer."""
    answer: str

    def __str__(self) -> str:
        return f"{self.model_dump(mode="json")}"

    def __rich__(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class StudentSearchResults(BaseModel):
    """Represent retrieval results for a complete student search."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent retrieval results and answers for a complete dataset."""
    search_results: List[MinimalAnswer]
    k: int
