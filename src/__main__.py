import fire, json
from .chunker import Chunker
from rich.traceback import install
from pathlib import Path
from .models import MinimalSource, MinimalSearchResults, MinimalSource, StudentSearchResults, RagDataset, StudentSearchResultsAndAnswer, MinimalAnswer, AnsweredQuestion
from tqdm import tqdm
from .LLM import llm_model
from .lexical import BM25
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
        self.lexical.index(self.chunker.save_path)
        self.semantic.index()
    def search(self, query: str, k: int = 10):
        #self.semantic.search(query, k)
        self.lexical.search(query, k)
        exit()
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
        
        return MinimalAnswer(
            question=minimal_source.question,
            retrieved_sources=minimal_source.retrieved_sources,
            answer=answer
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
            json.dump(
                ragadataset_answers.model_dump(mode="json"),
                file,
                indent=4
            )
if "__main__" == __name__:
    result = fire.Fire(RAG)
