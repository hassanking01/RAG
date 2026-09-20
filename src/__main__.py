import fire
from rich.traceback import install
import bm25s

install()

class RAG:
    def __init__(self):
        self.raw_path = "/home/hahchtar/Desktop/student/RAG/data/raw/vllm-0.10.1"
        self.bm25 = bm25s.BM25()

    def index(self, max_chunk_size: int = 2000):

        print("index", max_chunk_size)

    def search(self, query: str, k: int = 10):
        print("search", query, k)

if "__main__" == __name__:
    result = fire.Fire(RAG)
