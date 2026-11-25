*This project has been created as part of the 42 curriculum by hahchtar.*

# Retrieval-Augmented Generation

## Description

This project is a Retrieval-Augmented Generation (RAG) system. It answers questions about a local collection of source files by retrieving relevant pieces of those files and giving them to a language model as context.

The project is designed to answer questions about the files placed in `data/raw`. It supports Python, Markdown, and plain-text files. The system combines two retrieval approaches:

- lexical retrieval with a custom BM25 implementation;
- semantic retrieval with Sentence Transformers embeddings and cosine similarity.

The two ranked result lists are combined with Reciprocal Rank Fusion (RRF). The resulting context is passed to `Qwen/Qwen3-0.6B`, which generates an answer using only the retrieved text.

## System architecture

The pipeline is split into indexing and question answering:

```text
Files in data/raw
        |
        v
Document chunker
        |
        +--> chunks/chunks.json
        |         |
        |         +--> BM25 lexical index
        |         |
        |         +--> Sentence Transformer embeddings
        |                   |
        |                   +--> semantic index
        |
User question --> BM25 search ------+
                 semantic search ----+--> Reciprocal Rank Fusion
                                               |
                                               v
                                      Retrieved source chunks
                                               |
                                               v
                                      Qwen language model
                                               |
                                               v
                                             Answer
```

The main components are:

- `src/chunker.py`: segments supported files and stores character offsets.
- `src/lexical.py`: tokenizes chunks, removes stopwords, and ranks them with BM25.
- `src/semantic.py`: creates embeddings with `all-MiniLM-L6-v2` and ranks chunks using cosine similarity.
- `src/Rag.py`: builds or updates indexes, combines retrieval results with RRF, and creates answers.
- `src/LLM.py`: loads Qwen and generates an answer from the retrieved context.
- `src/api.py`: exposes an optional FastAPI endpoint.

## Chunking strategy

The default maximum chunk size is 2,000 characters and the allowed range is 200–2,000 characters.

- Python files are parsed with the Python AST. Top-level statements are kept together where possible. Large statements are split by lines and receive a small overlap of about 5% of the maximum chunk size.
- Markdown files are split at headings when the current section is large enough. Code blocks are protected so headings inside fenced code are not treated as document headings.
- Text files are split by lines until the maximum character size is reached.
- Every chunk stores its source file and character offsets. This allows the original text to be recovered for retrieval results and answer generation.

The indexer also records modification times and chunk hashes. When files change, only affected files and their indexes are reprocessed when possible.

## Retrieval method

For every query, the system performs two searches.

### BM25 retrieval

The lexical index uses lower-cased tokens, stopword removal, simple identifier handling, term frequency, inverse document frequency, and document-length normalization. It uses the standard BM25 parameters `k1=1.5` and `b=0.75`.

BM25 is useful when a query contains an exact function name, API name, filename, or technical term.

### Semantic retrieval

The query and every chunk are encoded with `sentence-transformers/all-MiniLM-L6-v2`. The implementation calculates cosine similarity between the query vector and each stored chunk vector, then keeps the highest-scoring chunks.

Semantic retrieval is useful when the query and the source use different words but express a similar meaning.

### Ranking combination

The two ranked lists are combined with Reciprocal Rank Fusion. For a result at rank `r`, the contribution is:

```text
1 / (60 + r)
```

Scores from both retrieval methods are added for the same chunk, and the final list is sorted by the combined score. The top `k` chunks are then used as the language-model context.

## Performance analysis

The project includes an `evaluate` command that measures recall at `k = 1, 3, 5, and 10`. A retrieved chunk counts as matching a reference source when their character ranges have an intersection-over-union of at least 0.05. Recall is then averaged over the reference sources for each question.

Run evaluation with:

```bash
uv run -m src evaluate \
  --student_search_results_path results/search.json \
  --dataset_path data/dataset.json
```

The command prints `Recall@1`, `Recall@3`, `Recall@5`, and `Recall@10`. The actual scores depend on the documents and evaluation dataset used, so they should be reported from the output of this command rather than assumed in advance.

The semantic search currently compares the query with all stored vectors, which is simple and appropriate for a small project but becomes slower as the number of chunks grows. BM25 uses an inverted token index and avoids comparing against every chunk for every query. Index files are persisted under `data/processed`, and unchanged files can be reused to reduce indexing time.

## Design decisions

- Hybrid retrieval was selected because exact keyword matching and semantic similarity solve different failure cases.
- Reciprocal Rank Fusion was selected because BM25 and cosine similarity have different score scales; combining ranks avoids requiring score calibration.
- Character offsets are stored instead of duplicating complete chunk text, reducing duplication and allowing the original file content to be used as context.
- Separate persisted lexical and semantic indexes make repeated searches faster than rebuilding the indexes for every query.
- A small local language model is used so the project can run without a hosted LLM API. This reduces external dependencies but requires model files and enough local memory.
- Incremental re-indexing uses file modification times and chunk identifiers to update changed files instead of always rebuilding everything.

## Challenges faced

- Keeping chunks useful for different file types required different strategies for Python syntax, Markdown headings, fenced code, and plain text.
- Combining lexical and semantic results required a ranking method that does not directly compare incompatible score values; RRF provides that normalization by rank.
- Updating indexes safely when a file is added, modified, or deleted required tracking each file's chunk identifiers.
- Embeddings and vector search were initially difficult concepts. Documentation, tutorials, and AI explanations were used to understand how text becomes vectors and how similarity search works.
- Retrieval quality depends heavily on chunk size, overlap, the embedding model, and the evaluation dataset. The evaluation command makes those trade-offs measurable with recall@k.

## Instructions

### Requirements

- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Enough disk space and memory to download and run the embedding and language models

### Installation

Install the dependencies with:

```bash
make install
```

Alternatively:

```bash
uv sync
```

Put the documents to be indexed in `data/raw`. Supported extensions are `.py`, `.md`, and `.txt`.

### Build the indexes

```bash
uv run -m src index
```

You can choose a different maximum chunk size:

```bash
uv run -m src index --max_chunk_size 1000
```

### Search for relevant chunks

```bash
uv run -m src search "How does the scheduler work?" --k 5
```

### Ask a question and generate an answer

```bash
uv run -m src answer "How does the scheduler work?" --k 5
```

### Run the API

```bash
make server
```

The API exposes `GET /get_response` and accepts a JSON body containing a `question` field. For example:

```bash
curl -X GET http://127.0.0.1:8000/get_response \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does the scheduler work?"}'
```

### Other Make targets

```bash
make run       # run the CLI
make lint      # run flake8 and mypy
make clean     # remove Python cache directories
```

## Resources

1. [Retrieval Augmented Generation — DeepLearning.AI](https://www.deeplearning.ai/courses/retrieval-augmented-generation) — course used to study RAG architecture, chunking, retrieval, hybrid search, evaluation, and deployment considerations.
2. [Okapi BM25 — Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25) — reference for the BM25 ranking function used by the lexical retriever.
3. [Semantic Search — Sentence Transformers](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) — reference for embeddings, semantic search, and cosine-similarity retrieval.
4. [RecursiveCharacterTextSplitter — LangChain reference](https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter) — useful reference for chunking and preserving meaningful text boundaries.
5. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — research paper describing the RAG approach of combining retrieval with generation.
6. [YouTube search: RAG and AI chatbot tutorials](https://www.youtube.com/results?search_query=RAG+AI+chatbot+embeddings+BM25) — video tutorials used for practical explanations of AI chatbots, embeddings, and retrieval systems.

### Use of AI

AI was used as a learning and writing aid. It helped explain difficult concepts such as embeddings, vector databases, semantic similarity, BM25, and the interaction between retrieval and generation. It was also used to help draft and organize this README. The project implementation, testing, design choices, and final verification remain the responsibility of the author.
