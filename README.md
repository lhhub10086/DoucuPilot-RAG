# DocuPilot-RAG

DocuPilot-RAG is an Agentic RAG and automatic evaluation system for technical documents. The current version includes the FastAPI application shell, document parsing/chunking, local retrieval indexes, and real reranker-based citation ordering.

## Stage 1 Scope

- Create the `docupilot-rag/` project structure.
- Provide a FastAPI app entrypoint in `main.py`.
- Implement `GET /health`.
- Add environment-based configuration in `app/core/config.py`.
- Add application logging in `app/core/logging.py`.
- Add initial Pydantic schemas in `app/core/schemas.py`.
- Add `.env.example` and `requirements.txt`.

## Stage 2 Scope

- Parse PDF files with PyMuPDF.
- Parse Markdown files while preserving heading sections.
- Parse txt files by paragraph.
- Generate sliding-window chunks.
- Generate section-based chunks.
- Save parsed documents to `data/parsed/`.
- Save generated chunks to `data/parsed/chunks/`.
- Provide `scripts/ingest_docs.py` for command-line ingestion.

## Stage 3 Scope

- Build a local vector index with `sentence-transformers`.
- Build a BM25 keyword index with `rank-bm25` and `jieba`.
- Persist indexes under `data/indexes/<chunk_strategy>/`.
- Provide `IndexManager` for vector, BM25, and hybrid RRF search.
- Provide naive and hybrid retrievers.
- Provide `scripts/ask_cli.py` for local question answering with citations.
- Rebuild stale indexes automatically when the embedding provider, model, or index version changes.

## Stage 4 Scope

- Retrieve hybrid candidates first.
- Rerank candidates with a real `sentence-transformers` CrossEncoder.
- Default reranker model: `BAAI/bge-reranker-base`.
- Return both `retrieved_chunk_ids` and `reranked_chunk_ids`.
- Include `section`, retrieval `score`, and `rerank_score` in each citation.
- Reranker sorting is metadata-aware: exact file-name matches and document-level priors affect ordering, while `rerank_score` remains the raw CrossEncoder score.
- Low-confidence rerank results return the conservative answer `当前文档中未找到充分依据。`.

Later stages will add automatic evaluation, failure analysis, Docker, and full documentation.

## Stage 5 Scope

Stage 5 adds Agentic RAG with LangGraph. Unlike `naive`, `hybrid`, and `rerank`, the `agentic` strategy keeps an explicit workflow state and records its execution route.

Workflow:

```text
classify_question
  -> retrieve_documents
  -> grade_documents
    -> passed: rerank_documents -> generate_answer -> grade_answer
    -> failed: rewrite_query -> retrieve_documents -> grade_documents
      -> passed: rerank_documents -> generate_answer -> grade_answer
      -> failed: fallback_response
```

Implemented nodes:

- `classify_question`: rule-based question type classification: `fact`, `summary`, `compare`, `method`, `unknown`.
- `retrieve_documents`: hybrid retrieval with original or rewritten query.
- `grade_documents`: rule-based retrieval quality grading.
- `rewrite_query`: one-time rule-based query rewrite.
- `rerank_documents`: reuses the existing CrossEncoder reranker.
- `generate_answer`: reuses the citation answer builder.
- `grade_answer`: rule-based evidence support check.
- `fallback_response`: returns `当前文档中未找到充分依据。`.

Run:

```bash
python scripts/ask_cli.py --question "文档主要讲了什么" --strategy agentic --chunk_strategy section
python scripts/ask_cli.py --question "这篇文档的核心方法是什么？" --strategy agentic --chunk_strategy section
python scripts/ask_cli.py --question "这个文档有没有提到火星移民计划？" --strategy agentic --chunk_strategy section
```

Agentic response fields:

- `answer`
- `citations`
- `strategy`
- `latency_ms`
- `retrieved_chunk_ids`
- `reranked_chunk_ids`
- `rewritten_question`
- `route_trace`
- `failure_reason`

Example route trace:

```json
[
  "classify_question: method",
  "retrieve_documents: 15 candidates",
  "grade_documents: passed",
  "rerank_documents: ... -> ...",
  "generate_answer",
  "grade_answer: supported"
]
```

Fallback example:

```json
{
  "answer": "当前文档中未找到充分依据。",
  "citations": [],
  "strategy": "agentic",
  "failure_reason": "low_retrieval_confidence"
}
```

Current Stage 5 limitations:

- Uses rule-based graders, not LLM graders.
- Uses rule-based query rewrite, not LLM rewrite.
- DeepEval/RAGAS evaluation is not implemented yet.
- FastAPI `/ask` is not implemented yet.
- MCP tooling is not part of this stage.

## Project Structure

```text
docupilot-rag/
├── app/
│   ├── api/
│   ├── core/
│   ├── parser/
│   ├── chunker/
│   ├── indexer/
│   ├── retriever/
│   ├── llm/
│   ├── agent/
│   ├── evaluator/
│   └── utils/
├── data/
│   ├── raw/
│   ├── parsed/
│   ├── indexes/
│   └── eval_results/
├── eval_sets/
├── scripts/
├── tests/
├── docs/
├── logs/
├── .env.example
├── requirements.txt
└── main.py
```

## Setup

```bash
cd C:\Users\admin\Desktop\llm-intern-projects\项目1\docupilot-rag
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Run

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

FastAPI docs will be available at:

```text
http://127.0.0.1:8000/docs
```

## API

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

## Document Parsing And Chunking

Put source files into `data/raw/`, then run one of the following commands:

```bash
python scripts/ingest_docs.py --input data/raw --chunk_strategy sliding
python scripts/ingest_docs.py --input data/raw --chunk_strategy section
```

Expected output:

```text
parsed documents: N
generated chunks: M
```

Supported first-version formats:

- PDF
- Markdown
- txt

Parsed JSON files are written to `data/parsed/`. Chunk JSON files are written to `data/parsed/chunks/<strategy>/`. Each chunk includes `chunk_id`, `doc_id`, text, source file name, page number, section, and chunk type.

## Retrieval CLI

Run ingestion first, then ask with one of the Stage 3 strategies:

```bash
python scripts/ingest_docs.py --input data/raw --chunk_strategy section
python scripts/ask_cli.py --question "文档主要讲了什么" --strategy naive --chunk_strategy section
python scripts/ask_cli.py --question "文档主要讲了什么" --strategy hybrid --chunk_strategy section
python scripts/ask_cli.py --question "文档主要讲了什么" --strategy rerank --chunk_strategy section
```

The CLI automatically builds indexes when `data/indexes/<chunk_strategy>/` does not exist. The response includes:

- `answer`
- `citations`
- `strategy`
- `latency_ms`
- `retrieved_chunk_ids`
- `reranked_chunk_ids`

Current embedding configuration:

- Provider: `sentence-transformers`
- Default model: `BAAI/bge-small-zh-v1.5`
- Vector dimension: `512`
- Dependency: `sentence-transformers`

The first CLI run may download the configured embedding model from Hugging Face. Tests use a deterministic hashing provider only to keep the unit test suite fast and offline-friendly.

Current reranker configuration:

- Provider: `sentence-transformers` CrossEncoder
- Default model: `BAAI/bge-reranker-base`
- Dependency: `sentence-transformers`

The first `--strategy rerank` run may download the configured reranker model from Hugging Face.

## Configuration

All model names, API endpoints, chunk parameters, and feature flags should be configured through `.env`. Do not hard-code private keys in source code.

Key variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `RERANKER_MODEL`
- `VECTOR_STORE`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K`
- `RRF_K`
- `ENABLE_RERANK`
- `ENABLE_AGENTIC_RAG`

## Development Plan

Stage 1 is complete when `uvicorn main:app --reload` starts and `GET /health` returns `{"status":"ok"}`.

Stage 2 is complete when:

```bash
python scripts/ingest_docs.py --input data/raw --chunk_strategy sliding
python scripts/ingest_docs.py --input data/raw --chunk_strategy section
pytest tests/
```

run successfully for supported sample documents.

Stage 3 is complete when:

```bash
python scripts/ask_cli.py --question "文档主要讲了什么" --strategy naive
python scripts/ask_cli.py --question "文档主要讲了什么" --strategy hybrid
pytest tests/
```

return answers, citations, retrieved chunk ids, and passing tests.

Stage 4 is complete when:

```bash
python scripts/ingest_docs.py --input data/raw --chunk_strategy section
python scripts/ask_cli.py --question "文档主要讲了什么" --strategy naive --chunk_strategy section
python scripts/ask_cli.py --question "文档主要讲了什么" --strategy hybrid --chunk_strategy section
python scripts/ask_cli.py --question "文档主要讲了什么" --strategy rerank --chunk_strategy section
pytest tests
```

The rerank response must include `retrieved_chunk_ids`, `reranked_chunk_ids`, and citations with `score` plus `rerank_score`.

Additional Stage 4 checks:

```bash
# Rebuild indexes from scratch.
python scripts/ingest_docs.py --input data/raw --chunk_strategy section
python scripts/ask_cli.py --question "sample.md 中提到了什么内容" --strategy rerank --chunk_strategy section

# Real PDF retrieval/rerank check.
python scripts/ask_cli.py --question "这篇文档的核心方法是什么？" --strategy naive --chunk_strategy section
python scripts/ask_cli.py --question "这篇文档的核心方法是什么？" --strategy hybrid --chunk_strategy section
python scripts/ask_cli.py --question "这篇文档的核心方法是什么？" --strategy rerank --chunk_strategy section

# Conservative no-answer check.
python scripts/ask_cli.py --question "这个文档有没有提到火星移民计划？" --strategy rerank --chunk_strategy section
```
