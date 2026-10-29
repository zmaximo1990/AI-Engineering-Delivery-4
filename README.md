# Code Delivery for Module #4 | Scalable RAG with Pinecone

Async Python pipeline that answers a question from a hybrid index. Files in `data/` (PDF, DOCX, Markdown, JSON, CSV, and plain text) are converted to Markdown, split into token-sized chunks, and embedded once. A LangChain LCEL chain fuses BM25 with vector similarity, asks the model to answer only from that context, and attaches the source filenames itself.

## Flows

Two passes, at a large scale. Each list links to its sequence diagram.

### Ingestion

Sequence diagram: [`diagrams/ingestion.mermaid`](diagrams/ingestion.mermaid)

1. The user starts the process with `python main.py`.
2. Configuration is loaded and the `Ingestor` is created, with its extractor and chunkener.
3. The vector index is opened or created (Pinecone or Chroma), together with the embedding model.
4. Existing chunks in the index are counted.
5. If the index already has data, ingestion stops. The files are not embedded again.
6. If the index is empty, supported files are listed. Text extraction is unified to Markdown, with page-by-page segmentation and consumption first when the format allows it. A format without pages becomes one segment page. Chunking runs on each `DocumentPage` after it has been converted to Markdown. Before the split, page-header lines that contain `Pág.` and a page number are removed, the `■` extraction artifact is dropped, extra blank lines collapse to one, and repeated spaces or tabs collapse to a single space. Line breaks stay so chunks can still break on paragraphs. Each chunk gets an id and metadata.
7. Chunks are embedded in batches and written to the store.
8. The number of chunks written is returned.

### Ask

Sequence diagram: [`diagrams/ask.mermaid`](diagrams/ask.mermaid)

1. The user asks a question.
2. The question is rejected when it is empty, and the LLM client is created.
3. The retriever is built. Documents in `data/` are read and chunked again, without writing the index. That corpus feeds an in-memory BM25 index and a vector retriever over the persisted embeddings.
4. Both retrievers are combined in an ensemble with equal weights (0.5 / 0.5).
5. On the question, BM25 and vector search run in parallel. The question is embedded before the similarity search.
6. The two rankings are fused with Reciprocal Rank Fusion. At most 5 fragments are kept.
7. Those fragments become the prompt context and are sent to the LLM as a structured `LLMAnswer`.
8. Empty, invalid, or truncated output is retried up to `RETRY_ATTEMPTS`.
9. The response is assembled from chunk metadata and returned: answer, sources, and fragment count.

### Main

The `main.py` invoke `get_rag_response()` after user enters a question, and it returns a `RAGAnswer`:

- `answer`: the model answer. If the context does not contain it, the model says it does not have access to that information.
- `sources`: filenames of the retrieved chunks. These come from document metadata, not from the model.
- `retrieved_fragments`: how many chunks were passed into the prompt. The hybrid retriever keeps at most 5.

The chain adds keys as it goes:

1. A lambda on `RunnablePassthrough.assign` stores the retrieved `documents`.
2. A second assign formats those documents into `context`.
3. `build_chain` calls `.with_structured_output(LLMAnswer)` (there is no `{formato}` variable), then checks the payload and retries truncated output.
4. A final `RunnableLambda` builds the `RAGAnswer` and its source list.

The hybrid retriever builds a BM25 index in memory from the chunked corpus and a `VectorStoreRetriever` over the persisted embeddings. `EnsembleRetriever` fuses the two rankings with equal weights (0.5 / 0.5) using Reciprocal Rank Fusion.

The default vector store is Pinecone Serverless (`aws` / `us-east-1`, cosine). Set `VECTOR_STORE=chroma` to use a local Chroma collection instead. Indexing and search share one embedding model. The default is `text-embedding-3-small` (1536 dimensions). A Pinecone index created at another dimension is refused. A non-empty namespace is left as-is on the next run.

The default chat provider is OpenAI (`gpt-4o-mini`). Pass `-p anthropic` or `-p gemini` to switch. Model names follow the Module 2 delivery (`claude-haiku-4-5-20251001`, `models/gemini-3.8-flash`). Gemini runs with at least `max_tokens=1024` so thinking tokens are less likely to truncate the object.

## Requirements

- **Python 3.12.13**, or Docker with Compose
- `PINECONE_API_KEY` when `VECTOR_STORE=pinecone` (the default)
- An API key for the chat provider you call:
  - `OPENAI_API_KEY` when `-p openai` (the default)
  - `ANTHROPIC_API_KEY` when `-p anthropic`
  - `GEMINI_API_KEY` when `-p gemini`

The default embeddings are OpenAI `text-embedding-3-small` (1536 dimensions) and need `OPENAI_API_KEY`. `EMBEDDING_PROVIDER=huggingface` uses `sentence-transformers/all-MiniLM-L6-v2` locally (384 dimensions) and does not use an API key.

## Setup

From this directory, with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Do NOT use "pip install --upgrade pip": it downloads the new pip through the old, vulnerable one.
# Install the pinned pip 26.2.2 wheel with a SHA256 check instead (eMudhra/emSign CA incident).
# See: https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576
python -m pip install "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
pip install -r requirements.txt
```

Copy the env template and set `PINECONE_API_KEY` plus the key for the chat provider you want to call. `.env` is gitignored.

```bash
cp .env.dist .env
```

The first query calls OpenAI to embed with `text-embedding-3-small`. `EMBEDDING_PROVIDER=huggingface` downloads `sentence-transformers/all-MiniLM-L6-v2` instead, if it is not already cached.

## Replicate the Pinecone index

`python main.py` opens the store before the prompt. `PineconeStoreManager` creates the Serverless index when `INDEX_NAME` does not exist, then upserts `data/` when that namespace is empty. Defaults:

| Setting | Default |
|---------|---------|
| `INDEX_NAME` | `delivery4-index` |
| `NAMESPACE` | `delivery4` |
| Metric | cosine |
| Cloud / region | `aws` / `us-east-1` |
| Dimension | 1536 (`text-embedding-3-small`). The live model must match `EMBEDDING_DIMENSION`. |

Each vector stores the chunk text in metadata under `text`, plus `source`, `doc_id` (the leading `NE-100` id, or the file stem), `format`, `chunk_index`, `char_count`, and `page` when the file has pages.

To build the index again, empty the namespace (or delete the index in the Pinecone console) and run `python main.py` once more. A namespace that already has vectors is skipped. If you change the embedding model, use a new `INDEX_NAME`: an existing index whose dimension does not match the model is rejected.

For a local index instead of Pinecone:

```bash
# in .env
VECTOR_STORE=chroma
```

Delete `vectorstore/` to reindex that collection.

## Project layout

| File | Role |
|------|------|
| `main.py` | Asks a question, runs one question, runs the demo, or scores both sets with `--eval` / `-e` |
| `samples.py` | Grounded question and trap question |
| `chain.py` | LCEL chain, `build_chain`, and `get_rag_response()` |
| `prompts.py` | Grounded system prompt. Context and question are template variables |
| `ingestion.py` | Lists `data/`, chunks each Markdown `DocumentPage`, and skips a populated index |
| `extraction.py` | `AssetExtractor`: PDF, DOCX, Markdown, JSON, CSV, and plain text to Markdown `DocumentPage`s, page by page when the format has pages |
| `chunking.py` | `Chunkener`: clean text (page headers, `■`, extra breaks, repeated spaces), then split at 500 tokens with 50 of overlap |
| `retriever.py` | BM25 plus vector search, fused by `EnsembleRetriever` |
| `vectorstore.py` | Pinecone and Chroma backends, upsert, and `as_retriever` |
| `embeddings.py` | Hugging Face and OpenAI embedding clients |
| `llm.py` | `OpenAIClient`, `AnthropicClient`, and `GeminiClient` |
| `factory.py` | Builds a chat client, an embeddings client, or the process-wide vector store |
| `schemas.py` | `LLMAnswer`, `RAGAnswer`, `DocumentPage`, providers, and `LLMConfig` |
| `settings.py` | Proyect settings loaded from the environment (`.env` file) like chunk size, `top_k`, Pinecone names, paths, and retry settings, loaded from the environment through `Settings` |
| `data/` | Nimbus engineering notes (PDF, DOCX, Markdown, JSON, and CSV) used as the corpus |
| `tests/evaluate.py` | Precision@k, Recall@k, MRR@k, Hit@k, and abstention on the unanswerable set |
| `tests/golden_set.json` | 33 questions with the source document ids used for Precision@k and Recall@k |
| `tests/unanswerable_set.json` | 10 questions the corpus does not answer, plus hallucination patterns |
| `diagrams/` | Sequence diagrams for ingestion and ask flows |
| `requirements.txt` | Dependencies |
| `.env.dist` | Env template with empty secrets |
| `Dockerfile` | Image with dependencies and the local embedding model |
| `docker-compose.yaml` | Runs `main.py` with `.env` and a mounted `vectorstore/` |
| `.dockerignore` | Keeps secrets and the local virtualenv out of the image |

## JSON to Markdown example

A JSON file becomes one Markdown page. A scalar is a bold label, and underscores in the key become spaces. A list of strings becomes bullets. A list of objects becomes sections: the heading is the first non-empty string among `term`, `title`, `name`, `heading`, and `id`, and that field is left out of the body. `null`, `""`, and `[]` are dropped.

```json
{
  "doc_code": "NE-700",
  "entries": [
    {
      "term": "nimbusd",
      "id": "GLS-001",
      "aliases": ["agente nimbusd"],
      "related_params": []
    }
  ]
}
```

```markdown
**doc code:** NE-700

## entries

### nimbusd

**id:** GLS-001

#### aliases

- agente nimbusd
```

## CSV to Markdown example

A CSV file becomes one Markdown page. The first non-empty row is the header. The delimiter is detected among comma, semicolon, and tab. Each data row is a section, and the fields of that row stay on consecutive lines so chunking can break between rows. The heading is the first non-empty cell in a column named `term`, `title`, `name`, `heading`, or `id`, ignoring capitalization, and that column is left out of the body. Blank cells are dropped. A cell that spans several lines is kept on one line. A file with no data rows produces no page.

```csv
id,version,fecha_lanzamiento,funcionalidad,parametros_relacionados,notas
NE800-001,4.0.0,2025-09-15,Particiones A/B con rollback automático,,La actualización se escribe siempre en la partición inactiva.
```

```markdown
## NE800-001

**version:** 4.0.0
**fecha lanzamiento:** 2025-09-15
**funcionalidad:** Particiones A/B con rollback automático
**notas:** La actualización se escribe siempre en la partición inactiva.
```

## Usage

Run from this directory. The index is opened before the prompt. With no arguments the program then reads questions from the terminal. An empty line, or the word `exit` or `quit` (any capitalization), stops the prompt.

```bash
python main.py

python main.py -p anthropic
python main.py --provider gemini

# One question, then exit
python main.py -q "¿Cuántos segundos hay que mantener presionado el botón RESET para restablecer de fábrica el gateway?"

# Two checks from samples.py, then exit
python main.py --demo

# Golden set and unanswerable set, then exit
python main.py --eval
python main.py -e -p anthropic
```

`--eval` cannot be combined with `--demo` or `--question`.

### Evaluation

`tests/evaluate.py` scores the hybrid retriever and the grounded answers. `python main.py --eval` (or `-e`) prepares the index, prints the report, and exits. The same report is available directly:

```bash
python tests/evaluate.py
python tests/evaluate.py -k 5
python tests/evaluate.py -p gemini
```

`main.py --eval` keeps `hybrid_k` chunks per question (5 by default). `-k` is only on `tests/evaluate.py`. The golden set does not call the chat model. The unanswerable set does, so it needs the key for the selected provider.

#### Golden set

`tests/golden_set.json` has 33 questions. Each one lists the source documents that contain the answer (`relevant_docs`, such as `NE-100`). Questions whose answer is in the version matrix also list `NE-800`. The hybrid retriever returns the top k chunks.

| Metric | Meaning |
|--------|---------|
| Precision@k | Share of the k retrieved chunks that come from an expected document |
| Recall@k | Share of the expected documents that appear among those chunks. With one expected document this is 1 when that document is in the top k, and 0 otherwise |
| MRR@k | Mean reciprocal rank. Each question contributes `1 / rank` of its first relevant chunk in the top k, or 0 when that chunk is missing. Rank 1 is 1.00 and rank 2 is 0.50 |
| Hit@k | Questions that retrieved at least one expected document |

Each question prints `P@k`, `R@k`, `RR@k` (that question's reciprocal rank), the retrieved document ids, and the expected ids. The report then prints the mean Precision@k, the mean Recall@k, MRR@k, and Hit@k.

#### Unanswerable set

`tests/unanswerable_set.json` has 10 questions the corpus does not answer. Each question runs through the RAG chain. A case passes when both of these hold:

- The answer says the information is not in the available documentation.
- The answer matches none of that case's hallucination patterns, such as an invented price, phone number, or USB version.

The report prints abstention, hallucination, and pass or fail for each question, then the totals. A failing answer is followed by a short excerpt.

### Docker

Create a `.env` first. Compose passes it into the container. `./vectorstore` is mounted so a Chroma index survives the container. Pinecone keeps the index in the cloud; the mount is only used when `VECTOR_STORE=chroma`.

```bash
docker compose build
docker compose run --rm app
docker compose run --rm app python main.py --demo
docker compose run --rm app python main.py --eval
docker compose run --rm app python main.py -p gemini
```

The image downloads `sentence-transformers/all-MiniLM-L6-v2` during the build.

## Example

A question about the factory reset should come back with the figure from `NE-100_Guia_Instalacion_GX-200.pdf`: hold RESET for 10 seconds and release it when the ERR LED flashes red three times. That file should appear in `sources`.

A question about the CEO's salary is not in the corpus. The answer has to say that the information is not available. It must not invent a number. `sources` still lists the chunks that were retrieved, because those are the fragments the model saw.
