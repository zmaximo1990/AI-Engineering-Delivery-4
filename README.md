# Code Delivery for Module #4 | Scalable RAG with Pinecone

Async Python pipeline that answers a question from a hybrid index. Files in `data/` (PDF, DOCX, Markdown, and plain text) are converted to Markdown, split into token-sized chunks, and embedded once. Extraction unifies every supported format to Markdown. When the format has pages, the file is segmented and consumed page by page first; a format without pages becomes a single segment. Each segment is a `DocumentPage`. Chunking runs on that Markdown page, not on the raw file. A LangChain LCEL chain fuses BM25 with vector similarity, asks the model to answer only from that context, and attaches the source filenames itself.

`get_rag_response()` returns a `RAGResponse`:

- `answer`: the model answer. If the context does not contain it, the model says it does not have access to that information.
- `sources`: filenames of the retrieved chunks. These come from document metadata, not from the model.
- `retrieved_fragments`: how many chunks were passed into the prompt. The hybrid retriever keeps at most 5.

The chain adds keys as it goes:

1. A lambda on `RunnablePassthrough.assign` stores the retrieved `documents`.
2. A second assign formats those documents into `context`.
3. `build_chain` calls `.with_structured_output(LLMAnswer)` (there is no `{formato}` variable), then checks the payload and retries truncated output.
4. A final `RunnableLambda` builds the `RAGResponse` and its source list.

The hybrid retriever builds a BM25 index in memory from the chunked corpus and a `VectorStoreRetriever` over the persisted embeddings. `EnsembleRetriever` fuses the two rankings with equal weights (0.5 / 0.5) using Reciprocal Rank Fusion.

The default vector store is Pinecone Serverless (`aws` / `us-east-1`, cosine). Set `VECTOR_STORE=chroma` to use a local Chroma collection instead. Indexing and search share one embedding model. The default is `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). A Pinecone index created at another dimension is refused. A non-empty namespace is left as-is on the next run.

The default chat provider is OpenAI (`gpt-4o-mini`). Pass `-p anthropic` or `-p gemini` to switch. Model names follow the Module 2 delivery (`claude-haiku-4-5-20251001`, `models/gemini-3.8-flash`). Gemini runs with at least `max_tokens=1024` so thinking tokens are less likely to truncate the object.

## Requirements

- **Python 3.12.13**, or Docker with Compose
- `PINECONE_API_KEY` when `VECTOR_STORE=pinecone` (the default)
- An API key for the chat provider you call:
  - `OPENAI_API_KEY` when `-p openai` (the default)
  - `ANTHROPIC_API_KEY` when `-p anthropic`
  - `GEMINI_API_KEY` when `-p gemini`

The default embeddings run locally and do not use an API key. `EMBEDDING_PROVIDER=openai` uses `text-embedding-3-small` (1536 dimensions) and needs `OPENAI_API_KEY`.

## Setup

From this directory, with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy the env template and set `PINECONE_API_KEY` plus the key for the chat provider you want to call. `.env` is gitignored.

```bash
cp .env.dist .env
```

The first query downloads `sentence-transformers/all-MiniLM-L6-v2` if it is not already cached.

## Replicate the Pinecone index

`python main.py` opens the store before the prompt. `PineconeStoreManager` creates the Serverless index when `INDEX_NAME` does not exist, then upserts `data/` when that namespace is empty. Defaults:

| Setting | Default |
|---------|---------|
| `INDEX_NAME` | `delivery4-index` |
| `NAMESPACE` | `delivery4` |
| Metric | cosine |
| Cloud / region | `aws` / `us-east-1` |
| Dimension | the live embedding model (384 for the default Hugging Face model) |

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
| `main.py` | Asks the user for a question, or runs one question / the two demo questions |
| `samples.py` | Grounded question and trap question |
| `chain.py` | LCEL chain, `build_chain`, and `get_rag_response()` |
| `prompts.py` | Grounded system prompt. Context and question are template variables |
| `ingestion.py` | Lists `data/`, chunks each Markdown `DocumentPage`, and skips a populated index |
| `extraction.py` | `AssetExtractor`: PDF, DOCX, Markdown, and plain text to Markdown `DocumentPage`s, page by page when the format has pages |
| `processing.py` | `Chunkener`: clean text, then split at 500 tokens with 50 of overlap |
| `retriever.py` | BM25 plus vector search, fused by `EnsembleRetriever` |
| `vectorstore.py` | Pinecone and Chroma backends, upsert, and `as_retriever` |
| `embeddings.py` | Hugging Face and OpenAI embedding clients |
| `llm.py` | `OpenAIClient`, `AnthropicClient`, and `GeminiClient` |
| `factory.py` | Builds a chat client or an embeddings client from config |
| `schemas.py` | `LLMAnswer`, `RAGResponse`, `DocumentPage`, providers, and `LLMConfig` |
| `settings.py` | Proyect settings loaded from the environment (`.env` file) like chunk size, `top_k`, Pinecone names, paths, and retry settings, loaded from the environment through `Settings` |
| `data/` | Nimbus engineering notes (PDF and DOCX) used as the corpus |
| `tests/golden_set.json` | Questions whose expected source document is known |
| `tests/unanswerable_set.json` | Questions the corpus does not answer |
| `diagrams/` | Sequence diagrams for ingestion and ask flows |
| `requirements.txt` | Dependencies |
| `.env.dist` | Env template with empty secrets |
| `Dockerfile` | Image with dependencies and the local embedding model |
| `docker-compose.yaml` | Runs `main.py` with `.env` and a mounted `vectorstore/` |
| `.dockerignore` | Keeps secrets and the local virtualenv out of the image |

## Usage

Run from this directory. The index is opened before the prompt. With no arguments the program then reads questions from the terminal. An empty line exits.

```bash
python main.py

python main.py -p anthropic
python main.py --provider gemini

# One question, then exit
python main.py -q "¿Cuántos segundos hay que mantener presionado el botón RESET para restablecer de fábrica el gateway?"

# Two checks from samples.py, then exit
python main.py --demo
```

### Docker

Create a `.env` first. Compose passes it into the container. `./vectorstore` is mounted so a Chroma index survives the container. Pinecone keeps the index in the cloud; the mount is only used when `VECTOR_STORE=chroma`.

```bash
docker compose build
docker compose run --rm app
docker compose run --rm app python main.py --demo
docker compose run --rm app python main.py -p gemini
```

The image downloads `sentence-transformers/all-MiniLM-L6-v2` during the build.

## Example

A question about the factory reset should come back with the figure from `NE-100_Guia_Instalacion_GX-200.pdf`: hold RESET for 10 seconds and release it when the ERR LED flashes red three times. That file should appear in `sources`.

A question about the CEO's salary is not in the corpus. The answer has to say that the information is not available. It must not invent a number. `sources` still lists the chunks that were retrieved, because those are the fragments the model saw.
