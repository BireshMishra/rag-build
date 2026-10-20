# rag-build

Small retrieval pipeline: PDFs → token-based chunks → Voyage embeddings → Chroma → similarity search.

## Setup

```
uv sync
```

Put your key in `src/rag_build/.env`:

```
VOYAGE_API_KEY=...
```

## Usage

1. Chunk PDFs. Writes `dataset/dataset_<pdf name>.jsonl` (created if missing) with one chunk per line. Pages and chunk ids are 1-based.

   ```
   uv run python src/rag_build/ingest_pdf.py RAGPdfs
   ```

2. Embed and query. Only chunks whose id (`file:page:chunk_id`) is not already in the collection are sent to Voyage, so re-runs are free.

   ```
   uv run python src/rag_build/embed.py "your question" dataset --top-k 5
   ```

   Options: `--db-path` (default `./chroma_db`), `--collection` (default `my_voyage_collection`), `--top-k` (default 20).

Chunks are keyed by id, so an existing chunk whose text changed is not re-embedded. Delete the collection directory to rebuild.

## Tests

```
uv run pytest
```

The tests use a fake embedding function and need no network or API key.

## Example run

Three papers from `RAGPdfs/` (`2026.aimecon-main.1`, `.2`, `.3`), ingested and queried with `"How can LLMs be used to generate test questions?"`:

```
dataset\dataset_2026.aimecon-main.1.jsonl: 17 chunks, 17 new, 0 already stored
dataset\dataset_2026.aimecon-main.2.jsonl: 16 chunks, 16 new, 0 already stored
dataset\dataset_2026.aimecon-main.3.jsonl: 48 chunks, 48 new, 0 already stored
collection 'my_voyage_collection' holds 81 chunks
```

`collection.count()` is 81. A second run reported `0 new` for all three files.

Top 5 (text shortened):

| # | id | distance | starts with |
|---|----|----------|-------------|
| 1 | `2026.aimecon-main.1:1:2` | 0.5087 | "...AI-enabled assessment environments, where large language models continuously generate items..." |
| 2 | `2026.aimecon-main.1:1:1` | 0.5470 | "Proceedings of the ... AIME-Con ..." |
| 3 | `2026.aimecon-main.1:5:3` | 0.5473 | "...selecting items for computerized adaptive tests..." (references) |
| 4 | `2026.aimecon-main.3:9:1` | 0.5494 | "...opportunities and challenges of large language models for education..." (references) |
| 5 | `2026.aimecon-main.1:5:2` | 0.5673 | "Acknowledgments and Declaration on the Use of Generative AI..." |

Only the first hit is clearly on topic. Several results are reference lists and front matter, which suggests chunking should drop references and boilerplate. With only three papers indexed, the weak ranking may also be down to corpus size.
