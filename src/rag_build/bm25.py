import bm25s
import chromadb
import argparse
import sys
from dotenv import load_dotenv

load_dotenv()


DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_COLLECTION = "my_voyage_collection"

def load_chunks(db_path: str = DEFAULT_DB_PATH, name: str = DEFAULT_COLLECTION):
    try:
        chroma_client = chromadb.PersistentClient(db_path)
        collection = chroma_client.get_collection(
            name=name
        )
        data = collection.get(include=["documents","metadatas"])
        ids, documents, metadatas = data["ids"], data["documents"], data["metadatas"]
        return ids, documents, metadatas
    except Exception as e:
        raise RuntimeError(f"could not load chunks from '{name}' in {db_path}: {e}. Run embed.py first.") from e


def build_index(documents: list[str]) -> bm25s.BM25:
    corpus_tokens = bm25s.tokenize(documents, lower=True)
    retriever = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens)
    return retriever


def search(retriever: bm25s.BM25, query: str, ids: list[str], documents: list[str], k: int = 20) -> list[dict]:
    k = min(k, len(ids))
    query_tokens = bm25s.tokenize(query, lower=True)
    indices, scores = retriever.retrieve(query_tokens, k=k)

    return [
        {"id": ids[i], "document": documents[i], "score": float(score)}
        for i, score in zip(indices[0], scores[0]) if score > 0
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="query to get tokens for")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="db path")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="collection")
    args = parser.parse_args()
    print(f"query arg is: {args.query}")
    try:
        ids, documents, metadatas = load_chunks(args.db_path, args.collection)
        retriever = build_index(documents)
        results = search(retriever, args.query, ids, documents)
    except Exception as error:
        sys.exit(f"error: {error}")
    for result in results:
        print(f"Score: {result['score']: .4f} | Document {result['document']}")