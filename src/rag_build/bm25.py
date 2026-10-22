import bm25s
import chromadb
import argparse
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
        print(f"Exception while loading chunks: {e}. Run embed.py first.")


def get_bm25_result(query: str, k: int = 2) -> list[dict]:
    ids, documents, metadatas = load_chunks()
    corpus_tokens = bm25s.tokenize(documents, lower=True)
    query_tokens = bm25s.tokenize(query, lower=True)

    retrievar = bm25s.BM25(method="lucene")
    retrievar.index(corpus_tokens)

    indices, scores = retrievar.retrieve(query_tokens, k=k)

    return [
        {"id": ids[i], "document": documents[i], "score": float(score)}
        for i, score in zip(indices[0], scores[0]) if score > 0
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="query to get tokens for")
    parser.add_argument("--db-path", type=str, help="db path")
    parser.add_argument("--collection", type=str, help="collection")
    args = parser.parse_args()
    print(f"query arg is: {args.query}")
    for result in get_bm25_result(args.query):
        print(f"Score: {result['score']: .4f} | Document {result['document']}")