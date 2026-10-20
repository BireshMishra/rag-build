import bm25s
import chromadb
import argparse
from dotenv import load_dotenv

load_dotenv()


DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_COLLECTION = "my_voyage_collection"

def get_collection(db_path: str = DEFAULT_DB_PATH, name: str = DEFAULT_COLLECTION, embedding_function=None):
    chroma_client = chromadb.PersistentClient(db_path)
    collection = chroma_client.get_collection(
        name=name
        )
    data = collection.get(include=["documents","metadatas"])
    ids, documents, metadatas = data["ids"], data["documents"], data["metadatas"]
    return ids, documents, metadatas


def get_bm25_result(query: str):
    print("getting collection")
    ids, documents, metadatas = get_collection()
    print("got collection")
    corpus_tokens = bm25s.tokenize(documents, lower=True)
    query_tokens = bm25s.tokenize(query, lower=True)

    retrievar = bm25s.BM25(method="lucene")
    retrievar.index(corpus_tokens)

    results, scores = retrievar.retrieve(query_tokens, corpus=documents, k=2)

    for doc, score in zip(results[0], scores[0]):
        print(f"Score: {score: .4f} | Document {doc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="query to get tokens for")
    args = parser.parse_args()
    print(f"query arg is: {args.query}")
    get_bm25_result(args.query)