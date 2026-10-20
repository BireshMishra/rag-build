import os
import chromadb
import argparse
import json
from chromadb.utils import embedding_functions
from pathlib import Path
from dotenv import load_dotenv
from cachetools import cached, TTLCache


load_dotenv()

cache_storage = TTLCache(maxsize=10000, ttl=600)

DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_COLLECTION = "my_voyage_collection"

def get_collection(db_path: str = DEFAULT_DB_PATH, name: str = DEFAULT_COLLECTION):
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_api_key:
        raise RuntimeError("VOYAGE_API_KEY is not set (add it to .env or the environment)")

    chroma_client = chromadb.PersistentClient(path=db_path)
    voyage_ef = embedding_functions.VoyageAIEmbeddingFunction(
        api_key=voyage_api_key,
        model_name="voyage-4-large"
    )
    return chroma_client.get_or_create_collection(
        name=name,
        embedding_function=voyage_ef
    )

def is_folder_empty(folder_path: str) -> bool:
    path = Path(folder_path)
    # Checks if the path exists, is a directory, and has no contents
    return path.exists() and path.is_dir() and not any(path.iterdir())

BATCH_SIZE = 100

def read_and_save_chunks(collection, file_path: str):
    ids, documents, metadatas = [], [], []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunk_json = json.loads(line)
                ids.append(f"{chunk_json.get('file')}:{chunk_json.get('page')}:{chunk_json.get('chunk_id')}")
                documents.append(chunk_json.get("chunk"))
                metadatas.append({
                    "file": chunk_json.get("file"),
                    "page": chunk_json.get("page"),
                    "chunk_id": chunk_json.get("chunk_id"),
                })

    print(f"{file_path}: {len(ids)} chunks")
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_ids = ids[start:end]
        existing = set(collection.get(ids=batch_ids, include=[])["ids"])
        new = [i for i, chunk_id in enumerate(batch_ids) if chunk_id not in existing]
        if not new:
            continue
        collection.upsert(
            ids=[batch_ids[i] for i in new],
            documents=[documents[start + i] for i in new],
            metadatas=[metadatas[start + i] for i in new],
        )


#@cached(cache_storage)
def save_embeddings(collection, file_path: str):
    if not is_folder_empty(file_path):
        return

    for chunk_file in Path(file_path).glob("*.jsonl"):
        data = read_and_save_chunks(collection, chunk_file)



def query_embeddings(collection, query: str):
    print("querying")
    return collection.query(
        query_texts=[query],
        n_results=20
    )

def main(query: str, file_path: str, db_path: str = DEFAULT_DB_PATH, name: str = DEFAULT_COLLECTION):
    collection = get_collection(db_path, name)
    save_embeddings(collection, file_path)
    results = query_embeddings(collection, query)
    print("received query result")
    print(results)
    for result in results:
        print(result[0:200])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="query name")
    parser.add_argument("file_path", type=str, help="file path to fetch chunks")

    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="chroma persistence directory")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="chroma collection name")

    args = parser.parse_args()
    main(args.name, args.file_path, args.db_path, args.collection)


