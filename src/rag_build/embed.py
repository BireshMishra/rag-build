import os
import chromadb
import argparse
import json
from chromadb.utils import embedding_functions
from pathlib import Path
from dotenv import load_dotenv
from cachetools import cached, TTLCache


load_dotenv()
voyage_api_key = os.getenv("VOYAGE_API_KEY")

cache_storage = TTLCache(maxsize=10000, ttl=600)

chroma_client = chromadb.PersistentClient(path = "./chroma_db")

voyage_ef = embedding_functions.VoyageAIEmbeddingFunction(
    api_key=voyage_api_key,
    model_name="voyage-4-large"
)

collection = chroma_client.get_or_create_collection(
    name="my_voyage_collection",
    embedding_function=voyage_ef
)

def is_folder_empty(folder_path: str) -> bool:
    path = Path(folder_path)
    # Checks if the path exists, is a directory, and has no contents
    return path.exists() and path.is_dir() and not any(path.iterdir())

BATCH_SIZE = 100

def read_and_save_chunks(file_path: str):
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
def save_embeddings(file_path: str):
    if not is_folder_empty(file_path):
        return

    for chunk_file in Path(file_path).glob("*.jsonl"):
        data = read_and_save_chunks(chunk_file)
        


def query_embeddings(query: str):
    print("querying")
    return collection.query(
        query_texts=[query],
        n_results=20
    )

def main(query: str, file_path: str):
    save_embeddings(file_path)
    results = query_embeddings(query)
    print("received query result")
    print(results)
    for result in results:
        print(result[0:200])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="query name")
    parser.add_argument("file_path", type=str, help="file path to fetch chunks")

    args = parser.parse_args()
    main(args.name, args.file_path)


