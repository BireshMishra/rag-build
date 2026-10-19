import os
import chromadb
import argparse
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


@cached(cache_storage)
def save_embeddings(file_path: str):
    if is_folder_empty(file_path):
        return

    for chunk_file in Path(file_path).glob("*.jsonl"):
        collection.add(chunk_file)


def query_embeddings(query: str):
    return collection.query(
        query_texts=query,
        n_results=1
    )

def main(query: str, file_path: str):
    save_embeddings(file_path)
    result = query_embeddings(query)

if __name__ == "__main()__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="query name")
    parser.add_argument("file_path", type=str, help="file path to fetch chunks")

    args = parser.parse_args()
    main(args.name, args.file_path)


