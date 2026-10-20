import argparse
import json

from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=500,
    chunk_overlap=50
) 

def ingest(file_path: str):
    file_name = Path(file_path).stem
    print(f"file name is: {file_name}")
    dataset=[]
    
    reader = PdfReader(file_path)

    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()

        chunks = text_splitter.split_text(text)
        for chunk_id, chunk in enumerate(chunks, 1):
            dataset.append({"file":file_name, "page":page_num, "chunk_id": chunk_id, "chunk":chunk})
    save_dataset(file_name, dataset)

def save_dataset(file_name: str, dataset: list[dict]):
    print("saving dataset")
    Path("dataset").mkdir(exist_ok=True)
    with open(f"dataset/dataset_{file_name}.jsonl", "w", encoding="utf-8") as file:
        for row in dataset:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    print("started")
    parser = argparse.ArgumentParser("cli arg")
    parser.add_argument("name", type=str, help="file path")
    args = parser.parse_args()
    print(f"file path is: {args.name}")
    for file in Path(args.name).glob("*.pdf"):
        print(f"filename during iteration is: {str(file)}/{file.name}")
        ingest(str(file))