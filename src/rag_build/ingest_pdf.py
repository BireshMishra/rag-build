import argparse
import pandas as pd
import datetime

from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

def ingest(file_path: str):
    file_name = Path(file_path).stem
    print(f"file name is: {file_name}")
    dataset=[]
    
    reader = PdfReader(file_path)

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        chunks = splitter.split_text(text)
        for chunk in chunks:
            dataset.append({"file":file_name, "page":page_num, "chunk":chunk})
        #dataset += create_chunks(file_name, page_num, text, 500, 50)
    save_dataset(file_name, dataset)

def create_chunks(file_name: str, page_num: int, text: str, chunk_size: int
                  ,  overlap_size: int = 50) -> list[dict]:
    print(f"creating chunk for page: {page_num}")
    dataset=[]
    text_len = len(text)
    step = chunk_size - overlap_size
    for i  in range(0, text_len, chunk_size):
        chunk = text[ i * step: i*step + chunk_size]

        if chunk.strip():
            dataset.append({"file":file_name, "page":page_num, "chunk":chunk})
    print(f"created chunks for page: {page_num}")
    return dataset

def save_dataset(file_name: str, dataset: list[dict]):
    print("saving dataset")
    df = pd.DataFrame(dataset)
    df.to_json(f"dataset_{file_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl", orient="records", lines=True)

if __name__ == "__main__":
    print("started")
    parser = argparse.ArgumentParser("cli arg")
    parser.add_argument("name", type=str, help="file path")
    args = parser.parse_args()
    print(f"file path is: {args.name}")
    ingest(args.name)