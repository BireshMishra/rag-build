import argparse
import pandas as pd
import datetime

from pathlib import Path
from pypdf import PdfReader


def ingest(file_path: str):
    file_name = Path(file_path).stem
    print(f"file name is: {file_name}")
    dataset=[]
    
    reader = PdfReader(file_path)

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        dataset += create_chunks(file_name, page_num, text, 500, 50)
    save_dataset(file_name, dataset)

def create_chunks(file_name: str, page_num: int, text: str, chunk_size: int
                  ,  overlap_size: int = 50) -> list[dict]:
    print(f"creating chunk for page: {page_num}")
    dataset=[]
    text_len = len(text)
    for i  in range(text_len//chunk_size + 1):
        chunk = text[max(0, i * chunk_size - overlap_size): (i+1) * chunk_size + overlap_size]
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