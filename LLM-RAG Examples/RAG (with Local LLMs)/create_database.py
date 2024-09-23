import os
import shutil
import argparse

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.schema.document import Document
from langchain_chroma import Chroma

from get_embedding_func import get_embedding_function


DATABASE = "database/chroma"


def load_document():
    loader = PyPDFDirectoryLoader("dataset")
    return loader.load()

def split_document(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def clear_db():
    if os.path.exists(DATABASE):
        shutil.rmtree(DATABASE)

def _calculate_and_update_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id  = f"{source}-{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}-{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database")

    args = parser.parse_args()

    if args.reset:
        print("Resetting the database")
        clear_db()

    documents = load_document()
    chunks = _calculate_and_update_chunk_ids(split_document(documents))

    db = Chroma(
        persist_directory=DATABASE, embedding_function=get_embedding_function()
    )

    # create or update the data store.
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Existing items: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding {len(new_chunks)} new items")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new items to add")
