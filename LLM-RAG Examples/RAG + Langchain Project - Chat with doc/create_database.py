import os
import openai
import shutil

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# load .env file
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_document():
    loader = DirectoryLoader("dataset/", glob="*.md")
    return loader.load()

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    # print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    # document = chunks[0]
    # print(document.page_content)
    # print(document.metadata)

    return chunks



if __name__ == "__main__":
    DB_PATH = "database/chroma"

    documents = load_document()
    chunks = split_text(documents)

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    # create a new db from the documents.
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory=DB_PATH,
    )
    print(f"Database created successfully: {len(chunks)} chunks saved!")
