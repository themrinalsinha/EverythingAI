import os
import openai
import argparse

from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.prompts import ChatPromptTemplate


# load .env file
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


PROMPT_TEMPLATE = """
Answer the question based on the following context:

{context}

----
Answer the question based on the context above. {question}
"""


if __name__ == "__main__":
    DB_PATH = "database/chroma"

    parser = argparse.ArgumentParser(description="Query the database")
    parser.add_argument("question", type=str, help="The question to ask the database")
    args = parser.parse_args()

    query_text = args.question

    # preparing the database
    embedding_fn = OpenAIEmbeddings()
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_fn)

    # search the database
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print(f"\nResults (chrome db): {results}\n")
    if len(results) == 0 or results[0][1] < 0.7:
        print("No relevant results found in the database.")

    else:
        context_text = "\n\n --- \n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=context_text,
            question=query_text,
        )
        print(f"\nPrompt: \n{prompt}\n")

        model = ChatOpenAI()
        response_text = model.invoke(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"\nResponse: {response_text}\n\nSources: {sources}\n"
        print(formatted_response)
