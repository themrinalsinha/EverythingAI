import argparse

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_func import get_embedding_function

DATABASE = "database/chroma"

PROMPT_TEMPLATE = """

Answer the question based only on the following context:

{context}

---
Answer the question based on the above context: {question}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Query to search for")
    args = parser.parse_args()

    user_input = args.query
    embedding_fn = get_embedding_function()

    db = Chroma(
        persist_directory=DATABASE, embedding_function=embedding_fn
    )

    # search the db
    results = db.similarity_search_with_score(user_input, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=args.query)

    # create a model
    model = Ollama(model="llama3.1")
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _source in results]
    formatted_response = f"\nResponse: {response_text}\n\nSources: {sources}\n"
    print(formatted_response)
