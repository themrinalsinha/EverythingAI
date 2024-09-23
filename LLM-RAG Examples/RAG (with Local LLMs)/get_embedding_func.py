from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


# creating embedding function
def get_embedding_function():
    # embedding = BedrockEmbeddings(
    #     credentials_profile_name="default",
    #     region_name="us-west-2",
    # )
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    return embedding
