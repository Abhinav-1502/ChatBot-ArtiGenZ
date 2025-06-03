from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
import json
import os


# Load .env to save OPEN AI api key
load_dotenv()

# Loads saved chunk JSON data from given path into a list of Document object
def load_chunks_from_json(path="chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in data]

# Embeds the data from docs and stores in FAISS - a local vector DB store
def embedDataToFAISS(docs):
    embedding_model = OpenAIEmbeddings() #sets the embedding model to small-text-embedding-3 by default

    print("\nIndexing embeddings to FAISS \n")
    # Create FAISS vector store from documents
    vectorstore = FAISS.from_documents(docs, embedding_model)

    # Save index to disk
    vectorstore.save_local("faiss_oracle_index")
    print("\n✅ FAISS index saved to 'faiss_oracle_index/ \n'")

    
if __name__ == "__main__":
    docs = load_chunks_from_json("resources/chunks.json")

    embedDataToFAISS(docs)