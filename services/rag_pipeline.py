import sys
import os
from langchain.schema import Document

# ✅ This line adds the project root (1 level up from this file) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','retrieving')))
from generation import generate # type: ignore
from retrieval import load_FAISS_retriever # type: ignore

def get_rag_response(question):

    ## Loading Faiss retriever 
    retriever = load_FAISS_retriever().as_retriever()

    response = generate(question, retriever)

    return response


#testing 
if __name__ == "__main__":
    response = get_rag_response("difference between an invoice and a purchase order")
    
    pages = [ctx.metadata for ctx in response["context"]]
    print(pages)