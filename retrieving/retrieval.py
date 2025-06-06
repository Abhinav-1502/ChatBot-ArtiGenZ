
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()


def load_FAISS_retriever():
    # Initialize embedding model (same as used before)
    embedding_model = OpenAIEmbeddings()

    # Load FAISS index from local folder
    vectorstore = FAISS.load_local(
        folder_path="faiss_oracle_index",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    return vectorstore

def getTopKChunks(question, vectorstore, k):

    results = vectorstore.similarity_search(question, k)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print("Metadata:", doc.metadata, "\n")
        print(doc.page_content)
        print("")


    


if __name__ == "__main__":

    vectorstore = load_FAISS_retriever()
    question = "what are the pre requisites for creating the supplier record manually? all the steps"

    getTopKChunks(question, vectorstore, 3)


