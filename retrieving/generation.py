from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from retrieval import load_FAISS_retriever
from dotenv import load_dotenv



def generate(question, retriever):
    llm = ChatOpenAI(model="gpt-4o")

    prompt = PromptTemplate.from_template("""
    You are an assistant answering questions based on Oracle documentation.

    Use only the context below to answer the user's question, 
    go through the entire context to find relevant information to present a complete answer to user. 
    If unsure, say "I don't know."

    Context:
    {context}

    Question: {input}
    """)

    # This handles injecting context into prompt
    stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    rag_chain = create_retrieval_chain(retriever, stuff_chain)

    response = rag_chain.invoke({"input": question})
    return response
    


if __name__ == "__main__":

    print("\n processing ... \n")

    faiss_retriever = load_FAISS_retriever().as_retriever()

    question = "what are the pre requisites for creating the supplier record manually?"

    print("\n generating response for question: 'what are the pre requisites for creating the supplier record manually?' ... \n")

    response = generate(question, faiss_retriever)


    print("\n🧠 #################Answer###################", response['answer'], "\n")

    print("\n#####################context#################:\n")
    for contextArr in response["context"]:
        print(contextArr)
        print("")
    