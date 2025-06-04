from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from retrieval import load_FAISS_retriever
from langchain_community.callbacks import get_openai_callback

def generate(question, retriever):
    llm = ChatOpenAI(model="gpt-4o")


############ PROMPT ####################
    prompt = PromptTemplate.from_template("""                            
                                          
    You are a helpful assistant specialized in Oracle documentation.

    Answer the user's question **only** using the provided context. 
    Read through **all** the context to construct the most accurate and complete response. 
    Do **not** rely on prior knowledge. If the answer cannot be found in the context, 
    reply with: "I'm sorry, I don't have enough information to answer that right now."

    Always aim to:
    - Provide a **clear and concise** explanation
    - Reference relevant Oracle-specific terms if applicable
    - Avoid adding assumptions or extra information
    
    Context:
    {context}

    Question: {input}
                                          
    """)

    # This handles injecting context into prompt
    stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    rag_chain = create_retrieval_chain(retriever, stuff_chain)

    response = rag_chain.invoke({"input": question})

    # # Measure token usage and cost
    # with get_openai_callback() as cb:

    #     response = rag_chain.invoke({"input": question})
    #     # print("🧠 Answer:", response["answer"])
        
    #     print("/n $$$$$$$$$$$$Prompt Cost $$$$$$$$$$$$")
    #     print("🔢 Prompt tokens:", cb.prompt_tokens)
    #     print("📝 Completion tokens:", cb.completion_tokens)
    #     print("📦 Total tokens:", cb.total_tokens)
    #     print("💰 Cost (USD):", cb.total_cost)
    #     print("/n $$$$$$$$$$$$Prompt Cost $$$$$$$$$$$$")

    return response

if __name__ == "__main__":

    print("\n processing ... \n")

    faiss_retriever = load_FAISS_retriever().as_retriever()

    question = "What is facebook?"

    print("\n generating response for question:",question," ... \n")  

    response = generate(question, faiss_retriever)

    print("\n🧠 #################   Answer   ###################\n\n", response['answer'], "\n")

    print("\n\n ####################### Context (Chunks retrieved) ####################### \n\n")

    for i in response['context']:
        print(i)
        print("\n\n")

    

