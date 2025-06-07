from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from retrieval import load_FAISS_retriever
from langchain_community.callbacks import get_openai_callback


######################################## PROMPT ########################################
def getRagPrompt():
    return PromptTemplate.from_template("""
    You are a helpful assistant specialized in Oracle documentation.

    Answer the user’s question as clearly and accurately as possible, using the available Oracle documentation you’ve been provided.
        •	If the answer can be reasonably inferred from the information you have, include it.
        •	If you can only answer part of the question, explain which part you can answer and provide that information.
        •	If no relevant information is available, reply with:
    “I’m sorry, I don’t have enough information to answer that right now.”

    Make sure your response:
        •	Uses clear, professional language
        •	Focuses on Oracle-specific terms or behavior when relevant
        •	Quotes directly from documentation when appropriate
        •	Avoids mentioning how the answer was found or processed
    
    Context:
    {context}

    Question: {input}                              
    """)

def generate(question, retriever):
    llm = ChatOpenAI(model="gpt-4.1-nano")

    prompt = getRagPrompt()

    print("\n\n ## RAG GENERATION LAYER: Question received:", question,"\n")

    try:

        # This handles injecting context into prompt
        stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

        rag_chain = create_retrieval_chain(retriever, stuff_chain)

        response = rag_chain.invoke({"input": question})

        print("\n\n ######## RAG Response ###### \n\n", response.get("answer"), "\n\n")

        return {
            "status": "success",
            "answer": response.get("answer", ""),
            "context": response.get("context", []),
            "error": None
        }

    except Exception as e:
        print(f"❌ RAG pipeline error: {str(e)}")
        return {
            "status": "error",
            "answer": None,
            "context": [],
            "error": str(e)
        }


## Helper function to get the token cost for the LLM call
def helper_getCost(rag_chain, question):
    # Measure token usage and cost
    with get_openai_callback() as cb:

        response = rag_chain.invoke({"input": question})
        # print("🧠 Answer:", response["answer"])
        
        print("/n $$$$$$$$$$$$Prompt Cost $$$$$$$$$$$$")
        print("🔢 Prompt tokens:", cb.prompt_tokens)
        print("📝 Completion tokens:", cb.completion_tokens)
        print("📦 Total tokens:", cb.total_tokens)
        print("💰 Cost (USD):", cb.total_cost)
        print("/n $$$$$$$$$$$$Prompt Cost $$$$$$$$$$$$")
    return response



# ## Test code if running this python file
# if __name__ == "__main__":

#     print("\n processing ... \n")

#     faiss_retriever = load_FAISS_retriever().as_retriever()

#     question = "what is the difference between a policy and a purhcase order?"

#     print("\n generating response for question:",question," ... \n")  

#     response = generate(question, faiss_retriever)

#     print("\n🧠 #################   Answer   ###################\n\n", response['answer'], "\n")

#     print("\n\n ####################### Context (Chunks retrieved) ####################### \n\n")

#     for i in response['context']:
#         print(i)
#         print("\n\n")