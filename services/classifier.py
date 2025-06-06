from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

## Defining the structure of output
def structer_output():
    classification_schema = ResponseSchema(
        name="classification",
        description="One of: sql, rag, or invalid"
    )

    rewritten_schema = ResponseSchema(
        name="rewritten_question",
        description="""Only rewrite the question if BOTH of these are true:
            1. The question is a vague follow-up.
            2. It cannot be understood without prior context.
            If the question is already clear and complete, return it exactly as it is."""
    )

    output_parser = StructuredOutputParser.from_response_schemas(
        [classification_schema, rewritten_schema]
    )

    return output_parser

## Defining the prompt 
def getPrompt():
    return PromptTemplate.from_template("""
        You are an AI assistant helping users interact with a system that consists of:
        1. A **relational database** with the following tables:

        - purchase_order(id, PO_Number, Date, Vendor_Name, Total_Price, Status, created_at)
        - invoices(id, Invoice_Number, Invoice_Date, Purchase_Order, Item_Description, Quantity, Unit_Price, Total_Price, Due_Date, Status, created_at)

        2. A **user manual**: *Oracle Payables User’s Guide for Release 12.2*, which covers topics related to using Oracle Payables within Oracle E-Business Suite.

        Your task is two-fold:
        1. **Classify** the question as one of:
        - `"sql"`: if the question can be answered by querying the database tables
        - `"rag"`: if it needs context from the Oracle documentation
        - `"invalid"`: if it is vague, unrelated, or unanswerable with the current system

        Rewrite this query to be as semantically clear and concise as possible for a document search system:

        ---
        Chat History:
        {history}

        User Question:
        {question}

        ---
        {format_instructions}
                                        
        """)

## Making the LLM call to classify the strategy based on the question and chat history
def classify_strat(user_question, chat_history):
    try:
        ## Initiallizing the output structure and prompt
        output_parser = structer_output()
        prompt = getPrompt()

        ## Coupling the output structure and prompt
        prompt = prompt.partial(format_instructions = output_parser.get_format_instructions())

        ## Making the llm call
        llm = ChatOpenAI(model="gpt-4o")
        combined_chain = prompt | llm | output_parser

        print("\n\n ## Classification Layer:  Question recieved", user_question,"\n")
        
        # Invoke the chain
        response = combined_chain.invoke({
            "history": chat_history,
            "question": user_question
        })

        ## Returns a json file with {classification, rewritten_question}
        return response
    
    except Exception as e:
        return {
            "message": "Error while classifying the strategy based on question",
            "error": str(e)
        }


## Test code if running this python file
if __name__ == "__main__":
    chat_history = """
        User: What is a purchace order?
        Bot: Explainantion about  purchace orders ....
        User: Show me pending invoices.
        Bot: These are the pending invoices: Invoice_101, Invoice_102.
        """

    question = "total number of purhcase orders?"
    
    response = classify_strat(question, chat_history)

    print("\nResponse from classifier:\n\n", response)
