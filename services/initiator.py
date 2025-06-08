

from models.response_model import ChatResponse, MetaData
from services.classifier import classify_strat
from services.sql_generator import generate_sql_response
from services.rag_pipeline import get_rag_response
import json

def format_history(history_list):
    return "\n".join([
        f"User: {m['user']}" if "user" in m else f"Bot: {m['bot']}"
        for m in history_list
    ])

#####  Main function to process user input and route accordingly 
async def handle_user_input(question: str, chat_history: str):

    ## Format chat history from list to string
    formatted_chat_history = format_history(chat_history)

    ## Get the classification and rewritten querry
    classification_response = classify_strat(question, formatted_chat_history)
    if "classification" in classification_response:
        classification = classification_response["classification"]
        rewritten = classification_response["rewritten_question"]
    ## ERROR RESPONSE FROM CLASSIFICATION
    else:
        return ChatResponse(
            status="error",
            source="classification",
            message=classification_response['message'],
            bot_response="Sorry the question seems vague or is not clear to generate a valid answer",
            meta= MetaData(
                raw_error=classification_response['error']
            )
        ).model_dump()
    
    ## Log the output from classification layer
    print(f"[Classifier Output] Classification: {classification} \n")
    print(f"[Classifier Output] Rewritten Question: {rewritten} \n")


    ## Replace the original question if we got the rewritten question 
    if rewritten != "N/A":
        question = rewritten


    ############## IF RAG ################################
    if classification == "rag":
        print(f"🔍 Initializing RAG pipeline for: {question}")
        # call_rag_pipeline(rewritten) — your logic
        return process_rag(question)


    ## If SQL then initiatlize sql generator
    elif classification == "sql":
        print(f"\n\n🧮 Initializing SQL generator for: {question}")
        
        return process_sql_generator(question)
    

    ## If the question is invalid and out of context then return response appropriately
    return ChatResponse(
        status="success",
        message="The question is invalid, classified as out of context",
        bot_response="Looks like you're asking about a topic that is out of context, " \
        "Feel free to ask any question regarding financial data",
        source="Classification",
        meta=MetaData(
            rewritten_query=question
        )
    ).model_dump()


"""
Process the RAG response 

Args: 
    question (str): User's question, raw or rewritten

Returns:
    Chat model response 
    ChatModel()

"""
def process_rag(question:str):
    rag_response = get_rag_response(question)

    ## Return error response
    if rag_response["status"]  == "error":
        return ChatResponse(
            status="error",
            source="RAG",
            message="Error while processing the question in RAG",
            bot_response="Sorry the question seems vague or is not clear to generate a valid answer",
            meta=MetaData(
                raw_error=rag_response["error"],
                rewritten_query=question
            )
        ).model_dump()

    ## Return a successful response from LLM
    return ChatResponse(
        status="success",
        source="RAG",
        message="Successfully got response using RAG strategy ",
        bot_response=rag_response['answer'],
        meta=MetaData(
            context_pages=[ctx.metadata for ctx in rag_response['context']],
            rewritten_query=question
        )
    ).model_dump()


"""
Processing SQL GENERATOR RESPONSE

ARGS:
    question: str
Response: 
    JSON object which changeing structures
"""
def process_sql_generator(question:str):
    ## Calls sql result generator
    response = generate_sql_response(question)

    # Return error response
    if response['status'] == "error":
        return ChatResponse(
            status='error',
            source='SQL',
            message=response.get('message', 'Error while processing SQL generator'),
            bot_response=response.get('result', "Couldn't process your question to generate an SQL query, Please try asking a different question :)"),
            meta=MetaData(
                sql_query=response.get('sql_query', ""),
                rewritten_query=question,
                raw_error=response.get('error', "")
            )
        ).model_dump()
    
    sql_data = response.get('results')

    sql_data = add_dollar_sign(sql_data)
    
    ## Return valid SQL table response
    return ChatResponse(
        status='success',
        source='SQL',
        message=response.get('message'),
        bot_response=sql_data,
        meta=MetaData(
            sql_query=response.get('sql_query'),
            rewritten_query=question
        )
    ).model_dump()


def add_dollar_sign(sql_data):
    

    for data in sql_data:
        if 'Total_Price' in data:
            data['Total_Price'] = "$"+str(data['Total_Price'])
        if 'Unit_Price' in data:
            data['Unit_Price'] = "$"+str(data['Unit_Price'])
    return sql_data
        



## Test the working of initiator
if __name__ == "__main__":

    chat_history = [
        {"user": "list 10 recent invoices with status"},
        {"bot": "{'id': 30080, 'Invoice_Number': '14899004', 'Invoice_Date': '2025-05-21', 'Purchase_Order': '72567994', 'Item_Description': 'Printer', 'Quantity': 14, 'Unit_Price': 202.38, 'Total_Price': 2833.32, 'Due_Date': '2025-09-01', 'Status': 'Pending', 'created_at': '2025-06-04T15:00:06'}, {'id': 16746, 'Invoice_Number': '48360530', 'Invoice_Date': '2025-05-21', 'Purchase_Order': '74686992', 'Item_Description': 'Tablet', 'Quantity': 15, 'Unit_Price': 1524.7, 'Total_Price': 22870.5, 'Due_Date': '2025-07-17', 'Status': 'Pending', 'created_at': '2025-06-04T14:59:57'}, {'id': 12093, 'Invoice_Number': '45588622', 'Invoice_Date': '2025-05-21', 'Purchase_Order': '99816646', 'Item_Description': 'Tablet', 'Quantity': 16, 'Unit_Price': 311.61, 'Total_Price': 4985.76, 'Due_Date': '2025-06-15', 'Status': 'Pending', 'created_at': '2025-06-04T14:59:54'}, {'id': 17630, 'Invoice_Number': '54651191', 'Invoice_Date': '2025-05-21', 'Purchase_Order': '44125352', 'Item_Description': 'Smartphone', 'Quantity': 3, 'Unit_Price': 911.32, 'Total_Price': 2733.96, 'Due_Date': '2025-06-14', 'Status': 'Pending', 'created_at': '2025-06-04T14:59:57'}, {'id': 18872, 'Invoice_Number': '65997174', 'Invoice_Date': '2025-05-21', 'Purchase_Order': '96618146', 'Item_Description': 'Desk Lamp', 'Quantity': 11, 'Unit_Price': 1670.62, 'Total_Price': 18376.82, 'Due_Date': '2025-05-30', 'Status': 'Pending', 'created_at': '2025-06-04T14:59:58'}, {'id': 15272, 'Invoice_Number': '38573575', 'Invoice_Date': '2025-05-21', 'Purchase_Order': '77730256', 'Item_Description': 'Smartphone', 'Quantity': 19, 'Unit_Price': 67.89, 'Total_Price': 1289.91, 'Due_Date': '2025-06-08', 'Status': 'Pending', 'created_at': '2025-06-04T14:59:56'}, {'id': 28296, 'Invoice_Number': '38267763', 'Invoice_Date': '2025-05-21', 'Purchase_Order': '27570689', 'Item_Description': 'Tablet', 'Quantity': 10, 'Unit_Price': 193.45, 'Total_Price': 1934.5, 'Due_Date': '2025-07-23', 'Status': 'Pending', 'created_at': '2025-06-04T15:00:04'}, {'id': 12001, 'Invoice_Number': '36318996', 'Invoice_Date': '2025-05-21', 'Purchase_Order': '26961923', 'Item_Description': 'Monitor', 'Quantity': 13, 'Unit_Price': 668.45, 'Total_Price': 8689.85, 'Due_Date': '2025-06-07', 'Status': 'Pending', 'created_at': '2025-06-04T14:59:53'}, {'id': 20339, 'Invoice_Number': '23441787', 'Invoice_Date': '2025-05-21', 'Purchase_Order': '69549444', 'Item_Description': 'Stationery', 'Quantity': 16, 'Unit_Price': 812.96, 'Total_Price': 13007.36, 'Due_Date': '2025-07-31', 'Status': 'Pending', 'created_at': '2025-06-04T14:59:59'}, {'id': 29163, 'Invoice_Number': '3327476', 'Invoice_Date': '2025-05-21', 'Purchase_Order': '20805307', 'Item_Description': 'Monitor', 'Quantity': 6, 'Unit_Price': 1110.25, 'Total_Price': 6661.5, 'Due_Date': '2025-06-09', 'Status': 'Pending', 'created_at': '2025-06-04T15:00:05'}"}
    ]
    
    question = "total number of invoices so far?"

    response = handle_user_input(question, chat_history)

    print(response)
