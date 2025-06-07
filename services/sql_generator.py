import os
import json
from dotenv import load_dotenv
import mysql.connector
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from openai import OpenAI
import datetime

load_dotenv()

def create_connection():
    # MySQL connection
    db_config = {
        'host': os.getenv("db_host"),
        'user': os.getenv("db_user"),
        'password': os.getenv("db_password"),  # Updated with your MySQL password
        'database': os.getenv("db_database")
    }
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return None
    
    
def generate_prompt(question):
    return f"""
You are a financial database query assistant. Convert ONLY finance-related questions into MySQL queries.
DATABASE SCHEMA WITH ACTUAL DEFINITIONS:
Table: purchase_order
- id: INT AUTO_INCREMENT PRIMARY KEY (unique identifier)
- PO_Number: VARCHAR(20) NOT NULL (purchase order number, indexed)
- Date: DATE NOT NULL (order placement date)
- Vendor_Name: VARCHAR(100) NOT NULL (supplier company name, indexed)
- Total_Price: DECIMAL(10,2) NOT NULL (total order value)
- Status: VARCHAR(20) NOT NULL (order status, indexed)
- created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP (record creation time)

Table: invoices
- id: INT AUTO_INCREMENT PRIMARY KEY (unique identifier)
- Invoice_Number: VARCHAR(20) NOT NULL (invoice number, indexed)
- Invoice_Date: DATE NOT NULL (invoice generation date)
- Purchase_Order: VARCHAR(20) NOT NULL (references purchase_order.PO_Number, indexed)
- Item_Description: VARCHAR(200) NOT NULL (detailed item description)
- Quantity: INT NOT NULL (number of items, must be positive)
- Unit_Price: DECIMAL(10,2) NOT NULL (price per single unit)
- Total_Price: DECIMAL(10,2) NOT NULL (calculated amount)
- Due_Date: DATE NOT NULL (payment deadline, indexed)
- Status: VARCHAR(20) NOT NULL (payment status, indexed)
- created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP (record creation time)

INDEXED FIELDS (for optimized queries):
- purchase_order: PO_Number, Vendor_Name, Status
- invoices: Invoice_Number, Purchase_Order, Status, Due_Date

RELATIONSHIP:
invoices.Purchase_Order → purchase_order.PO_Number (foreign key relationship)

BUSINESS CONTEXT:
- Purchase orders are created first, then invoices reference them
- Multiple invoices can belong to one purchase order
- Status values: Pending (awaiting), Approved/Paid (completed), Overdue (late), Cancelled (void)
- Dates use MySQL DATE format (YYYY-MM-DD)
- All prices in decimal format (10,2) for currency precision

QUERY GUIDELINES:
- Use proper JOINs when relating tables
- Use aggregate functions (SUM, COUNT, AVG) for totals
- Use WHERE clauses for filtering by status, dates, vendors
- Use GROUP BY for summaries by vendor/status/date
- Use DATE functions like CURDATE(), DATE_SUB() for date comparisons

STRICT VALIDATION RULES:
1. ONLY process questions about: purchase orders, invoices, vendors, payments, financial totals, dates, status
2. REJECT questions about: weather, personal topics, math problems, general knowledge, non-business topics
3. REJECT incomplete/nonsensical input: single letters, "hello", random text, empty questions
4. Response format: Return ONLY valid SQL SELECT query OR exactly "INVALID_QUERY"
5. NO explanations, NO markdown, NO extra text

VALID QUESTION PATTERNS:
✓ "show me all pending purchase orders"
✓ "total spent on vendor ABC"
✓ "overdue invoices this month"
✓ "count of completed orders"
✓ "average invoice amount"
✓ "orders placed last week"

INVALID PATTERNS (return INVALID_QUERY):
✗ "hello", "w", "weather today", "how are you", "what is 2+2", "tell me a joke"
    
    
    User question: {question}
    """


def generate_sql_response(question: str):
    # data = request.json
    # question = data.get('question', '')
    
    if not question:
        return {
            'status': 'error',
            'message': 'No question recived by sql generator',
            'error': 'No question provided',
            'answer': "No question provided"
            }
    
    try:
        # Get SQL query from OpenAI using the new client
        print("Generating response for ", question, "......\n\n")

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that converts natural language to MySQL queries."},
                {"role": "user", "content": generate_prompt(question)}
            ],
            temperature=0.2
        )
        print(f"OpenAI raw response:\n{response}")  #print openai response
      
        sql_query = response.choices[0].message.content.strip()
        
        # Remove any markdown formatting
        if sql_query.startswith("```sql"):
            sql_query = sql_query[7:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        sql_query = sql_query.strip()
        
        print(f"\n\nGenerated SQL query: {sql_query}") #print sql queries
        
        # Check if the query is invalid
        if sql_query == "INVALID_QUERY":
            return {
                'status': 'error',
                'sql_query': sql_query,
                'message': "LLM responded with INVALID_QUERY for user's question",
                'error': "Stopped before execution: input not recognized as a question.",
            }
        
        results = None
        # Execute the query against the database
        conn = create_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(sql_query)
                results = cursor.fetchall()

                ## Date JSON error fix
                for row in results:
                    for key, value in row.items():
                        if isinstance(value, (datetime.date, datetime.datetime)):
                            row[key] = value.isoformat()

                cursor.close()
                conn.close()
            except Exception as db_err:
                print(f"Database query error: {str(db_err)}")
                return {
                    'status': 'error',
                    'error': f'Database query error: {str(db_err)}',
                    'sql_query': sql_query,
                    'message': "Error while executing the SQL query in the database"
                }
        else:
            return {
                'status': 'error',
                'error': 'Database connection failed',
                'sql_query': sql_query,
                'message': "Database connection error"
            }
        
        if not results:
            message = "I couldn't find any data matching your query. Please try asking a different question.",
            message1 = "there is no such data in the db"
            return {
                'status': 'error',
                'results': message,
                'sql_query': sql_query,
                'message': message1,
                'error' :  message1
            }
        
        ## Return valid SQL table response
        return {
            'status': 'success',
            'results': results,
            'sql_query': sql_query,
            'message': 'Successfully returned a valid SQL result'
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'message': "Error while processing SQL generator"
        }


if __name__ == "__main__":
    response = generate_sql_response("Can you show me the status of Purchase Order")
    print("\n\n",json.loads(response.body.decode()))

