from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
from pydantic import BaseModel
import os
from typing import List, TypedDict
import json
import time
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
import tiktoken
from dotenv import load_dotenv
import uvicorn


load_dotenv()

app = FastAPI(title="RAG API", description="REST API for RAG system with NICE Guidelines")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
db_config = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

MODEL = "gpt-4o-mini"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic models for request/response
class Query(BaseModel):
    text: str
    conversation_history: Optional[List[Dict[str, str]]] = []

class RAGResponse(BaseModel):
    answer: str
    context: Optional[str]
    results: List[dict]
    category: str
    processing_time: float

# Utility functions (moved from your original code)
def calculate_token_count(messages, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    token_count = 0
    for message in messages:
        content = None
        if isinstance(message, dict):
            content = message.get("content", "")
        elif hasattr(message, "content"):
            content = message.content
        if content:
            token_count += len(encoding.encode(str(content)))
    return token_count

def classify_query(query: str) -> dict:
    classification_prompt = (
        "You are an expert in classifying queries into one of the following categories:\n"
        "1. 'greeting' - If the query is a greeting like 'hello', 'hi', 'good morning', etc.\n"
        "2. 'goodbye' - If the query is a farewell like 'goodbye', 'bye', etc.\n"
        "3. 'medical' - If the query relates to medical topics, diseases, medications, symptoms\n"
        "4. 'non-forbes' - For riddles queries, gibberish, weather updates, queries like 'who are you', 'what is your name', 'who i am talking to'\n"
        "5. 'medical' - For ambiguous cases in which you can not decide if its non-medical then default it to medical\n\n"
        f"{query}\n"
        "Respond only with valid JSON. Example: {'category': 'medical'}"
    )
    
    try:
        response = openai_client.chat.completions.create(
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=50,
            temperature=0.1,
            model=MODEL,
        )
        
        response_content = response.choices[0].message.content.strip()
        response_content = response_content.replace("```json", "").replace("```", "").strip()
        response_content = response_content.replace("'", '"')
        
        try:
            response_json = json.loads(response_content)
            category = response_json.get("category", "").lower()
            return {"category": category}
        except json.JSONDecodeError:
            return {"category": "medical"}
    except Exception:
        return {"category": "medical"}

def retrieve(query: str) -> dict:
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        conn = psycopg2.connect(**db_config)
        register_vector(conn)
        cur = conn.cursor()
        
        query_embedding = model.encode(query).tolist()
        
        cur.execute("""
            SELECT md_file, pdf_link, chunk_index, chunk_text
            FROM document_embeddings
            ORDER BY embedding <-> %s::vector
            LIMIT 2;
        """, (query_embedding,))
        
        results = cur.fetchall()
        
        formatted_results = [{
            "md_file": row[0],
            "pdf_link": row[1],
            "chunk_index": row[2],
            "chunk_text": row[3]
        } for row in results]
        
        cur.close()
        conn.close()
        
        serialized = "\n\n".join(doc["chunk_text"] for doc in formatted_results)
        return {"content": serialized, "results": formatted_results}
    except Exception as e:
        return {"content": "", "results": []}

def run_rag_conversation(query: str, conversation_history: list) -> dict:
    system_prompt = """You MUST follow these rules:
    1. FIRST check the provided context THOROUGHLY. If the user's query already exists or is completely answered in the context, respond using ONLY that context WITHOUT retrieval.
    2. If answer is not found in provided context, ALWAYS trigger tool call to retrieve from NICE Guidelines database
    3. NEVER respond using your own internal knowledge outside these sources(tool call, provided context)
    4. Tool Calls:
       - When rephrasing the user query try to keep the rephrased queries of medium size and include all information of user query.
       - Never create more than two rephrased queries
       - Include relevant context from the user query for better retrieval.
    5. Final Responses:
       - Do not include statements like 'Based on the context' or 'The context information' or 'According to the context'.\n"
       - Do not cite any links or resources in the final response.
       - Structure the response with clear headings whenever possible(e.g., '1. Immediate Actions', '2. Investigations', etc.).\n"
       - Use bullet points for clarity and readability.\n"""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "retrieve",
                "description": "Retrieve relevant information from the NICE Guidelines database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to retrieve information",
                        }
                    },
                    "required": ["query"],
                }
            }
        }
    ]
    
    messages = [{"role": "system", "content": system_prompt}] + conversation_history + [{"role": "user", "content": query}]
    
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    messages.append(response_message)
    
    retrieved_results = []
    context = None
    
    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "retrieve":
                function_response = retrieve(function_args.get("query"))
                retrieved_results = function_response.get("results", [])
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                })
                context = function_response.get("content", "")
    
    final_response = openai_client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    answer = final_response.choices[0].message.content
    
    return {
        "answer": answer,
        "context": context,
        "results": retrieved_results
    }

# FastAPI endpoints
@app.post("/api/query", response_model=RAGResponse)
async def process_query(query: Query):
    start_time = time.time()
    
    # Classify query
    classification = classify_query(query.text)
    category = classification.get("category", "medical")
    
    response = {
        "answer": "",
        "context": None,
        "results": [],
        "category": category,
        "processing_time": 0
    }
    
    if category in ["greeting", "goodbye", "non-medical"]:
        responses = {
            "greeting": "Hello! How can I assist you today?",
            "goodbye": "Goodbye! Feel free to ask more questions anytime.",
            "non-medical": "Please ask a question related to NICE Guidelines."
        }
        response["answer"] = responses[category]
    
    elif category == "medical":
        result = run_rag_conversation(query.text, query.conversation_history)
        response.update(result)
    
    response["processing_time"] = time.time() - start_time
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)