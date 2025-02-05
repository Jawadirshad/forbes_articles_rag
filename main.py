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

app = FastAPI(title="RAG API", description="REST API for RAG system with Forbes Guidelines")

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
openai_api_key = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print(openai_api_key)

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
    """
    Calculate token count for a list of messages using tiktoken.
    """
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





def classify_query(query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    classification_prompt = (
        "You are an expert in classifying queries into one of the following categories:\n"
        "1. 'greeting' - If the query is a greeting like 'hello', 'hi', 'good morning', etc.\n"
        "2. 'goodbye' - If the query is a farewell like 'goodbye', 'bye', etc.\n"
        "3. 'forbes' - If the query relates to ai topics and knowledge, big data, cloud topics and knowledge, consumer tech and topics\n"
        "4. 'non-forbes' - For riddles queries, gibberish, weather updates, queries like 'who are you', 'what is your name', 'who i am talking to'\n"
        "5. 'forbes' - For ambiguous cases in which you can not decide if its non-forbes then default it to forbes\n\n"

        f"{query}\n"
        "Respond only with valid JSON. Example: {'category': 'forbes'}"
    )
    
    try:
        print("Sending classification request...")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=50,
            temperature=0.1,
            model=MODEL,
        )
        
        response_content = response.choices[0].message.content.strip()
        print(f"Raw classification response: {response_content}")

        # **Fix: Remove triple backticks and language tags if present**
        response_content = response_content.replace("```json", "").replace("```", "").strip()

        # **Fix: Ensure JSON formatting is valid before parsing**
        response_content = response_content.replace("'", '"')
        
        try:
            response_json = json.loads(response_content)
            category = response_json.get("category", "").lower()
            print(f"Parsed category: {category}")
            return {"category": category}
        except json.JSONDecodeError:
            print("!! JSON decode error, defaulting to forbes")
            return {"category": "forbes"}
    except Exception as e:
        print(f"!! Classification error: {str(e)}")
        return {"category": "forbes"}

def retrieve(query: str):
    """Retrieve information related to a query."""
    print(f"\n=== RETRIEVAL PHASE ===")
    print(f"Query: '{query}'")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Connecting to database...")
        conn = psycopg2.connect(**db_config)
        register_vector(conn)
        cur = conn.cursor()
        
        print("Generating embeddings...")
        query_embedding = model.encode(query).tolist()
        
        print("Executing similarity search...")
        cur.execute("""
            SELECT md_file, source_link, chunk_index, chunk_text
            FROM document_embeddings
            ORDER BY embedding <-> %s::vector
            LIMIT 2;
        """, (query_embedding,))
        
        results = cur.fetchall()
        print(f"Found {len(results)} matching chunks")
        
        formatted_results = [{
            "md_file": row[0],
            "source_link": row[1],
            "chunk_index": row[2],
            "chunk_text": row[3]
        } for row in results]
        
        cur.close()
        conn.close()
        
        serialized = "\n\n".join(doc["chunk_text"] for doc in formatted_results)
        print("Retrieval completed successfully")
        return json.dumps({"content": serialized, "results": formatted_results})
    except Exception as e:
        print(f"!! Retrieval error: {str(e)}")
        return json.dumps({"content": "", "results": []})




def run_rag_conversation(query: str, conversation_history: list):
    print(f"\n=== RAG PROCESSING ===")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = """You MUST follow these rules:
    1. FIRST check the provided context THOROUGHLY. If the user's query already exists or is completely answered in the context, respond using ONLY that context WITHOUT retrieval.
    2. If answer is not found in provided context, ALWAYS trigger tool call to retrieve from Forbes Articles database
    3. NEVER respond using your own internal knowledge outside these sources(tool call, provided context)
    4. Tool Calls:
       - When rephrasing the user query try to keep the rephrased queries of medium size and include all information of user query.
       - Never create more than two rephrased queries
       - Include relevant context from the user query for better retrieval.
    5. Final Responses:
       - Do not include statements like 'Based on the context' or 'The context information' or 'According to the context'.\n"
       - Do not cite any links or resources in the final response\n.
       - Structure the response with clear headings whenever possible.\n"
       - Use bullet points for clarity and readability.\n"""  
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "retrieve",
                "description": "Retrieve relevant information from the Forbes Articles database, the retrieval tool call is must unless answer is fully present in provided context",
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

    token_count = calculate_token_count(messages, model=MODEL)
    print(f"Token count for current request: {token_count} tokens")

    response = client.chat.completions.create(
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
        print(f"2. Tool Calls Detected ({len(tool_calls)})")
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            print(f" - Tool: {function_name}")
            print(f"   Args: {function_args}")
            
            if function_name == "retrieve":
                print("3. Executing Retrieval...")
                start_time = time.time()
                function_response = retrieve(function_args.get("query"))
                retrieved_data = json.loads(function_response)
                retrieved_results = retrieved_data.get("results", [])
                retrieval_time = time.time() - start_time
                print(f"   Retrieval took {retrieval_time:.2f}s")
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
                context = json.loads(function_response).get("content", "")
                print(context)

    token_count_final = calculate_token_count(messages, model=MODEL)
    print(f"Token count for final request: {token_count_final} tokens")
    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    answer = final_response.choices[0].message.content
    print("5. Response Generation Complete")
    print(retrieved_results)
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
    print(query)

    classification = classify_query(query.text)
    category = classification.get("category", "forbes")
    print(classification)
    response = {
        "answer": "",
        "context": None,
        "results": [],
        "category": category,
        "processing_time": 0
    }
    
    if category in ["greeting", "goodbye", "non-forbes"]:
        responses = {
            "greeting": "Hello! How can I assist you today?",
            "goodbye": "Goodbye! Feel free to ask more questions anytime.",
            "non-forbes": "Please ask a question related to Forbes articles."
        }
        response["answer"] = responses[category]
    
    elif category == "forbes":
        print(query)
        result = run_rag_conversation(query.text, query.conversation_history)
        response.update(result)
    
    response["processing_time"] = time.time() - start_time
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)