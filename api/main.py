from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import requests
from typing import List, Optional
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
SEARCH_URL = os.environ.get("SEARCH_URL", "http://localhost:8000")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_URL = os.environ.get("LLM_URL")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3-12b-it-qat")

# Global OpenAI client
llm_client = None

# No need for collection parameters as we're using the indexer_api

app = FastAPI(title="RAG API Service")

# Use FastAPI lifespan instead of on_event
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup logic
    logger.info("Starting up API service")

    # Initialize OpenAI client
    global llm_client

    if OPENAI_API_KEY:
        logger.info("Initializing OpenAI client with API key")
        llm_client = OpenAI(api_key=OPENAI_API_KEY)
    elif LLM_URL:
        logger.info(f"Initializing OpenAI client with custom base URL: {LLM_URL}")
        llm_client = OpenAI(base_url=LLM_URL, api_key="lmstudio")
    else:
        logger.warning("No LLM configuration provided (OPENAI_API_KEY or LLM_URL)")
        llm_client = None

    yield

    # Shutdown logic
    logger.info("Shutting down API service")

# Update FastAPI app with lifespan
app = FastAPI(title="RAG API Service", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_hybrid: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None

def get_embedding(text):
    """Get embedding from the embedder service"""
    try:
        response = requests.post(
            f"{SEARCH_URL}/embed",
            json={"texts": [text]},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

def search_milvus(_, query_text, top_k=3, use_hybrid=True, vector_weight=0.5, keyword_weight=0.5):
    """Search Milvus using hybrid search (vector + BM25)"""
    try:
        # Use the indexer_api's search endpoint for hybrid search
        search_params = {
            "query": query_text,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
        }

        response = requests.post(
            f"{SEARCH_URL}/search",
            json=search_params,
            timeout=30
        )
        response.raise_for_status()

        # Process results
        search_results = response.json()["results"]

        # Format results to match the expected structure
        results = []
        for hit in search_results:
            results.append({
                "question": hit["question"],
                "answer": hit["answer"],
                "score": hit["score"]
            })

        return results
    except Exception as e:
        logger.error(f"Error searching with hybrid search: {str(e)}")
        raise

def generate_llm_response(query, context_data):
    """Generate response using LLM"""
    try:
        # Check if LLM client is initialized
        if llm_client is None:
            raise Exception("LLM client is not initialized. Please check your configuration.")

        # Prepare context from retrieved QA pairs
        context = ""
        for i, item in enumerate(context_data):
            context += f"Context {i+1}:\nQuestion: {item['question']}\nAnswer: {item['answer']}\n"

        # Create prompt
        prompt = f"""Ты эксперт-криминалист. Используйте приведённые ниже контекстные материалы (Context 1, Context 2 и т. д.) для ответа на вопрос пользователя. 
Если на основании контекста невозможно сформулировать ответ, скажите «У меня недостаточно информации, чтобы ответить на этот вопрос».

{context}

User Question: {query}

Answer:"""

        # Log which LLM model we're using
        logger.info(f"Using LLM model: {LLM_MODEL}, Prompt: {prompt}")

        # Make the API call using the global client
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Ты эксперт-криминалист"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        raise

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        # Get embedding for the query
        query_embedding = get_embedding(request.question)

        # Search Milvus with hybrid search
        search_results = search_milvus(
            query_embedding,
            request.question,
            request.top_k,
            request.use_hybrid,
        )

        if not search_results:
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question."
            )

        # Generate LLM response
        llm_response = generate_llm_response(request.question, search_results)

        # Return response with sources
        return QueryResponse(
            answer=llm_response,
            sources=[{
                "question": item["question"],
                "answer": item["answer"][:200] + "..." if len(item["answer"]) > 200 else item["answer"],
                "score": item["score"]
            } for item in search_results]
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "llm_model": LLM_MODEL,
        "llm_url": LLM_URL if LLM_URL else "Using OpenAI API",
        "openai_api_key": "Configured" if OPENAI_API_KEY else "Not configured",
        "llm_client_initialized": llm_client is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
