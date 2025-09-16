from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import pandas as pd
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
from pymilvus import (
    utility,
    DataType,
    Collection,
    MilvusClient,
    Function,
    FunctionType,
    connections,
    AnnSearchRequest,
    RRFRanker,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
MILVUS_ENABLE_HYBRID_SEARCH = os.environ.get("MILVUS_ENABLE_HYBRID_SEARCH", "True").lower() in ("true", "1", "t")
EXCEL_PATH = os.environ.get("EXCEL_PATH", "/data/qa_data.xlsx")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Hugging Face cache environment variables
HF_HOME = os.environ.get("HF_HOME", "/cache/huggingface")

# Collection parameters
COLLECTION_NAME = "qa_collection"

# Load the embedding model
model = None
milvus_client = None

@asynccontextmanager
async def lifespan(_):
    # Startup: initialize resources
    global model, milvus_client
    try:
        # Log Hugging Face cache configuration
        logger.info(f"Hugging Face cache configuration:")
        logger.info(f"  HF_HOME: {HF_HOME}")

        # Load the embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")

        # Initialize Milvus client
        milvus_client = initialize_milvus()
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

    yield

    # Shutdown: clean up resources
    logger.info("Shutting down the application")

# Initialize FastAPI with lifespan
app = FastAPI(title="Indexer and Embedder API", lifespan=lifespan)

def initialize_milvus():
    """Initialize Milvus client"""
    try:
        # Create Milvus client with URI and authentication if provided
        client_params = {"uri": MILVUS_URI}

        # Add authentication parameters if provided
        # if MILVUS_TOKEN:
        #     client_params["token"] = MILVUS_TOKEN
        # elif MILVUS_USER and MILVUS_PASSWORD:
        #     client_params["user"] = MILVUS_USER
        #     client_params["password"] = MILVUS_PASSWORD

        client = MilvusClient(**client_params)
        connections.connect(**client_params)
        logger.info(f"Initialized Milvus client with URI: {MILVUS_URI}")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Milvus client: {str(e)}")
        return None

def create_collection(dimension):
    """Create Milvus collection if it doesn't exist"""

    if milvus_client.has_collection(COLLECTION_NAME):
        logger.info(f"Collection {COLLECTION_NAME} already exists")
        return Collection(COLLECTION_NAME)

    # Create schema
    schema = milvus_client.create_schema()

    # Add fields
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=8000, enable_analyzer=True)
    schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=40000)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # Add BM25 function
    bm25_function = Function(
        name="question_bm25_emb",  # Function name
        input_field_names=["question"],  # Name of the VARCHAR field containing raw text data
        output_field_names=["sparse"],  # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
        function_type=FunctionType.BM25,  # Set to `BM25`
    )

    schema.add_function(bm25_function)

    # Prepare index parameters
    index_params = milvus_client.prepare_index_params()

    # Add dense vector index
    index_params.add_index(
        field_name="embedding",
        index_name="embedding_index",
        index_type="FLAT",
        metric_type="IP"
    )

    # Add sparse vector index for BM25
    index_params.add_index(
        field_name="sparse",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",  # Algorithm for building and querying the index
            "bm25_k1": 1.2,
            "bm25_b": 0.75
        },
    )

    # Create collection
    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )

    logger.info(f"Collection {COLLECTION_NAME} created successfully with BM25 support")
    return Collection(COLLECTION_NAME)

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class IndexStatusResponse(BaseModel):
    status: str
    message: str

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate embeddings for the provided texts"""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")

        if model is None:
            raise HTTPException(status_code=500, detail="Embedding model not loaded")

        logger.info(f"Generating embeddings for {len(request.texts)} texts")

        # Generate embeddings
        embeddings = model.encode(request.texts)

        # Convert numpy arrays to lists for JSON serialization
        embeddings_list = embeddings.tolist()

        logger.info(f"Successfully generated embeddings with shape: {embeddings.shape}")

        return EmbedResponse(embeddings=embeddings_list)

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@app.post("/index/excel", response_model=IndexStatusResponse)
async def index_excel(background_tasks: BackgroundTasks, file_path: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    """Index data from an Excel file into Milvus"""
    try:
        # Determine the file path
        excel_path = None
        if file:
            # Save uploaded file
            temp_file_path = f"/data/uploaded_{file.filename}"
            with open(temp_file_path, "wb") as buffer:
                buffer.write(await file.read())
            excel_path = temp_file_path
            logger.info(f"Uploaded file saved to {excel_path}")
        elif file_path:
            excel_path = file_path
            logger.info(f"Using provided file path: {excel_path}")
        else:
            excel_path = EXCEL_PATH
            logger.info(f"Using default file path from environment: {excel_path}")

        # Start indexing
        background_tasks.add_task(index_data_task, excel_path)
        # await index_data_task(excel_path)

        return IndexStatusResponse(
            status="processing",
            message=f"Indexing started for file: {excel_path}"
        )

    except Exception as e:
        logger.error(f"Error starting indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting indexing: {str(e)}")

async def index_data_task(excel_path):
    """Background task to index data"""
    try:
        # Load Excel data
        df = load_excel_data(excel_path)
        if df is None:
            logger.error(f"Failed to load data from {excel_path}")
            return

        # Get embedding dimension
        dimension = model.get_sentence_embedding_dimension()

        # Create or get collection
        collection = create_collection(dimension)

        # Index data
        success = index_data(collection, df)

        if success:
            logger.info("Indexing completed successfully")
        else:
            logger.error("Indexing failed")

    except Exception as e:
        logger.error(f"Error in indexing task: {str(e)}")

def load_excel_data(excel_path):
    """Load data from Excel file"""
    try:
        logger.info(f"Loading data from {excel_path}")
        df = pd.read_excel(excel_path)

        # Check if required columns exist
        required_columns = ["question", "answer"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in Excel file")
                return None

        # Remove rows with missing values
        df = df.dropna(subset=required_columns)
        logger.info(f"Loaded {len(df)} QA pairs from Excel")

        return df
    except Exception as e:
        logger.error(f"Error loading Excel data: {str(e)}")
        return None

def index_data(_, df):
    """Index data into Milvus"""
    try:
        # Prepare data
        questions = df["question"].tolist()
        answers = df["answer"].tolist()

        # Generate embeddings for questions
        logger.info("Generating embeddings for questions...")
        embeddings = model.encode(questions)

        # Insert data into collection
        # The sparse vectors for BM25 will be generated automatically by the BM25 function
        data = []
        for i in range(len(questions)):
            data.append({
                "question": questions[i],
                "answer": answers[i],
                "embedding": embeddings[i].tolist()
            })

        milvus_client.insert(collection_name=COLLECTION_NAME, data=data)

        logger.info(f"Successfully indexed {len(questions)} QA pairs")
        return True
    except Exception as e:
        logger.error(f"Error indexing data: {str(e)}")
        return False

@app.get("/collection/status")
async def collection_status():
    """Get the status of the collection"""
    try:
        if not utility.has_collection(COLLECTION_NAME):
            return {"status": "not_exists", "message": f"Collection {COLLECTION_NAME} does not exist"}

        collection = Collection(COLLECTION_NAME)
        stats = collection.schema.to_dict()

        return {
            "status": "exists",
            "name": COLLECTION_NAME,
            "schema": str(stats),
            "embedding_dimension": model.get_sentence_embedding_dimension() if model else "unknown"
        }
    except Exception as e:
        logger.error(f"Error getting collection status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting collection status: {str(e)}")

@app.delete("/collection")
async def delete_collection():
    """Delete the collection"""
    try:
        if not utility.has_collection(COLLECTION_NAME):
            return {"status": "not_exists", "message": f"Collection {COLLECTION_NAME} does not exist"}

        utility.drop_collection(COLLECTION_NAME)

        return {"status": "deleted", "message": f"Collection {COLLECTION_NAME} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_hybrid: bool = True

class SearchResponse(BaseModel):
    results: List[dict]

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search the collection using hybrid search (vector + BM25)"""
    try:
        if not utility.has_collection(COLLECTION_NAME):
            raise HTTPException(status_code=404, detail=f"Collection {COLLECTION_NAME} does not exist")

        # Generate embedding for the query
        query_embedding = model.encode(request.query).tolist()

        if request.use_hybrid:
            # Hybrid search (vector + BM25)
            ranker = RRFRanker(100)

            search_param_1 = {
                "data": [query_embedding],
                "anns_field": "embedding",
                "param": {
                    "metric_type": "IP",
                },
                "limit": request.top_k,
            }
            request_1 = AnnSearchRequest(**search_param_1)

            search_param_2 = {
                "data": [request.query],
                "anns_field": "sparse",
                "param": {
                    "metric_type": "BM25",
                },
                "limit": request.top_k,
            }
            request_2 = AnnSearchRequest(**search_param_2)


            results = milvus_client.hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=[request_1, request_2],
                ranker=ranker,
                limit=request.top_k,
                output_fields=["question", "answer"],
            )

        else:
            # Vector search only
            results = milvus_client.search(
                collection_name=COLLECTION_NAME,
                data=[query_embedding],
                anns_field="embedding",
                search_params={"metric_type": "IP",},
                limit=request.top_k,
                output_fields=["question", "answer"]
            )

        # Format results
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                "id": hit["id"],
                "question": hit["entity"]["question"],
                "answer": hit["entity"]["answer"],
                "score": hit["distance"]
            })

        return SearchResponse(results=formatted_results)

    except Exception as e:
        logger.error(f"Error searching collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching collection: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "milvus_uri": MILVUS_URI,
        "model_loaded": model is not None,
        "model_name": EMBEDDING_MODEL,
        "embedding_dimension": model.get_sentence_embedding_dimension() if model else None,
        "huggingface_cache": {
            "hf_home": HF_HOME,
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
