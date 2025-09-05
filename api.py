from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import tempfile
import shutil
from pathlib import Path
import asyncio
import logging

# Import your modules
from doc_process import process_document
from doc_retrieval import DocumentRetriever, RetrievalConfig, enhanced_search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Processing and Retrieval API",
    description="Upload documents and query them using Azure OpenAI",
    version="1.0.0"
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    max_results: int = 5
    min_similarity: float = 0.6
    source_filter: Optional[str] = None
    enable_enhancement: bool = True
    enable_context: bool = True

class QueryResponse(BaseModel):
    query: str
    enhanced_query: Optional[str]
    total_results: int
    search_time: float
    results: list

class ProcessResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

# Global retriever instance
retriever: Optional[DocumentRetriever] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the document retriever on startup."""
    global retriever
    try:
        config = RetrievalConfig(
            max_results=10,
            min_similarity_threshold=0.5,
            enable_query_enhancement=True,
            llm_model="azure_openai/gpt-4"  # Using Azure OpenAI
        )
        retriever = DocumentRetriever(config)
        await retriever.initialize()
        logger.info("Document retriever initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global retriever
    if retriever:
        try:
            await retriever.close()
            logger.info("Document retriever closed successfully")
        except Exception as e:
            logger.error(f"Error closing retriever: {e}")

@app.post("/upload", response_model=ProcessResponse)
async def upload_document(
    file: UploadFile = File(...),
    description: str = Form(None)
):
    """
    Upload and process a document.
    
    Args:
        file: The document file to upload
        description: Optional description for the document
    
    Returns:
        Processing results and statistics
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file type (basic validation)
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Allowed: {allowed_extensions}"
        )
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = None
    
    try:
        # Save uploaded file to temporary location
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing file: {file.filename}")
        
        # Process the document
        result = await process_document(temp_file_path)
        
        return ProcessResponse(
            status="success",
            message=f"Document '{file.filename}' processed successfully",
            details=result
        )
        
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query processed documents using semantic search.
    
    Args:
        request: Query parameters including search text and options
        
    Returns:
        Search results with similarity scores and metadata
    """
    global retriever
    
    if not retriever:
        raise HTTPException(status_code=500, detail="Document retriever not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Perform enhanced search using the retriever
        results = await enhanced_search(
            query=request.query,
            source_filter=request.source_filter,
            max_results=request.max_results,
            min_similarity=request.min_similarity,
            enable_context=request.enable_context,
            enable_enhancement=request.enable_enhancement
        )
        
        # Convert to response format
        response = QueryResponse(
            query=results.query,
            enhanced_query=results.enhanced_query,
            total_results=results.total_results,
            search_time=results.search_time,
            results=[r.to_dict(full_content=False, preview_chars=500) for r in results.results]
        )
        
        logger.info(f"Query completed: {results.total_results} results in {results.search_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/query/{query_text}")
async def quick_query(
    query_text: str,
    max_results: int = 5,
    min_similarity: float = 0.6
):
    """
    Quick query endpoint using GET method.
    
    Args:
        query_text: The search query
        max_results: Maximum number of results to return
        min_similarity: Minimum similarity threshold
        
    Returns:
        Search results
    """
    request = QueryRequest(
        query=query_text,
        max_results=max_results,
        min_similarity=min_similarity
    )
    return await query_documents(request)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global retriever
    
    status = {
        "status": "healthy",
        "retriever_initialized": retriever is not None,
        "message": "API is running"
    }
    
    if retriever:
        try:
            # Test database connection
            db_health = await retriever.vector_db.health_check()
            status["database"] = db_health
        except Exception as e:
            status["database"] = {"status": "error", "message": str(e)}
            status["status"] = "degraded"
    
    return status

@app.get("/stats")
async def get_database_stats():
    """Get database statistics."""
    global retriever
    
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    try:
        stats = await retriever.vector_db.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# Or use the code below if running directly

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI server
    uvicorn.run(
        app,  # Direct reference to the app instance
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )