from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import asyncio
import os
from functools import lru_cache
from dotenv import load_dotenv
from src.errors import UpstreamServiceError
from src.query_budget import reserve_query, DailyLimitReached

load_dotenv()


@lru_cache(maxsize=1)
def get_workflow():
    # Defer expensive model loading until the first query.
    from src.graph import app as langgraph_app

    return langgraph_app


query_lock = asyncio.Lock()
last_upstream_error = None

# ==========================================
# 1. INITIALIZE FASTAPI APP
# ==========================================
app = FastAPI(
    title="⚖️ CompliGraph AI API",
    description="RESTful backend for the Agentic Regulatory & Financial Compliance Assistant",
    version="1.0.0",
)

# Configure CORS so Streamlit (or React) can safely communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 2. PYDANTIC SCHEMAS (API CONTRACTS)
# ==========================================
class QueryRequest(BaseModel):
    """Expected JSON payload from the frontend."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The compliance question from the user.",
    )


class QueryResponse(BaseModel):
    """The JSON response structure sent back to the frontend."""

    question: str
    generation: str
    documents: List[str]
    web_search_used: bool


# ==========================================
# 3. API ENDPOINTS
# ==========================================
@app.get("/")
async def health_check():
    """Simple health check to verify the server is running."""
    missing = [
        key for key in ("GOOGLE_API_KEY", "TAVILY_API_KEY") if not os.getenv(key)
    ]
    return {
        "status": (
            "configuration_required"
            if missing
            else ("dependency_unavailable" if last_upstream_error else "active")
        ),
        "upstream_error": last_upstream_error,
        "missing_configuration": missing,
        "model": os.getenv("GEMINI_MODEL", "gemini-3.7-flash"),
        "vector_store": "remote" if os.getenv("QDRANT_URL") else "local",
        "message": "CompliGraph AI Backend is running. Access /docs for Swagger UI.",
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Ingests a user question, triggers the LangGraph state machine,
    and returns the generated answer and retrieved documents.
    """
    global last_upstream_error
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Configure Gemini and Tavily credentials before querying.",
        )
    if query_lock.locked():
        raise HTTPException(
            status_code=429,
            detail="Another question is being processed. Please retry shortly.",
        )
    try:
        # 1. Initialize the state memory with the user's question
        initial_state = {"question": request.question}

        # 2. Execute the LangGraph agent pipeline
        # One workflow at a time keeps peak RAM bounded on the 4 GB VM.
        async with query_lock:
            await asyncio.to_thread(reserve_query)
            langgraph_app = await asyncio.to_thread(get_workflow)
            result = await langgraph_app.ainvoke(initial_state)
            last_upstream_error = None

        # 3. Package and return the results
        raw_docs = result.get("documents", [])
        serialized_docs = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in raw_docs
        ]

        return QueryResponse(
            question=result.get("question", request.question),
            generation=result.get("generation", "Error generating response."),
            documents=serialized_docs,
            # 3. FIXED: Mismatched state key mapped correctly
            web_search_used=result.get("web_search_used", False),
        )
    except DailyLimitReached as e:
        raise HTTPException(status_code=429, detail=str(e)) from None
    except UpstreamServiceError as e:
        last_upstream_error = str(e)
        raise HTTPException(status_code=503, detail=str(e)) from None
    except Exception as e:
        # Generic error for the client; log the actual traceback internally
        print(f"Internal Graph Error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal Server Error processing request."
        )


# ==========================================
# 4. LOCAL DEVELOPMENT EXECUTION
# ==========================================
if __name__ == "__main__":
    import uvicorn

    # Runs the server on http://localhost:8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
