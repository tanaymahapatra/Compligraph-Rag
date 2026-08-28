from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

# Import the compiled LangGraph workflow
from src.graph import app as langgraph_app

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
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 2. PYDANTIC SCHEMAS (API CONTRACTS)
# ==========================================
class QueryRequest(BaseModel):
    """Expected JSON payload from the frontend."""

    question: str = Field(..., description="The compliance question from the user.")


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
    return {
        "status": "active",
        "message": "CompliGraph AI Backend is running. Access /docs for Swagger UI.",
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Ingests a user question, triggers the LangGraph state machine,
    and returns the generated answer and retrieved documents.
    """
    try:
        # 1. Initialize the state memory with the user's question
        initial_state = {"question": request.question}

        # 2. Execute the LangGraph agent pipeline
        result = await langgraph_app.ainvoke(initial_state)

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
