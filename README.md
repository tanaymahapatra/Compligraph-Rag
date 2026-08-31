<div align="center">

<h1>⚖️ CompliGraph AI</h1>

<p><strong>Agentic RAG for Financial & Regulatory Compliance — Live on Google Cloud</strong></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-StateGraph-7B2FBE?style=for-the-badge&logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C?style=for-the-badge&logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini-3.7%20Flash-FF6D00?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Cloud-Deployed-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white" />
  <img src="https://img.shields.io/badge/RAGAS-Evaluated-22C55E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-6366F1?style=for-the-badge" />
</p>

<h3>
  🌐 <a href="https://your-compligraph-demo.com">Live Demo</a> &nbsp;|&nbsp;
  📖 <a href="#-quick-start">Quick Start</a> &nbsp;|&nbsp;
  📊 <a href="#-evaluation-results">Evaluation Results</a> &nbsp;|&nbsp;
  🏗️ <a href="#️-architecture">Architecture</a>
</h3>

> 🔗 **Live App:** https://35.209.123.193

<p><em>A production-grade, Google Cloud-deployed agentic RAG system that answers financial and regulatory compliance questions with <strong>verifiable</strong>, source-backed precision — not hallucinated confidence.</em></p>

</div>

---

## 📖 What is CompliGraph AI?

Regulatory documents are notoriously difficult to retrieve from. Answers depend on **exact thresholds, circular identifiers, clause numbers, and dates**. A wrong or partially-relevant passage produces an answer that is *confidently incorrect* — which in a compliance setting is far worse than no answer at all.

**CompliGraph AI** is a fully deployed, end-to-end agentic RAG system that solves this with a three-pronged approach:

| Problem | Solution |
|---|---|
| Dense retrieval misses exact regulatory IDs | **Hybrid dense + sparse (BM25) retrieval** fused with Reciprocal Rank Fusion |
| Reranking is arbitrary | **Cross-encoder reranking** (`BAAI/bge-reranker-base`) over fused candidates |
| LLMs hallucinate regulatory facts | **LLM Compliance Auditor** that verifies every claim before returning an answer |
| Single-query retrieval is brittle | **Multi-query expansion** rewrites questions into up to 3 regulatory search queries |
| Vector store may return nothing useful | **Tavily web-search fallback** when retrieval confidence is low |
| API abuse and runaway costs | **Daily query budget** (`query_budget.py`) and **input guardrail** with domain filtering |

The corpus includes **SEC Form 10-K filings**, **RBI KYC and banking circulars**, and **Basel III / academic banking-regulation working papers**.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌──────────────────────┐
│   Input Guardrail    │  ← Rejects off-domain queries & prompt injection
└───────────┬──────────┘
            │ safe
            ▼
┌──────────────────────┐
│   Query Rewriter     │  ← Rewrites into 1–3 precise regulatory search queries
└───────────┬──────────┘
            │
            ▼
┌───────────────────────────────────────────────────┐
│               Hybrid Retrieval (Qdrant)            │
│  Dense (BAAI/bge-small-en-v1.5) ──┐               │
│                                    ├─ RRF Fusion   │
│  Sparse (BM25 via FastEmbed) ─────┘               │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Cross-Encoder        │  ← BAAI/bge-reranker-base (top 11 of 14)
            │  Reranking            │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  Document Grader      │  ← Are retrieved docs relevant enough?
            └──────┬──────┬─────────┘
               Yes │      │ No
                   │      ▼
                   │  ┌───────────────────┐
                   │  │ Tavily Web Search  │  ← Live fallback
                   │  └──────┬────────────┘
                   │         │
                   └────┬────┘
                        ▼
             ┌──────────────────────┐
             │   Generate Answer    │  ← Gemini 3.7 Flash (Pydantic structured)
             └──────────┬───────────┘
                        │
             ┌──────────▼────────────┐
             │  Compliance Auditor   │  ← Second LLM call verifies every claim
             └──────────┬────────────┘
                   Pass │    Fail → regenerate (max 2 retries)
                        ▼
                   Final Answer
```

### State Machine (LangGraph)

The entire pipeline is a **`StateGraph`** with typed `GraphState` and explicit conditional routers between all 8 nodes. This enables:
- **Deterministic, inspectable control flow** — no magic prompt chaining
- **Bounded audit loops** — max 2 compliance-check regeneration attempts
- **Full state snapshot** at every node for debugging and observability

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| 🔍 **Hybrid Retrieval** | Dense + BM25 sparse retrieval, server-side RRF fusion in Qdrant |
| 🏆 **Cross-Encoder Reranking** | `BAAI/bge-reranker-base` selects the best 11 from 14 fused candidates |
| 🔄 **Multi-Query Expansion** | LLM expands question into up to 3 distinct regulatory queries |
| 🔒 **Input Guardrail** | Domain + prompt-injection classification before entering the pipeline |
| 🕵️ **LLM Compliance Auditor** | Separate Gemini call verifies every factual claim with structured feedback |
| 🌐 **Tavily Web-Search Fallback** | Auto-triggers when vector retrieval returns low-confidence results |
| 💰 **Daily Query Budget** | `src/query_budget.py` — rate-limits API usage to prevent cost overruns |
| 🚨 **Structured Error Handling** | `src/errors.py` — `UpstreamServiceError` with propagation to frontend |
| 📐 **Pydantic Structured Outputs** | All LLM calls use `.with_structured_output()` — zero free-text parsing |
| 📄 **Docling PDF/HTML Parsing** | Handles complex regulatory PDFs with structural fidelity |
| 📊 **RAGAS Evaluation** | Full benchmark suite on 100 Q&A pairs with `final_evaluation_report.csv` |
| 🎛️ **Optuna Hyperparameter Tuning** | Retrieval parameters tuned against benchmark, not guessed |
| ☁️ **Google Cloud Deployed** | Live production deployment — backend URL configurable via env var |

---

## 📂 Project Structure

```
Compligraph-Rag/
├── main.py                          # Streamlit chat UI (cloud-ready, env-configurable)
├── server.py                        # FastAPI REST backend (lazy-loading, query budget, CORS via env)
├── requirements.txt                 # Full pinned dependency list
├── .gitignore
│
├── src/
│   ├── graph.py                     # LangGraph StateGraph — 8 nodes, 3 routers
│   ├── nodes.py                     # All pipeline node implementations
│   ├── state.py                     # GraphState — typed shared state schema
│   ├── errors.py                    # Custom exceptions (UpstreamServiceError)
│   └── query_budget.py              # Daily API rate-limiter (DailyLimitReached)
│
├── scripts/
│   └── injegst_data.py              # Data ingestion: PDF/HTML → Qdrant vectors
│
└── evaluation/
    ├── compliance_benchmark.csv     # 100-question Q&A benchmark dataset
    ├── evaluator.py                 # RAGAS evaluation runner
    ├── run_benchmark.py             # End-to-end benchmark execution script
    ├── tune_retriever.py            # Optuna hyperparameter search
    ├── final_evaluation_report.csv  # Full per-question RAGAS scores (100 rows × 13 cols)
    └── review_evaluations.ipynb     # Results analysis notebook
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- A [Qdrant](https://qdrant.tech/) instance (cloud or local)
- API keys for **Google Gemini** and **Tavily Search**

### 1. Clone the repository

```bash
git clone https://github.com/tanaymahapatra/Compligraph-Rag.git
cd Compligraph-Rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
# ── Google Gemini ────────────────────────────────────────────
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=gemini-3.7-flash           # Optional: override the model

# ── Qdrant Vector Database ───────────────────────────────────
QDRANT_URL=https://your-instance.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=compligraph_docs      # Optional: override collection name

# ── Tavily Web Search ─────────────────────────────────────────
TAVILY_API_KEY=your_tavily_api_key

# ── FastAPI Backend URL (for Streamlit frontend) ──────────────
COMPLIGRAPH_API_URL=http://127.0.0.1:8001   # or your Cloud Run URL

# ── CORS (comma-separated allowed origins) ────────────────────
CORS_ORIGINS=http://localhost:8501,https://your-streamlit-app.com

# ── FastEmbed model cache (optional) ─────────────────────────
FASTEMBED_CACHE_PATH=/path/to/cache
```

### 5. Ingest your regulatory documents

```bash
python scripts/ingest_data.py
```

> Parses PDF/HTML regulatory documents and indexes them into Qdrant with both dense (`BAAI/bge-small-en-v1.5`) and sparse (`Qdrant/bm25`) vectors.

### 6. Start the FastAPI backend

```bash
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

- API root: `http://localhost:8001/`
- Interactive docs: `http://localhost:8001/docs`

### 7. Start the Streamlit frontend

```bash
streamlit run main.py
```

Open `http://localhost:8501` to start chatting with CompliGraph AI.

---

## ☁️ Google Cloud Deployment

The backend and frontend are both designed for Google Cloud deployment with **zero hardcoded URLs**.

### Backend (Cloud Run)

```bash
# Build and deploy server.py as a Cloud Run service
gcloud run deploy compligraph-api \
  --source . \
  --region us-central1 \
  --set-env-vars GOOGLE_API_KEY=...,QDRANT_URL=...,TAVILY_API_KEY=... \
  --set-env-vars CORS_ORIGINS=https://your-streamlit-url.com
```

### Frontend (Streamlit Cloud / Cloud Run)

Set the environment variable to point to your deployed backend:

```env
COMPLIGRAPH_API_URL=https://compligraph-api-xxxx-uc.a.run.app
```

The frontend performs a **live health check** on startup and displays backend status in the sidebar — showing `Connected` or a meaningful warning if API credentials are missing.

---

## 🔌 API Reference

### `GET /`

Health check — verifies server is running and all required API keys are present.

**Response (healthy):**
```json
{ "status": "active" }
```

**Response (missing credentials):**
```json
{ "status": "degraded", "upstream_error": "Missing env vars: TAVILY_API_KEY" }
```

---

### `POST /query`

Submit a compliance question to the full agentic pipeline.

**Request:**
```json
{
  "question": "What are the KYC requirements for high-risk customers under RBI circular UBD.BPD.CO./NSB1/11/12.03.000?"
}
```

**Response:**
```json
{
  "question": "What are the KYC requirements...",
  "generation": "According to RBI circular UBD.BPD..., high-risk customers must...",
  "documents": ["Source chunk 1...", "Source chunk 2..."],
  "web_search_used": false
}
```

**Error — Daily limit reached (429):**
```json
{ "detail": "Daily query limit reached. Please try again tomorrow." }
```

---

## 📊 Evaluation Results

CompliGraph AI was evaluated with **RAGAS** on a **100-question compliance benchmark** covering SEC 10-K filings, RBI KYC circulars, and Basel III documents.

### 🏅 Benchmark Scores

> Evaluated with Gemini 3.7 Flash as judge LLM. Dataset: `evaluation/final_evaluation_report.csv` — 100 rows × 13 columns.

| Metric | Score | Interpretation |
|---|---|---|
| 🎯 **Faithfulness** | **0.956** | 95.6% of claims are directly supported by retrieved sources — near-zero hallucination |
| ✅ **Answer Correctness** | **0.832** | 83.2% factual alignment with verified ground-truth answers |
| 🔍 **Context Precision** | **0.788** | ~79% of retrieved chunks are directly relevant — hybrid RRF paying off |
| 📖 **Context Recall** | **0.860** | Pipeline retrieves 86% of information needed to correctly answer each question |

```
faithfulness          0.956
answer_correctness    0.832
context_precision     0.788
context_recall        0.860
```

### 📈 Score Interpretation

| Score | Why It Matters |
|---|---|
| **Faithfulness 0.956** | In compliance, hallucination = liability. 95.6% groundedness is production-grade |
| **Answer Correctness 0.832** | Regulatory Q&A is among the hardest NLP tasks; 0.83+ is strong |
| **Context Precision 0.788** | Confirms hybrid retrieval + reranking surfaces relevant, low-noise context |
| **Context Recall 0.860** | Multi-query expansion is successfully capturing coverage across the corpus |

### ▶️ Reproduce the Evaluation

```bash
# Run end-to-end evaluation against the live pipeline
python evaluation/run_benchmark.py

# Tune retrieval hyperparameters with Optuna
python evaluation/tune_retriever.py

# Review results interactively
jupyter notebook evaluation/review_evaluations.ipynb
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini 3.7 Flash |
| **Orchestration** | LangGraph `StateGraph` |
| **Vector DB** | Qdrant (cloud or local) |
| **Dense Embeddings** | `BAAI/bge-small-en-v1.5` via FastEmbed |
| **Sparse Embeddings** | `Qdrant/bm25` via FastEmbed |
| **Retrieval Fusion** | Reciprocal Rank Fusion (server-side in Qdrant) |
| **Reranker** | `BAAI/bge-reranker-base` via sentence-transformers |
| **Web Search** | Tavily Search API |
| **PDF Parsing** | Docling |
| **Validation** | Pydantic v2 structured outputs |
| **API Backend** | FastAPI + Uvicorn |   
| **Frontend** | Streamlit |
| **Evaluation** | RAGAS |
| **Hyperparameter Tuning** | Optuna |
| **Deployment** | Google Cloud Run |

---

## ⚙️ Configuration Reference

### Retrieval parameters (`src/nodes.py`)

| Parameter | Default | Description |
|---|---|---|
| `DENSE_LIMIT` | `14` | Max dense retrieval candidates from Qdrant |
| `SPARSE_LIMIT` | `14` | Max BM25 sparse retrieval candidates |
| `RRF_LIMIT` | `14` | Max candidates after RRF fusion |
| `FINAL_DOCS` | `11` | Docs passed to generation after reranking |
| `RERANK_THRESHOLD` | `0.00` | Minimum cross-encoder score to retain a document |
| `MAX_AUDIT_ATTEMPTS` | `2` | Max compliance-audit regeneration loops |

### Environment variables (`server.py` / `main.py`)

| Variable | Default | Description |
|---|---|---|
| `COMPLIGRAPH_API_URL` | `http://127.0.0.1:8001` | Backend URL for Streamlit frontend |
| `CORS_ORIGINS` | `http://localhost:8501,...` | Comma-separated allowed CORS origins |
| `GEMINI_MODEL` | `gemini-3.7-flash` | Gemini model override |
| `QDRANT_URL` | _(empty → local)_ | Qdrant cloud URL |
| `QDRANT_API_KEY` | — | Qdrant cloud API key |
| `QDRANT_COLLECTION` | `compligraph_docs` | Qdrant collection name |
| `FASTEMBED_CACHE_PATH` | — | Local cache dir for embedding models |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Open a pull request with a clear description

Please open an **Issue** first for major changes.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

| Project | Role |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful multi-step agent orchestration |
| [Qdrant](https://qdrant.tech/) | Hybrid vector search with RRF fusion |
| [FastEmbed](https://github.com/qdrant/fastembed) | Lightweight dense + sparse embeddings |
| [BAAI](https://huggingface.co/BAAI) | BGE embedding & reranker models |
| [Docling](https://github.com/DS4SD/docling) | Intelligent regulatory PDF parsing |
| [RAGAS](https://docs.ragas.io/) | RAG evaluation framework |
| [Optuna](https://optuna.org/) | Hyperparameter optimisation |
| [Google Cloud Run](https://cloud.google.com/run) | Serverless container deployment |

---

<div align="center">
<p>Built with ❤️ by <a href="https://github.com/tanaymahapatra">tanaymahapatra</a></p>
<p><em>If this project helped you, consider giving it a ⭐!</em></p>
</div>
