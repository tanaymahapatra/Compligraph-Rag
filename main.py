import streamlit as st
import requests

API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="CompliGraph AI", page_icon="⚖️", layout="wide")

st.title("⚖️ CompliGraph AI")
st.markdown("""
*An Agentic Regulatory & Financial Compliance Assistant powered by **LangGraph**, **Qdrant**, **FastEmbed**, and **Gemini**.*
""")
st.divider()

with st.sidebar:
    st.header("⚙️ System Status")
    st.success("FastAPI Backend: Connected")
    st.info("Embedding Model: `BAAI/bge-small-en-v1.5`")
    st.info("LLM: `gemini-3.7-flash`")

    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if "sources" in msg and msg["sources"]:
            with st.expander("📄 Retrieved Regulatory Context"):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"**Source Chunk {idx+1}:**\n>{src}\n")

        if "web_search" in msg and msg["web_search"]:
            st.caption("🌐 *Information supplemented by live web search.*")

if user_query := st.chat_input(
    "Ask a question about SEC 10-K, RBI KYC directives, or Basel III..."
):
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Consulting FastAPI Backend..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": user_query},
                    timeout=120,
                )

                response.raise_for_status()

                data = response.json()

                generation = data.get("generation", "No response generated.")
                documents = data.get("documents", [])
                web_search_used = data.get("web_search_used", False)

                st.markdown(generation)

                if documents:
                    with st.expander("📄 Retrieved Regulatory Context"):
                        for idx, doc in enumerate(documents):
                            st.markdown(f"**Source Chunk {idx+1}:**\n{doc}\n")

                if web_search_used:
                    st.caption("🌐 *Information supplemented by live web search.*")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": generation,
                        "sources": documents,
                        "web_search": web_search_used,
                    }
                )

            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the FastAPI backend.")

            except requests.exceptions.Timeout:
                st.error("⏳ Request timed out. The backend may still be waking up.")

            except Exception as e:
                st.error(f"🚨 An error occurred: {str(e)}")
