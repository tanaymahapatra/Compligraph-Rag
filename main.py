import streamlit as st
import requests

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="CompliGraph AI", page_icon="⚖️", layout="wide")

st.title("⚖️ CompliGraph AI")
st.markdown("""
*An Agentic Regulatory & Financial Compliance Assistant powered by **LangGraph**, **Qdrant**, **FastEmbed**, and **Gemini**.*
""")
st.divider()

# Sidebar options
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("FastAPI Backend: Connected (`localhost:8000`)")
    st.info("Embedding Model: `BAAI/bge-small-en-v1.5`")
    st.info("LLM: `gemini-3.7-flash`")

    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 2. CHAT SESSION STATE MANAGEMENT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 Retrieved Regulatory Context"):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"**Source Chunk {idx+1}:**\n>{src}\n")
        if "web_search" in msg and msg["web_search"]:
            st.caption("🌐 *Information supplemented by live web search.*")

# ==========================================
# 3. CHAT INPUT & HTTP REQUEST
# ==========================================
if user_query := st.chat_input(
    "Ask a question about SEC 10-K, RBI KYC directives, or Basel III..."
):

    # Render user message in UI
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Fetch response from FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("🤖 Consulting FastAPI Backend..."):
            try:
                # Send HTTP POST request to your local server
                response = requests.post(
                    "http://localhost:8000/query",
                    json={"question": user_query},
                    timeout=60,
                )
                response.raise_for_status()  # Raise exception for 4xx/5xx status codes

                # Parse the JSON response
                data = response.json()
                generation = data.get("generation", "No response generated.")
                documents = data.get("documents", [])
                web_search_used = data.get("web_search_used", False)

                # Display the answer
                st.markdown(generation)

                # Display expandable sources
                if documents:
                    with st.expander("📄 Retrieved Regulatory Context"):
                        for idx, doc in enumerate(documents):
                            st.markdown(f"**Source Chunk {idx+1}:**\n{doc}\n")

                # Highlight if Tavily was triggered
                if web_search_used:
                    st.caption("🌐 *Information supplemented by live web search.*")

                # Save assistant message to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": generation,
                        "sources": documents,
                        "web_search": web_search_used,
                    }
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "🚨 Connection Error: Ensure your FastAPI server (`server.py`) is running on port 8000."
                )
            except requests.exceptions.Timeout:
                st.error(
                    "⏳ Request Timed Out: The agent took too long to process the query."
                )
            except Exception as e:
                st.error(f"🚨 An error occurred: {str(e)}")
