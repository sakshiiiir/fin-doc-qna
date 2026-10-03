import os

import requests
import streamlit as st

API_URL = os.environ.get("FINSIGHT_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 300  # local inference on a laptop can take a while

st.set_page_config(page_title="Ask your financial documents", page_icon="📊")
st.title("Ask your financial documents")

# ── session state ──────────────────────────────────────────────────────────
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
    st.session_state.filename = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str, "sources": list|None}


def esc(text: str) -> str:
    """Streamlit treats $...$ as LaTeX, which swallows dollar signs in financial text."""
    return text.replace("$", "\\$")


def call_api(path: str, **kwargs):
    """POST to the backend. Returns (json, None) on success or (None, friendly_error)."""
    try:
        res = requests.post(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError:
        return None, f"Can't reach the backend at {API_URL}. Is it running?"
    except requests.exceptions.Timeout:
        return None, "The backend took too long to respond."
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"
    if not res.ok:
        try:
            detail = res.json().get("detail", res.text)
        except ValueError:
            detail = res.text
        return None, f"{res.status_code}: {detail}"
    return res.json(), None


def backend_status() -> str:
    try:
        data = requests.get(f"{API_URL}/health", timeout=3).json()
    except requests.exceptions.RequestException:
        return "down"
    if data.get("error"):
        return "failed"
    return "ready" if data.get("model_ready", True) else "loading"


# ── sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    status = backend_status()
    st.caption(
        {
            "ready": "🟢 Backend ready",
            "loading": "🟡 Model loading, give it a minute and refresh",
            "failed": "🔴 Model failed to load, check the backend log",
            "down": "🔴 Backend not reachable",
        }[status]
    )

    st.header("Document")
    uploaded_file = st.file_uploader("Upload a 10-K or earnings PDF", type="pdf")
    if uploaded_file and uploaded_file.name != st.session_state.filename:
        with st.spinner("Indexing document..."):
            data, err = call_api(
                "/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            )
        if err:
            st.error(f"Upload failed: {err}")
        else:
            st.session_state.doc_id = data["doc_id"]
            st.session_state.filename = uploaded_file.name
            st.session_state.messages = []  # new doc → fresh chat
            st.success(f"Indexed {data['num_chunks']} chunks")

    st.header("Settings")
    model_choice = st.selectbox("Model", ["Fine-Tuned FinSight", "Base Llama"])
    top_k = st.slider("Top-k chunks", min_value=1, max_value=6, value=3)

# ── chat area ──────────────────────────────────────────────────────────────
if not st.session_state.doc_id:
    st.info("Upload a financial PDF from the sidebar to get started.")
else:
    # Render existing conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="📊" if msg["role"] == "assistant" else None):
            st.markdown(esc(msg["content"]))
            if msg.get("sources"):
                with st.expander("Source chunks"):
                    for chunk in msg["sources"]:
                        st.markdown(f"> {esc(chunk)}")

    # Chat input at the bottom
    if question := st.chat_input("Ask a question about this filing"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(esc(question))

        # Generate answer
        model_param = "finetuned" if model_choice == "Fine-Tuned FinSight" else "base"
        with st.chat_message("assistant", avatar="📊"):
            with st.spinner("Thinking..."):
                data, err = call_api(
                    "/query",
                    json={
                        "doc_id": st.session_state.doc_id,
                        "question": question,
                        "top_k": top_k,
                        "model": model_param,
                    },
                )
            if err:
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ {err}", "sources": None}
                )
            else:
                st.markdown(esc(data["answer"]))
                st.caption(f"Model: {data['model_used']}")
                sources = data.get("sources", [])
                if sources:
                    with st.expander("Source chunks"):
                        for chunk in sources:
                            st.markdown(f"> {esc(chunk)}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": data["answer"], "sources": sources}
                )
