"""
MedAssist — Interface Streamlit
================================
Interface web pour interroger les documents médicaux.
Utilise le pipeline RAG de rag_medical.py.
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from rag_medical import (
    load_documents,
    split_documents,
    create_vectorstore,
    build_prompt,
    answer_question,
    DATA_DIR,
    GROQ_MODEL,
    TOP_K,
)
from langchain_groq import ChatGroq

# --- Configuration de la page ---
st.set_page_config(
    page_title="MedAssist — Assistant Médical RAG",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Chargement du CSS ---
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_rag():
    """Initialise le pipeline RAG une seule fois (cache Streamlit)."""
    if "rag_ready" not in st.session_state:
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            st.error("⚠️ Clé GROQ_API_KEY manquante. Créez un fichier `.env`.")
            st.stop()

        with st.spinner("Chargement des documents médicaux..."):
            documents = load_documents(DATA_DIR)
            chunks = split_documents(documents)
            vectorstore = create_vectorstore(chunks)

            st.session_state.retriever = vectorstore.as_retriever(
                search_kwargs={"k": TOP_K}
            )
            st.session_state.llm = ChatGroq(
                model=GROQ_MODEL,
                temperature=0,
                api_key=groq_api_key,
            )
            st.session_state.prompt = build_prompt()
            st.session_state.num_chunks = len(chunks)
            st.session_state.num_docs = len(
                set(d.metadata.get("source", "") for d in documents)
            )
            st.session_state.rag_ready = True

    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar():
    """Barre latérale avec infos et configuration."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-header">
                <div class="logo-icon">🏥</div>
                <h2>MedAssist</h2>
                <p class="subtitle">Assistant Médical RAG</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Infos sur le système
        st.markdown("### 📊 Base de connaissances")
        if "rag_ready" in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", st.session_state.num_docs)
            with col2:
                st.metric("Chunks", st.session_state.num_chunks)

        st.markdown("---")

        # Liste des documents
        st.markdown("### 📁 Documents chargés")
        data_path = Path("data")
        if data_path.exists():
            for f in sorted(data_path.iterdir()):
                if f.suffix in [".pdf", ".txt"]:
                    icon = "📄" if f.suffix == ".pdf" else "📝"
                    st.markdown(f"{icon} `{f.name}`")

        st.markdown("---")

        # Pipeline RAG info
        st.markdown("### ⚙️ Pipeline RAG")
        st.markdown(
            """
            <div class="pipeline-info">
                <div class="pipeline-step">📄 Chargement docs</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">✂️ Chunking</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">🧮 Embeddings</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">🔍 FAISS Retrieval</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">🤖 Groq LLM</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            """
            <div class="tech-stack">
                <span class="tech-badge">LangChain</span>
                <span class="tech-badge">FAISS</span>
                <span class="tech-badge">Groq</span>
                <span class="tech-badge">Llama 3.3</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat():
    """Interface de chat principale."""
    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🏥 MedAssist</h1>
            <p>Posez vos questions sur les documents médicaux</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Affichage de l'historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

            # Afficher les sources si présentes
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Chunks récupérés", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"**{i}. {src['source']}** (page {src['page']})"
                        )
                        st.markdown(f"> {src['preview']}")

    # Input utilisateur
    if question := st.chat_input("Posez votre question médicale..."):
        # Ajouter la question
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(question)

        # Générer la réponse
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Recherche dans les documents..."):
                answer, docs = answer_question(
                    question,
                    st.session_state.retriever,
                    st.session_state.llm,
                    st.session_state.prompt,
                )

            st.markdown(answer)

            # Sources
            sources = []
            with st.expander("📚 Chunks récupérés", expanded=False):
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("source", "?")
                    page = doc.metadata.get("page", "?")
                    if isinstance(page, int):
                        page = page + 1
                    preview = doc.page_content[:200].replace("\n", " ")

                    st.markdown(f"**{i}. {source}** (page {page})")
                    st.markdown(f"> {preview}...")

                    sources.append(
                        {"source": source, "page": page, "preview": preview}
                    )

        # Sauvegarder la réponse
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


def main():
    init_rag()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
