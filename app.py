"""
MedAssist — Interface Streamlit v2
====================================
Interface web avec mémoire de conversation, reranking et agent web.
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from rag_medical import (
    load_documents,
    split_documents,
    create_vectorstore,
    create_retriever,
    build_prompt,
    format_history,
    get_embeddings,
    DATA_DIR,
    CHROMA_DIR,
    GROQ_MODEL,
    TOP_K_FINAL,
)
from agent import agent_answer
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

        with st.spinner("Initialisation du pipeline RAG..."):
            embeddings = get_embeddings()

            # Vérifier si ChromaDB existe déjà
            if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
                vectorstore = create_vectorstore(embeddings=embeddings)
                # Compter les documents dans la collection
                collection = vectorstore._collection
                num_chunks = collection.count()
            else:
                documents = load_documents(DATA_DIR)
                chunks = split_documents(documents)
                vectorstore = create_vectorstore(
                    chunks=chunks, embeddings=embeddings
                )
                num_chunks = len(chunks)

            retriever = create_retriever(vectorstore)

            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.session_state.llm = ChatGroq(
                model=GROQ_MODEL,
                temperature=0,
                api_key=groq_api_key,
            )
            st.session_state.prompt = build_prompt()
            st.session_state.num_chunks = num_chunks
            st.session_state.num_docs = len(
                list(DATA_DIR.glob("*.pdf")) + list(DATA_DIR.glob("*.txt"))
            )
            st.session_state.web_enrichments = 0
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
                <p class="subtitle">Assistant Médical RAG v2</p>
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

            if st.session_state.web_enrichments > 0:
                st.metric(
                    "🌐 Enrichissements web",
                    st.session_state.web_enrichments,
                )

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

        # Pipeline RAG v2
        st.markdown("### ⚙️ Pipeline RAG v2")
        st.markdown(
            """
            <div class="pipeline-info">
                <div class="pipeline-step">📄 Chargement docs</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">✂️ Chunking</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">🧮 Embeddings</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step step-new">🗄️ ChromaDB</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step step-new">🔀 Reranking</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step step-new">🤖 Agent</div>
                <div class="pipeline-arrow">↓</div>
                <div class="pipeline-step">📚 Docs / 🌐 Web</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            """
            <div class="tech-stack">
                <span class="tech-badge">LangChain</span>
                <span class="tech-badge">ChromaDB</span>
                <span class="tech-badge">Groq</span>
                <span class="tech-badge">Reranking</span>
                <span class="tech-badge">Agent</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Bouton reset conversation
        if st.button("🗑️ Effacer la conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_chat():
    """Interface de chat principale."""
    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🏥 MedAssist</h1>
            <p>Assistant médical intelligent avec recherche documentaire et web</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Affichage de l'historique
    for msg in st.session_state.messages:
        avatar = "🧑‍⚕️" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            # Badge source
            if msg["role"] == "assistant" and "source_type" in msg:
                if msg["source_type"] == "web":
                    st.markdown(
                        '<span class="source-badge web-badge">🌐 Recherche Web</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span class="source-badge doc-badge">📚 Documents</span>',
                        unsafe_allow_html=True,
                    )

            st.markdown(msg["content"])

            # Afficher les sources si présentes
            if msg["role"] == "assistant" and "sources" in msg:
                label = "🌐 Sources web" if msg.get("source_type") == "web" else "📚 Chunks récupérés"
                with st.expander(label, expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"**{i}. {src['source']}** (page {src['page']})"
                        )
                        st.markdown(f"> {src['preview']}")

            # Notification enrichissement
            if msg["role"] == "assistant" and msg.get("web_added", 0) > 0:
                st.success(
                    f"✅ {msg['web_added']} chunk(s) ajouté(s) à ChromaDB depuis le web"
                )

    # Input utilisateur
    if question := st.chat_input("Posez votre question médicale..."):
        # Ajouter la question
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(question)

        # Générer la réponse via l'agent
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Recherche et analyse..."):
                history = format_history(st.session_state.messages)

                answer, docs, source_type, web_added = agent_answer(
                    question=question,
                    retriever=st.session_state.retriever,
                    llm=st.session_state.llm,
                    prompt=st.session_state.prompt,
                    vectorstore=st.session_state.vectorstore,
                    history=history,
                )

            # Badge source
            if source_type == "web":
                st.markdown(
                    '<span class="source-badge web-badge">🌐 Recherche Web</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="source-badge doc-badge">📚 Documents</span>',
                    unsafe_allow_html=True,
                )

            st.markdown(answer)

            # Sources
            sources = []
            label = "🌐 Sources web" if source_type == "web" else "📚 Chunks récupérés"
            with st.expander(label, expanded=False):
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

            # Notification enrichissement
            if web_added > 0:
                st.success(
                    f"✅ {web_added} chunk(s) ajouté(s) à ChromaDB depuis le web"
                )
                st.session_state.web_enrichments += web_added
                st.session_state.num_chunks += web_added

        # Sauvegarder la réponse
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "source_type": source_type,
            "web_added": web_added,
        })


def main():
    init_rag()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
