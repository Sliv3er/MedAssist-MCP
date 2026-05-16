"""
MedAssist — RAG Médical Avancé
================================
Pipeline RAG avec ChromaDB, Reranking et Mémoire de conversation.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# --- Paramètres ---
DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 10   # nombre de docs récupérés avant reranking
TOP_K_FINAL = 3        # nombre de docs après reranking


def get_embeddings():
    """Retourne le modèle d'embeddings HuggingFace."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_documents(data_dir: Path):
    """
    Charge tous les documents du dossier data/.
    Supporte les fichiers PDF (.pdf) et texte (.txt).
    """
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    txt_paths = sorted(data_dir.glob("*.txt"))

    all_paths = list(pdf_paths) + list(txt_paths)

    if not all_paths:
        raise FileNotFoundError(
            "Aucun document trouve dans le dossier 'data/'. "
            "Ajoutez des fichiers PDF ou TXT."
        )

    documents = []

    for file_path in all_paths:
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        pages = loader.load()

        for page in pages:
            page.metadata["source"] = file_path.name

        documents.extend(pages)

    return documents


def split_documents(documents):
    """Découpe les documents en chunks avec overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks=None, embeddings=None):
    """
    Crée ou charge la base ChromaDB persistante.
    Si chunks sont fournis, les indexe. Sinon, charge la base existante.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    if chunks:
        # Créer la base avec les documents
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_name="medassist_docs",
        )
        return vectorstore
    else:
        # Charger la base existante
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name="medassist_docs",
        )
        return vectorstore


def create_retriever(vectorstore):
    """
    Crée un retriever avec reranking Cross-Encoder.
    1. ChromaDB récupère TOP_K_RETRIEVAL documents (10)
    2. Cross-Encoder reranke et garde TOP_K_FINAL (3)
    """
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K_RETRIEVAL}
    )

    # Cross-Encoder pour le reranking
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(
        model=cross_encoder,
        top_n=TOP_K_FINAL,
    )

    # Compression retriever = retrieval + reranking
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    return retriever


def add_to_vectorstore(vectorstore, text, source="web_search"):
    """
    Ajoute du texte au ChromaDB (enrichissement depuis le web).
    """
    doc = Document(
        page_content=text,
        metadata={"source": source, "page": "web"},
    )

    chunks = split_documents([doc])
    vectorstore.add_documents(chunks)
    return len(chunks)


def build_prompt():
    """Construit le PromptTemplate avec mémoire de conversation."""
    template = """
Tu es un assistant médical spécialisé dans la recherche documentaire.
Tu dois répondre à partir du contexte fourni et de l'historique de conversation.

Consignes importantes :
- Réponds en français.
- Si l'information n'est pas présente dans le contexte, dis clairement :
  "Je ne trouve pas cette information dans les documents fournis."
- Donne une réponse claire, structurée et concise.
- Termine par une ligne "Sources :" avec les fichiers utilisés.

Historique de conversation :
{history}

Contexte :
{context}

Question :
{question}
"""
    return PromptTemplate.from_template(template)


def format_context(docs):
    """Formate les chunks récupérés en contexte lisible."""
    parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "source_inconnue")
        page = doc.metadata.get("page", "?")
        if isinstance(page, int):
            page = page + 1

        parts.append(
            f"[Extrait {i} | source={source} | page={page}]\n{doc.page_content}"
        )

    return "\n\n".join(parts)


def format_history(messages, max_turns=5):
    """Formate l'historique de conversation pour le prompt."""
    if not messages:
        return "Aucun historique."

    # Garder les N derniers échanges
    recent = messages[-(max_turns * 2):]
    parts = []

    for msg in recent:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        parts.append(f"{role} : {msg['content'][:300]}")

    return "\n".join(parts)


def answer_question(question, retriever, llm, prompt, history=""):
    """Récupère les chunks pertinents et génère une réponse."""
    docs = retriever.invoke(question)
    context = format_context(docs)

    formatted_prompt = prompt.format(
        context=context,
        question=question,
        history=history,
    )
    response = llm.invoke(formatted_prompt)

    return response.content, docs


def init_pipeline():
    """Initialise tout le pipeline RAG (pour usage console ou import)."""
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY manquante dans .env")

    embeddings = get_embeddings()

    # Vérifier si la base existe déjà
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print("Chargement de la base ChromaDB existante...")
        vectorstore = create_vectorstore(embeddings=embeddings)
    else:
        print("Indexation des documents...")
        documents = load_documents(DATA_DIR)
        chunks = split_documents(documents)
        vectorstore = create_vectorstore(chunks=chunks, embeddings=embeddings)
        print(f"Indexe {len(chunks)} chunks dans ChromaDB.")

    retriever = create_retriever(vectorstore)

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=groq_api_key,
    )
    prompt = build_prompt()

    return vectorstore, retriever, llm, prompt


def main():
    """Point d'entrée principal — mode console interactif."""
    vectorstore, retriever, llm, prompt = init_pipeline()

    print("\nAssistant RAG medical pret.")
    print("Tapez votre question ou 'quit' pour quitter.\n")

    history_msgs = []

    while True:
        question = input("Question > ").strip()

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Fin du programme.")
            break

        history = format_history(history_msgs)
        answer, docs = answer_question(question, retriever, llm, prompt, history)

        history_msgs.append({"role": "user", "content": question})
        history_msgs.append({"role": "assistant", "content": answer})

        print("\n--- Reponse ---")
        print(answer)

        print("\n--- Chunks recuperes (apres reranking) ---")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            if isinstance(page, int):
                page = page + 1
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"{i}. {source} | page {page}")
            print(f"   {preview}...")

        print()


if __name__ == "__main__":
    main()
