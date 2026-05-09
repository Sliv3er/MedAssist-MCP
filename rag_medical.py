"""
MedAssist — RAG Médical avec LangChain, FAISS et Groq
=====================================================
Pipeline RAG simple pour interroger des documents médicaux.
Basé sur le TP2 du module IA Générative.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- Paramètres ---
DATA_DIR = Path("data")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3


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


def create_vectorstore(chunks):
    """Crée les embeddings et l'index FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def build_prompt():
    """Construit le PromptTemplate pour le domaine médical."""
    template = """
Tu es un assistant médical spécialisé dans la recherche documentaire.
Tu dois répondre uniquement à partir du contexte fourni.

Consignes importantes :
- Réponds en français.
- Si l'information n'est pas présente dans le contexte, dis clairement :
  "Je ne trouve pas cette information dans les documents fournis."
- Donne une réponse claire, structurée et concise.
- Termine par une ligne "Sources :" avec les fichiers utilisés.

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


def answer_question(question, retriever, llm, prompt):
    """Récupère les chunks pertinents et génère une réponse."""
    docs = retriever.invoke(question)
    context = format_context(docs)

    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)

    return response.content, docs


def main():
    """Point d'entrée principal — mode console interactif."""
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERREUR : La cle GROQ_API_KEY est absente.")
        print("Creez un fichier .env avec : GROQ_API_KEY=votre_cle")
        return

    print("Chargement des documents...")
    documents = load_documents(DATA_DIR)
    print(f"Nombre total de pages chargees : {len(documents)}")

    print("Decoupage en chunks...")
    chunks = split_documents(documents)
    print(f"Nombre total de chunks : {len(chunks)}")

    print("Creation des embeddings et de l'index FAISS...")
    vectorstore = create_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=groq_api_key,
    )
    prompt = build_prompt()

    print("\nAssistant RAG medical pret.")
    print("Tapez votre question ou 'quit' pour quitter.\n")

    while True:
        question = input("Question > ").strip()

        if not question:
            print("Veuillez saisir une question.\n")
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Fin du programme.")
            break

        answer, docs = answer_question(question, retriever, llm, prompt)

        print("\n--- Reponse ---")
        print(answer)

        print("\n--- Chunks recuperes ---")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "source_inconnue")
            page = doc.metadata.get("page", "?")
            if isinstance(page, int):
                page = page + 1

            preview = doc.page_content[:250].replace("\n", " ")
            print(f"{i}. {source} | page {page}")
            print(f"   {preview}...")

        print()


if __name__ == "__main__":
    main()
