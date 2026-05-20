"""
MedAssist — Inspecteur ChromaDB
================================
Lance ce script pour voir le contenu de ta base ChromaDB.

Usage : python inspect_db.py
"""

import chromadb
from pathlib import Path

CHROMA_DIR = "chroma_db"

def main():
    if not Path(CHROMA_DIR).exists():
        print("❌ Le dossier chroma_db/ n'existe pas encore.")
        print("   Lance l'app d'abord : python -m streamlit run app.py")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection("medassist_docs")
    total = col.count()

    print(f"\n{'='*60}")
    print(f"  ChromaDB MedAssist — {total} chunks total")
    print(f"{'='*60}\n")

    data = col.get(include=["documents", "metadatas"])

    # Grouper par source
    sources = {}
    for doc, meta in zip(data["documents"], data["metadatas"]):
        src = meta.get("source", "?")
        page = meta.get("page", "?")
        if src not in sources:
            sources[src] = []
        sources[src].append({"text": doc, "page": page})

    for src, chunks in sources.items():
        is_web = src.startswith("web:")
        icon = "🌐" if is_web else "📄"
        print(f"{icon} {src} — {len(chunks)} chunk(s)")
        print(f"   {'─'*50}")
        for j, c in enumerate(chunks):
            preview = c["text"][:150].replace("\n", " ")
            print(f"   [{j+1}] {preview}...")
        print()

    # Stats
    doc_chunks = sum(len(c) for s, c in sources.items() if not s.startswith("web:"))
    web_chunks = sum(len(c) for s, c in sources.items() if s.startswith("web:"))
    print(f"{'='*60}")
    print(f"  📄 Documents : {doc_chunks} chunks")
    print(f"  🌐 Web       : {web_chunks} chunks")
    print(f"  📊 Total     : {total} chunks")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
