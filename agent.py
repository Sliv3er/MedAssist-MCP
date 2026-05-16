"""
MedAssist — Agent avec Recherche Web
======================================
Quand les documents ne contiennent pas la réponse,
l'agent cherche sur le web (DuckDuckGo) et enrichit ChromaDB.
"""

from langchain_core.documents import Document
from duckduckgo_search import DDGS

from rag_medical import (
    format_context,
    add_to_vectorstore,
)


def evaluate_context_relevance(docs, question, llm):
    """
    Demande au LLM d'évaluer si les chunks récupérés sont pertinents.
    Retourne True si le contexte est suffisant, False sinon.
    """
    if not docs:
        return False

    context = format_context(docs)

    eval_prompt = f"""Évalue si le contexte suivant contient des informations
pertinentes pour répondre à cette question.

Question : {question}

Contexte :
{context}

Réponds UNIQUEMENT par "OUI" ou "NON".
- OUI = le contexte contient des informations utiles pour répondre
- NON = le contexte ne contient pas d'informations pertinentes"""

    response = llm.invoke(eval_prompt)
    answer = response.content.strip().upper()

    return "OUI" in answer


def web_search(query, max_results=3):
    """
    Recherche sur le web avec DuckDuckGo.
    Retourne une liste de résultats (titre, body, href).
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        print(f"Erreur recherche web : {e}")
        return []


def format_web_results(results):
    """Formate les résultats web en texte lisible."""
    if not results:
        return ""

    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        parts.append(f"[Résultat web {i}]\nTitre: {title}\nContenu: {body}\nURL: {href}")

    return "\n\n".join(parts)


def agent_answer(question, retriever, llm, prompt, vectorstore, history=""):
    """
    Agent intelligent :
    1. Cherche dans ChromaDB (avec reranking)
    2. Évalue la pertinence du contexte
    3. Si pertinent → répond depuis les documents
    4. Si non pertinent → cherche sur le web → répond → enrichit ChromaDB
    
    Retourne : (answer, docs, source_type, web_added)
    - source_type : "documents" ou "web"
    - web_added : nombre de chunks ajoutés à ChromaDB (0 si documents)
    """
    # Étape 1 : Retrieval depuis ChromaDB (avec reranking)
    docs = retriever.invoke(question)

    # Étape 2 : Évaluer la pertinence
    is_relevant = evaluate_context_relevance(docs, question, llm)

    if is_relevant:
        # Étape 3a : Répondre depuis les documents
        context = format_context(docs)
        formatted_prompt = prompt.format(
            context=context,
            question=question,
            history=history,
        )
        response = llm.invoke(formatted_prompt)
        return response.content, docs, "documents", 0

    else:
        # Étape 3b : Recherche web
        web_results = web_search(question)

        if not web_results:
            # Pas de résultats web non plus
            context = format_context(docs)
            formatted_prompt = prompt.format(
                context=context,
                question=question,
                history=history,
            )
            response = llm.invoke(formatted_prompt)
            return response.content, docs, "documents", 0

        # Formater les résultats web comme contexte
        web_context = format_web_results(web_results)

        # Prompt spécial pour les résultats web
        web_prompt = f"""Tu es un assistant médical. L'information n'a pas été trouvée
dans les documents locaux, mais voici des résultats de recherche web.

Consignes :
- Réponds en français.
- Base ta réponse sur les résultats web fournis.
- Indique clairement que l'information provient d'une recherche web.
- Donne une réponse claire et structurée.
- Termine par "Sources : Recherche web" avec les URLs utilisées.

Historique :
{history}

Résultats web :
{web_context}

Question :
{question}"""

        response = llm.invoke(web_prompt)

        # Étape 4 : Enrichir ChromaDB avec les résultats web
        web_added = 0
        for r in web_results:
            body = r.get("body", "")
            title = r.get("title", "")
            href = r.get("href", "")
            if body:
                full_text = f"{title}\n\n{body}\n\nSource: {href}"
                added = add_to_vectorstore(
                    vectorstore,
                    text=full_text,
                    source=f"web:{href[:60]}",
                )
                web_added += added

        # Créer des docs factices pour l'affichage des sources web
        web_docs = []
        for r in web_results:
            web_docs.append(Document(
                page_content=r.get("body", ""),
                metadata={
                    "source": r.get("href", "web"),
                    "page": "web",
                    "title": r.get("title", ""),
                },
            ))

        return response.content, web_docs, "web", web_added
