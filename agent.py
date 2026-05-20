"""
MedAssist — Agent avec Recherche Web
======================================
Quand les documents ne contiennent pas la réponse,
l'agent cherche sur le web (DuckDuckGo) et enrichit ChromaDB.
"""

import streamlit as st
from langchain_core.documents import Document

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False

from rag_medical import (
    format_context,
    add_to_vectorstore,
)


def normalize_text(text):
    """Supprime les accents et met en minuscule pour comparaison."""
    import unicodedata
    text = text.lower()
    # Décomposer les caractères accentués puis supprimer les accents
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def evaluate_context_relevance(docs, question, llm):
    """
    Évalue si les chunks récupérés sont pertinents pour la question.
    Approche hybride :
    1. D'abord vérifier le chevauchement de mots-clés (rapide, fiable)
    2. Si ambigu, demander au LLM (plus lent mais plus intelligent)
    """
    if not docs:
        return False, "Aucun chunk récupéré"

    # Normaliser la question (sans accents)
    q_normalized = normalize_text(question)
    question_words = set(
        w.strip(".,;:!?()\"'")
        for w in q_normalized.split()
        if len(w) >= 4
    )

    # Normaliser les chunks (sans accents)
    all_chunk_text = " ".join(d.page_content for d in docs)
    chunk_normalized = normalize_text(all_chunk_text)
    chunk_words = set(
        w.strip(".,;:!?()\"'")
        for w in chunk_normalized.split()
        if len(w) >= 4
    )

    # Compter le chevauchement
    common_words = question_words & chunk_words
    overlap_ratio = len(common_words) / max(len(question_words), 1)

    # Si chevauchement suffisant → pertinent (pas besoin du LLM)
    if overlap_ratio >= 0.2:
        return True, f"Pertinent (mots communs : {', '.join(list(common_words)[:5])})"

    # Si aucun chevauchement et question assez longue → pas pertinent
    if overlap_ratio == 0 and len(question_words) > 3:
        return False, "Aucun mot-clé commun entre la question et les chunks"

    # Zone grise → demander au LLM
    context = format_context(docs)
    eval_prompt = f"""Le contexte ci-dessous est-il utile pour répondre à cette question ?

Question : {question}

Contexte (premiers 500 caractères) :
{context[:500]}

Réponds par OUI ou NON uniquement."""

    response = llm.invoke(eval_prompt)
    answer = response.content.strip().upper()
    is_relevant = "OUI" in answer
    reason = f"LLM dit {'OUI' if is_relevant else 'NON'} (overlap={overlap_ratio:.0%})"
    return is_relevant, reason


def web_search(query, max_results=5):
    """
    Recherche sur le web avec DuckDuckGo.
    Filtre les résultats inutiles (YouTube, réseaux sociaux).
    """
    if not DDGS_AVAILABLE:
        st.warning("⚠️ Module duckduckgo-search non installé.")
        return []

    # Exclure YouTube et réseaux sociaux de la recherche
    filtered_query = f"{query} -site:youtube.com -site:tiktok.com -site:facebook.com -site:instagram.com"

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(filtered_query, max_results=max_results))

        # Filtrer les résultats indésirables
        blocked_domains = [
            "youtube.com", "youtu.be", "tiktok.com", "facebook.com",
            "instagram.com", "twitter.com", "x.com", "reddit.com",
            "pinterest.com", "dailymotion.com"
        ]

        filtered = []
        for r in raw_results:
            href = r.get("href", "").lower()
            if not any(domain in href for domain in blocked_domains):
                filtered.append(r)

        # Garder max 3 résultats propres
        return filtered[:3]

    except Exception as e:
        st.warning(f"⚠️ Erreur recherche web : {str(e)}")
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

    Retourne : (answer, docs, source_type, web_added, agent_log)
    """
    agent_log = []

    # Étape 1 : Retrieval depuis ChromaDB (avec reranking)
    agent_log.append("🔍 **Étape 1 — Retrieval + Reranking** : Recherche dans ChromaDB (top 10 → reranking → top 3)...")
    docs = retriever.invoke(question)
    agent_log.append(f"   → {len(docs)} chunk(s) récupéré(s) après reranking")

    if docs:
        sources_found = set(d.metadata.get("source", "?") for d in docs)
        agent_log.append(f"   → Sources : {', '.join(sources_found)}")

    # Étape 2 : Évaluer la pertinence (hybride : mots-clés + LLM)
    agent_log.append("🧠 **Étape 2 — Grading** : Évaluation de la pertinence (mots-clés + LLM)...")
    is_relevant, reason = evaluate_context_relevance(docs, question, llm)
    agent_log.append(f"   → Verdict : **{'PERTINENT ✅' if is_relevant else 'NON PERTINENT ❌'}**")
    agent_log.append(f"   → Raison : {reason}")

    if is_relevant:
        # Étape 3a : Répondre depuis les documents
        agent_log.append("📚 **Étape 3 — Génération** : Réponse depuis les documents locaux")
        agent_log.append("💬 **Mémoire** : Historique de conversation injecté dans le prompt")
        context = format_context(docs)
        formatted_prompt = prompt.format(
            context=context,
            question=question,
            history=history,
        )
        response = llm.invoke(formatted_prompt)
        return response.content, docs, "documents", 0, agent_log

    else:
        # Étape 3b : Recherche web
        agent_log.append("🌐 **Étape 3 — Recherche Web** : Lancement de DuckDuckGo (YouTube/réseaux sociaux exclus)...")
        web_results = web_search(question)

        if not web_results:
            agent_log.append("   → ⚠️ Aucun résultat web pertinent, fallback sur les documents")
            context = format_context(docs)
            formatted_prompt = prompt.format(
                context=context,
                question=question,
                history=history,
            )
            response = llm.invoke(formatted_prompt)
            return response.content, docs, "documents", 0, agent_log

        agent_log.append(f"   → {len(web_results)} résultat(s) web pertinent(s) trouvé(s)")
        for r in web_results:
            agent_log.append(f"   → 🔗 {r.get('title', '?')[:60]}")

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

        agent_log.append("💬 **Mémoire** : Historique de conversation injecté dans le prompt")
        response = llm.invoke(web_prompt)

        # Étape 4 : Enrichir ChromaDB avec les résultats web
        agent_log.append("💾 **Étape 4 — Enrichissement ChromaDB** : Ajout des résultats web dans la base...")
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

        agent_log.append(f"   → ✅ {web_added} chunk(s) ajouté(s) à ChromaDB")

        # Créer des docs pour l'affichage des sources web
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

        return response.content, web_docs, "web", web_added, agent_log
