# 🏥 MedAssist

Assistant médical intelligent basé sur le RAG avancé (Retrieval-Augmented Generation).

## Pipeline RAG v2

```
Documents (PDF/TXT) → Chunking → Embeddings → ChromaDB → Reranking → Agent → Réponse
                                                                        ↓
                                                              🌐 Recherche Web (fallback)
                                                                        ↓
                                                              Enrichissement ChromaDB
```

## Fonctionnalités

| Fonctionnalité | Description |
|---------------|-------------|
| **ChromaDB** | Base vectorielle persistante (données sauvegardées entre sessions) |
| **Reranking** | Cross-Encoder pour réordonner les résultats (meilleure pertinence) |
| **Mémoire** | L'assistant se souvient des échanges précédents |
| **Agent Web** | Si l'info n'est pas dans les documents, recherche automatique sur le web |
| **Enrichissement** | Les résultats web sont ajoutés à ChromaDB pour les prochaines questions |

## Technologies

| Outil | Rôle |
|-------|------|
| LangChain | Orchestration du pipeline RAG |
| ChromaDB | Base vectorielle persistante |
| HuggingFace Embeddings | Transformation texte → vecteurs |
| Cross-Encoder | Reranking des résultats |
| Groq (Llama 3.3 70B) | Génération de réponses |
| DuckDuckGo | Recherche web (fallback agent) |
| Streamlit | Interface web |

## Installation

```bash
# 1. Créer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer la clé API Groq
copy .env.example .env
# Modifier .env avec votre clé : https://console.groq.com
```

## Utilisation

### Interface web (Streamlit)
```bash
python -m streamlit run app.py
```

### Mode console
```bash
python rag_medical.py
```

## Structure du projet

```
MedAssist/
├── data/                  # Documents médicaux (PDF/TXT)
├── chroma_db/             # Base ChromaDB persistante (auto-générée)
├── .streamlit/            # Configuration Streamlit
├── rag_medical.py         # Pipeline RAG (ChromaDB + Reranking)
├── agent.py               # Agent web search + enrichissement
├── app.py                 # Interface Streamlit
├── style.css              # Thème visuel
├── requirements.txt       # Dépendances Python
└── README.md
```

## Paramètres RAG

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| CHUNK_SIZE | 800 | Taille des morceaux de texte |
| CHUNK_OVERLAP | 150 | Chevauchement entre chunks |
| TOP_K_RETRIEVAL | 10 | Chunks récupérés avant reranking |
| TOP_K_FINAL | 3 | Chunks gardés après reranking |
| Embeddings | all-MiniLM-L6-v2 | Modèle de vectorisation |
| Reranker | ms-marco-MiniLM-L-6-v2 | Cross-Encoder pour le reranking |
| LLM | llama-3.3-70b-versatile | Modèle de génération |
