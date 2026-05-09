# 🏥 MedAssist

Assistant médical intelligent basé sur le RAG (Retrieval-Augmented Generation).

Ce projet utilise **LangChain**, **FAISS** et **Groq** pour interroger des documents médicaux en langage naturel.

## Pipeline RAG

```
Documents (PDF/TXT) → Chunking → Embeddings → FAISS → Retrieval → Groq LLM → Réponse
```

## Technologies

| Outil | Rôle |
|-------|------|
| LangChain | Orchestration du pipeline |
| FAISS | Base vectorielle (recherche sémantique) |
| HuggingFace Embeddings | Transformation texte → vecteurs |
| Groq (Llama 3.3 70B) | Génération de réponses |
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

### Mode console
```bash
python rag_medical.py
```

### Interface web (Streamlit)
```bash
streamlit run app.py
```

## Documents médicaux

Le dossier `data/` contient les documents à interroger :
- `diabete_type2.txt` — Guide clinique sur le diabète de type 2
- `hypertension_arterielle.txt` — Guide sur l'hypertension artérielle
- `antibiotiques_guide.txt` — Guide de prescription des antibiotiques

Vous pouvez ajouter vos propres fichiers PDF ou TXT dans ce dossier.

## Structure du projet

```
MedAssist/
├── data/                  # Documents médicaux
├── .streamlit/            # Configuration Streamlit
├── rag_medical.py         # Pipeline RAG (script principal)
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
| TOP_K | 3 | Nombre de chunks récupérés |
| Modèle embeddings | all-MiniLM-L6-v2 | Modèle de vectorisation |
| Modèle LLM | llama-3.3-70b-versatile | Modèle de génération |
