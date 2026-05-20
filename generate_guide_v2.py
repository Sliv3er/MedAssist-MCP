"""Generate MedAssist v2 Walkthrough PDF"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
import os

OUT = os.path.join(os.path.dirname(__file__), "guide_medassist_v2.pdf")

# Colors
BLUE = HexColor("#3b82f6")
PURPLE = HexColor("#8b5cf6")
GREEN = HexColor("#10b981")
ORANGE = HexColor("#f59e0b")
RED = HexColor("#ef4444")
GRAY = HexColor("#94a3b8")
DARK = HexColor("#1e293b")

def S():
    s = {}
    s['T'] = ParagraphStyle('T', fontSize=26, textColor=BLUE, alignment=TA_CENTER, spaceAfter=4, fontName='Helvetica-Bold')
    s['ST'] = ParagraphStyle('ST', fontSize=11, textColor=GRAY, alignment=TA_CENTER, spaceAfter=20, fontName='Helvetica')
    s['H1'] = ParagraphStyle('H1', fontSize=17, textColor=BLUE, spaceBefore=16, spaceAfter=8, fontName='Helvetica-Bold')
    s['H2'] = ParagraphStyle('H2', fontSize=13, textColor=PURPLE, spaceBefore=12, spaceAfter=6, fontName='Helvetica-Bold')
    s['H3'] = ParagraphStyle('H3', fontSize=11, textColor=GREEN, spaceBefore=8, spaceAfter=4, fontName='Helvetica-Bold')
    s['B'] = ParagraphStyle('B', fontSize=9.5, textColor=DARK, spaceAfter=5, leading=13, fontName='Helvetica')
    s['C'] = ParagraphStyle('C', fontSize=8, textColor=HexColor("#e2e8f0"), backColor=HexColor("#1e293b"), spaceAfter=6, leading=11, fontName='Courier', leftIndent=8, rightIndent=8, borderPadding=7)
    s['Q'] = ParagraphStyle('Q', fontSize=9.5, textColor=RED, spaceBefore=8, spaceAfter=3, fontName='Helvetica-Bold')
    s['A'] = ParagraphStyle('A', fontSize=9.5, textColor=DARK, spaceAfter=6, leading=13, fontName='Helvetica', leftIndent=12)
    s['SM'] = ParagraphStyle('SM', fontSize=8, textColor=GRAY, alignment=TA_CENTER, fontName='Helvetica')
    return s

def box(text, color, s):
    t = Table([[Paragraph(text, s['B'])]], colWidths=[16*cm])
    bg = HexColor("#f0f9ff") if color==BLUE else (HexColor("#faf5ff") if color==PURPLE else (HexColor("#f0fdf4") if color==GREEN else HexColor("#fffbeb")))
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),bg),('BOX',(0,0),(-1,-1),1.5,color),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    return t

def tbl(headers, rows, cw=None):
    data = [headers]+rows
    if not cw: cw=[16*cm/len(headers)]*len(headers)
    t = Table(data, colWidths=cw, repeatRows=1)
    style=[('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),('ALIGN',(0,0),(-1,-1),'LEFT'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('GRID',(0,0),(-1,-1),0.5,HexColor("#cbd5e1")),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),6)]
    for i in range(1,len(data)):
        bg=HexColor("#f8fafc") if i%2==0 else white
        style.append(('BACKGROUND',(0,i),(-1,i),bg))
    t.setStyle(TableStyle(style))
    return t

def build():
    doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=1.8*cm, bottomMargin=1.8*cm, leftMargin=2*cm, rightMargin=2*cm)
    s = S()
    st = []

    # ===== COVER =====
    st.append(Spacer(1,3.5*cm))
    st.append(Paragraph("MedAssist v2", s['T']))
    st.append(Paragraph("Guide Complet du Projet RAG Avance", s['ST']))
    st.append(Spacer(1,0.8*cm))
    st.append(box("<b>Module :</b> IA Generative &nbsp;|&nbsp; <b>Sujet :</b> RAG Medical Avance<br/><b>Technologies :</b> LangChain, ChromaDB, Reranking, Agent Web, Groq<br/><b>GitHub :</b> github.com/Sliv3er/MedAssist", BLUE, s))
    st.append(Spacer(1,1*cm))
    st.append(box("<b>Pipeline v2 :</b> Documents → Chunking → Embeddings → ChromaDB → Reranking → Agent → Documents / Web → Enrichissement ChromaDB", PURPLE, s))
    st.append(Spacer(1,1.5*cm))
    st.append(tbl(["Fonctionnalite","Description"],
        [["ChromaDB","Base vectorielle persistante (donnees sauvegardees entre sessions)"],
         ["Reranking Cross-Encoder","Reordonne les resultats pour meilleure pertinence"],
         ["Memoire conversation","L'assistant se souvient des echanges precedents"],
         ["Agent Web (DuckDuckGo)","Si l'info n'est pas dans les docs, recherche automatique web"],
         ["Enrichissement auto","Les resultats web sont ajoutes a ChromaDB"]],
        [4.5*cm,11.5*cm]))
    st.append(PageBreak())

    # ===== HOW TO RUN =====
    st.append(Paragraph("0. Comment Lancer le Projet (autre PC)", s['H1']))
    st.append(box("<b>Etape 1 :</b> Cloner le repo<br/><font face='Courier' size='8'>git clone https://github.com/Sliv3er/MedAssist.git</font><br/><font face='Courier' size='8'>cd MedAssist</font><br/><br/><b>Etape 2 :</b> Installer les dependances<br/><font face='Courier' size='8'>pip install -r requirements.txt</font><br/><br/><b>Etape 3 :</b> Creer le fichier .env<br/><font face='Courier' size='8'>copy .env.example .env</font><br/>Modifier .env avec votre cle Groq : <font face='Courier' size='8'>GROQ_API_KEY=gsk_votre_cle</font><br/><br/><b>Etape 4 :</b> Lancer l'application<br/><font face='Courier' size='8'>python -m streamlit run app.py</font><br/><br/><b>Note :</b> Si 'streamlit' n'est pas reconnu, utiliser <font face='Courier' size='8'>python -m streamlit run app.py</font>", BLUE, s))
    st.append(Spacer(1,0.3*cm))

    st.append(Paragraph("Ou est la base de donnees ChromaDB ?", s['H2']))
    st.append(Paragraph("Le dossier <b>chroma_db/</b> est cree automatiquement au premier lancement dans le repertoire du projet. Il contient les fichiers SQLite et les vecteurs. Ce dossier <b>persiste</b> entre les sessions — pas besoin de re-indexer.", s['B']))
    st.append(Paragraph("Pour reinitialiser la base : supprimer le dossier chroma_db/ et relancer.", s['B']))
    st.append(Spacer(1,0.3*cm))

    st.append(Paragraph("Comment voir la recherche web en action ?", s['H2']))
    st.append(box("<b>1.</b> Lancer l'app avec <font face='Courier' size='8'>python -m streamlit run app.py</font><br/><b>2.</b> Poser une question medicale (ex: 'traitements du diabete') → badge vert '📚 Documents'<br/><b>3.</b> Poser une question HORS des documents (ex: 'C'est quoi le machine learning') → l'agent detecte que les chunks ne sont pas pertinents → lance DuckDuckGo → badge orange '🌐 Recherche Web' → message vert '✅ chunks ajoutes a ChromaDB'<br/><b>4.</b> Reposer la meme question → maintenant la reponse vient directement de ChromaDB (badge vert '📚 Documents') car les resultats web ont ete ajoutes !", GREEN, s))
    st.append(PageBreak())

    # ===== PART 1: VUE D'ENSEMBLE =====
    st.append(Paragraph("1. Vue d'Ensemble du Projet", s['H1']))
    st.append(Paragraph("MedAssist est un assistant medical intelligent base sur le RAG avance (Retrieval-Augmented Generation). Il lit des documents medicaux, les stocke dans ChromaDB, et repond aux questions en se basant sur ces documents. Si l'information n'est pas disponible, il cherche automatiquement sur le web et enrichit sa base.", s['B']))
    st.append(Spacer(1,0.3*cm))

    st.append(Paragraph("Architecture des fichiers", s['H2']))
    st.append(tbl(["Fichier","Role"],
        [["rag_medical.py","Pipeline RAG : chargement, chunking, ChromaDB, reranking, prompt, memoire"],
         ["agent.py","Agent intelligent : evaluation pertinence, recherche web, enrichissement ChromaDB"],
         ["app.py","Interface Streamlit : chat, badges, sidebar, historique"],
         ["style.css","Theme glassmorphism sombre avec badges colores"],
         ["data/","3 documents medicaux TXT (diabete, hypertension, antibiotiques)"],
         ["chroma_db/","Base vectorielle persistante (auto-generee)"],
         ["requirements.txt","Dependances Python"],
         [".env","Cle API Groq (secret)"]],
        [3.5*cm,12.5*cm]))
    st.append(PageBreak())

    # ===== PART 2: CE QUI A CHANGE =====
    st.append(Paragraph("2. Ce Qui a Change (v1 → v2)", s['H1']))
    st.append(tbl(["Aspect","Avant (v1)","Apres (v2)","Pourquoi"],
        [["Base vectorielle","FAISS (en memoire)","ChromaDB (persistant)","Pas de re-indexation au redemarrage"],
         ["Retrieval","Direct top-3","Top-10 → Reranking → Top-3","Resultats plus pertinents"],
         ["Memoire","Aucune","5 derniers echanges","Comprend les references"],
         ["Fallback","'Je ne sais pas'","Agent → DuckDuckGo","Ne reste jamais bloque"],
         ["Enrichissement","Non","Auto (web → ChromaDB)","Base de connaissances grandit"]],
        [2.5*cm,3.5*cm,4.5*cm,5.5*cm]))
    st.append(PageBreak())

    # ===== PART 3: PIPELINE DETAILLE =====
    st.append(Paragraph("3. Pipeline RAG Detaille", s['H1']))

    st.append(Paragraph("Etape 1 : Chargement des documents", s['H2']))
    st.append(Paragraph("<b>Fonction :</b> load_documents() dans rag_medical.py<br/>Parcourt le dossier data/, lit chaque fichier PDF (PyPDFLoader) ou TXT (TextLoader). Retourne une liste d'objets Document avec le contenu et les metadonnees (nom du fichier, page).", s['B']))

    st.append(Paragraph("Etape 2 : Chunking (Decoupage)", s['H2']))
    st.append(Paragraph("<b>Fonction :</b> split_documents() dans rag_medical.py<br/><b>Parametres :</b> CHUNK_SIZE=800, CHUNK_OVERLAP=150<br/>Le RecursiveCharacterTextSplitter decoupe intelligemment : d'abord sur les paragraphes, puis les lignes, puis les phrases. L'overlap de 150 caracteres evite de couper une information entre deux chunks. Resultat : 3 documents → 16 chunks.", s['B']))

    st.append(Paragraph("Etape 3 : Embeddings + ChromaDB", s['H2']))
    st.append(Paragraph("<b>Fonction :</b> create_vectorstore() dans rag_medical.py<br/><b>Modele :</b> all-MiniLM-L6-v2 (384 dimensions)<br/>Chaque chunk est transforme en vecteur numerique (384 nombres) qui capture le sens du texte. Ces vecteurs sont stockes dans ChromaDB (dossier chroma_db/) — une base persistante. Au prochain lancement, la base est chargee depuis le disque sans re-indexation.", s['B']))
    st.append(box("<b>Difference cle :</b> FAISS stocke tout en RAM (perdu au redemarrage). ChromaDB stocke sur disque dans chroma_db/ (persist entre sessions).", ORANGE, s))

    st.append(Paragraph("Etape 4 : Retrieval + Reranking", s['H2']))
    st.append(Paragraph("<b>Fonction :</b> create_retriever() dans rag_medical.py<br/>Le retrieval se fait en 2 phases :", s['B']))
    st.append(box("<b>Phase 1 — Retrieval large :</b> ChromaDB cherche les 10 chunks les plus proches par similarite cosinus (rapide mais approximatif).<br/><br/><b>Phase 2 — Reranking precis :</b> Le Cross-Encoder (ms-marco-MiniLM-L-6-v2) prend la question ET chaque chunk ensemble, les evalue, et ne garde que les 3 meilleurs.<br/><br/><b>Analogie :</b> Un recruteur recoit 100 CV (base), en pre-selectionne 10 (cosinus), puis les lit en detail pour garder les 3 meilleurs (cross-encoder).", PURPLE, s))

    st.append(Paragraph("Etape 5 : Agent (Decision)", s['H2']))
    st.append(Paragraph("<b>Fonction :</b> agent_answer() dans agent.py<br/>L'agent demande au LLM : 'Ces chunks sont-ils pertinents pour la question ? OUI/NON'<br/>- Si OUI → repond depuis les documents (badge vert)<br/>- Si NON → recherche web DuckDuckGo → repond → enrichit ChromaDB (badge orange)", s['B']))

    st.append(Paragraph("Etape 6 : Memoire de conversation", s['H2']))
    st.append(Paragraph("<b>Fonction :</b> format_history() dans rag_medical.py<br/>Les 5 derniers echanges (question + reponse) sont injectes dans le prompt. Le LLM comprend les references aux messages precedents. Ex: 'Et les effets secondaires ?' → comprend qu'on parle du medicament mentionne avant.", s['B']))

    st.append(Paragraph("Etape 7 : Enrichissement automatique", s['H2']))
    st.append(Paragraph("<b>Fonction :</b> add_to_vectorstore() dans rag_medical.py<br/>Quand l'agent fait une recherche web, chaque resultat (titre + contenu + URL) est decoupe en chunks et ajoute a ChromaDB. La prochaine fois, la meme question sera repondue directement depuis la base locale.", s['B']))
    st.append(PageBreak())

    # ===== PART 4: FONCTIONS =====
    st.append(Paragraph("4. Toutes les Fonctions du Projet", s['H1']))
    st.append(tbl(["Fonction","Fichier","Role","Nouveau"],
        [["load_documents()","rag_medical.py","Charge les fichiers PDF/TXT de data/","Non"],
         ["split_documents()","rag_medical.py","Decoupe en chunks (800 car, overlap 150)","Non"],
         ["get_embeddings()","rag_medical.py","Retourne le modele HuggingFace","Non"],
         ["create_vectorstore()","rag_medical.py","Cree/charge la base ChromaDB persistante","OUI"],
         ["create_retriever()","rag_medical.py","Retriever + CrossEncoder reranking","OUI"],
         ["add_to_vectorstore()","rag_medical.py","Ajoute du texte web dans ChromaDB","OUI"],
         ["build_prompt()","rag_medical.py","Template avec historique conversation","Modifie"],
         ["format_context()","rag_medical.py","Formate les chunks en texte lisible","Non"],
         ["format_history()","rag_medical.py","Formate les 5 derniers echanges","OUI"],
         ["answer_question()","rag_medical.py","Retrieval + LLM + reponse","Modifie"],
         ["init_pipeline()","rag_medical.py","Initialise tout le pipeline","OUI"],
         ["evaluate_context_relevance()","agent.py","LLM evalue si chunks pertinents","OUI"],
         ["web_search()","agent.py","Recherche DuckDuckGo","OUI"],
         ["format_web_results()","agent.py","Formate resultats web en texte","OUI"],
         ["agent_answer()","agent.py","Logique agent : docs ou web","OUI"],
         ["init_rag()","app.py","Init pipeline dans Streamlit","Modifie"],
         ["render_sidebar()","app.py","Sidebar avec stats et pipeline v2","Modifie"],
         ["render_chat()","app.py","Chat avec badges et enrichissement","Modifie"]],
        [3.8*cm,2.5*cm,6.7*cm,1.5*cm]))
    st.append(PageBreak())

    # ===== PART 5: QUESTIONS =====
    st.append(Paragraph("5. Questions Techniques du Professeur", s['H1']))

    qas = [
        ("C'est quoi la difference entre FAISS et ChromaDB ?","FAISS stocke les vecteurs en memoire (RAM) — quand on eteint le programme, tout est perdu et il faut re-indexer. ChromaDB est une vraie base de donnees vectorielle qui persiste sur le disque dans le dossier chroma_db/. Au prochain lancement, on charge directement la base existante."),
        ("C'est quoi le reranking ?","C'est une deuxieme phase de filtrage apres le retrieval. Le retriever classique (similarite cosinus) est rapide mais approximatif. Le Cross-Encoder prend la question et chaque chunk ensemble, les lit vraiment, et donne un score plus precis. On recupere 10 candidats, le Cross-Encoder les reordonne, et on garde les 3 meilleurs."),
        ("C'est quoi un Cross-Encoder ?","C'est un modele qui prend deux textes en entree (question + document) et donne un score de similarite. Contrairement aux embeddings qui encodent les textes separement, le Cross-Encoder les traite ensemble — c'est plus lent mais beaucoup plus precis."),
        ("Pourquoi 10 puis 3 et pas directement 3 ?","Si on prend directement 3, on risque de rater des chunks pertinents qui seraient classes 4eme ou 5eme par la similarite cosinus. En prenant 10 puis en reranking, on a plus de chances de trouver les 3 meilleurs vrais resultats."),
        ("Comment fonctionne la memoire de conversation ?","On garde les 5 derniers echanges (messages utilisateur + reponses) et on les injecte dans le prompt. Le LLM peut ainsi comprendre les pronoms et les references. Ex: 'Et les effets secondaires ?' → comprend qu'on parle du medicament mentionne avant."),
        ("Comment l'agent decide de chercher sur le web ?","L'agent envoie les chunks au LLM avec la question et lui demande 'est-ce pertinent ? OUI ou NON'. Si NON (les chunks ne parlent pas du sujet), l'agent lance DuckDuckGo automatiquement."),
        ("Pourquoi DuckDuckGo et pas Google ?","DuckDuckGo est gratuit et ne necessite aucune cle API. Google Search API est payant. Pour un projet academique, DuckDuckGo est parfait."),
        ("Comment fonctionne l'enrichissement ?","Quand l'agent fait une recherche web, il prend chaque resultat (titre + contenu + URL), le decoupe en chunks, et l'ajoute dans ChromaDB. La prochaine fois, la meme question sera repondue directement sans recherche web."),
        ("C'est quoi le ContextualCompressionRetriever ?","C'est un composant LangChain qui combine un retriever de base (ChromaDB) avec un compresseur (Cross-Encoder). Il recupere d'abord les documents, puis les passe au compresseur qui les filtre et reordonne."),
        ("Que se passe-t-il si le web ne trouve rien ?","Si DuckDuckGo ne retourne rien (ou erreur reseau), l'agent revient aux chunks de ChromaDB meme s'ils ne sont pas parfaits, et le LLM fait de son mieux. Il n'y a jamais de blocage."),
        ("Est-ce que le LLM apprend de vos documents ?","NON. Le LLM (Llama 3.3) est fige. On lui donne les documents en contexte a chaque question, mais ses poids ne changent pas. C'est la difference entre RAG et fine-tuning."),
        ("Difference entre RAG et fine-tuning ?","RAG : on donne les docs au moment de la question, le modele ne change pas. Fine-tuning : on reentraine le modele sur nos donnees. RAG est prefere quand les donnees changent souvent."),
        ("Pourquoi temperature=0 ?","Temperature=0 rend le modele deterministe : pas de creativite ni d'invention. Pour le medical, on veut des reponses fiables et reproductibles."),
        ("C'est quoi les tokens ?","Unite de texte traitee par le LLM (environ un mot). Les LLM ont une limite (ex: 8192 tokens), d'ou l'importance du chunking et de la limite de 5 echanges dans l'historique."),
        ("Comment ameliorer le systeme ?","1) Ajouter un upload de documents dans l'interface. 2) Supporter plus de formats (DOCX, HTML). 3) Utiliser un LLM local. 4) Ajouter une evaluation automatique des reponses. 5) Deployer sur le cloud."),
    ]
    for q,a in qas:
        st.append(Paragraph(f"Q : {q}", s['Q']))
        st.append(Paragraph(a, s['A']))

    st.append(PageBreak())

    # ===== PART 6: VOCABULAIRE =====
    st.append(Paragraph("6. Vocabulaire a Maitriser", s['H1']))
    st.append(tbl(["Terme","Definition"],
        [["RAG","Retrieval-Augmented Generation — combiner recherche + generation"],
         ["LLM","Large Language Model — gros modele de langage (Llama, GPT)"],
         ["Embedding","Vecteur numerique representant le sens d'un texte (384 dim.)"],
         ["Chunk","Morceau de texte decoupe d'un document (800 car.)"],
         ["Overlap","Chevauchement entre chunks consecutifs (150 car.)"],
         ["ChromaDB","Base vectorielle persistante (stockee sur disque)"],
         ["FAISS","Facebook AI Similarity Search (stocke en RAM uniquement)"],
         ["Cross-Encoder","Modele qui evalue question+document ensemble pour reranking"],
         ["Reranking","Reordonnement des resultats pour meilleure pertinence"],
         ["Retriever","Composant qui cherche les chunks pertinents"],
         ["Agent","Systeme qui decide quelle action effectuer (docs ou web)"],
         ["Grading","Evaluation de la pertinence des chunks par le LLM"],
         ["Enrichissement","Ajout de nouveaux contenus (web) dans la base vectorielle"],
         ["PromptTemplate","Template structure pour formater les requetes au LLM"],
         ["Similarite cosinus","Mesure de ressemblance entre deux vecteurs"],
         ["Hallucination","Quand un LLM invente une reponse fausse"],
         ["Temperature","Parametre de creativite (0=deterministe, 1=creatif)"],
         ["Persistance","Capacite a sauvegarder les donnees entre les sessions"]],
        [3.5*cm,12.5*cm]))

    st.append(Spacer(1,1*cm))
    st.append(box("<b>Conseil :</b> Parle naturellement. Si le prof demande quelque chose, ne recite pas — explique avec tes mots. Utilise des analogies : 'le reranking c'est comme un recruteur qui pre-selectionne 10 CV puis les lit en detail pour garder les 3 meilleurs'.", GREEN, s))

    doc.build(st)
    print(f"PDF : {OUT}")

if __name__=="__main__":
    build()
