import streamlit as st #outil qui fait la liaison entre le code et l'interface utilisateur

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

# Load environment variables from .env file
load_dotenv()

# Récupère le texte de la réponse, qu'elle soit un texte simple ou une liste de blocs
def extraire_texte(reponse):
    contenu = reponse.content
    if isinstance(contenu, str):          # cas normal : c'est déjà du texte
        return contenu.strip()
    # sinon, c'est une liste de blocs : on assemble le texte de chacun
    morceaux = []
    for bloc in contenu:
        if isinstance(bloc, dict):
            morceaux.append(bloc.get("text", ""))
        else:
            morceaux.append(str(bloc))
    return " ".join(morceaux).strip()

#charger l'assistant de google genai une seule fois 
@st.cache_resource 
def charger_assistant():
    embeddings = FastEmbedEmbeddings()
    vectorstore = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
        )
    #retriever est un objet qui permet de faire des recherches dans la base de données vectorielle. 
    # search_kwargs={"k": 6} signifie que nous voulons récupérer les 6 résultats les plus pertinents pour chaque requête.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    #créer une instance de ChatGoogleGenerativeAI, qui est un modèle de langage développé par Google.
    #temperature=0.2 signifie que nous voulons que les réponses générées soient relativement cohérentes et moins créatives, ce qui est souvent souhaitable pour des applications d'assistance.
    assistant = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.2
    )

    #on retoune le retriever et l'assistant pour les utiliser dans l'application Streamlit.
    return retriever, assistant

retriever, assistant = charger_assistant()

#fonction pour reformuler la question de l'utilisateur  pour le rendre plus clair et précis pour le modèle de langage.
def reformuler_question(question):

    reformulation = f"""Reformule la question suivante en une phrase claire et complète,
    en développant les abréviations (par exemple : IA -> intelligence artificielle).
    Donne UNIQUEMENT la reponse à la question reformulée, rien d'autre.
   
    Question : {question}
    Question reformulée :"""
    response = assistant.invoke(reformulation)
    return extraire_texte(response)

#fonction qui permet de générer une réponse à partir de la question posée et des documents récupérés par le retriever.
def repondre_a_la_question(question):
    #reformuler la question pour la rendre plus claire et précise pour le modèle de langage.
    question_reformulee = reformuler_question(question)

    #récupérer les passages pertinents à partir de la base de données vectorielle en utilisant le retriever.
    passages = retriever.invoke(question_reformulee)

    contexte = ""
    for passage in passages:
        contexte += passage.page_content + "\n\n"

    #générer une réponse à partir de la question reformulée et du contexte récupéré.
    message = f"""Réponds à la question en utilisant SEULEMENT le contexte ci-dessous.
    Si la réponse n'y est pas, dis simplement que tu n''a pas les connaissances nécessaires pour répondre à la question.

    Contexte : {contexte}

    Question : {question_reformulee}
    Réponse :"""
    reponse = assistant.invoke(message)
    sources = [passage.metadata.get("source", "inconnue") for passage in passages]
    return extraire_texte(reponse), sources, question_reformulee


#l'interface utilisateur de l'application Streamlit.
st.title("Assistant de recherche RAG sur l'IA")
st.write("Posez une question sur l'intelligence artificielle, le machine learning ou le deep learning et obtenez une réponse basée sur les documents de la base de données.")

question = st.text_input("Entrez votre question :")

if st.button("Obtenir la réponse"):
    if question:
        with st.spinner("Recherche en cours..."):
            reponse, sources, question_reformulee = repondre_a_la_question(question)
        
        st.caption(f"Question reformulée pour : {question_reformulee}")
        st.subheader("Réponse :")
        st.write(reponse)
        
        st.subheader("Sources utilisées par le modèle :")
        sources_uniques = list(dict.fromkeys(sources))  # pour éviter les doublons en gardant l'ordre
        for source in sources_uniques:
            st.write(f"- {source}")
    else:
        st.warning("Veuillez entrer une question avant de cliquer sur le bouton.")