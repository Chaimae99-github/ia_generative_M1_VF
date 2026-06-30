import streamlit as st
import json
import sys
import os
import pandas as pd

# Permet import depuis dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nlp.embeddings import EmbeddingModel
from nlp.scoring import compute_block_scores, compute_global_score
from nlp.recommender import recommend_jobs

from rag.context_builder import build_context
from rag.genai_api import generate_plan, generate_bio

def reset_form():
    st.session_state["niveau_python"] = 3
    st.session_state["utilise_ml"] = "Non"
    st.session_state["competences_selection"] = []
    st.session_state["gestion_projet"] = False
    st.session_state["user_text"] = ""

# ----------------------
# CONFIGURATION PAGE
# ----------------------

st.set_page_config(
    page_title="AISCA - Agent IA",
    page_icon="🤖",
    layout="wide"
)

if "reset" not in st.session_state:
    st.session_state.reset = False

st.title("🤖 AISCA - Agent Intelligent de Cartographie des Compétences")
st.markdown("### Analyse sémantique & Recommandation de Métiers")


# ----------------------
# CHARGEMENT DONNÉES
# ----------------------

with open("data/competences.json", "r", encoding="utf-8") as f:
    competences_data = json.load(f)

with open("data/jobs.json", "r", encoding="utf-8") as f:
    jobs_data = json.load(f)

blocks = competences_data["blocks"]
jobs = jobs_data["jobs"]


# ----------------------
# QUESTIONNAIRE AMÉLIORÉ
# ----------------------

st.header("📋 Questionnaire")

col1, col2 = st.columns(2)

with col1:
    niveau_python = st.slider("Évaluez votre niveau en Python (1-5)", 1, 5, 3, key="niveau_python")
    utilise_ml = st.radio(
        "Avez-vous déjà entraîné un modèle de Machine Learning ?",
        ["Oui", "Non"], key="utilise_ml"
    )

with col2:
    competences_selection = st.multiselect(
        "Technologies utilisées :",
        ["Python", "SQL", "TensorFlow", "PyTorch", "Pandas", "Scikit-learn"],
        default=[],
        key="competences_selection"
    )

    gestion_projet = st.checkbox(
        "Avez-vous déjà travaillé sur un projet en équipe ?", key="gestion_projet"
    )

user_text = st.text_area(
    "Décrivez un projet ou vos expériences en data / IA :",
    height=150, key="user_text"
)


# ----------------------
# BOUTON ANALYSE
# ----------------------

if st.button("🚀 Analyser mon profil"):

    if user_text.strip() == "":
        st.warning("Veuillez décrire au moins une expérience.")
    else:

        # Construire texte enrichi
        texte_complet = user_text

        texte_complet += f" Mon niveau en Python est {niveau_python}/5."

        if utilise_ml == "Oui":
            texte_complet += " J'ai déjà entraîné des modèles de machine learning."

        if competences_selection:
            texte_complet += " J'ai utilisé : " + ", ".join(competences_selection) + "."

        if gestion_projet:
            texte_complet += " J'ai travaillé sur des projets en équipe."

        # Initialiser modèle
        embedding_model = EmbeddingModel()

        # Calculs
        block_scores = compute_block_scores(texte_complet, blocks, embedding_model)
        block_scores = {k: round(v,3) for k,v in block_scores.items()}
        global_score = compute_global_score(block_scores)
        recommended_jobs = recommend_jobs(block_scores, jobs)
        recommended_jobs = [(job, round(score,3)) for job, score in recommended_jobs]
        


        ## partie genération AI de plan et bio
        context = build_context(block_scores, recommended_jobs)

        plan = generate_plan(context)
        bio = generate_bio(context)


        st.subheader("Bio professionnelle")
        st.write(bio)

        st.subheader("Plan de progression")
        st.write(plan)

        # ----------------------
        # AFFICHAGE RÉSULTATS
        # ----------------------

        st.divider()
        st.header("📊 Résultats de l'analyse")

        colA, colB = st.columns([2,1])

        with colA:
            df_scores = pd.DataFrame(
                block_scores.items(),
                columns=["Bloc", "Score"]
            )

            st.subheader("Scores par bloc")
            st.bar_chart(df_scores.set_index("Bloc"))

        with colB:
            st.subheader("Score global")
            st.metric(label="Couverture globale", value=round(global_score, 3))

        st.subheader("🎯 Top 3 Métiers Recommandés")

        for job, score in recommended_jobs[:3]:
            st.success(f"{job}  →  Score : {round(score,3)}")




# if st.button("🔄 Réinitialiser le questionnaire"):
#     for key in list(st.session_state.keys()):
#         del st.session_state[key]
#     st.rerun()


st.button(
    "🔄 Réinitialiser le questionnaire",
    on_click=reset_form
)
    

