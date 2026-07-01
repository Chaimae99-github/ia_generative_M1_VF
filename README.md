```markdown
# 🤖 AISCA - Agent Intelligent Sémantique et Génératif pour la Cartographie des Compétences

**Analyse Sémantique & Recommandation de Métiers via NLP et IA Générative**

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Objectifs du projet](#objectifs-du-projet)
3. [Architecture générale](#architecture-générale)
4. [Technologies utilisées](#technologies-utilisées)
5. [Structure du projet](#structure-du-projet)
6. [Installation](#installation)
7. [Utilisation](#utilisation)
8. [Composants clés](#composants-clés)
9. [Pipeline de traitement](#pipeline-de-traitement)
10. [Données de référence](#données-de-référence)
11. [Exigences fonctionnelles](#exigences-fonctionnelles)
12. [Déploiement](#déploiement)
13. [Contribution](#contribution)

---

## 🎯 Vue d'ensemble

**AISCA** est une application web intelligente qui analyse les compétences et expériences d'un utilisateur via un questionnaire, puis :

- 📊 **Calcule automatiquement** un score de couverture sémantique par bloc de compétences
- 🎯 **Recommande les 3 métiers** les plus adaptés au profil
- 📈 **Génère un plan de progression** personnalisé via IA générative
- 👤 **Produit une biographie professionnelle** synthétique

Le système repose sur :
- **NLP sémantique** : embeddings SBERT + similarité cosinus
- **RAG (Retrieval-Augmented Generation)** : contexte structuré + génération IA contrôlée
- **Interface web** : Streamlit pour prototypage rapide

---

## 🎓 Objectifs du projet

### Objectifs pédagogiques

- Appliquer le **prétraitement textuel** et les **embeddings sémantiques**
- Distinguer l'analyse numérique brute de l'analyse **sémantique contextualisée**
- Implémenter un **moteur de similarité** basé sur SBERT
- Structurer un **référentiel de compétences** en format professionnel
- Développer une **interface web interactive** (Streamlit)
- Intégrer **l'IA générative** de manière responsable, contrôlée et économique
- Concevoir un **pipeline NLP complet** : Scoring → Recommandation → GenAI
- Travailler en **équipe** et présenter une solution professionnelle

### Compétences RNCP visées

**Bloc 2 - RNCP40875** : *Piloter et implémenter des solutions d'IA en s'aidant notamment de l'IA générative*

- Collecter, analyser et préparer des données structurées et non structurées
- Concevoir et implémenter des modèles Data Science / NLP / IA
- Évaluer et optimiser les modèles
- Prototyper des solutions IA (API, NLP, RAG, embeddings, GenAI)
- Développer des pipelines data de bout en bout
- Industrialiser une solution (architecture, performance, contraintes coût)
- Documenter et présenter une solution technique complète

---

## 🏗️ Architecture générale

```
┌─────────────────────────────────────────────────────────────┐
│                    INTERFACE WEB (Streamlit)                 │
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │  Questionnaire   │→ │   Formulaire     │→ │   Analyse  │ │
│  │   (Likert +      │  │   Multiselect    │  │    Lancer  │ │
│  │   Text Free)     │  │   + Checkbox     │  │            │ │
│  └──────────────────┘  └──────────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              PIPELINE NLP SÉMANTIQUE (Core Engine)           │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  EmbeddingModel (SBERT Local - Zero Cost)             │ │
│  │  • SentenceTransformer                                │ │
│  │  • Encode user text + competency blocks               │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Scoring Module (Similarité Cosinus)                  │ │
│  │  • compute_block_scores()                             │ │
│  │  • compute_global_score()                             │ │
│  │  • Formule pondérée : Coverage = Σ(Wi * Si) / Σ(Wi)  │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Recommender (Job Matching)                           │ │
│  │  • recommend_jobs()                                   │ │
│  │  • Top 3 métiers basés sur scores blocs               │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│             AUGMENTATION GENAI (RAG Module)                  │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Context Builder (Retrieval)                          │ │
│  │  • Extraction des scores + métiers recom.             │ │
│  │  • Identification des écarts de compétences           │ │
│  │  • Construction du contexte enrichi                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  GenAI API (Gemini 2.5 Flash / OpenAI)               │ │
│  │  • Génération Plan de Progression (1 appel)           │ │
│  │  • Génération Bio Professionnelle (1 appel)           │ │
│  │  • Cache JSON automatique                             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              AFFICHAGE DES RÉSULTATS (Output)                │
│                                                               │
│  • Bio professionnelle (texte)                               │
│  • Plan de progression (texte structuré)                     │
│  • Scores par bloc (graphique barres)                        │
│  • Score global (métrique)                                   │
│  • Top 3 métiers recommandés (badges)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Technologies utilisées

### Backend & NLP
- **Python 3.10+** : langage principal
- **SentenceTransformers (SBERT)** : embeddings sémantiques contextuels
- **NumPy** : calculs vectoriels (similarité cosinus)
- **Pandas** : structuration des données

### Frontend
- **Streamlit** : interface web interactive (recommandé pour prototypes rapides)
- **Markdown** : mise en forme du contenu

### IA Générative
- **Google Gemini 2.5 Flash** : génération de texte (recommandé - gratuit & rapide)
- *Alternative* : OpenAI API, Ollama (local), ou autre LLM free-tier

### Gestion des données
- **JSON** : fichiers de configuration (competences.json, jobs.json)
- **Cache JSON** : stockage local des réponses GenAI (réduction des appels API)

### Versioning & Déploiement
- **Git / GitHub** : versioning du code
- **Moodle** : submission finale

---

## 📁 Structure du projet

```
AISCA/
├── data/
│   ├── competences.json          # Blocs de compétences (référentiel)
│   └── jobs.json                 # Profils de métiers
│
├── nlp/
│   ├── __init__.py
│   ├── embeddings.py             # EmbeddingModel - chargement SBERT
│   ├── scoring.py                # Calcul scores par bloc & global
│   ├── recommender.py            # Recommandation de métiers
│   └── similarity.py             # Utilitaires similarité cosinus
│
├── rag/
│   ├── __init__.py
│   ├── context_builder.py        # Construction contexte enrichi
│   ├── genai_api.py              # Appels API GenAI (Gemini/OpenAI)
│   └── cache.json                # Cache des réponses GenAI
│
├── ui/
│   ├── __init__.py
│   └── app.py                    # Application Streamlit principale
│
├── main.py                       # Script d'exemple (CLI)
├── test_sbert.py                 # Tests SBERT basiques
├── requirements.txt              # Dépendances Python
├── .gitignore                    # Fichiers à ignorer
└── README.md                     # Ce fichier
```

---

## 🚀 Installation

### Prérequis

- **Python 3.10+** (testé sur 3.10, 3.11, 3.12)
- **pip** (gestionnaire de paquets Python)
- Windows / macOS / Linux
- Connexion internet (téléchargement modèles + API)

### Étapes d'installation

#### 1. Cloner le repository

```powershell
git clone https://github.com/VOTRE_USERNAME/AISCA.git
cd AISCA
```

#### 2. Créer un environnement virtuel

```powershell
# Windows
py -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Mettre à jour pip

```powershell
pip install --upgrade pip
```

#### 4. Installer les dépendances

**Option A : Avec requirements.txt**

```powershell
pip install -r requirements.txt
```

**Option B : Installation manuelle**

```powershell
pip install streamlit pandas numpy
pip install sentence-transformers torch
pip install openai google-generativeai
```

#### 5. Configurer les variables d'environnement (GenAI)

Créer un fichier `.env` à la racine du projet :

```bash
# .env
GEMINI_API_KEY=votre_clé_gemini_ici
# OU
OPENAI_API_KEY=votre_clé_openai_ici
```

**Obtenir les clés API :**

- **Google Gemini** : https://aistudio.google.com/app/apikey
- **OpenAI** : https://platform.openai.com/api-keys

#### 6. Vérifier l'installation

```powershell
py -c "import streamlit; import sentence_transformers; print('✅ Installation OK')"
```

---

## 🎮 Utilisation

### Démarrer l'application Streamlit

```powershell
py -m streamlit run ui/app.py
```

L'application s'ouvre automatiquement sur :
```
http://localhost:8501
```

### Flux utilisateur

1. **Remplir le questionnaire** :
   - Évaluer votre niveau Python (1-5)
   - Répondre "Oui/Non" : Avez-vous entraîné un modèle ML ?
   - Sélectionner technologies utilisées
   - Indiquer si expérience travail en équipe
   - Décrire un projet ou expérience Data/IA (texte libre)

2. **Cliquer "🚀 Analyser mon profil"**

3. **Consulter les résultats** :
   - Bio professionnelle générée
   - Plan de progression personnalisé
   - Scores par bloc de compétences (graphique)
   - Score global
   - Top 3 métiers recommandés

4. **Réinitialiser** le formulaire avec le bouton "🔄 Réinitialiser"

### Scripts d'exemple

#### Via CLI (test_sbert.py)

```powershell
py test_sbert.py
```

Affiche les scores blocs et recommandations métiers pour un texte d'exemple.

#### Via pipeline complet (main.py)

```powershell
py main.py
```

Exécute le pipeline NLP complet sans interface web.

---

## 🔧 Composants clés

### 1. **EmbeddingModel** (embeddings.py)

Charge le modèle SBERT et encode les textes.

```python
from nlp.embeddings import EmbeddingModel

embedding_model = EmbeddingModel()
user_embedding = embedding_model.encode("J'ai experience Python")
```

**Modèle utilisé** : `paraphrase-multilingual-MiniLM-L12-v2` (multilingue, ~60MB)

**Propriétés** :
- Gratuit et open-source
- Téléchargé une seule fois (cache local)
- Dimension : 384 dimensions
- Temps d'inférence : ~10-50ms par texte

---

### 2. **Scoring Module** (scoring.py)

Calcule les scores de similarité par bloc et le score global.

**Fonction clé** : `compute_block_scores(user_text, blocks, embedding_model)`

```python
block_scores = {
    "Data Analysis": 0.82,
    "Machine Learning": 0.75,
    "NLP": 0.45
}
```

**Formule de score global** :

$$\text{Coverage Score} = \frac{\sum_{i=1}^{n} W_i \times S_i}{\sum_{i=1}^{n} W_i}$$

Où :
- $S_i$ = score de similarité pour le bloc $i$
- $W_i$ = poids pour le bloc $i$ (par défaut = 1)

---

### 3. **Recommender** (recommender.py)

Recommande les 3 meilleurs métiers basés sur les scores blocs.

```python
recommended_jobs = recommend_jobs(block_scores, jobs)
# [("Data Scientist", 0.78), ("ML Engineer", 0.71), ("Data Analyst", 0.65)]
```

**Logique** : Moyenne pondérée des scores des blocs requis pour chaque métier.

---

### 4. **RAG Pipeline**

#### Context Builder (context_builder.py)

Construit un contexte enrichi à transmettre au LLM :

```python
context = {
    "block_scores": {...},
    "top_jobs": [...],
    "weak_blocks": [...],
    "user_profile": "..."
}
```

#### GenAI API (genai_api.py)

Génère le plan de progression et la bio via API.

**Fonctions** :
- `generate_plan(context)` : Plan de progression personnalisé (1 appel API)
- `generate_bio(context)` : Biographie professionnelle (1 appel API)

**Caching** :
- Chaque prompt + réponse sauvegardés dans `cache.json`
- Requête identique = lecture du cache (0 appel API)

---

## 📊 Pipeline de traitement

### Étape 1 : Collecte des données

**Input** : Réponses du questionnaire Streamlit

```json
{
  "niveau_python": 4,
  "utilise_ml": "Oui",
  "competences_selection": ["Python", "TensorFlow", "Pandas"],
  "gestion_projet": true,
  "user_text": "J'ai travaillé sur un projet de prédiction..."
}
```

---

### Étape 2 : Enrichissement du texte

Fusion de toutes les entrées en un texte complet :

```python
texte_complet = user_text
texte_complet += f" Mon niveau en Python est 4/5."
texte_complet += " J'ai déjà entraîné des modèles de machine learning."
texte_complet += " J'ai utilisé : Python, TensorFlow, Pandas."
texte_complet += " J'ai travaillé sur des projets en équipe."
```

---

### Étape 3 : Encodage sémantique (SBERT)

```python
# Encoder le texte utilisateur
user_embedding = embedding_model.encode(texte_complet)

# Encoder chaque bloc de compétences
for block, competences in blocks.items():
    block_embeddings = embedding_model.encode(competences)
    similarities = cosine_similarity(user_embedding, block_embeddings)
    block_scores[block] = mean(similarities)
```

**Output** :
```python
block_scores = {
    "Data Analysis": 0.82,
    "Machine Learning": 0.75,
    "NLP": 0.45
}
```

---

### Étape 4 : Calcul du score global

```python
global_score = mean(list(block_scores.values()))  # 0.67
```

---

### Étape 5 : Recommandation de métiers

```python
job_scores = {}
for job, required_blocks in jobs.items():
    scores = [block_scores[b] for b in required_blocks]
    job_scores[job] = mean(scores)

# Trier et retourner les 3 meilleurs
recommended_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)[:3]
```

**Output** :
```python
[
    ("Data Scientist", 0.78),
    ("ML Engineer", 0.71),
    ("Data Analyst", 0.65)
]
```

---

### Étape 6 : Construction du contexte RAG

```python
context = build_context(block_scores, recommended_jobs)

context = {
    "scores_par_bloc": {"Data Analysis": 0.82, ...},
    "metiers_recommandes": ["Data Scientist", "ML Engineer", ...],
    "blocs_faibles": ["NLP"],
    "blocs_forts": ["Data Analysis"],
    "score_global": 0.67
}
```

---

### Étape 7 : Génération GenAI

**Appel 1** : Plan de progression

```python
prompt = f"""
Basé sur le profil suivant:
{context}

Générez un plan de progression détaillé identifiant:
1. Les compétences prioritaires à développer
2. Les ressources recommandées
3. Un timeline estimé
"""

plan = generate_plan(context)  # 1 seul appel API
```

**Appel 2** : Bio professionnelle

```python
bio = generate_bio(context)  # 1 seul appel API
```

---

### Étape 8 : Affichage des résultats

Streamlit affiche :
- Bio professionnelle (texte)
- Plan de progression (texte structuré)
- Graphique des scores blocs
- Métrique du score global
- Badges des 3 métiers

---

## 📚 Données de référence

### Format : competences.json

Structure des blocs de compétences :

```json
{
  "blocks": {
    "Data Analysis": [
      "data cleaning and preprocessing",
      "data visualization with Python",
      "statistical analysis",
      "Python programming"
    ],
    "Machine Learning": [
      "classification algorithms",
      "regression models",
      "neural networks and deep learning",
      "model evaluation and cross-validation"
    ],
    "NLP": [
      "text tokenization and preprocessing",
      "word embeddings and semantic analysis",
      "transformer models (BERT, GPT)",
      "sentiment analysis and text classification"
    ],
    "Project Management": [
      "agile methodologies",
      "team coordination",
      "documentation and reporting"
    ]
  }
}
```

### Format : jobs.json

Profils de métiers et leurs compétences requises :

```json
{
  "jobs": {
    "Data Analyst": ["Data Analysis", "Project Management"],
    "ML Engineer": ["Data Analysis", "Machine Learning"],
    "NLP Engineer": ["NLP", "Machine Learning"],
    "Data Scientist": ["Data Analysis", "Machine Learning", "NLP"]
  }
}
```

### Sources de données recommandées

- **ROME** : https://www.data.gouv.fr/datasets/repertoire-operationnel-des-metiers-et-des-emplois-rome/
- **e-Competence Framework** : https://assets.ide-conseil-webmarketing.fr/wp-content/uploads/2019/05/European-e-Competence-Framework-3.0_FR.pdf

---

## ✅ Exigences fonctionnelles

### EF1 : Acquisition de la donnée

| Exigence | Description | Implémentation |
|----------|-------------|-----------------|
| **EF1.1** | Questionnaire hybride (Likert + texte libre) | Streamlit multiselect, slider, text_area |
| **EF1.2** | Stockage structuré des réponses | JSON ou CSV en session Streamlit |

### EF2 : Moteur NLP sémantique (Cœur - Coût zéro)

| Exigence | Description | Implémentation |
|----------|-------------|-----------------|
| **EF2.1** | Référentiel de compétences | competences.json + jobs.json |
| **EF2.2** | Embeddings sémantiques | SentenceTransformer (SBERT) |
| **EF2.3** | Mesure de similarité | Similarité cosinus (NumPy) |

### EF3 : Système de scoring et recommandation

| Exigence | Description | Implémentation |
|----------|-------------|-----------------|
| **EF3.1** | Formule de score pondérée | Coverage = Σ(Wi × Si) / Σ(Wi) |
| **EF3.2** | Recommandation Top 3 métiers | Tri et classement des scores |

### EF4 : Augmentation par GenAI

| Exigence | Description | Implémentation | Appels API |
|----------|-------------|-----------------|------------|
| **EF4.1** | Enrichissement du texte court (optionnel) | Pre-processing avec GenAI | Conditionnel (si < 5 mots) |
| **EF4.2** | Plan de progression | `generate_plan(context)` | **1 seul appel** |
| **EF4.3** | Bio professionnelle | `generate_bio(context)` | **1 seul appel** |

### EF5 : Caching & Optimisation

| Exigence | Description | Implémentation |
|----------|-------------|-----------------|
| **EF5.1** | Cache automatique | JSON local (cache.json) |
| **EF5.2** | Limitation des appels API | Stratégie RAG : contexte enrichi → 2 appels max |
| **EF5.3** | Performance | Encodages cachés, models pré-chargés |

---

## 🌐 Déploiement

### Local (Développement)

```powershell
py -m streamlit run ui/app.py
```

Accès : `http://localhost:8501`

### Streamlit Cloud (Production recommandée)

1. Pousser le code sur GitHub
2. Connecter le repository à Streamlit Cloud
3. Déployer automatiquement

**Fichier `streamlit/config.toml`** (optionnel) :

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
headless = true
runOnSave = true
```

### Docker (Optionnel)

Créer un `Dockerfile` :

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py"]
```

Build & run :

```powershell
docker build -t aisca:latest .
docker run -p 8501:8501 aisca:latest
```

---

## 🤝 Contribution

### Workflow Git

```bash
# 1. Créer une branche
git checkout -b feature/mon-amelioration

# 2. Faire les modifications
# ... code ...

# 3. Commit
git add .
git commit -m "feat: description de la modification"

# 4. Push
git push origin feature/mon-amelioration

# 5. Créer un Pull Request sur GitHub
```

### Standards de code

- **PEP 8** : Style de code Python
- **Type hints** : Annotations de type recommandées
- **Docstrings** : Documentation des fonctions (format Google)

Exemple :

```python
def compute_block_scores(user_text: str, blocks: dict, embedding_model) -> dict:
    """
    Calcule les scores de similarité sémantique pour chaque bloc de compétences.
    
    Args:
        user_text (str): Texte de l'utilisateur à analyser
        blocks (dict): Dictionnaire {nom_bloc: [competences]}
        embedding_model: Modèle d'embeddings SBERT
    
    Returns:
        dict: {nom_bloc: score_similarite}
    """
    ...
```

---

## 📋 Requirements

Fichier `requirements.txt` :

```txt
streamlit==1.31.0
pandas==2.1.3
numpy==1.24.3
sentence-transformers==2.2.2
torch==2.1.1
scikit-learn==1.3.2
google-generativeai==0.3.0
python-dotenv==1.0.0
```

---

## 🔐 Sécurité et bonnes pratiques

### Gestion des clés API

❌ **Ne jamais** commit les clés API :

```bash
# Ajouter au .gitignore
.env
*.key
cache.json  # (optionnel, contient les réponses GenAI)
```

✅ **Utiliser** des variables d'environnement :

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
```

### Rate limiting & Quotas

- **Free Tier Gemini** : ~60 requêtes/minute
- **Free Tier OpenAI** : Limité (vérifier votre compte)
- **Caching** : Réutilise les réponses pour éviter le dépassement

---

## 📞 Support & FAQ

### Q : Où télécharger les modèles SBERT ?

**R :** Automatiquement lors du 1er lancement. Le modèle se cache dans `~/.cache/huggingface/`.

### Q : Comment modifier les blocs de compétences ?

**R :** Éditez competences.json directement.

### Q : Puis-je déployer sans clé GenAI ?

**R :** Oui ! Le scoring NLP fonctionne 100% en local (gratuit). GenAI est optionnel pour enrichissement.

### Q : Comment ajouter une nouvelle langue ?

**R :** Le modèle SBERT est multilingue. Il suffit d'ajouter du texte dans `competences.json`.

---

## 📄 Licence

Projet académique - EFREI 2025-26

---

## 👥 Auteurs

**Binôme AISCA** :
- Membre 1 : [Nom]
- Membre 2 : [Nom]

**Encadrant** : Sarah Malaeb

**Promotion** : Data Engineering & AI 2025-26

---

## 📅 Dates importantes

- **Début du projet** : [Date]
- **Deadline de submission** : [Date]
- **Présentation** : [Date/Dernière séance du module]

---

## 🚀 Prochaines étapes

- [ ] Améliorer la couverture des métiers (ajouter +50 métiers)
- [ ] Intégrer un système de feedback (utilisateurs évaluent les recommandations)
- [ ] Dashboard analytics (visualiser les profils les plus courants)
- [ ] API REST pour intégration tiers
- [ ] Application mobile React Native
- [ ] Support multi-langue avancé

---

## 📞 Contact

Pour toute question :
- 📧 Moodle du cours
- 💬 GitHub Issues


