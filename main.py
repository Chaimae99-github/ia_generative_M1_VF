import json
from nlp.embeddings import EmbeddingModel
from nlp.scoring import compute_block_scores, compute_global_score
from nlp.recommender import recommend_jobs

# Charger données
with open("data/competences.json", "r", encoding="utf-8") as f:
    competences_data = json.load(f)

with open("data/jobs.json", "r", encoding="utf-8") as f:
    jobs_data = json.load(f)

blocks = competences_data["blocks"]
jobs = jobs_data["jobs"]

# Initialiser modèle
embedding_model = EmbeddingModel()

# Exemple texte utilisateur
user_text = "J'ai réalisé un projet de nettoyage de données avec Python et créé des visualisations."

# Calcul scores
block_scores = compute_block_scores(user_text, blocks, embedding_model)
global_score = compute_global_score(block_scores)

# Recommandation
recommended_jobs = recommend_jobs(block_scores, jobs)

# Affichage
print("\n--- Scores par bloc ---")
for block, score in block_scores.items():
    print(f"{block} --> {score:.3f}")

print("\nScore global :", round(global_score, 3))

print("\n--- Recommandation métiers ---")
for job, score in recommended_jobs:
    print(f"{job} --> {score:.3f}")