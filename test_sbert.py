from sentence_transformers import SentenceTransformer, util
import json
import numpy as np

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

user_text = "J'ai réalisé un projet de nettoyage de données avec Python et créé des visualisations."

with open("data/competences.json", "r", encoding="utf-8") as f:
    data = json.load(f)

user_embedding = model.encode(user_text, convert_to_tensor=True)

block_scores = {}

for block_name, competences in data["blocks"].items():
    competences_embeddings = model.encode(competences, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, competences_embeddings)

    max_score = float(similarities.max())
    block_scores[block_name] = max_score

print("\n--- Scores par bloc ---\n")
for block, score in block_scores.items():
    print(f"{block} --> {score:.3f}")

# Score global
final_score = np.mean(list(block_scores.values()))

print("\nScore global :", round(final_score, 3))


# Charger les métiers
with open("data/jobs.json", "r", encoding="utf-8") as f:
    jobs_data = json.load(f)

job_scores = {}

for job, required_blocks in jobs_data["jobs"].items():
    scores = [block_scores[block] for block in required_blocks]
    job_score = np.mean(scores)
    job_scores[job] = job_score

# Trier les métiers
sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)

print("\n--- Recommandation métiers ---\n")
for job, score in sorted_jobs:
    print(f"{job} --> {score:.3f}")


    