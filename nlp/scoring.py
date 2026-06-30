from sentence_transformers import util
import numpy as np

def compute_block_scores(user_text, blocks, model):
    user_embedding = model.encode(user_text)

    block_scores = {}

    for block_name, competences in blocks.items():
        competences_embeddings = model.encode(competences)
        similarities = util.cos_sim(user_embedding, competences_embeddings)
        # max_score = float(similarities.max())
        # block_scores[block_name] = max_score
        

        similarity_values = similarities[0].cpu().numpy()

        # garder seulement les similarités pertinentes
        relevant_scores = [s for s in similarity_values if s > 0.3]

        if relevant_scores:
            block_score = sum(relevant_scores) / len(relevant_scores)
        else:
            block_score = 0

        block_scores[block_name] = block_score



    return block_scores


def compute_global_score(block_scores):
    return np.mean(list(block_scores.values()))