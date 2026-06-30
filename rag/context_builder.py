def build_context(block_scores, recommended_jobs):

    # blocs les plus faibles
    sorted_blocks = sorted(block_scores.items(), key=lambda x: x[1])

    weak_blocks = [b for b, s in sorted_blocks[:2]]

    top_job = recommended_jobs[0][0]

    context = {
        "top_job": top_job,
        "weak_blocks": weak_blocks,
        "scores": block_scores
    }

    return context