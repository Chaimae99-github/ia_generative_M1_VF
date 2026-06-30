import numpy as np

def recommend_jobs(block_scores, jobs_dict):
    job_scores = {}

    for job, required_blocks in jobs_dict.items():
        scores = [block_scores[block] for block in required_blocks]
        job_score = np.mean(scores)
        job_scores[job] = job_score

    sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_jobs