import google.generativeai as genai
import json
import os

# configuration Gemini
genai.configure(api_key="AIzaSyAq_5xqOmVcdHO6gVq403XA_8r7UHOAF1s") 

model = genai.GenerativeModel("gemini-2.5-flash")

CACHE_FILE = "rag/cache.json"


def load_cache():
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except:
        return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def normalize_context(context):

    normalized_scores = {
        k: round(float(v), 3)
        for k, v in context["scores"].items()
    }

    return {
        "top_job": context["top_job"],
        "weak_blocks": context["weak_blocks"],
        "scores": normalized_scores
    }

def generate_plan(context):

    context = normalize_context(context)

    cache = load_cache()
    key = "plan_" + json.dumps(context, sort_keys=True)

    # vérifier si la réponse existe déjà
    if key in cache:
        return cache[key]
    prompt = f"""
    L'utilisateur souhaite devenir {context['top_job']}.

    Ses compétences les plus faibles sont :
    {context['weak_blocks']}.

    Génère un plan de progression simple et clair :

    - 3 étapes maximum
    - chaque étape doit contenir 2 actions concrètes
    - réponse courte (8 lignes maximum)
    - format en liste

    Réponds uniquement avec le plan.
    """

    response = model.generate_content(prompt)

    result = response.text

    # sauvegarder dans le cache
    cache[key] = result
    save_cache(cache)

    return result


def generate_bio(context):

    context = normalize_context(context)

    cache = load_cache()
    key = "bio_" + json.dumps(context, sort_keys=True)

    if key in cache:
        return cache[key]

    prompt = f"""
    Voici les scores de compétences d'un profil data :

    {context['scores']}

    Génère une bio professionnelle :

    - 2 phrases maximum
    - style professionnel
    - mentionner les principales compétences
    """

    response = model.generate_content(prompt)

    result = response.text

    cache[key] = result
    save_cache(cache)

    return result