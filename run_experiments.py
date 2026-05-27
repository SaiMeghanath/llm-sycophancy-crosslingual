# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE: Free options — Groq API (llama3-70b), Ollama (llama3-8b, mistral)
#          Paid optional — OpenAI API for GPT-4o only (~$5-10 total)
#          Sign up free: console.groq.com | api.together.ai
# ─────────────────────────────────────────────────────────────────────────────
import os
"""
Full Experiment Runner — Base vs Instruction-Tuned LLMs on Noisy Text
Gupta et al. 2020/2021 methodology extended to modern LLMs.

Models:
- Llama-3-8B (base) vs Llama-3-8B-Instruct
- Mistral-Small (base) vs Mistral-Small-Instruct  
- Qwen-32B (base) vs Qwen-32B-Instruct
- Llama-70B (base) vs Llama-70B-Instruct
- GPT-4o (instruct only)

Inference backends:
- Ollama: Llama-3-8B, Mistral-Small (local)
- Groq API: Llama-70B (free tier)
- OpenAI API: GPT-4o
- Together.ai API: Qwen-32B

Usage:
    # Small scale test (15 samples)
    python run_experiments.py --scale small --task imdb

    # Full scale
    python run_experiments.py --scale full --task imdb
"""

import csv
import json
import random
import argparse
from noise_injection import inject_qwerty_noise

# ── CONFIG ──────────────────────────────────────────────────────────────────
NOISE_LEVELS = [0, 5, 10, 15, 20, 25]
SMALL_SCALE_N = 15
RANDOM_SEED = 42

# API Keys — set via environment variables
# Groq is FREE — sign up at console.groq.com
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Together.ai free tier — sign up at api.together.ai
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
# OpenAI is PAID — optional, only needed for GPT-4o
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── MODEL REGISTRY ──────────────────────────────────────────────────────────
MODELS = {
    # Ollama (local)
    "llama3-8b-base":        {"backend": "ollama", "model": "llama3:8b"},
    "llama3-8b-instruct":    {"backend": "ollama", "model": "llama3:8b-instruct"},
    "mistral-base":          {"backend": "ollama", "model": "mistral"},
    "mistral-instruct":      {"backend": "ollama", "model": "mistral:instruct"},

    # Groq (free API)
    "llama3-70b-base":       {"backend": "groq",   "model": "llama3-70b-8192"},
    "llama3-70b-instruct":   {"backend": "groq",   "model": "llama-3.1-70b-versatile"},

    # Together.ai
    "qwen-32b-base":         {"backend": "together", "model": "Qwen/Qwen2.5-32B"},
    "qwen-32b-instruct":     {"backend": "together", "model": "Qwen/Qwen2.5-32B-Instruct"},

    # OpenAI
    "gpt-4o":                {"backend": "openai",  "model": "gpt-4o"},
}

# ── PROMPT TEMPLATES ─────────────────────────────────────────────────────────
PROMPTS = {
    "sentiment": {
        "base": "Sentiment of this review (POSITIVE or NEGATIVE):\n{text}\nAnswer:",
        "instruct": "Classify the sentiment of the following movie review.\nReview: {text}\nRespond with only: POSITIVE or NEGATIVE"
    },
    "similarity": {
        "base": "Similarity score 0-5 for these sentences:\nSentence 1: {s1}\nSentence 2: {s2}\nScore:",
        "instruct": "Rate the semantic similarity between these two sentences on a scale of 0 to 5.\nSentence 1: {s1}\nSentence 2: {s2}\nRespond with only a number between 0 and 5."
    }
}

# ── INFERENCE BACKENDS ───────────────────────────────────────────────────────
def query_ollama(model, prompt):
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Ollama error ({model}): {e}")
        return None

def query_openai(model, prompt):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error ({model}): {e}")
        return None

def query_groq(model, prompt):
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error ({model}): {e}")
        return None

def query_together(model, prompt):
    try:
        from together import Together
        client = Together(api_key=TOGETHER_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Together error ({model}): {e}")
        return None

def run_inference(model_key, prompt):
    config = MODELS[model_key]
    backend = config["backend"]
    model = config["model"]
    if backend == "ollama":
        return query_ollama(model, prompt)
    elif backend == "openai":
        return query_openai(model, prompt)
    elif backend == "groq":
        return query_groq(model, prompt)
    elif backend == "together":
        return query_together(model, prompt)
    return None

# ── LABEL PARSING ────────────────────────────────────────────────────────────
def parse_sentiment(response):
    if response is None:
        return -1
    r = response.upper()
    if "POSITIVE" in r:
        return 1
    if "NEGATIVE" in r:
        return 0
    return -1

# ── METRICS ──────────────────────────────────────────────────────────────────
def compute_f1(predictions, labels):
    valid = [(p, l) for p, l in zip(predictions, labels) if p != -1]
    if not valid:
        return 0.0
    preds, labs = zip(*valid)
    tp = sum(1 for p, l in zip(preds, labs) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labs) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labs) if p == 0 and l == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return round(f1, 4)

# ── SAMPLE DATA (small scale) ────────────────────────────────────────────────
IMDB_SAMPLES = [
    {"text": "This film is a masterpiece. Every scene is carefully crafted and the performances are outstanding.", "label": 1},
    {"text": "An absolute waste of time. The plot makes no sense and the acting is terrible.", "label": 0},
    {"text": "One of the best movies I have ever seen. The story is deeply moving and beautifully told.", "label": 1},
    {"text": "I fell asleep halfway through. Boring, predictable, and completely forgettable.", "label": 0},
    {"text": "A stunning achievement in filmmaking. The director brings such vision to every frame.", "label": 1},
    {"text": "Dreadful in every way. The script is awful and the characters are completely unlikeable.", "label": 0},
    {"text": "Genuinely one of the most entertaining films of the decade. I laughed, I cried, I loved it.", "label": 1},
    {"text": "Painfully bad. Nothing about this movie works. Avoid at all costs.", "label": 0},
    {"text": "A beautiful and thought-provoking film that stays with you long after it ends.", "label": 1},
    {"text": "Terrible pacing, weak dialogue, and zero chemistry between the leads.", "label": 0},
    {"text": "Brilliant performances all around. This is exactly the kind of cinema we need more of.", "label": 1},
    {"text": "A complete disappointment. The trailer promised so much but the film delivers nothing.", "label": 0},
    {"text": "Emotionally resonant and visually spectacular. A triumph of modern storytelling.", "label": 1},
    {"text": "I cannot believe how bad this was. Two hours of my life I will never get back.", "label": 0},
    {"text": "Exceptional filmmaking from start to finish. Highly recommended.", "label": 1},
]

# ── MAIN RUNNER ──────────────────────────────────────────────────────────────
def run_experiment(models_to_run, scale="small", task="sentiment"):
    random.seed(RANDOM_SEED)

    samples = IMDB_SAMPLES[:SMALL_SCALE_N] if scale == "small" else IMDB_SAMPLES
    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]

    results = []

    for model_key in models_to_run:
        is_instruct = "instruct" in model_key or "gpt" in model_key
        prompt_type = "instruct" if is_instruct else "base"

        print(f"\n{'='*60}")
        print(f"Model: {model_key} | Scale: {scale} | Task: {task}")
        print(f"{'='*60}")

        for noise_level in NOISE_LEVELS:
            noisy_texts = [inject_qwerty_noise(t, noise_level) for t in texts]
            predictions = []

            for i, text in enumerate(noisy_texts):
                prompt = PROMPTS[task][prompt_type].format(text=text)
                response = run_inference(model_key, prompt)
                pred = parse_sentiment(response)
                predictions.append(pred)
                print(f"  D{noise_level} [{i+1}/{len(texts)}] pred={pred} label={labels[i]} | {str(response)[:40]}")

            f1 = compute_f1(predictions, labels)
            result = {
                "model": model_key,
                "noise_level": noise_level,
                "f1": f1,
                "n_samples": len(texts),
                "task": task,
                "scale": scale
            }
            results.append(result)
            print(f"  → D{noise_level} F1: {f1}")

    # Save CSV
    output_file = f"results_{task}_{scale}.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "noise_level", "f1", "n_samples", "task", "scale"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_file}")

    # Print summary table
    print(f"\n{'Model':<30} {'D0':>6} {'D5':>6} {'D10':>6} {'D15':>6} {'D20':>6} {'D25':>6}")
    print("-" * 72)
    for model_key in models_to_run:
        model_results = {r["noise_level"]: r["f1"] for r in results if r["model"] == model_key}
        scores = [str(model_results.get(l, "N/A")) for l in NOISE_LEVELS]
        print(f"{model_key:<30} {' '.join(f'{s:>6}' for s in scores)}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="small", choices=["small", "full"])
    parser.add_argument("--task", default="sentiment", choices=["sentiment", "similarity"])
    parser.add_argument("--models", nargs="+", default=["llama3-70b-base", "llama3-70b-instruct"],
                        help="Models to run. Default uses Groq free API. Add 'gpt-4o' only if you have an OpenAI API key.")
    args = parser.parse_args()

    run_experiment(args.models, scale=args.scale, task=args.task)
