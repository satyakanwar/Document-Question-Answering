import json
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import os
import Levenshtein

# -----------------------
# Config
# -----------------------
IMAGE_DIR = "multidoc/images"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

CHUNK_START = 0
CHUNK_END = 200

# -----------------------
# Metrics
# -----------------------
def exact_match_strict(pred, answers):
    pred = pred.lower().strip()
    answers = [a.lower().strip() for a in answers]
    return int(any(pred == a for a in answers))

def exact_match_relaxed(pred, answers):
    pred = pred.lower().strip()
    answers = [a.lower().strip() for a in answers]
    return int(any(pred == a or a in pred for a in answers))

def anls(pred, answers, tau=0.5):
    pred = pred.lower().strip()
    scores = []
    for gt in answers:
        gt = gt.lower().strip()
        dist = Levenshtein.distance(pred, gt)
        score = 1 - dist / max(len(pred), len(gt))
        scores.append(score if score >= tau else 0)
    return max(scores) if scores else 0

# -----------------------
# Heuristic scoring
# -----------------------
def score_prediction(pred, question):
    pred = pred.lower().strip()
    question_words = question.lower().split()

    score = 0

    overlap = sum(1 for w in question_words if w in pred)
    score += overlap * 3

    score -= len(pred) * 0.1

    word_count = len(pred.split())
    if 1 <= word_count <= 5:
        score += 5
    if word_count > 10:
        score -= 5

    bad_phrases = ["based on", "according to", "the document", "we can see"]
    for phrase in bad_phrases:
        if phrase in pred:
            score -= 4

    if any(c.isdigit() for c in pred):
        score += 2

    return score

# -----------------------
# Load model
# -----------------------
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_ID)

# -----------------------
# Summary generation (NEW)
# -----------------------
def get_page_summary(page_id):
    image_path = os.path.join(IMAGE_DIR, page_id + ".jpg")

    if not os.path.exists(image_path):
        return ""

    prompt = "Give a short summary of this page in 5 words."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

    summary = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0].strip().lower()

    return summary

# -----------------------
# Page selection (NEW)
# -----------------------
def select_best_page(page_ids, question):
    best_page = None
    best_score = -1

    for page_id in page_ids:
        summary = get_page_summary(page_id)

        if not summary:
            continue

        score = sum(1 for w in question.lower().split() if w in summary)

        if score > best_score:
            best_score = score
            best_page = page_id

    return best_page

# -----------------------
# Load dataset
# -----------------------
with open("multi_val.json", "r") as f:
    dataset = json.load(f)["data"]

dataset = dataset[CHUNK_START:CHUNK_END]

# -----------------------
# Stats
# -----------------------
results = []
total_em_strict = 0
total_em_relaxed = 0
total_anls = 0
count = 0

# -----------------------
# Main loop
# -----------------------
for sample in tqdm(dataset):

    question = sample["question"]
    answers = sample["answers"]
    page_ids = sample["page_ids"]

    # -----------------------
    # Step 1: Smart page selection
    # -----------------------
    best_page = select_best_page(page_ids, question)

    if best_page is None:
        continue

    image_path = os.path.join(IMAGE_DIR, best_page + ".jpg")

    # -----------------------
    # Step 2: Generate answer
    # -----------------------
    prompt = (
        "Read the document carefully and answer exactly using text from the document. "
        "Give a short answer only. Do not explain.\n"
        f"Question: {question}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )

    output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

    prediction = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0].strip()

    # -----------------------
    # Metrics
    # -----------------------
    em_strict = exact_match_strict(prediction, answers)
    em_relaxed = exact_match_relaxed(prediction, answers)
    anls_score = anls(prediction, answers)

    total_em_strict += em_strict
    total_em_relaxed += em_relaxed
    total_anls += anls_score
    count += 1

    # -----------------------
    # Save results
    # -----------------------
    results.append({
        "questionId": sample["questionId"],
        "question": question,
        "prediction": prediction,
        "answers": answers,
        "page_selected": best_page,
        "page_index_selected": page_ids.index(best_page) if best_page in page_ids else -1,
        "exact_match_strict": em_strict,
        "exact_match_relaxed": em_relaxed,
        "anls": anls_score
    })

# -----------------------
# Final results
# -----------------------
accuracy_strict = total_em_strict / count if count else 0
accuracy_relaxed = total_em_relaxed / count if count else 0
mean_anls = total_anls / count if count else 0

print("\n===== FINAL RESULTS =====")
print(f"Samples: {count}")
print(f"Strict EM:  {accuracy_strict:.4f}")
print(f"Relaxed EM: {accuracy_relaxed:.4f}")
print(f"ANLS:       {mean_anls:.4f}")

# -----------------------
# Save
# -----------------------
with open("mp_docvqa_smart_selection_350_380.json", "w") as f:
    json.dump({
        "overall": {
            "exact_match_strict": accuracy_strict,
            "exact_match_relaxed": accuracy_relaxed,
            "anls": mean_anls
        },
        "results": results
    }, f, indent=2)

print(" Saved mp_docvqa_smart_selection_350_380.json")