import json
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import os
import Levenshtein
from collections import defaultdict

# ==============================
#  CHANGE THIS
# ==============================
TARGET_TYPE = "layout"
# Examples:
# "handwritten"
# "table/list"
# "layout"
# "form"
# "figure/diagram"
# "Yes/No"
# ==============================


# -----------------------
# Improvement functions
# -----------------------
def clean_prediction(pred):
    pred = pred.strip()

    # remove trailing punctuation
    pred = pred.rstrip(".:,;-")

    # extract numbers
    import re
    num = re.search(r"\b\d+(\.\d+)?\b", pred)
    if num:
        return num.group()

    # yes/no
    if "yes" in pred.lower():
        return "yes"
    if "no" in pred.lower():
        return "no"

    return pred




# -----------------------
# Metric functions
# -----------------------
def exact_match(pred, answers):
    pred = pred.lower().strip()
    answers = [a.lower().strip() for a in answers]
    return int(any(pred == a for a in answers))

def anls(pred, answers, tau=0.5):
    pred = pred.lower().strip()
    scores = []
    for gt in answers:
        gt = gt.lower().strip()
        dist = Levenshtein.distance(pred, gt)
        score = 1 - dist / max(len(pred), len(gt))
        scores.append(score if score >= tau else 0)
    return max(scores)


# -----------------------
# Load model
# -----------------------
model_id = "Qwen/Qwen3-VL-4B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id)


# -----------------------
# Load dataset
# -----------------------
with open("val.json", "r") as f:
    dataset = json.load(f)["data"]

# -----------------------
# Filter dataset by type
# -----------------------
filtered_data = [
    s for s in dataset
    if TARGET_TYPE in s.get("question_types", [])
]

print(f"\nRunning inference for type: {TARGET_TYPE}")
print(f"Total samples: {len(filtered_data)}")

# -----------------------
# Stats containers
# -----------------------
results = []
total_em = 0
total_anls = 0
count = 0

# -----------------------
# Inference loop
# -----------------------
for sample in tqdm(filtered_data):
    image_path = sample["image"]
    question = sample["question"]
    answers = sample["answers"]

    if not os.path.exists(image_path):
        print(f"Missing image: {image_path}")
        continue

    # Short answer prompt
    # if "Yes/No" in qtypes:
    #     prompt = f"Answer only Yes or No.\nQuestion: {question}"
    # else:
    #     prompt = (
    #         "Read the document carefully and answer exactly using text from the document. "
    #         "Give a short answer only. Do not explain.\n"
    #         f"Question: {question}"
    #     )

    
    # prompt = (
    #     "Read the document carefully and answer exactly using text from the document. "
    #     "Give a short answer only. Do not explain.\n"
    #     f"Question: {question}"
    # )
    prompt = (
        "Read the document carefully and answer exactly using text from the document. "
        "Give a short answer only. Do not explain.\n"
        "Do not add extra 0 after decimal place."
        "Do not add full stop if not required"
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

    prediction = clean_prediction(prediction)  #added new improvement

    # -----------------------
    # Metrics
    # -----------------------
    em = exact_match(prediction, answers)
    anls_score = anls(prediction, answers)

    total_em += em
    total_anls += anls_score
    count += 1

    results.append({
        "questionId": sample["questionId"],
        "question": question,
        "question_types": sample["question_types"],
        "prediction": prediction,
        "answers": answers,
        "exact_match": em,
        "anls": anls_score
    })

# -----------------------
# Final metrics
# -----------------------
accuracy = total_em / count if count else 0
mean_anls = total_anls / count if count else 0

print("\n===== RESULTS =====")
print(f"Question Type: {TARGET_TYPE}")
print(f"Samples evaluated: {count}")
print(f"Accuracy: {accuracy:.4f}")
print(f"ANLS: {mean_anls:.4f}")

# -----------------------
# Save results
# -----------------------
output_file = f"results_{TARGET_TYPE.replace('/','_')}.json"

with open(output_file, "w") as f:
    json.dump({
        "type": TARGET_TYPE,
        "accuracy": accuracy,
        "anls": mean_anls,
        "results": results
    }, f, indent=2)

print(f"\nSaved: {output_file}")
