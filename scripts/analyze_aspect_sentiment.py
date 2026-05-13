import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


device = "cuda" if torch.cuda.is_available() else "cpu"

asc_model_path = "/content/drive/MyDrive/absa_self_train_phase1/asc_teacher_phase1"

asc_tokenizer = AutoTokenizer.from_pretrained(asc_model_path, use_fast=True)
asc_model = AutoModelForSequenceClassification.from_pretrained(asc_model_path).to(device)
asc_model.eval()


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def mark_aspect(sentence, aspect):
    sentence = clean_text(sentence)
    aspect = clean_text(aspect)

    pattern = re.compile(re.escape(aspect), flags=re.IGNORECASE)
    return pattern.sub(f"[ASP] {aspect} [/ASP]", sentence, count=1)


def predict_asc(
    sentences,
    aspects,
    polar_threshold=0.90,
    neutral_threshold=0.60,
    batch_size=64,
    max_length=192,
):
    single_input = isinstance(sentences, str)

    if single_input:
        sentences = [sentences]
        aspects = [aspects]

    sentences = [clean_text(x) for x in sentences]
    aspects = [clean_text(x) for x in aspects]

    inputs = [
        mark_aspect(sentence, aspect)
        for sentence, aspect in zip(sentences, aspects)
    ]

    probs_all = []

    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]

        enc = asc_tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = asc_model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        probs_all.append(probs)

    probs_all = np.concatenate(probs_all, axis=0)

    results = []

    for probs in probs_all:
        confidences = probs.tolist()
        pred_id = int(np.argmax(confidences))
        confidence = float(confidences[pred_id])

        threshold = neutral_threshold if pred_id == 1 else polar_threshold
        is_high_confidence = confidence >= threshold

        results.append((confidences, is_high_confidence))

    return results


# ============================
# Ví dụ sử dụng
#
# Output của predict_asc là list[tuple]:
# [
#   ([conf_negative, conf_neutral, conf_positive], is_high_confidence),
#   ...
# ]
#
# Trong đó:
# - conf_negative: xác suất sentiment negative
# - conf_neutral : xác suất sentiment neutral
# - conf_positive: xác suất sentiment positive
# - is_high_confidence:
#     True nếu confidence của nhãn dự đoán đạt ngưỡng:
#       negative / positive >= polar_threshold
#       neutral >= neutral_threshold
#
# Một câu
#
# result = predict_asc(
#     "the battery life is excellent but the screen is dim",
#     "screen",
#     polar_threshold=0.90,
#     neutral_threshold=0.60,
# )
#
# print(result)
#
# Ví dụ output:
# [
#   ([0.01, 0.02, 0.97], True)
# ]
#
#
# Batch nhiều câu
#
# sentences = [
#     "the battery life is excellent but the screen is dim",
#     "the keyboard feels cheap",
#     "the price is okay",
# ]
#
# aspects = [
#     "screen",
#     "keyboard",
#     "price",
# ]
#
# results = predict_asc(
#     sentences,
#     aspects,
#     polar_threshold=0.90,
#     neutral_threshold=0.60,
#     batch_size=64,
#     max_length=192,
# )
#
# print(results)
#
# Ví dụ output:
# [
#   ([0.01, 0.02, 0.97], True),
#   ([0.94, 0.03, 0.03], True),
#   ([0.20, 0.65, 0.15], True)
# ]
