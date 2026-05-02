import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification


device = "cuda" if torch.cuda.is_available() else "cpu"

gate_model_path = "/content/drive/MyDrive/absa_self_train_phase1/ate_gate_teacher_phase1/"
ate_model_path = "/content/drive/MyDrive/absa_self_train_phase1/ate_teacher_phase1"

gate_tokenizer = AutoTokenizer.from_pretrained(gate_model_path, use_fast=True)
gate_model = AutoModelForSequenceClassification.from_pretrained(gate_model_path).to(device)
gate_model.eval()

ate_tokenizer = AutoTokenizer.from_pretrained(ate_model_path, use_fast=True)
ate_model = AutoModelForTokenClassification.from_pretrained(ate_model_path).to(device)
ate_model.eval()

TOKEN_RE = re.compile(r"\b\w+(?:'\w+)?\b")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_words(text):
    return [m.group(0) for m in TOKEN_RE.finditer(text)]

def normalize_span(span):
    span = clean_text(span)
    span = re.sub(r"\s+", " ", span)
    return span.strip(" \t\n\r.,;:!?()[]{}\"'")

def decode_bio_spans(tokens, labels, confidences):
    """
    Decode BIO labels thành aspect spans.
    confidence của span = mean confidence của token trong span.
    """
    spans = []
    i = 0

    while i < len(labels):
        if labels[i] == "B-ASP":
            start = i
            i += 1

            while i < len(labels) and labels[i] == "I-ASP":
                i += 1

            end = i - 1
            text = normalize_span(" ".join(tokens[start:end + 1]))

            if text:
                conf = float(np.mean(confidences[start:end + 1]))
                spans.append({
                    "aspect": text,
                    "confidence": conf,
                    "start_token": start,
                    "end_token": end,
                })
        else:
            i += 1

    # Deduplicate theo aspect text, giữ confidence cao nhất
    best = {}
    for span in spans:
        key = span["aspect"]
        if key not in best or span["confidence"] > best[key]["confidence"]:
            best[key] = span

    return sorted(best.values(), key=lambda x: (-x["confidence"], x["aspect"]))

def predict_gate_proba(sentences, batch_size=64, max_length=192):
    probs_all = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        enc = gate_tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = gate_model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        probs_all.extend(probs.tolist())

    return probs_all

def predict_ate_one(sentence, max_length=192):
    tokens = tokenize_words(sentence)

    if not tokens:
        return {
            "tokens": [],
            "labels": [],
            "token_confidences": [],
            "spans": [],
        }

    enc = ate_tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    word_ids = enc.word_ids()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = ate_model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred_ids = probs.argmax(axis=-1)

    id2label = ate_model.config.id2label

    word_labels = ["O"] * len(tokens)
    word_confidences = [0.0] * len(tokens)

    seen_word_ids = set()

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_word_ids:
            continue

        seen_word_ids.add(word_id)

        pred_id = int(pred_ids[token_idx])
        label = id2label[pred_id]
        conf = float(probs[token_idx, pred_id])

        word_labels[word_id] = label
        word_confidences[word_id] = conf

    spans = decode_bio_spans(tokens, word_labels, word_confidences)

    return {
        "tokens": tokens,
        "labels": word_labels,
        "token_confidences": word_confidences,
        "spans": spans,
    }

def predict_aspects(
    texts,
    gate_threshold=0.90,
    span_threshold=0.90,
    gate_batch_size=64,
    max_length=192,
    return_debug=False,
):
    """
    Input:
        texts: str hoặc list[str]

    Output:
        DataFrame gồm:
        - sentence
        - gate_confidence
        - aspects
        - confidences

    Logic:
        Nếu gate p(has_aspect) < gate_threshold -> aspects = []
        Nếu gate qua threshold -> chạy ATE, giữ span có confidence >= span_threshold
    """
    single_input = isinstance(texts, str)

    if single_input:
        sentences = [clean_text(texts)]
    else:
        sentences = [clean_text(x) for x in texts]

    gate_probs = predict_gate_proba(
        sentences,
        batch_size=gate_batch_size,
        max_length=max_length,
    )

    rows = []

    for sentence, gate_p in zip(sentences, gate_probs):
        if gate_p < gate_threshold:
            row = {
                "sentence": sentence,
                "gate_confidence": float(gate_p),
                "aspects": [],
                "confidences": [],
            }

            if return_debug:
                row.update({
                    "tokens": [],
                    "labels": [],
                    "token_confidences": [],
                    "raw_spans": [],
                })

            rows.append(row)
            continue

        ate_pred = predict_ate_one(sentence, max_length=max_length)

        kept_spans = [
            span for span in ate_pred["spans"]
            if span["confidence"] >= span_threshold
        ]

        row = {
            "sentence": sentence,
            "gate_confidence": float(gate_p),
            "aspects": [x["aspect"] for x in kept_spans],
            "confidences": [x["confidence"] for x in kept_spans],
        }

        if return_debug:
            row.update({
                "tokens": ate_pred["tokens"],
                "labels": ate_pred["labels"],
                "token_confidences": ate_pred["token_confidences"],
                "raw_spans": ate_pred["spans"],
            })

        rows.append(row)

    result_df = pd.DataFrame(rows)

    return result_df



# ============================
# Ví dụ sử dụng
# 
# Batch nhiều câu
# 
# sentences = [
#     "perfect stand for a novation launchpad x",
#     "who needs a proofreader",
#     "the battery life is excellent but the screen is dim",
# ]

# predict_aspects(
#     sentences,
#     gate_threshold=0.90,
#     span_threshold=0.85,
# )
# 
# Hàm trả về list[tuple(gate_confidence, aspects, confidences)]
# Ví dụ:
# [
#   (0.9758333563804626, ['stand'], [0.8594013452529907]),
#   (0.01565667986869812, [], []),
#   (0.993351936340332, ['battery life', 'screen'], [0.9818900525569916, 0.9659952521324158])
# ]
