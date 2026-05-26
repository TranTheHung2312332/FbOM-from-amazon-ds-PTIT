from __future__ import annotations

import argparse
import ast
import html
import json
import logging
import math
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


LOGGER = logging.getLogger("benchmark")
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
ASC_ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}

np = None
pd = None
plt = None
torch = None
accuracy_score = None
classification_report = None
confusion_matrix = None
precision_recall_fscore_support = None
AutoModelForSequenceClassification = None
AutoModelForTokenClassification = None
AutoTokenizer = None

TOKEN_RE = None
asc_clean_text = None
mark_aspect = None
ate_clean_text = None
decode_bio_spans = None
preprocess_normalize_text = None
_TOKENIZER_TEMP_DIRS = []


@dataclass
class AspectTerm:
    term: str
    polarity: str | None
    start: int | None
    end: int | None


@dataclass
class SentenceExample:
    sentence_id: str
    text: str
    aspects: list[AspectTerm]


@dataclass
class PredictedAspect:
    term: str
    start: int | None
    end: int | None
    confidence: float | None
    start_token: int | None
    end_token: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ATE and ASC independently without Gate.",
    )
    parser.add_argument("--data-path", required=True, help="Benchmark dataset path. SemEval XML is preferred.")
    parser.add_argument("--ate-model", required=True, help="ATE token-classification model path.")
    parser.add_argument("--asc-model", required=True, help="ASC sequence-classification model path.")
    parser.add_argument("--output-dir", default="outputs/benchmark", help="Directory for benchmark outputs.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--max-length", type=int, default=192, help="Tokenizer max length.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_runtime_dependencies() -> None:
    global np, pd, plt, torch
    global accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    global AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as matplotlib_pyplot
        import numpy as numpy_module
        import pandas as pandas_module
        import torch as torch_module
        from sklearn.metrics import (
            accuracy_score as sklearn_accuracy_score,
            classification_report as sklearn_classification_report,
            confusion_matrix as sklearn_confusion_matrix,
            precision_recall_fscore_support as sklearn_precision_recall_fscore_support,
        )
        from transformers import (
            AutoModelForSequenceClassification as HFAutoModelForSequenceClassification,
            AutoModelForTokenClassification as HFAutoModelForTokenClassification,
            AutoTokenizer as HFAutoTokenizer,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing benchmark dependency. Please install torch, transformers, pandas, numpy, "
            f"scikit-learn, and matplotlib. Original error: {exc}"
        ) from exc

    np = numpy_module
    pd = pandas_module
    plt = matplotlib_pyplot
    torch = torch_module
    accuracy_score = sklearn_accuracy_score
    classification_report = sklearn_classification_report
    confusion_matrix = sklearn_confusion_matrix
    precision_recall_fscore_support = sklearn_precision_recall_fscore_support
    AutoModelForSequenceClassification = HFAutoModelForSequenceClassification
    AutoModelForTokenClassification = HFAutoModelForTokenClassification
    AutoTokenizer = HFAutoTokenizer


def load_local_pipeline_logic() -> None:
    global TOKEN_RE, asc_clean_text, mark_aspect, ate_clean_text, decode_bio_spans
    global preprocess_normalize_text

    from analyze_aspect_sentiment import clean_text as local_asc_clean_text
    from analyze_aspect_sentiment import mark_aspect as local_mark_aspect
    from extract_aspect import TOKEN_RE as local_token_re
    from extract_aspect import clean_text as local_ate_clean_text
    from extract_aspect import decode_bio_spans as local_decode_bio_spans
    from preprocess_sentence import normalize_text as local_preprocess_normalize_text

    TOKEN_RE = local_token_re
    asc_clean_text = local_asc_clean_text
    mark_aspect = local_mark_aspect
    ate_clean_text = local_ate_clean_text
    decode_bio_spans = local_decode_bio_spans
    preprocess_normalize_text = local_preprocess_normalize_text


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def stringify_id(value: Any, fallback: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return str(fallback)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_polarity(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    aliases = {
        "neg": "negative",
        "neu": "neutral",
        "pos": "positive",
    }
    return aliases.get(text, text)


def parse_serialized_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, float) and math.isnan(value):
        return []

    text = str(value).strip()
    if not text or text == "nan":
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]

    return parsed if isinstance(parsed, list) else [parsed]


def sentiment_from_value(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        normalized = normalize_polarity(value)
        return normalized if normalized in SENTIMENT_LABELS or normalized == "conflict" else None

    if isinstance(value, (int, np.integer)):
        return ASC_ID_TO_LABEL.get(int(value))

    if isinstance(value, (float, np.floating)) and not math.isnan(float(value)):
        idx = int(value)
        return ASC_ID_TO_LABEL.get(idx)

    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
        idx = int(np.argmax(np.asarray(value, dtype=float)[:3]))
        return ASC_ID_TO_LABEL.get(idx)

    return None


def find_term_offsets(sentence: str, terms: list[str]) -> list[tuple[int | None, int | None]]:
    offsets: list[tuple[int | None, int | None]] = []
    cursor = 0

    for term in terms:
        clean_term = str(term)
        match = re.search(re.escape(clean_term), sentence[cursor:], flags=re.IGNORECASE)
        if match is not None:
            start = cursor + match.start()
            end = cursor + match.end()
            cursor = end
            offsets.append((start, end))
            continue

        match = re.search(re.escape(clean_term), sentence, flags=re.IGNORECASE)
        if match is not None:
            offsets.append((match.start(), match.end()))
        else:
            offsets.append((None, None))

    return offsets


def load_semeval_xml(path: Path) -> list[SentenceExample]:
    tree = ET.parse(path)
    root = tree.getroot()
    examples: list[SentenceExample] = []

    for idx, sentence_el in enumerate(root.findall(".//sentence")):
        sentence_id = sentence_el.get("id") or str(idx)
        text_el = sentence_el.find("text")
        sentence_text = html.unescape(text_el.text if text_el is not None and text_el.text else "")

        aspects: list[AspectTerm] = []
        aspect_terms_el = sentence_el.find("aspectTerms")
        if aspect_terms_el is not None:
            for aspect_el in aspect_terms_el.findall("aspectTerm"):
                term = html.unescape(aspect_el.get("term") or "")
                polarity = normalize_polarity(aspect_el.get("polarity"))
                start = parse_optional_int(aspect_el.get("from"))
                end = parse_optional_int(aspect_el.get("to"))
                aspects.append(AspectTerm(term=term, polarity=polarity, start=start, end=end))

        examples.append(SentenceExample(sentence_id=sentence_id, text=sentence_text, aspects=aspects))

    return examples


def load_csv(path: Path) -> list[SentenceExample]:
    df = pd.read_csv(path)
    columns_lower = {column.lower(): column for column in df.columns}

    if {"id", "sentence", "aspect term"}.issubset(columns_lower):
        id_col = columns_lower["id"]
        sentence_col = columns_lower["sentence"]
        aspect_col = columns_lower["aspect term"]
        polarity_col = columns_lower.get("polarity")
        from_col = columns_lower.get("from")
        to_col = columns_lower.get("to")

        examples = []
        grouped = df.groupby([id_col, sentence_col], dropna=False, sort=False)
        for fallback, ((sentence_id, sentence), group) in enumerate(grouped):
            aspects = []
            for _, row in group.iterrows():
                term = "" if pd.isna(row[aspect_col]) else str(row[aspect_col])
                if not term:
                    continue
                aspects.append(
                    AspectTerm(
                        term=term,
                        polarity=normalize_polarity(row[polarity_col]) if polarity_col else None,
                        start=parse_optional_int(row[from_col]) if from_col else None,
                        end=parse_optional_int(row[to_col]) if to_col else None,
                    )
                )
            examples.append(
                SentenceExample(
                    sentence_id=stringify_id(sentence_id, fallback),
                    text="" if pd.isna(sentence) else str(sentence),
                    aspects=aspects,
                )
            )
        return examples

    if {"sentence_id", "sentence_text", "aspects"}.issubset(columns_lower):
        id_col = columns_lower["sentence_id"]
        sentence_col = columns_lower["sentence_text"]
        aspects_col = columns_lower["aspects"]
        sentiments_col = columns_lower.get("sentiments")
        examples = []

        for idx, row in df.iterrows():
            sentence = "" if pd.isna(row[sentence_col]) else str(row[sentence_col])
            terms = [str(term) for term in parse_serialized_list(row[aspects_col])]
            sentiment_values = parse_serialized_list(row[sentiments_col]) if sentiments_col else []
            offsets = find_term_offsets(sentence, terms)

            aspects = []
            for term_idx, term in enumerate(terms):
                polarity = sentiment_from_value(sentiment_values[term_idx]) if term_idx < len(sentiment_values) else None
                start, end = offsets[term_idx]
                aspects.append(AspectTerm(term=term, polarity=polarity, start=start, end=end))

            examples.append(
                SentenceExample(
                    sentence_id=stringify_id(row[id_col], idx),
                    text=sentence,
                    aspects=aspects,
                )
            )
        return examples

    raise ValueError(f"Unsupported CSV schema in {path}. Expected SemEval CSV or gold_with_sentiments CSV.")


def load_dataset(path: Path) -> list[SentenceExample]:
    suffix = path.suffix.lower()
    if suffix == ".xml":
        return load_semeval_xml(path)
    if suffix == ".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}. Use .xml or .csv.")


def resolve_dataset_path(path: Path) -> Path:
    if path.suffix.lower() == ".csv":
        sibling_xml = path.with_suffix(".xml")
        if sibling_xml.exists():
            LOGGER.info("Found sibling SemEval XML for CSV input. Using %s", sibling_xml)
            return sibling_xml
    return path


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def softmax_np(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()


def load_tokenizer(model_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        LOGGER.warning(
            "Tokenizer config at %s uses legacy list extra_special_tokens. "
            "Loading from a sanitized temporary config.",
            model_path,
        )
        return load_tokenizer_from_sanitized_config(model_path)


def load_tokenizer_from_sanitized_config(model_path: str):
    source_dir = Path(model_path)
    if not source_dir.is_dir():
        raise ValueError(f"Sanitized tokenizer fallback requires a local model directory: {model_path}")

    tokenizer_config_path = source_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        raise FileNotFoundError(f"Missing tokenizer_config.json in {source_dir}")

    tmp = tempfile.TemporaryDirectory(prefix="absa_tokenizer_")
    _TOKENIZER_TEMP_DIRS.append(tmp)
    tmp_dir = Path(tmp.name)

    tokenizer_files = [
        "config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    for file_name in tokenizer_files:
        source_file = source_dir / file_name
        if source_file.exists():
            shutil.copy2(source_file, tmp_dir / file_name)

    with tokenizer_config_path.open("r", encoding="utf-8") as file:
        tokenizer_config = json.load(file)

    extra_special_tokens = tokenizer_config.pop("extra_special_tokens", None)
    if isinstance(extra_special_tokens, list):
        additional_tokens = tokenizer_config.get("additional_special_tokens", [])
        if not isinstance(additional_tokens, list):
            additional_tokens = []
        tokenizer_config["additional_special_tokens"] = list(
            dict.fromkeys(additional_tokens + extra_special_tokens)
        )

    with (tmp_dir / "tokenizer_config.json").open("w", encoding="utf-8") as file:
        json.dump(tokenizer_config, file, ensure_ascii=False, indent=2)

    return AutoTokenizer.from_pretrained(str(tmp_dir), use_fast=True)


def load_sequence_classifier(model_path: str, device: torch.device):
    tokenizer = load_tokenizer(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model


def load_token_classifier(model_path: str, device: torch.device):
    tokenizer = load_tokenizer(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model


def infer_gate_positive_id(model: AutoModelForSequenceClassification) -> int:
    label2id = getattr(model.config, "label2id", None) or {}
    for label, idx in label2id.items():
        normalized = str(label).lower()
        if normalized in {"1", "label_1", "has_aspect", "aspect", "positive", "true"}:
            return int(idx)

    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        normalized = str(label).lower()
        if normalized in {"1", "label_1", "has_aspect", "aspect", "positive", "true"}:
            return int(idx)

    return 1


def predict_gate(
    sentences: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> tuple[list[int], list[list[float]], list[list[float]]]:
    positive_id = infer_gate_positive_id(model)
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    logits_all: list[list[float]] = []

    for start in range(0, len(sentences), batch_size):
        batch_sentences = [ate_clean_text(sentence) for sentence in sentences[start : start + batch_size]]
        enc = tokenizer(
            batch_sentences,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = move_batch_to_device(enc, device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = softmax_np(logits)
            pred_ids = np.argmax(logits.detach().cpu().numpy(), axis=-1)

        logits_all.extend(logits.detach().cpu().numpy().tolist())
        probabilities.extend(probs.tolist())
        predictions.extend([1 if int(pred_id) == positive_id else 0 for pred_id in pred_ids])

    return predictions, probabilities, logits_all


def tokenize_words_with_offsets(sentence: str) -> tuple[list[str], list[tuple[int, int]]]:
    matches = list(TOKEN_RE.finditer(sentence))
    tokens = [match.group(0) for match in matches]
    offsets = [(match.start(), match.end()) for match in matches]
    return tokens, offsets


def predict_ate(
    sentences: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    id2label = {int(idx): label for idx, label in model.config.id2label.items()}

    tokenized = [tokenize_words_with_offsets(sentence) for sentence in sentences]

    for start in range(0, len(sentences), batch_size):
        batch_items = tokenized[start : start + batch_size]
        batch_tokens = [tokens for tokens, _ in batch_items]
        non_empty_positions = [idx for idx, tokens in enumerate(batch_tokens) if tokens]

        batch_results: list[dict[str, Any] | None] = [None] * len(batch_tokens)
        for idx, tokens in enumerate(batch_tokens):
            if not tokens:
                batch_results[idx] = {
                    "tokens": [],
                    "offsets": [],
                    "labels": [],
                    "token_confidences": [],
                    "raw_spans": [],
                }

        if non_empty_positions:
            model_tokens = [batch_tokens[idx] for idx in non_empty_positions]
            enc = tokenizer(
                model_tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            enc_on_device = move_batch_to_device(enc, device)

            with torch.no_grad():
                logits = model(**enc_on_device).logits
                probs = softmax_np(logits)
                pred_ids = np.argmax(logits.detach().cpu().numpy(), axis=-1)

            for model_row, original_pos in enumerate(non_empty_positions):
                tokens = batch_tokens[original_pos]
                offsets = batch_items[original_pos][1]
                word_ids = enc.word_ids(batch_index=model_row)
                word_labels = ["O"] * len(tokens)
                word_confidences = [0.0] * len(tokens)
                seen_word_ids = set()

                for token_idx, word_id in enumerate(word_ids):
                    if word_id is None or word_id in seen_word_ids or word_id >= len(tokens):
                        continue
                    seen_word_ids.add(word_id)

                    pred_id = int(pred_ids[model_row, token_idx])
                    label = id2label[pred_id]
                    confidence = float(probs[model_row, token_idx, pred_id])
                    word_labels[word_id] = label
                    word_confidences[word_id] = confidence

                decoded_spans = decode_bio_spans(tokens, word_labels, word_confidences)
                raw_spans = []
                for span in decoded_spans:
                    start_token = int(span["start_token"])
                    end_token = int(span["end_token"])
                    char_start = offsets[start_token][0] if start_token < len(offsets) else None
                    char_end = offsets[end_token][1] if end_token < len(offsets) else None
                    raw_spans.append(
                        PredictedAspect(
                            term=str(span["aspect"]),
                            start=char_start,
                            end=char_end,
                            confidence=float(span["confidence"]),
                            start_token=start_token,
                            end_token=end_token,
                        )
                    )

                batch_results[original_pos] = {
                    "tokens": tokens,
                    "offsets": offsets,
                    "labels": word_labels,
                    "token_confidences": word_confidences,
                    "raw_spans": raw_spans,
                }

        predictions.extend([result for result in batch_results if result is not None])

    return predictions


def empty_ate_prediction() -> dict[str, Any]:
    return {
        "ran": False,
        "tokens": [],
        "offsets": [],
        "labels": [],
        "token_confidences": [],
        "raw_spans": [],
    }


def mark_aspect_with_offsets(sentence: str, aspect: AspectTerm) -> str:
    if aspect.start is not None and aspect.end is not None:
        if 0 <= aspect.start <= aspect.end <= len(sentence):
            surface = sentence[aspect.start : aspect.end]
            if surface.lower() == aspect.term.lower():
                marked = f"{sentence[:aspect.start]}[ASP] {surface} [/ASP]{sentence[aspect.end:]}"
                return asc_clean_text(marked)
    return mark_aspect(sentence, aspect.term)


def predict_asc(
    samples: list[tuple[SentenceExample, AspectTerm]],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    inputs = [mark_aspect_with_offsets(example.text, aspect) for example, aspect in samples]

    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start : start + batch_size]
        enc = tokenizer(
            batch_inputs,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = move_batch_to_device(enc, device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = softmax_np(logits)
            pred_ids = np.argmax(logits.detach().cpu().numpy(), axis=-1)

        for offset, pred_id in enumerate(pred_ids):
            probabilities = probs[offset].tolist()
            predictions.append(
                {
                    "marked_input": batch_inputs[offset],
                    "pred_id": int(pred_id),
                    "pred_label": ASC_ID_TO_LABEL.get(int(pred_id), str(pred_id)),
                    "probabilities": probabilities,
                    "logits": logits[offset].detach().cpu().numpy().tolist(),
                }
            )

    return predictions


def aspect_to_tuple(sentence_id: str, aspect: AspectTerm | PredictedAspect) -> tuple[str, int, int] | None:
    if aspect.start is None or aspect.end is None:
        return None
    return (sentence_id, int(aspect.start), int(aspect.end))


def aspect_spans_to_token_binary(
    token_offsets: list[tuple[int, int]],
    aspects: list[AspectTerm] | list[PredictedAspect],
) -> list[int]:
    labels = [0] * len(token_offsets)

    for aspect in aspects:
        if aspect.start is None or aspect.end is None:
            continue

        span_start = int(aspect.start)
        span_end = int(aspect.end)
        if span_start >= span_end:
            continue

        for token_idx, (token_start, token_end) in enumerate(token_offsets):
            if token_start < span_end and token_end > span_start:
                labels[token_idx] = 1

    return labels


def predicted_labels_to_token_binary(
    token_offsets: list[tuple[int, int]],
    ate_output: dict[str, Any],
    fallback_predicted_aspects: list[PredictedAspect],
) -> list[int]:
    labels = ate_output.get("labels", [])
    if len(labels) == len(token_offsets):
        return [0 if label == "O" else 1 for label in labels]

    return aspect_spans_to_token_binary(token_offsets, fallback_predicted_aspects)


def compute_ate_metrics(
    examples: list[SentenceExample],
    predicted_aspects: list[list[PredictedAspect]],
    ate_outputs: list[dict[str, Any]],
) -> dict[str, float | int]:
    sentence_matches = []
    gold_global = set()
    pred_global = set()
    token_true_positive = 0
    token_false_positive = 0
    token_false_negative = 0
    token_true_negative = 0

    for example, pred_spans, ate_output in zip(examples, predicted_aspects, ate_outputs):
        gold_set = {span for span in (aspect_to_tuple(example.sentence_id, aspect) for aspect in example.aspects) if span}
        pred_set = {span for span in (aspect_to_tuple(example.sentence_id, aspect) for aspect in pred_spans) if span}
        sentence_matches.append(gold_set == pred_set)
        gold_global.update(gold_set)
        pred_global.update(pred_set)

        token_offsets = ate_output.get("offsets", [])
        if not token_offsets:
            _, token_offsets = tokenize_words_with_offsets(example.text)

        gold_token_labels = aspect_spans_to_token_binary(token_offsets, example.aspects)
        pred_token_labels = predicted_labels_to_token_binary(token_offsets, ate_output, pred_spans)

        for gold_label, pred_label in zip(gold_token_labels, pred_token_labels):
            if gold_label == 1 and pred_label == 1:
                token_true_positive += 1
            elif gold_label == 0 and pred_label == 1:
                token_false_positive += 1
            elif gold_label == 1 and pred_label == 0:
                token_false_negative += 1
            else:
                token_true_negative += 1

    span_true_positive = len(gold_global & pred_global)
    span_false_positive = len(pred_global - gold_global)
    span_false_negative = len(gold_global - pred_global)

    precision = (
        token_true_positive / (token_true_positive + token_false_positive)
        if token_true_positive + token_false_positive
        else 0.0
    )
    recall = (
        token_true_positive / (token_true_positive + token_false_negative)
        if token_true_positive + token_false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    token_total = token_true_positive + token_false_positive + token_false_negative + token_true_negative
    token_accuracy = (
        (token_true_positive + token_true_negative) / token_total
        if token_total
        else 0.0
    )
    span_precision = (
        span_true_positive / (span_true_positive + span_false_positive)
        if span_true_positive + span_false_positive
        else 0.0
    )
    span_recall = (
        span_true_positive / (span_true_positive + span_false_negative)
        if span_true_positive + span_false_negative
        else 0.0
    )
    span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall) if span_precision + span_recall else 0.0

    return {
        "accuracy_exact_match": float(np.mean(sentence_matches)) if sentence_matches else 0.0,
        "metric_level": "token_binary",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "token_accuracy": float(token_accuracy),
        "token_true_positive": int(token_true_positive),
        "token_false_positive": int(token_false_positive),
        "token_false_negative": int(token_false_negative),
        "token_true_negative": int(token_true_negative),
        "token_support": int(token_total),
        "span_precision_exact": float(span_precision),
        "span_recall_exact": float(span_recall),
        "span_f1_exact": float(span_f1),
        "span_true_positive_exact": int(span_true_positive),
        "span_false_positive_exact": int(span_false_positive),
        "span_false_negative_exact": int(span_false_negative),
        "gold_spans": int(len(gold_global)),
        "predicted_spans": int(len(pred_global)),
    }


def empty_asc_metrics() -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    per_class = {
        label: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
        for label in SENTIMENT_LABELS
    }
    report_rows = {
        label: {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
        for label in SENTIMENT_LABELS
    }
    report_rows["accuracy"] = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
    report_rows["macro avg"] = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
    report_rows["weighted avg"] = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}

    metrics = {
        "accuracy": 0.0,
        "precision_macro": 0.0,
        "precision_weighted": 0.0,
        "recall_macro": 0.0,
        "recall_weighted": 0.0,
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "per_class": per_class,
        "labels": SENTIMENT_LABELS,
        "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "support": 0,
    }
    return metrics, pd.DataFrame.from_dict(report_rows, orient="index"), np.zeros((3, 3), dtype=int)


def compute_asc_metrics(y_true: list[str], y_pred: list[str]) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    if not y_true:
        return empty_asc_metrics()

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=SENTIMENT_LABELS,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=SENTIMENT_LABELS,
        average="weighted",
        zero_division=0,
    )
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=SENTIMENT_LABELS,
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=SENTIMENT_LABELS)
    report = classification_report(
        y_true,
        y_pred,
        labels=SENTIMENT_LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    per_class = {}
    for idx, label in enumerate(SENTIMENT_LABELS):
        per_class[label] = {
            "precision": float(per_precision[idx]),
            "recall": float(per_recall[idx]),
            "f1": float(per_f1[idx]),
            "support": int(per_support[idx]),
        }

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class,
        "labels": SENTIMENT_LABELS,
        "confusion_matrix": cm.tolist(),
        "support": int(len(y_true)),
    }
    return metrics, report_df, cm


def flatten_metrics(component: str, values: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows = []
    if isinstance(values, dict):
        for key, value in values.items():
            key_prefix = f"{prefix}.{key}" if prefix else key
            rows.extend(flatten_metrics(component, value, key_prefix))
    elif isinstance(values, (int, float, np.integer, np.floating)):
        rows.append({"component": component, "metric": prefix, "value": float(values)})
    return rows


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(jsonable(data), file, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")


def save_confusion_matrix_png(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(SENTIMENT_LABELS)),
        yticks=np.arange(len(SENTIMENT_LABELS)),
        xticklabels=SENTIMENT_LABELS,
        yticklabels=SENTIMENT_LABELS,
        ylabel="Gold label",
        xlabel="Predicted label",
        title="ASC Confusion Matrix",
    )

    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                int(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_metrics_summary_png(metrics: dict[str, Any], output_path: Path) -> None:
    rows = []
    colors = []

    if "ate" in metrics:
        rows.extend([
            ("ATE exact accuracy", metrics["ate"]["accuracy_exact_match"]),
            ("ATE F1", metrics["ate"]["f1"]),
        ])
        colors.extend(["#059669", "#059669"])

    if "asc" in metrics:
        rows.extend([
            ("ASC accuracy", metrics["asc"]["accuracy"]),
            ("ASC macro F1", metrics["asc"]["f1_macro"]),
            ("ASC weighted F1", metrics["asc"]["f1_weighted"]),
        ])
        colors.extend(["#dc2626", "#dc2626", "#dc2626"])

    if "end_to_end" in metrics:
        rows.extend([
            ("E2E exact accuracy", metrics["end_to_end"]["accuracy_exact_match"]),
            ("E2E precision", metrics["end_to_end"]["precision"]),
            ("E2E recall", metrics["end_to_end"]["recall"]),
            ("E2E F1", metrics["end_to_end"]["f1"]),
        ])
        colors.extend(["#7c3aed", "#7c3aed", "#7c3aed", "#7c3aed"])

    if not rows:
        return

    labels = [row[0] for row in rows]
    values = [float(row[1]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("ABSA Benchmark Summary")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.02, 0.98),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_independent_prediction_records(
    examples: list[SentenceExample],
    ate_outputs: list[dict[str, Any]],
    predicted_aspects: list[list[PredictedAspect]],
    asc_by_sentence: dict[tuple[str, int, int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for idx, example in enumerate(examples):
        asc_predictions = []
        for aspect in example.aspects:
            key = (
                example.sentence_id,
                -1 if aspect.start is None else aspect.start,
                -1 if aspect.end is None else aspect.end,
                aspect.term,
            )
            if key in asc_by_sentence:
                asc_predictions.append(asc_by_sentence[key])

        records.append(
            {
                "sentence_id": example.sentence_id,
                "sentence": example.text,
                "preprocessed_sentence": preprocess_normalize_text(example.text),
                "gold_aspects": [asdict(aspect) for aspect in example.aspects],
                "ate_ran": bool(ate_outputs[idx].get("ran", False)),
                "ate_tokens": ate_outputs[idx]["tokens"],
                "ate_offsets": ate_outputs[idx]["offsets"],
                "ate_labels": ate_outputs[idx]["labels"],
                "ate_raw_spans": [asdict(span) for span in ate_outputs[idx]["raw_spans"]],
                "predicted_aspects": [asdict(span) for span in predicted_aspects[idx]],
                "asc_predictions": asc_predictions,
            }
        )
    return records


def build_independent_error_records(
    examples: list[SentenceExample],
    predicted_aspects: list[list[PredictedAspect]],
    asc_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    errors = []

    for idx, example in enumerate(examples):
        gold_set = {span for span in (aspect_to_tuple(example.sentence_id, aspect) for aspect in example.aspects) if span}
        pred_set = {span for span in (aspect_to_tuple(example.sentence_id, aspect) for aspect in predicted_aspects[idx]) if span}
        if gold_set != pred_set:
            errors.append(
                {
                    "error_type": "ate",
                    "sentence_id": example.sentence_id,
                    "sentence": example.text,
                    "missing_spans": sorted(gold_set - pred_set),
                    "extra_spans": sorted(pred_set - gold_set),
                    "gold_aspects": [asdict(aspect) for aspect in example.aspects],
                    "predicted_aspects": [asdict(aspect) for aspect in predicted_aspects[idx]],
                }
            )

    for record in asc_records:
        if record["gold_sentiment"] != record["pred_sentiment"]:
            errors.append(
                {
                    "error_type": "asc",
                    "sentence_id": record["sentence_id"],
                    "sentence": record["sentence"],
                    "aspect": record["aspect"],
                    "gold": record["gold_sentiment"],
                    "predicted": record["pred_sentiment"],
                    "probabilities": record["probabilities"],
                }
            )

    return errors


def main() -> None:
    setup_logging()
    args = parse_args()
    load_runtime_dependencies()
    load_local_pipeline_logic()

    requested_data_path = Path(args.data_path)
    data_path = resolve_dataset_path(requested_data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    LOGGER.info("Loading dataset from %s", data_path)
    examples = load_dataset(data_path)
    if not examples:
        raise ValueError(f"No benchmark examples found in {data_path}")

    sentences = [example.text for example in examples]

    LOGGER.info("Loaded %d sentences and %d gold aspects", len(examples), sum(len(x.aspects) for x in examples))
    LOGGER.info("Using device: %s", device)

    LOGGER.info("Loading ATE model from %s", args.ate_model)
    ate_tokenizer, ate_model = load_token_classifier(args.ate_model, device)
    LOGGER.info("Running independent ATE inference on all sentences with argmax labels")
    ate_outputs = predict_ate(
        sentences,
        ate_tokenizer,
        ate_model,
        device,
        args.batch_size,
        args.max_length,
    )
    for ate_output in ate_outputs:
        ate_output["ran"] = True
    predicted_aspects = [list(output["raw_spans"]) for output in ate_outputs]

    asc_samples: list[tuple[SentenceExample, AspectTerm]] = []
    for example in examples:
        for aspect in example.aspects:
            if aspect.polarity == "conflict":
                continue
            if aspect.polarity not in SENTIMENT_LABELS:
                continue
            asc_samples.append((example, aspect))

    LOGGER.info("Loading ASC model from %s", args.asc_model)
    asc_tokenizer, asc_model = load_sequence_classifier(args.asc_model, device)
    LOGGER.info("Running ASC inference on %d gold aspects with argmax labels", len(asc_samples))
    asc_predictions = predict_asc(
        asc_samples,
        asc_tokenizer,
        asc_model,
        device,
        args.batch_size,
        args.max_length,
    )

    asc_gold_labels = [aspect.polarity for _, aspect in asc_samples if aspect.polarity is not None]
    asc_pred_labels = [prediction["pred_label"] for prediction in asc_predictions]
    ate_metrics = compute_ate_metrics(examples, predicted_aspects, ate_outputs)
    asc_metrics, asc_report_df, asc_cm = compute_asc_metrics(asc_gold_labels, asc_pred_labels)

    asc_records = []
    asc_by_sentence: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for (example, aspect), prediction in zip(asc_samples, asc_predictions):
        record = {
            "sentence_id": example.sentence_id,
            "sentence": example.text,
            "aspect": asdict(aspect),
            "gold_sentiment": aspect.polarity,
            "pred_sentiment": prediction["pred_label"],
            "pred_id": prediction["pred_id"],
            "probabilities": prediction["probabilities"],
            "logits": prediction["logits"],
            "marked_input": prediction["marked_input"],
        }
        asc_records.append(record)
        asc_by_sentence[
            (
                example.sentence_id,
                -1 if aspect.start is None else aspect.start,
                -1 if aspect.end is None else aspect.end,
                aspect.term,
            )
        ] = record

    all_metrics = {
        "dataset": {
            "requested_path": str(requested_data_path),
            "path": str(data_path),
            "num_sentences": len(examples),
            "num_gold_aspects": int(sum(len(example.aspects) for example in examples)),
            "num_asc_eval_aspects": int(len(asc_samples)),
            "num_conflict_aspects_skipped_for_asc": int(
                sum(1 for example in examples for aspect in example.aspects if aspect.polarity == "conflict")
            ),
        },
        "models": {
            "ate_model": args.ate_model,
            "asc_model": args.asc_model,
            "device": str(device),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
        "ate": ate_metrics,
        "asc": asc_metrics,
    }

    prediction_records = build_independent_prediction_records(
        examples,
        ate_outputs,
        predicted_aspects,
        asc_by_sentence,
    )
    error_records = build_independent_error_records(
        examples,
        predicted_aspects,
        asc_records,
    )

    write_json(output_dir / "benchmark_metrics.json", all_metrics)

    metric_rows = []
    metric_rows.extend(flatten_metrics("dataset", all_metrics["dataset"]))
    metric_rows.extend(flatten_metrics("ate", ate_metrics))
    metric_rows.extend(flatten_metrics("asc", asc_metrics))
    pd.DataFrame(metric_rows).to_csv(output_dir / "benchmark_metrics.csv", index=False)

    asc_report_df.to_csv(output_dir / "asc_classification_report.csv")
    save_confusion_matrix_png(asc_cm, output_dir / "asc_confusion_matrix.png")
    save_metrics_summary_png(all_metrics, output_dir / "metrics_summary.png")
    write_jsonl(output_dir / "benchmark_predictions.jsonl", prediction_records)
    write_jsonl(output_dir / "benchmark_errors.jsonl", error_records)

    LOGGER.info("Benchmark complete. Outputs written to %s", output_dir)
    LOGGER.info("ATE F1: %.4f | ASC macro F1: %.4f", ate_metrics["f1"], asc_metrics["f1_macro"])


if __name__ == "__main__":
    main()
