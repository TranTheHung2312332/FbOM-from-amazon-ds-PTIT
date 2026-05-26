from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark as bm


LOGGER = logging.getLogger("benchmark_pipeline")
THRESHOLDS = [round(value / 10, 1) for value in range(2, 10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ABSA pipeline end-to-end with Gate threshold sweep.",
    )
    parser.add_argument("--data-path", required=True, help="Benchmark dataset path. SemEval XML is preferred.")
    parser.add_argument("--gate-model", required=True, help="Gate sequence-classification model path.")
    parser.add_argument("--ate-model", required=True, help="ATE token-classification model path.")
    parser.add_argument("--asc-model", required=True, help="ASC sequence-classification model path.")
    parser.add_argument("--output-dir", default="outputs/benchmark_pipeline", help="Directory for pipeline outputs.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--max-length", type=int, default=192, help="Tokenizer max length.")
    return parser.parse_args()


def gold_end_to_end_tuples(example: bm.SentenceExample) -> set[tuple[str, int, int, str]]:
    tuples = set()
    for aspect in example.aspects:
        if aspect.polarity not in bm.SENTIMENT_LABELS:
            continue
        if aspect.start is None or aspect.end is None:
            continue
        tuples.add((example.sentence_id, int(aspect.start), int(aspect.end), aspect.polarity))
    return tuples


def predicted_end_to_end_tuples(
    sentence_id: str,
    predicted_aspects: list[dict[str, Any]],
) -> set[tuple[str, int, int, str]]:
    tuples = set()
    for aspect in predicted_aspects:
        if aspect.get("polarity") not in bm.SENTIMENT_LABELS:
            continue
        if aspect.get("start") is None or aspect.get("end") is None:
            continue
        tuples.add((sentence_id, int(aspect["start"]), int(aspect["end"]), aspect["polarity"]))
    return tuples


def compute_end_to_end_metrics(
    gold_by_sentence: list[set[tuple[str, int, int, str]]],
    pred_by_sentence: list[set[tuple[str, int, int, str]]],
) -> dict[str, float | int]:
    sentence_matches = [gold == pred for gold, pred in zip(gold_by_sentence, pred_by_sentence)]
    gold_global = set().union(*gold_by_sentence) if gold_by_sentence else set()
    pred_global = set().union(*pred_by_sentence) if pred_by_sentence else set()

    true_positive = len(gold_global & pred_global)
    false_positive = len(pred_global - gold_global)
    false_negative = len(gold_global - pred_global)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy_exact_match": float(bm.np.mean(sentence_matches)) if sentence_matches else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "gold_tuples": int(len(gold_global)),
        "predicted_tuples": int(len(pred_global)),
    }


def build_pred_sets_for_threshold(
    examples: list[bm.SentenceExample],
    gate_scores: list[float],
    candidate_predictions: list[list[dict[str, Any]]],
    threshold: float,
) -> list[set[tuple[str, int, int, str]]]:
    pred_by_sentence = []
    for example, gate_score, predictions in zip(examples, gate_scores, candidate_predictions):
        if gate_score < threshold:
            pred_by_sentence.append(set())
            continue
        pred_by_sentence.append(predicted_end_to_end_tuples(example.sentence_id, predictions))
    return pred_by_sentence


def select_best_threshold(threshold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        threshold_metrics,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["recall"],
        ),
    )


def save_threshold_curve(threshold_metrics: list[dict[str, Any]], output_path: Path) -> None:
    thresholds = [row["threshold"] for row in threshold_metrics]
    f1_values = [row["f1"] for row in threshold_metrics]
    precision_values = [row["precision"] for row in threshold_metrics]
    recall_values = [row["recall"] for row in threshold_metrics]

    fig, ax = bm.plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1_values, marker="o", label="F1")
    ax.plot(thresholds, precision_values, marker="o", label="Precision")
    ax.plot(thresholds, recall_values, marker="o", label="Recall")
    ax.set_xlabel("Gate threshold")
    ax.set_ylabel("End-to-end score")
    ax.set_title("Pipeline Threshold Sweep")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    bm.plt.close(fig)


def build_prediction_records(
    examples: list[bm.SentenceExample],
    gate_scores: list[float],
    candidate_predictions: list[list[dict[str, Any]]],
    gold_by_sentence: list[set[tuple[str, int, int, str]]],
    pred_by_sentence: list[set[tuple[str, int, int, str]]],
    threshold: float,
) -> list[dict[str, Any]]:
    records = []
    for example, gate_score, candidate, gold_tuples, pred_tuples in zip(
        examples,
        gate_scores,
        candidate_predictions,
        gold_by_sentence,
        pred_by_sentence,
    ):
        gate_pass = gate_score >= threshold
        records.append(
            {
                "sentence_id": example.sentence_id,
                "sentence": example.text,
                "gate_score": gate_score,
                "gate_threshold": threshold,
                "gate_pass": gate_pass,
                "gold_tuples": sorted(gold_tuples),
                "predicted_aspects": candidate if gate_pass else [],
                "predicted_tuples": sorted(pred_tuples),
            }
        )
    return records


def build_error_records(
    prediction_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    errors = []
    for record in prediction_records:
        gold = {tuple(item) for item in record["gold_tuples"]}
        pred = {tuple(item) for item in record["predicted_tuples"]}
        if gold == pred:
            continue
        errors.append(
            {
                "sentence_id": record["sentence_id"],
                "sentence": record["sentence"],
                "gate_score": record["gate_score"],
                "gate_threshold": record["gate_threshold"],
                "gate_pass": record["gate_pass"],
                "missing_tuples": sorted(gold - pred),
                "extra_tuples": sorted(pred - gold),
            }
        )
    return errors


def main() -> None:
    bm.setup_logging()
    args = parse_args()
    bm.load_runtime_dependencies()
    bm.load_local_pipeline_logic()

    requested_data_path = Path(args.data_path)
    data_path = bm.resolve_dataset_path(requested_data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = bm.resolve_device(args.device)

    LOGGER.info("Loading dataset from %s", data_path)
    examples = bm.load_dataset(data_path)
    if not examples:
        raise ValueError(f"No benchmark examples found in {data_path}")

    sentences = [example.text for example in examples]
    gold_by_sentence = [gold_end_to_end_tuples(example) for example in examples]

    LOGGER.info("Loaded %d sentences and %d end-to-end gold tuples", len(examples), sum(len(x) for x in gold_by_sentence))
    LOGGER.info("Using device: %s", device)

    LOGGER.info("Loading Gate model from %s", args.gate_model)
    gate_tokenizer, gate_model = bm.load_sequence_classifier(args.gate_model, device)
    positive_id = bm.infer_gate_positive_id(gate_model)
    LOGGER.info("Running Gate inference once for threshold sweep")
    _, gate_probs, gate_logits = bm.predict_gate(
        sentences,
        gate_tokenizer,
        gate_model,
        device,
        args.batch_size,
        args.max_length,
    )
    gate_scores = [float(probs[positive_id]) for probs in gate_probs]

    min_threshold = min(THRESHOLDS)
    candidate_indices = [idx for idx, score in enumerate(gate_scores) if score >= min_threshold]
    LOGGER.info(
        "Gate threshold sweep range is %.1f..%.1f. Running ATE for %d/%d sentences passing %.1f.",
        min(THRESHOLDS),
        max(THRESHOLDS),
        len(candidate_indices),
        len(examples),
        min_threshold,
    )

    candidate_ate_outputs = [bm.empty_ate_prediction() for _ in examples]
    candidate_aspects: list[list[bm.PredictedAspect]] = [[] for _ in examples]
    if candidate_indices:
        LOGGER.info("Loading ATE model from %s", args.ate_model)
        ate_tokenizer, ate_model = bm.load_token_classifier(args.ate_model, device)
        ate_outputs = bm.predict_ate(
            [sentences[idx] for idx in candidate_indices],
            ate_tokenizer,
            ate_model,
            device,
            args.batch_size,
            args.max_length,
        )
        for original_idx, ate_output in zip(candidate_indices, ate_outputs):
            ate_output["ran"] = True
            candidate_ate_outputs[original_idx] = ate_output
            candidate_aspects[original_idx] = list(ate_output["raw_spans"])

    asc_samples: list[tuple[bm.SentenceExample, bm.PredictedAspect]] = []
    asc_keys: list[tuple[int, int]] = []
    for example_idx, spans in enumerate(candidate_aspects):
        for span_idx, span in enumerate(spans):
            asc_samples.append((examples[example_idx], span))
            asc_keys.append((example_idx, span_idx))

    LOGGER.info("Loading ASC model from %s", args.asc_model)
    asc_tokenizer, asc_model = bm.load_sequence_classifier(args.asc_model, device)
    LOGGER.info("Running ASC inference on %d predicted aspects with argmax labels", len(asc_samples))
    asc_predictions = bm.predict_asc(
        asc_samples,
        asc_tokenizer,
        asc_model,
        device,
        args.batch_size,
        args.max_length,
    ) if asc_samples else []

    candidate_predictions: list[list[dict[str, Any]]] = [[] for _ in examples]
    for (example_idx, span_idx), asc_prediction in zip(asc_keys, asc_predictions):
        span = candidate_aspects[example_idx][span_idx]
        candidate_predictions[example_idx].append(
            {
                "term": span.term,
                "start": span.start,
                "end": span.end,
                "confidence": span.confidence,
                "start_token": span.start_token,
                "end_token": span.end_token,
                "polarity": asc_prediction["pred_label"],
                "asc_pred_id": asc_prediction["pred_id"],
                "asc_probabilities": asc_prediction["probabilities"],
                "asc_logits": asc_prediction["logits"],
            }
        )

    threshold_metrics = []
    pred_sets_by_threshold = {}
    for threshold in THRESHOLDS:
        pred_by_sentence = build_pred_sets_for_threshold(
            examples,
            gate_scores,
            candidate_predictions,
            threshold,
        )
        metrics = compute_end_to_end_metrics(gold_by_sentence, pred_by_sentence)
        row = {"threshold": threshold, **metrics}
        threshold_metrics.append(row)
        pred_sets_by_threshold[threshold] = pred_by_sentence

    best_row = select_best_threshold(threshold_metrics)
    best_threshold = float(best_row["threshold"])
    best_metrics = {key: value for key, value in best_row.items() if key != "threshold"}
    best_pred_by_sentence = pred_sets_by_threshold[best_threshold]

    prediction_records = build_prediction_records(
        examples,
        gate_scores,
        candidate_predictions,
        gold_by_sentence,
        best_pred_by_sentence,
        best_threshold,
    )
    error_records = build_error_records(prediction_records)

    all_metrics = {
        "dataset": {
            "requested_path": str(requested_data_path),
            "path": str(data_path),
            "num_sentences": len(examples),
            "num_gold_tuples": int(sum(len(tuples) for tuples in gold_by_sentence)),
            "num_conflict_aspects_skipped": int(
                sum(1 for example in examples for aspect in example.aspects if aspect.polarity == "conflict")
            ),
        },
        "models": {
            "gate_model": args.gate_model,
            "ate_model": args.ate_model,
            "asc_model": args.asc_model,
            "device": str(device),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
        "threshold_selection": {
            "range": THRESHOLDS,
            "selected_threshold": best_threshold,
            "selection_metric": "max_end_to_end_f1",
        },
        "end_to_end": best_metrics,
        "threshold_metrics": threshold_metrics,
    }

    bm.write_json(output_dir / "pipeline_metrics.json", all_metrics)
    bm.pd.DataFrame(threshold_metrics).to_csv(output_dir / "pipeline_threshold_metrics.csv", index=False)
    bm.write_jsonl(output_dir / "pipeline_predictions.jsonl", prediction_records)
    bm.write_jsonl(output_dir / "pipeline_errors.jsonl", error_records)
    bm.save_metrics_summary_png({"end_to_end": best_metrics}, output_dir / "pipeline_metrics_summary.png")
    save_threshold_curve(threshold_metrics, output_dir / "pipeline_threshold_curve.png")

    LOGGER.info("Pipeline benchmark complete. Outputs written to %s", output_dir)
    LOGGER.info(
        "Selected Gate threshold %.1f | E2E precision %.4f | recall %.4f | F1 %.4f",
        best_threshold,
        best_metrics["precision"],
        best_metrics["recall"],
        best_metrics["f1"],
    )


if __name__ == "__main__":
    main()
