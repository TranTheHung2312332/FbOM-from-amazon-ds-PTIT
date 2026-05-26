from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark as bm


DEFAULT_GATE_MODEL = str(PROJECT_ROOT / "models" / "gate")
DEFAULT_ATE_MODEL = str(PROJECT_ROOT / "models" / "ate")
DEFAULT_ASC_MODEL = str(PROJECT_ROOT / "models" / "asc" / "model")
DEMO_BATCH_SIZE = 1


@st.cache_resource(show_spinner=False)
def prepare_runtime() -> bool:
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    bm.np = np
    bm.torch = torch
    bm.AutoModelForSequenceClassification = AutoModelForSequenceClassification
    bm.AutoModelForTokenClassification = AutoModelForTokenClassification
    bm.AutoTokenizer = AutoTokenizer
    bm.load_local_pipeline_logic()
    return True


@st.cache_resource(show_spinner=False)
def load_sequence_model(model_path: str, device_name: str):
    prepare_runtime()
    return bm.load_sequence_classifier(model_path, bm.torch.device(device_name))


@st.cache_resource(show_spinner=False)
def load_token_model(model_path: str, device_name: str):
    prepare_runtime()
    return bm.load_token_classifier(model_path, bm.torch.device(device_name))


def resolve_device_name(device_arg: str) -> str:
    prepare_runtime()
    return str(bm.resolve_device(device_arg))


def format_float(value: float | None) -> str | None:
    if value is None:
        return None

    text = format(float(value), ".18f").rstrip("0").rstrip(".")
    if not text:
        return "0.0"
    if text == "-0":
        return "0.0"
    if "." not in text:
        text = f"{text}.0"
    return text


def probability_dict(probabilities: list[float], labels: list[str]) -> dict[str, float]:
    return {
        label: float(probabilities[idx]) if idx < len(probabilities) else 0.0
        for idx, label in enumerate(labels)
    }


def display_probability_dict(probabilities: list[float], labels: list[str]) -> dict[str, str | None]:
    return {
        label: format_float(float(probabilities[idx])) if idx < len(probabilities) else format_float(0.0)
        for idx, label in enumerate(labels)
    }


def run_pipeline(
    sentence: str,
    use_gate: bool,
    gate_threshold: float,
    gate_model_path: str,
    ate_model_path: str,
    asc_model_path: str,
    device_arg: str,
    max_length: int,
) -> dict[str, Any]:
    prepare_runtime()
    device_name = resolve_device_name(device_arg)
    device = bm.torch.device(device_name)

    result: dict[str, Any] = {
        "sentence": sentence,
        "device": device_name,
        "gate": {
            "enabled": use_gate,
            "threshold": gate_threshold,
            "score": None,
            "pass": True,
            "probabilities": None,
            "logits": None,
        },
        "ate": {
            "ran": False,
            "tokens": [],
            "labels": [],
            "token_confidences": [],
            "spans": [],
        },
        "asc": [],
    }

    if use_gate:
        gate_tokenizer, gate_model = load_sequence_model(gate_model_path, device_name)
        positive_id = bm.infer_gate_positive_id(gate_model)
        _, gate_probs, gate_logits = bm.predict_gate(
            [sentence],
            gate_tokenizer,
            gate_model,
            device,
            DEMO_BATCH_SIZE,
            max_length,
        )
        gate_score = float(gate_probs[0][positive_id])
        gate_pass = gate_score >= gate_threshold
        result["gate"].update(
            {
                "score": gate_score,
                "pass": gate_pass,
                "positive_id": positive_id,
                "probabilities": gate_probs[0],
                "logits": gate_logits[0],
            }
        )
        if not gate_pass:
            return result

    ate_tokenizer, ate_model = load_token_model(ate_model_path, device_name)
    ate_output = bm.predict_ate(
        [sentence],
        ate_tokenizer,
        ate_model,
        device,
        DEMO_BATCH_SIZE,
        max_length,
    )[0]
    result["ate"] = {
        "ran": True,
        "tokens": ate_output["tokens"],
        "offsets": ate_output["offsets"],
        "labels": ate_output["labels"],
        "token_confidences": ate_output["token_confidences"],
        "spans": [bm.asdict(span) for span in ate_output["raw_spans"]],
    }

    spans = list(ate_output["raw_spans"])
    if not spans:
        return result

    example = bm.SentenceExample(sentence_id="demo", text=sentence, aspects=[])
    asc_tokenizer, asc_model = load_sequence_model(asc_model_path, device_name)
    asc_outputs = bm.predict_asc(
        [(example, span) for span in spans],
        asc_tokenizer,
        asc_model,
        device,
        DEMO_BATCH_SIZE,
        max_length,
    )

    asc_rows = []
    for span, asc_output in zip(spans, asc_outputs):
        asc_rows.append(
            {
                "aspect": span.term,
                "from": span.start,
                "to": span.end,
                "ate_confidence": span.confidence,
                "ate_confidence_display": format_float(span.confidence),
                "sentiment": asc_output["pred_label"],
                "asc_pred_id": asc_output["pred_id"],
                **probability_dict(asc_output["probabilities"], bm.SENTIMENT_LABELS),
                **{
                    f"{label}_display": value
                    for label, value in display_probability_dict(
                        asc_output["probabilities"],
                        bm.SENTIMENT_LABELS,
                    ).items()
                },
            }
        )
    result["asc"] = asc_rows
    return result


def render_gate(result: dict[str, Any]) -> None:
    gate = result["gate"]
    st.subheader("Gate")
    if not gate["enabled"]:
        st.write("Gate disabled. ATE is run directly.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Score", format_float(gate["score"]) if gate["score"] is not None else "n/a")
    col2.metric("Threshold", f"{gate['threshold']:.2f}")
    col3.metric("Pass", "yes" if gate["pass"] else "no")

    if gate["probabilities"] is not None:
        st.json(
            {
                "positive_id": gate.get("positive_id"),
                "probabilities": gate["probabilities"],
                "probabilities_display": [
                    format_float(value) for value in gate["probabilities"]
                ],
            }
        )


def render_ate(result: dict[str, Any]) -> None:
    st.subheader("ATE")
    ate = result["ate"]
    if not ate["ran"]:
        st.write("ATE was skipped because Gate did not pass.")
        return

    token_rows = [
        {
            "token": token,
            "label": label,
            "confidence": format_float(confidence),
        }
        for token, label, confidence in zip(
            ate["tokens"],
            ate["labels"],
            ate["token_confidences"],
        )
    ]
    st.dataframe(token_rows, use_container_width=True, hide_index=True)

    if ate["spans"]:
        st.write("Predicted aspect spans")
        span_rows = []
        for span in ate["spans"]:
            row = dict(span)
            row["confidence"] = format_float(row.get("confidence"))
            span_rows.append(row)
        st.dataframe(span_rows, use_container_width=True, hide_index=True)
    else:
        st.write("No aspect predicted.")


def render_asc(result: dict[str, Any]) -> None:
    st.subheader("ASC")
    if result["asc"]:
        display_rows = []
        for row in result["asc"]:
            display_rows.append(
                {
                    "aspect": row["aspect"],
                    "from": row["from"],
                    "to": row["to"],
                    "ate_confidence": row["ate_confidence_display"],
                    "sentiment": row["sentiment"],
                    "negative": row["negative_display"],
                    "neutral": row["neutral_display"],
                    "positive": row["positive_display"],
                }
            )
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
    else:
        st.write("No ASC prediction because no predicted aspect was available.")


def main() -> None:
    st.set_page_config(page_title="ABSA Pipeline Demo", layout="wide")
    st.title("ABSA Pipeline Demo")

    with st.sidebar:
        st.header("Parameters")
        gate_model_path = st.text_input("Gate model path", DEFAULT_GATE_MODEL)
        ate_model_path = st.text_input("ATE model path", DEFAULT_ATE_MODEL)
        asc_model_path = st.text_input("ASC model path", DEFAULT_ASC_MODEL)

        use_gate = st.checkbox("Use Gate", value=True)
        gate_threshold = st.slider("Gate threshold", 0.0, 1.0, 0.5, 0.01)
        device_arg = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
        max_length = st.number_input("Max length", min_value=16, max_value=512, value=192, step=8)

        if st.button("Clear model cache"):
            st.cache_resource.clear()
            st.rerun()

    sentence = st.text_area(
        "Input sentence",
        value="The battery life is excellent but the screen is dim.",
        height=120,
    )

    run_clicked = st.button("Run inference", type="primary")
    if not run_clicked:
        return

    sentence = sentence.strip()
    if not sentence:
        st.error("Input sentence is empty.")
        return

    try:
        with st.spinner("Running pipeline..."):
            result = run_pipeline(
                sentence=sentence,
                use_gate=use_gate,
                gate_threshold=gate_threshold,
                gate_model_path=gate_model_path,
                ate_model_path=ate_model_path,
                asc_model_path=asc_model_path,
                device_arg=device_arg,
                max_length=int(max_length),
            )
    except Exception as exc:
        st.exception(exc)
        return

    render_gate(result)
    render_ate(result)
    render_asc(result)

    with st.expander("Raw output"):
        st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")


if __name__ == "__main__":
    main()
