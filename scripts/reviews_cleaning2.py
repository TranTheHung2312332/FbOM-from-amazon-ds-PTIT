
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterator, List

import pandas as pd
from pyspark.sql import SparkSession, Window, functions as F, types as T

REQUIRED_COLUMNS = ["parent_asin", "review_id", "rating", "sentence_id", "sentence_text"]
OUTPUT_COLUMNS = ["parent_asin", "review_id", "sentence_id", "sentence_text", "rating"]

GENERIC_NOUNS = {
    "thing", "things", "stuff", "item", "items", "product", "products", "object", "objects",
    "part", "parts", "piece", "pieces", "kind", "kinds", "type", "types", "sort", "sorts",
    "one", "ones", "something", "anything", "everything", "nothing", "someone", "anyone",
    "everyone", "nobody", "aspect", "issue", "issues", "lot", "bit", "matter", "way",
    "case", "point", "side", "version", "time", "times", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "problem", "problems", "reason", "reasons",
    "result", "results", "experience", "review", "reviews", "rating", "ratings", "star", "stars",
}

DOMAIN_NOISE = {
    "amazon", "seller", "sellers", "shipping", "delivery", "shipment", "shipments", "package",
    "packages", "packaging", "box", "boxes", "wrapper", "wrappers", "return", "returns",
    "refund", "refunds", "replacement", "replacements",
}

CONTRACTIONS = {
    "ain't": "is not", "aren't": "are not", "can't": "can not", "can't've": "can not have",
    "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it'd": "it would", "it'll": "it will", "it's": "it is", "let's": "let us",
    "might've": "might have", "mightn't": "might not", "must've": "must have", "mustn't": "must not",
    "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "that'd": "that would", "that's": "that is",
    "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will",
    "we're": "we are", "we've": "we have", "weren't": "were not", "what's": "what is",
    "where's": "where is", "who's": "who is", "won't": "will not", "would've": "would have",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are",
    "you've": "you have",
}

CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)) + r")\b"
)
TOKEN_RE = re.compile(r"[a-z0-9']+")
KEEP_CHARS_RE = re.compile(r"[^a-z0-9\s']")
MULTISPACE_RE = re.compile(r"\s+")

OPINION_VERBS = {
    "adore", "admire", "annoy", "appreciate", "avoid", "confuse", "despise", "disappoint",
    "dislike", "disturb", "enjoy", "favor", "hate", "impress", "like", "love", "need",
    "prefer", "recommend", "regret", "satisfy", "surprise", "value", "want", "wish",
}
LINKING_VERBS = {"be", "become", "feel", "look", "remain", "seem", "smell", "sound", "stay", "taste"}
SOCIAL_PATTERNS = [
    re.compile(r"^(thank|thanks|thank you)( so much)?$"),
    re.compile(r"^(thanks again)$"),
    re.compile(r"^(have a nice day)$"),
    re.compile(r"^(best regards|kind regards|warm regards)$"),
    re.compile(r"^(happy holidays|merry christmas)$"),
]
PRONOUNS = {"i", "it", "this", "that", "these", "those", "they", "he", "she", "them", "him", "her"}

AUDIT_SCHEMA = T.StructType([
    T.StructField("parent_asin", T.StringType(), True),
    T.StructField("review_id", T.StringType(), True),
    T.StructField("rating", T.DoubleType(), True),
    T.StructField("orig_sentence_id", T.LongType(), True),
    T.StructField("original_sentence_text", T.StringType(), True),
    T.StructField("normalized_text", T.StringType(), True),
    T.StructField("final_text", T.StringType(), True),
    T.StructField("keep", T.BooleanType(), True),
    T.StructField("primary_reason", T.StringType(), True),
    T.StructField("generic_token_hits", T.IntegerType(), True),
    T.StructField("domain_noise_hits", T.IntegerType(), True),
    T.StructField("char_count", T.IntegerType(), True),
    T.StructField("token_count", T.IntegerType(), True),
    T.StructField("flag_too_short", T.BooleanType(), True),
    T.StructField("flag_no_noun_or_np", T.BooleanType(), True),
    T.StructField("flag_adj_only", T.BooleanType(), True),
    T.StructField("flag_social", T.BooleanType(), True),
    T.StructField("flag_pronoun_target_only", T.BooleanType(), True),
    T.StructField("flag_all_targets_generic_or_noise", T.BooleanType(), True),
    T.StructField("flag_no_dependency_pattern", T.BooleanType(), True),
    T.StructField("matched_patterns", T.StringType(), True),
])

NLP = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ABSA reviews cleaning - part 2 with Spark on Linux/WSL")
    parser.add_argument("--input", required=True, help="Input parquet from cleaning 1")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--category-name", required=True, help="Category name for report filename")
    parser.add_argument("--report-dir", default="outputs/reports", help="Directory for report txt")
    parser.add_argument("--debug-output", default=None, help="Optional parquet path for audit/debug rows")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model name/path")
    parser.add_argument("--spacy-batch-size", type=int, default=256, help="spaCy nlp.pipe batch size per partition")
    parser.add_argument("--master", default="local[*]", help='Spark master, e.g. "local[*]"')
    parser.add_argument("--max-rows", type=int, default=None, help="Only process first N rows for testing")
    parser.add_argument("--input-partitions", type=int, default=None, help="Optional repartition before NLP")
    parser.add_argument("--output-partitions", type=int, default=None, help="Optional repartition before write")
    parser.add_argument("--shuffle-partitions", type=int, default=None, help="spark.sql.shuffle.partitions")
    parser.add_argument("--cache-audit", action="store_true", help="Cache audit dataframe; useful because report uses repeated actions")
    parser.add_argument("--disable-adaptive", action="store_true", help="Disable Spark adaptive query execution")
    return parser.parse_args()


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def path_size_bytes(path_str: str) -> int:
    path = Path(path_str)
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total
    return 0


def normalize_contractions(text: str) -> str:
    return CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group(0)], text)


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    value = str(text)
    if value.lower() == "nan":
        return ""
    value = value.replace("’", "'").replace("`", "'").replace("´", "'")
    value = value.lower().strip()
    value = normalize_contractions(value)
    value = KEEP_CHARS_RE.sub(" ", value)
    value = MULTISPACE_RE.sub(" ", value).strip()
    return value


def simple_tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def get_nlp(model_name: str):
    global NLP
    if NLP is None:
        import spacy
        NLP = spacy.load(model_name, disable=["ner"])
    return NLP


def token_key(token) -> str:
    lemma = token.lemma_.lower().strip()
    if lemma and lemma != "-pron-":
        return lemma
    return token.text.lower().strip()


def is_aspect_noun(token) -> bool:
    return token.pos_ in {"NOUN", "PROPN"}


def has_noun_or_noun_phrase(doc) -> bool:
    if any(is_aspect_noun(tok) for tok in doc):
        return True
    try:
        return any(True for _ in doc.noun_chunks)
    except Exception:
        return False


def is_adj_only_or_phrase(doc) -> bool:
    lexical = [tok for tok in doc if not tok.is_space]
    if not lexical:
        return False
    allowed = {"ADJ", "ADV", "CCONJ", "SCONJ", "PART", "INTJ"}
    blocked = {"NOUN", "PROPN", "VERB", "AUX", "PRON"}
    if any(tok.pos_ in blocked for tok in lexical):
        return False
    return all(tok.pos_ in allowed or tok.is_punct for tok in lexical)


def pronoun_target_only(doc) -> bool:
    pron_target = False
    noun_target = False
    for tok in doc:
        if tok.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj", "attr"}:
            if tok.pos_ in {"NOUN", "PROPN"}:
                noun_target = True
            elif tok.pos_ == "PRON" or tok.text.lower() in PRONOUNS:
                pron_target = True
    return pron_target and not noun_target


def extract_aspect_targets(doc):
    targets = []
    for tok in doc:
        if tok.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj", "attr", "conj"} and tok.pos_ in {"NOUN", "PROPN", "PRON"}:
            targets.append(tok)
    return targets


def is_opinion_predicate(token) -> bool:
    lemma = token.lemma_.lower()
    if token.pos_ == "ADJ":
        return True
    if token.pos_ == "VERB":
        if lemma in OPINION_VERBS or lemma in LINKING_VERBS:
            return True
        if any(child.dep_ in {"acomp", "attr", "oprd", "xcomp"} and child.pos_ == "ADJ" for child in token.children):
            return True
    if token.pos_ == "AUX" and lemma in LINKING_VERBS:
        return True
    return False


def pattern_amod(doc) -> bool:
    return any(tok.dep_ == "amod" and tok.pos_ == "ADJ" and is_aspect_noun(tok.head) for tok in doc)


def pattern_nsubj_opinion_predicate(doc) -> bool:
    for tok in doc:
        if tok.dep_ in {"nsubj", "nsubjpass"} and is_aspect_noun(tok):
            head = tok.head
            if is_opinion_predicate(head):
                return True
            if head.pos_ in {"VERB", "AUX"} and any(
                child.dep_ in {"acomp", "attr", "oprd", "xcomp"} and child.pos_ == "ADJ"
                for child in head.children
            ):
                return True
    return False


def pattern_obj_opinion_verb(doc) -> bool:
    for tok in doc:
        if tok.dep_ in {"dobj", "obj"} and is_aspect_noun(tok) and tok.head.pos_ == "VERB":
            if tok.head.lemma_.lower() in OPINION_VERBS:
                return True
    return False


def pattern_copula_adj_noun_subject(doc) -> bool:
    for tok in doc:
        if tok.pos_ == "ADJ":
            has_cop = any(child.dep_ == "cop" for child in tok.children)
            has_nsubj = any(child.dep_ in {"nsubj", "nsubjpass"} and is_aspect_noun(child) for child in tok.children)
            if has_cop and has_nsubj:
                return True
            if tok.head.pos_ in {"VERB", "AUX"} and tok.head.lemma_.lower() in LINKING_VERBS:
                if any(child.dep_ in {"nsubj", "nsubjpass"} and is_aspect_noun(child) for child in tok.head.children):
                    return True
    return False


def pattern_conj_opinion(doc) -> bool:
    return any(tok.dep_ == "conj" and tok.pos_ == "ADJ" and tok.head.pos_ == "ADJ" for tok in doc)


def pattern_conj_aspect(doc) -> bool:
    return any(tok.dep_ == "conj" and is_aspect_noun(tok) and is_aspect_noun(tok.head) for tok in doc)


def pattern_neg_opinion(doc) -> bool:
    for tok in doc:
        if tok.dep_ == "neg":
            head = tok.head
            if head.pos_ == "ADJ" or head.lemma_.lower() in OPINION_VERBS or head.lemma_.lower() in LINKING_VERBS:
                return True
    return False


def pattern_obj_xcomp(doc) -> bool:
    for tok in doc:
        if tok.dep_ in {"dobj", "obj"} and is_aspect_noun(tok):
            head = tok.head
            if head.pos_ == "VERB":
                for child in head.children:
                    if child.dep_ == "xcomp" and child.pos_ == "ADJ":
                        return True
    return False


PATTERNS = [
    ("amod(adj,noun)", pattern_amod),
    ("nsubj(noun,opinion_predicate)", pattern_nsubj_opinion_predicate),
    ("obj(noun,opinion_verb)", pattern_obj_opinion_verb),
    ("copula_adj_noun_subject", pattern_copula_adj_noun_subject),
    ("conj(opinion1,opinion2)", pattern_conj_opinion),
    ("conj(aspect1,aspect2)", pattern_conj_aspect),
    ("neg(negation,opinion_word)", pattern_neg_opinion),
    ("obj(noun,verb)+xcomp(adj,verb)", pattern_obj_xcomp),
]


def audit_one_sentence(doc, normalized_text: str) -> Dict[str, object]:
    tokens = simple_tokens(normalized_text)
    char_count = len(normalized_text)
    token_count = len(tokens)

    generic_hits = 0
    noise_hits = 0
    final_tokens = []

    for tok in doc:
        if tok.is_space:
            continue
        key = token_key(tok)
        if key in GENERIC_NOUNS:
            generic_hits += 1
            final_tokens.append("[GENERIC_NOUN]")
        elif key in DOMAIN_NOISE:
            noise_hits += 1
            final_tokens.append("[DOMAIN_NOISE]")
        else:
            final_tokens.append(tok.text.lower())

    final_text = MULTISPACE_RE.sub(" ", " ".join(final_tokens)).strip()

    flag_too_short = char_count < 5 or token_count < 2
    flag_social = any(p.fullmatch(normalized_text) for p in SOCIAL_PATTERNS)
    flag_no_noun_or_np = not has_noun_or_noun_phrase(doc)
    flag_adj_only = is_adj_only_or_phrase(doc)
    flag_pronoun_target_only = pronoun_target_only(doc)

    targets = extract_aspect_targets(doc)
    noun_targets = [tok for tok in targets if tok.pos_ in {"NOUN", "PROPN"}]
    flag_all_targets_generic_or_noise = bool(noun_targets) and all(
        token_key(tok) in GENERIC_NOUNS or token_key(tok) in DOMAIN_NOISE for tok in noun_targets
    )

    matched_patterns = [name for name, fn in PATTERNS if fn(doc)]
    flag_no_dependency_pattern = len(matched_patterns) == 0

    keep = True
    reason = "kept"
    ordered_reasons = [
        ("too_short", flag_too_short),
        ("social", flag_social),
        ("no_noun_or_noun_phrase", flag_no_noun_or_np),
        ("adjective_only_phrase", flag_adj_only),
        ("pronoun_target_only", flag_pronoun_target_only),
        ("all_targets_generic_or_noise", flag_all_targets_generic_or_noise),
        ("no_dependency_pattern", flag_no_dependency_pattern),
    ]
    for label, cond in ordered_reasons:
        if cond:
            keep = False
            reason = label
            break

    return {
        "normalized_text": normalized_text,
        "final_text": final_text,
        "keep": keep,
        "primary_reason": reason,
        "generic_token_hits": int(generic_hits),
        "domain_noise_hits": int(noise_hits),
        "char_count": int(char_count),
        "token_count": int(token_count),
        "flag_too_short": bool(flag_too_short),
        "flag_no_noun_or_np": bool(flag_no_noun_or_np),
        "flag_adj_only": bool(flag_adj_only),
        "flag_social": bool(flag_social),
        "flag_pronoun_target_only": bool(flag_pronoun_target_only),
        "flag_all_targets_generic_or_noise": bool(flag_all_targets_generic_or_noise),
        "flag_no_dependency_pattern": bool(flag_no_dependency_pattern),
        "matched_patterns": "|".join(matched_patterns),
    }


def make_partition_processor(spacy_model: str, batch_size: int):
    def _process(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        nlp = get_nlp(spacy_model)
        for pdf in iterator:
            if pdf.empty:
                yield pd.DataFrame(columns=[f.name for f in AUDIT_SCHEMA.fields])
                continue

            local = pdf.copy()
            local["original_sentence_text"] = local["sentence_text"].astype(str)
            local["normalized_text"] = local["sentence_text"].map(normalize_text)

            docs = list(nlp.pipe(local["normalized_text"].tolist(), batch_size=batch_size))
            rows = []
            for row, doc, norm in zip(local.itertuples(index=False), docs, local["normalized_text"].tolist()):
                audit = audit_one_sentence(doc, norm)
                rows.append({
                    "parent_asin": None if pd.isna(row.parent_asin) else str(row.parent_asin),
                    "review_id": None if pd.isna(row.review_id) else str(row.review_id),
                    "rating": None if pd.isna(row.rating) else float(row.rating),
                    "orig_sentence_id": None if pd.isna(row.sentence_id) else int(row.sentence_id),
                    "original_sentence_text": "" if pd.isna(row.sentence_text) else str(row.sentence_text),
                    **audit,
                })
            yield pd.DataFrame(rows)
    return _process


def validate_columns(df) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_report(stats: Dict[str, object]) -> str:
    lines = [
        f"Category: {stats['category_name']}",
        f"Số review ban đầu: {stats['initial_reviews']:,}",
        f"Số câu trước khi xử lý: {stats['sentences_before']:,}",
        f"Số câu sau khi xử lý: {stats['sentences_after']:,}",
        f"Số review còn lại (có ít nhất 1 câu giữ lại): {stats['remaining_reviews']:,}",
        f"Tỉ lệ review có thể có ích: {stats['helpful_review_ratio']:.6f}",
        f"Kích thước ban đầu: {stats['input_size_human']}",
        f"Kích thước cuối cùng: {stats['output_size_human']}",
        "",
        "=== Thống kê rule / marker ===",
        f"Số lượng GENERIC_NOUNS được đánh dấu: {stats['generic_hits']:,}",
        f"Số lượng DOMAIN_NOISE được đánh dấu: {stats['domain_noise_hits']:,}",
        f"Số câu không có danh từ hoặc cụm danh từ: {stats['cnt_no_noun_or_np']:,}",
        f"Số câu chứa toàn tính từ hoặc cụm tính từ: {stats['cnt_adj_only']:,}",
        f"Số câu chỉ mang mục đích xã giao: {stats['cnt_social']:,}",
        f"Số câu có số lượng từ < 5: {stats['cnt_lt5_tokens']:,}",
        f"Số câu target chỉ là pronoun: {stats['cnt_pronoun_target_only']:,}",
        f"Số câu có target nhưng tất cả đều generic/noise: {stats['cnt_all_targets_generic_or_noise']:,}",
        f"Số câu không khớp dependency pattern nào: {stats['cnt_no_dependency_pattern']:,}",
        "",
        "=== Primary removal reasons ===",
    ]
    for k, v in stats["reason_counts"].items():
        lines.append(f"{k}: {v:,}")
    lines += ["", "=== Matched dependency patterns ==="]
    for k, v in stats["pattern_counts"].items():
        lines.append(f"{k}: {v:,}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    input_path = str(Path(args.input).expanduser().resolve())
    output_path = str(Path(args.output).expanduser().resolve())
    report_dir = Path(args.report_dir).expanduser().resolve()
    debug_output_path = str(Path(args.debug_output).expanduser().resolve()) if args.debug_output else None

    builder = SparkSession.builder.appName(f"reviews_cleaning2_{args.category_name}").master(args.master)
    if args.shuffle_partitions is not None:
        builder = builder.config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    if args.disable_adaptive:
        builder = builder.config("spark.sql.adaptive.enabled", "false")

    spark = builder.getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    input_size = path_size_bytes(input_path)

    df = spark.read.parquet(input_path)
    validate_columns(df)

    df = df.select(
        F.col("parent_asin").cast("string"),
        F.col("review_id").cast("string"),
        F.col("rating").cast("double"),
        F.col("sentence_id").cast("long"),
        F.col("sentence_text").cast("string"),
    )

    if args.max_rows is not None:
        df = df.orderBy("review_id", "sentence_id").limit(args.max_rows)

    if args.input_partitions is not None and args.input_partitions > 0:
        df = df.repartition(args.input_partitions)

    initial_reviews = df.select("review_id").distinct().count()
    sentences_before = df.count()

    processor = make_partition_processor(args.spacy_model, args.spacy_batch_size)
    audit_df = df.mapInPandas(processor, schema=AUDIT_SCHEMA)

    if args.cache_audit:
        audit_df = audit_df.cache()
        _ = audit_df.count()

    if debug_output_path:
        Path(debug_output_path).parent.mkdir(parents=True, exist_ok=True)
        audit_df.write.mode("overwrite").parquet(debug_output_path)

    kept_df = (
        audit_df
        .filter(F.col("keep") == True)
        .select(
            "parent_asin",
            "review_id",
            F.col("rating").cast("double").alias("rating"),
            F.col("orig_sentence_id").cast("long").alias("orig_sentence_id"),
            F.col("final_text").alias("sentence_text"),
        )
        .filter(F.length(F.trim(F.col("sentence_text"))) > 0)
    )

    w = Window.partitionBy("review_id").orderBy(F.col("orig_sentence_id").asc_nulls_last())
    final_df = (
        kept_df
        .withColumn("sentence_id", F.row_number().over(w))
        .select(*OUTPUT_COLUMNS)
    )

    if args.output_partitions is not None and args.output_partitions > 0:
        final_df = final_df.repartition(args.output_partitions, "parent_asin")
    else:
        current_partitions = final_df.rdd.getNumPartitions()
        target = max(1, min(current_partitions, 256))
        final_df = final_df.repartition(target, "parent_asin")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.write.mode("overwrite").parquet(output_path)
    output_size = path_size_bytes(output_path)

    sentences_after = final_df.count()
    remaining_reviews = final_df.select("review_id").distinct().count()
    helpful_review_ratio = (remaining_reviews / initial_reviews) if initial_reviews else 0.0

    agg_row = audit_df.agg(
        F.sum(F.col("generic_token_hits")).alias("generic_hits"),
        F.sum(F.col("domain_noise_hits")).alias("domain_noise_hits"),
        F.sum(F.when(F.col("flag_no_noun_or_np"), 1).otherwise(0)).alias("cnt_no_noun_or_np"),
        F.sum(F.when(F.col("flag_adj_only"), 1).otherwise(0)).alias("cnt_adj_only"),
        F.sum(F.when(F.col("flag_social"), 1).otherwise(0)).alias("cnt_social"),
        F.sum(F.when(F.col("token_count") < 5, 1).otherwise(0)).alias("cnt_lt5_tokens"),
        F.sum(F.when(F.col("flag_pronoun_target_only"), 1).otherwise(0)).alias("cnt_pronoun_target_only"),
        F.sum(F.when(F.col("flag_all_targets_generic_or_noise"), 1).otherwise(0)).alias("cnt_all_targets_generic_or_noise"),
        F.sum(F.when(F.col("flag_no_dependency_pattern"), 1).otherwise(0)).alias("cnt_no_dependency_pattern"),
    ).collect()[0].asDict()

    reason_rows = audit_df.groupBy("primary_reason").count().orderBy(F.desc("count")).collect()
    reason_counts = {r["primary_reason"]: int(r["count"]) for r in reason_rows}

    pattern_rows = (
        audit_df
        .select(F.explode_outer(F.split(F.col("matched_patterns"), r"\|")).alias("pattern"))
        .filter(F.col("pattern").isNotNull() & (F.col("pattern") != ""))
        .groupBy("pattern")
        .count()
        .orderBy(F.desc("count"))
        .collect()
    )
    pattern_counts = {r["pattern"]: int(r["count"]) for r in pattern_rows}

    report_stats = {
        "category_name": args.category_name,
        "initial_reviews": int(initial_reviews),
        "sentences_before": int(sentences_before),
        "sentences_after": int(sentences_after),
        "remaining_reviews": int(remaining_reviews),
        "helpful_review_ratio": float(helpful_review_ratio),
        "input_size_human": human_size(input_size),
        "output_size_human": human_size(output_size),
        "generic_hits": int(agg_row["generic_hits"] or 0),
        "domain_noise_hits": int(agg_row["domain_noise_hits"] or 0),
        "cnt_no_noun_or_np": int(agg_row["cnt_no_noun_or_np"] or 0),
        "cnt_adj_only": int(agg_row["cnt_adj_only"] or 0),
        "cnt_social": int(agg_row["cnt_social"] or 0),
        "cnt_lt5_tokens": int(agg_row["cnt_lt5_tokens"] or 0),
        "cnt_pronoun_target_only": int(agg_row["cnt_pronoun_target_only"] or 0),
        "cnt_all_targets_generic_or_noise": int(agg_row["cnt_all_targets_generic_or_noise"] or 0),
        "cnt_no_dependency_pattern": int(agg_row["cnt_no_dependency_pattern"] or 0),
        "reason_counts": reason_counts,
        "pattern_counts": pattern_counts,
    }

    report_text = build_report(report_stats)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"reviews_cleaning2_report_{args.category_name}.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print()
    print(f"Saved cleaned parquet: {output_path}")
    if debug_output_path:
        print(f"Saved debug parquet: {debug_output_path}")
    print(f"Saved report: {report_path}")

    spark.stop()


if __name__ == "__main__":
    main()
