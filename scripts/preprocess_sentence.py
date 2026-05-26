from __future__ import annotations

import re
from collections.abc import Iterable
from typing import List, Union

# =========================
# Constants
# =========================

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

LINKING_VERBS = {
    "be", "become", "feel", "look", "remain", "seem", "smell", "sound", "stay", "taste"
}

SOCIAL_PATTERNS = [
    re.compile(r"^(thank|thanks|thank you)( so much)?$"),
    re.compile(r"^(thanks again)$"),
    re.compile(r"^(have a nice day)$"),
    re.compile(r"^(best regards|kind regards|warm regards)$"),
    re.compile(r"^(happy holidays|merry christmas)$"),
]

PRONOUNS = {
    "i", "it", "this", "that", "these", "those", "they", "he", "she", "them", "him", "her"
}

_NLP_CACHE = {}


# =========================
# Normalize helpers
# =========================

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
    if model_name not in _NLP_CACHE:
        import spacy
        _NLP_CACHE[model_name] = spacy.load(model_name, disable=["ner"])
    return _NLP_CACHE[model_name]


# =========================
# spaCy rule helpers
# =========================

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
    pronoun_target = False
    noun_target = False

    for tok in doc:
        if tok.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj", "attr"}:
            if tok.pos_ in {"NOUN", "PROPN"}:
                noun_target = True
            elif tok.pos_ == "PRON" or tok.text.lower() in PRONOUNS:
                pronoun_target = True

    return pronoun_target and not noun_target


def extract_aspect_targets(doc):
    targets = []

    for tok in doc:
        if (
            tok.dep_ in {"nsubj", "nsubjpass", "dobj", "obj", "pobj", "attr", "conj"}
            and tok.pos_ in {"NOUN", "PROPN", "PRON"}
        ):
            targets.append(tok)

    return targets


def is_opinion_predicate(token) -> bool:
    lemma = token.lemma_.lower()

    if token.pos_ == "ADJ":
        return True

    if token.pos_ == "VERB":
        if lemma in OPINION_VERBS or lemma in LINKING_VERBS:
            return True

        if any(
            child.dep_ in {"acomp", "attr", "oprd", "xcomp"} and child.pos_ == "ADJ"
            for child in token.children
        ):
            return True

    if token.pos_ == "AUX" and lemma in LINKING_VERBS:
        return True

    return False


# =========================
# Dependency patterns
# =========================

def pattern_amod(doc) -> bool:
    return any(
        tok.dep_ == "amod" and tok.pos_ == "ADJ" and is_aspect_noun(tok.head)
        for tok in doc
    )


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
            has_nsubj = any(
                child.dep_ in {"nsubj", "nsubjpass"} and is_aspect_noun(child)
                for child in tok.children
            )

            if has_cop and has_nsubj:
                return True

            if tok.head.pos_ in {"VERB", "AUX"} and tok.head.lemma_.lower() in LINKING_VERBS:
                if any(
                    child.dep_ in {"nsubj", "nsubjpass"} and is_aspect_noun(child)
                    for child in tok.head.children
                ):
                    return True

    return False


def pattern_conj_opinion(doc) -> bool:
    return any(
        tok.dep_ == "conj" and tok.pos_ == "ADJ" and tok.head.pos_ == "ADJ"
        for tok in doc
    )


def pattern_conj_aspect(doc) -> bool:
    return any(
        tok.dep_ == "conj" and is_aspect_noun(tok) and is_aspect_noun(tok.head)
        for tok in doc
    )


def pattern_neg_opinion(doc) -> bool:
    for tok in doc:
        if tok.dep_ == "neg":
            head = tok.head

            if (
                head.pos_ == "ADJ"
                or head.lemma_.lower() in OPINION_VERBS
                or head.lemma_.lower() in LINKING_VERBS
            ):
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
    pattern_amod,
    pattern_nsubj_opinion_predicate,
    pattern_obj_opinion_verb,
    pattern_copula_adj_noun_subject,
    pattern_conj_opinion,
    pattern_conj_aspect,
    pattern_neg_opinion,
    pattern_obj_xcomp,
]


# =========================
# Core preprocessing
# =========================

def should_keep_sentence(doc, normalized_text: str) -> bool:
    tokens = simple_tokens(normalized_text)
    char_count = len(normalized_text)
    token_count = len(tokens)

    flag_too_short = char_count < 5 or token_count < 2
    flag_social = any(pattern.fullmatch(normalized_text) for pattern in SOCIAL_PATTERNS)
    flag_no_noun_or_np = not has_noun_or_noun_phrase(doc)
    flag_adj_only = is_adj_only_or_phrase(doc)
    flag_pronoun_target_only = pronoun_target_only(doc)

    targets = extract_aspect_targets(doc)
    noun_targets = [tok for tok in targets if tok.pos_ in {"NOUN", "PROPN"}]

    flag_all_targets_generic_or_noise = bool(noun_targets) and all(
        token_key(tok) in GENERIC_NOUNS or token_key(tok) in DOMAIN_NOISE
        for tok in noun_targets
    )

    flag_no_dependency_pattern = not any(pattern(doc) for pattern in PATTERNS)

    reject_flags = [
        flag_too_short,
        flag_social,
        flag_no_noun_or_np,
        flag_adj_only,
        flag_pronoun_target_only,
        flag_all_targets_generic_or_noise,
        flag_no_dependency_pattern,
    ]

    return not any(reject_flags)


def build_final_text(doc) -> str:
    """
    Không thêm marker [GENERIC_NOUN] / [DOMAIN_NOISE].
    Chỉ trả về câu đã normalize/tokenize lại theo spaCy.
    """
    tokens = []

    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue

        text = tok.text.lower().strip()

        if text:
            tokens.append(text)

    return MULTISPACE_RE.sub(" ", " ".join(tokens)).strip()


def preprocess_sentences(
    sentences: Union[str, Iterable[object]],
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 256,
) -> List[str]:
    """
    Nhận 1 câu hoặc batch câu.
    Luôn trả về list[str] có cùng số lượng phần tử với input.

    - Câu được giữ lại: trả về câu đã tiền xử lý.
    - Câu bị loại bỏ: trả về "".
    """

    if isinstance(sentences, str) or sentences is None:
        sentence_list = [sentences]
    else:
        sentence_list = list(sentences)

    normalized_sentences = [normalize_text(sentence) for sentence in sentence_list]

    nlp = get_nlp(spacy_model)
    docs = nlp.pipe(normalized_sentences, batch_size=batch_size)

    results = []

    for doc, normalized_text in zip(docs, normalized_sentences):
        if not normalized_text:
            results.append("")
            continue

        if should_keep_sentence(doc, normalized_text):
            results.append(build_final_text(doc))
        else:
            results.append("")

    return results


if __name__ == "__main__":
    test_one = "The battery life is excellent!"
    print("Single sentence:")
    print(preprocess_sentences(test_one))

    test_batch = [
        "The battery life is excellent!",
        "Thanks again",
        "It is great",
        "Bad",
        "I love the screen quality.",
        "The package arrived late.",
        "The sound and display are amazing.",
        "I don't like the keyboard.",
    ]

    print("\nBatch:")
    for original, processed in zip(test_batch, preprocess_sentences(test_batch)):
        print(f"{original!r} -> {processed!r}")