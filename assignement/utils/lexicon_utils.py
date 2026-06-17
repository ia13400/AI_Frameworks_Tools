import json
import re
from pathlib import Path

import nltk
import torch

from config import (
    INPUT_DIR,
    MODEL_NAME,
    NEGATIVE_WORDS_PATH,
    POSITIVE_WORDS_PATH,
    SENTIMENT_CHALLENGE_ANNOTATED_PATH,
    SENTIMENT_CHALLENGE_TOP_TOKENS_PATH,
    SENTIMENT_CHALLENGE_TOP_TOKENS_REPORT_PATH,
)
from plotting_utils import (
    plot_filtered_sentiment_challenge_category_heatmap,
    plot_sentiment_challenge_category_accuracy,
    plot_sentiment_challenge_category_confusion_matrices,
    plot_sentiment_challenge_category_heatmap,
)


NLTK_DATA_DIR = INPUT_DIR / "nltk_data"
OPINION_LEXICON_DIR = NLTK_DATA_DIR / "corpora" / "opinion_lexicon"
SENTIMENT_CHALLENGE_LABELS = {
    0: "very negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very positive",
}
NEGATIVE_SENTIMENT_LABELS = {0, 1}
POSITIVE_SENTIMENT_LABELS = {3, 4}
BINARY_SENTIMENT_LABELS = {
    "negative": "negative",
    "positive": "positive",
}


def read_hu_liu_word_file(filename: str):
    """Read one Hu & Liu word-list file and ignore comments/blank lines."""
    path = OPINION_LEXICON_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Hu & Liu word file not found: {path}")

    words = []
    for line in path.read_text(encoding="ISO-8859-1").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        words.append(line)
    return words


def import_hu_liu_dataset():
    """Ensure the Hu & Liu Opinion Lexicon is available locally and load it."""
    nltk_data_path = str(NLTK_DATA_DIR.resolve())
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)

    positive_path = OPINION_LEXICON_DIR / "positive-words.txt"
    negative_path = OPINION_LEXICON_DIR / "negative-words.txt"

    if not positive_path.exists() or not negative_path.exists():
        NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Hu & Liu Opinion Lexicon to: {NLTK_DATA_DIR}")
        nltk.download("opinion_lexicon", download_dir=nltk_data_path, quiet=True)
    else:
        print(f"Hu & Liu Opinion Lexicon already available in: {NLTK_DATA_DIR}")

    positive = read_hu_liu_word_file("positive-words.txt")
    negative = read_hu_liu_word_file("negative-words.txt")

    print(f"Positive lexicon words: {len(positive)}")
    print(f"Negative lexicon words: {len(negative)}")
    return positive, negative


def import_sentiment_challenge_dataset(
    dataset_path: str | Path | None = None,
    verbose: bool = True,
):
    """Load the Barnes et al. challenge dataset with its error annotations."""
    path = (
        Path(dataset_path)
        if dataset_path is not None
        else SENTIMENT_CHALLENGE_ANNOTATED_PATH
    )
    if not path.exists():
        raise FileNotFoundError(f"Sentiment challenge dataset not found: {path}")

    records = []
    category_to_records = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) != 6:
            raise ValueError(
                f"Expected 6 tab-separated fields in {path} line {line_number}, "
                f"but found {len(parts)}."
            )

        sentence_index, source_dataset, source_index, gold_label, text, annotations = parts
        annotation_labels = [
            annotation.strip()
            for annotation in annotations.split("::")
            if annotation.strip()
        ]
        gold_label_id = int(gold_label)
        record = {
            "sentence_index": int(sentence_index),
            "source_dataset": source_dataset,
            "source_index": int(source_index),
            "gold_label": gold_label_id,
            "gold_label_name": SENTIMENT_CHALLENGE_LABELS.get(
                gold_label_id,
                f"label {gold_label_id}",
            ),
            "text": text,
            "annotations": annotation_labels,
        }
        records.append(record)

        for annotation in annotation_labels:
            category_to_records.setdefault(annotation, []).append(record)

    if verbose:
        print(f"Loaded sentiment challenge examples: {len(records)}")
        print(f"Annotation categories: {len(category_to_records)}")
        print(f"Dataset file: {path}")
    return records, category_to_records


def records_by_challenge_category(records):
    """Group challenge records by each annotation category."""
    category_to_records = {}
    for record in records:
        for annotation in record["annotations"]:
            category_to_records.setdefault(annotation, []).append(record)
    return category_to_records


def binary_sentiment_label(gold_label: int):
    """Map challenge sentiment scores to binary sentiment labels."""
    if gold_label in NEGATIVE_SENTIMENT_LABELS:
        return "negative"
    if gold_label in POSITIVE_SENTIMENT_LABELS:
        return "positive"
    return None


def filter_sentiment_challenge_dataset(dataset_path: str | Path | None = None):
    """Remove neutral rows, convert labels to binary sentiment, and plot a heatmap."""
    records, _ = import_sentiment_challenge_dataset(dataset_path, verbose=False)
    filtered_records = []

    for record in records:
        sentiment = binary_sentiment_label(record["gold_label"])
        if sentiment is None:
            continue

        filtered_records.append(
            {
                "sentence_index": record["sentence_index"],
                "source_dataset": record["source_dataset"],
                "source_index": record["source_index"],
                "sentiment": sentiment,
                "text": record["text"],
                "annotations": record["annotations"],
            }
        )

    filtered_category_to_records = records_by_challenge_category(filtered_records)
    plot_filtered_sentiment_challenge_category_heatmap(filtered_category_to_records)
    return filtered_records, filtered_category_to_records


def import_data_set(dataset_path: str | Path | None = None):
    """Import the original challenge dataset and save its category heatmap PNG."""
    records, category_to_records = import_sentiment_challenge_dataset(
        dataset_path,
        verbose=False,
    )
    plot_sentiment_challenge_category_heatmap(
        category_to_records,
        SENTIMENT_CHALLENGE_LABELS,
    )
    return records, category_to_records


def filter_data_set(dataset_path: str | Path | None = None):
    """Import, filter to binary sentiment labels, and save the filtered heatmap PNG."""
    return filter_sentiment_challenge_dataset(dataset_path)


def form_sentiment_challenge_prompts(filtered_records, sentiment_suffix: str):
    """Create model prompts from filtered challenge sentences."""
    return [
        {
            "sentence_index": record["sentence_index"],
            "true_sentiment": record["sentiment"],
            "annotations": record["annotations"],
            "prompt": record["text"] + sentiment_suffix,
        }
        for record in filtered_records
    ]


def classify_top_tokens_by_hu_liu(top_tokens, hu_liu_lookup):
    """Classify top-k next tokens by summed Hu & Liu sentiment probability."""
    sentiment_scores = {"positive": 0.0, "negative": 0.0}
    matched_tokens = []

    for token in top_tokens:
        key = normalized_word_key(token["token"])
        if key not in hu_liu_lookup:
            continue

        sentiment = hu_liu_lookup[key]["sentiment"]
        sentiment_scores[sentiment] += token["probability"]
        matched_tokens.append(
            {
                "token": token["token"],
                "probability": token["probability"],
                "hu_liu_word": hu_liu_lookup[key]["word"],
                "sentiment": sentiment,
            }
        )

    if not matched_tokens:
        predicted_sentiment = "no sentiment"
    elif sentiment_scores["positive"] > sentiment_scores["negative"]:
        predicted_sentiment = "positive"
    elif sentiment_scores["negative"] > sentiment_scores["positive"]:
        predicted_sentiment = "negative"
    else:
        predicted_sentiment = matched_tokens[0]["sentiment"]

    return predicted_sentiment, sentiment_scores, matched_tokens


def write_top_token_report(results):
    """Write a readable prompt/token/Hu-Liu-hit report for inspection."""
    lines = [
        "Sentiment Challenge top-10 next-token report",
        "=" * 52,
        (
            "Prediction rule: sum the probability mass of top-10 tokens that "
            "match the Hu & Liu positive or negative lexicon. If neither side "
            "has more mass, label the result as no sentiment."
        ),
        "",
    ]

    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                f"Example {index}",
                "-" * 52,
                f"sentence_index: {result['sentence_index']}",
                f"true_sentiment: {result['true_sentiment']}",
                f"predicted_sentiment: {result['predicted_sentiment']}",
                f"annotations: {', '.join(result['annotations'])}",
                f"prompt: {result['prompt']}",
                "",
                "top_10_next_tokens:",
            ]
        )

        for token in result["top_tokens"]:
            lines.append(
                f"  {token['rank']:>2}. {token['token']!r:<18} "
                f"token_id={token['token_id']:<6} "
                f"probability={token['probability']:.6f}"
            )

        lines.append("")
        lines.append("hu_liu_hits:")
        if result["matched_hu_liu_tokens"]:
            for match in result["matched_hu_liu_tokens"]:
                lines.append(
                    f"  {match['token']!r:<18} "
                    f"word={match['hu_liu_word']:<18} "
                    f"sentiment={match['sentiment']:<8} "
                    f"probability={match['probability']:.6f}"
                )
        else:
            lines.append("  none")

        scores = result["hu_liu_probability_mass"]
        lines.extend(
            [
                "",
                "hu_liu_probability_mass:",
                f"  negative={scores['negative']:.6f}",
                f"  positive={scores['positive']:.6f}",
                "",
            ]
        )

    SENTIMENT_CHALLENGE_TOP_TOKENS_REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def top_k_next_tokens_for_prompts(
    model,
    tokenizer,
    device,
    prompt_records,
    positive_words,
    negative_words,
    top_k: int = 10,
    batch_size: int = 8,
):
    """Run next-token inference and classify top-k tokens through Hu & Liu."""
    hu_liu_lookup = build_hu_liu_lookup(positive_words, negative_words)
    results = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    for start in range(0, len(prompt_records), batch_size):
        batch_records = prompt_records[start : start + batch_size]
        prompts = [record["prompt"] for record in batch_records]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        attention_lengths = inputs["attention_mask"].sum(dim=1) - 1
        for batch_index, record in enumerate(batch_records):
            final_token_index = int(attention_lengths[batch_index].item())
            logits = outputs.logits[batch_index, final_token_index, :]
            probabilities = torch.softmax(logits, dim=-1)
            top_probabilities, top_indices = torch.topk(probabilities, top_k)
            top_tokens = []

            for rank, (token_id, probability) in enumerate(
                zip(top_indices, top_probabilities),
                start=1,
            ):
                token_text = tokenizer.decode([int(token_id)])
                top_tokens.append(
                    {
                        "rank": rank,
                        "token": token_text,
                        "token_id": int(token_id),
                        "probability": float(probability),
                    }
                )

            predicted, scores, matches = classify_top_tokens_by_hu_liu(
                top_tokens,
                hu_liu_lookup,
            )
            results.append(
                {
                    **record,
                    "predicted_sentiment": predicted,
                    "hu_liu_probability_mass": scores,
                    "matched_hu_liu_tokens": matches,
                    "top_tokens": top_tokens,
                }
            )

    SENTIMENT_CHALLENGE_TOP_TOKENS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_top_token_report(results)
    print(
        "Top-token prompt details were saved to:\n"
        f"- {SENTIMENT_CHALLENGE_TOP_TOKENS_REPORT_PATH}\n"
        f"- {SENTIMENT_CHALLENGE_TOP_TOKENS_PATH}"
    )
    return results


def build_category_confusion_counts(prediction_results):
    """Count true-vs-predicted sentiment for every annotation category."""
    categories = sorted(
        {category for result in prediction_results for category in result["annotations"]}
    )
    true_labels = ["negative", "positive"]
    predicted_labels = ["negative", "positive", "no sentiment"]
    counts = {
        category: [
            [0 for _ in predicted_labels]
            for _ in true_labels
        ]
        for category in categories
    }
    true_index = {label: index for index, label in enumerate(true_labels)}
    predicted_index = {label: index for index, label in enumerate(predicted_labels)}

    for result in prediction_results:
        for category in result["annotations"]:
            counts[category][true_index[result["true_sentiment"]]][
                predicted_index[result["predicted_sentiment"]]
            ] += 1

    return counts


def build_category_accuracy_rows(prediction_results):
    """Compute binary sentiment accuracy for each annotation category."""
    categories = sorted(
        {category for result in prediction_results for category in result["annotations"]}
    )
    rows = []

    for category in categories:
        category_results = [
            result
            for result in prediction_results
            if category in result["annotations"]
        ]
        correct = sum(
            result["true_sentiment"] == result["predicted_sentiment"]
            for result in category_results
        )
        total = len(category_results)
        rows.append(
            {
                "category": category,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        )

    return rows


def run_sentiment_challenge_top_token_analysis(
    model,
    tokenizer,
    device,
    prompt_records,
    positive_words,
    negative_words,
    top_k: int = 10,
    batch_size: int = 8,
):
    """Save top-k token predictions and plot category confusion matrices."""
    prediction_results = top_k_next_tokens_for_prompts(
        model,
        tokenizer,
        device,
        prompt_records,
        positive_words,
        negative_words,
        top_k=top_k,
        batch_size=batch_size,
    )
    category_confusion_counts = build_category_confusion_counts(prediction_results)
    category_accuracy_rows = build_category_accuracy_rows(prediction_results)
    plot_sentiment_challenge_category_confusion_matrices(category_confusion_counts)
    plot_sentiment_challenge_category_accuracy(category_accuracy_rows)
    return prediction_results, category_confusion_counts, category_accuracy_rows


def format_challenge_sentence(record, missing_text: str, text_width: int = 140):
    """Return a compact sentence string for notebook printing."""
    if record is None:
        return missing_text

    text = record["text"]
    if len(text) > text_width:
        text = text[: text_width - 3].rstrip() + "..."
    return text


def print_sentiment_challenge_examples(category_to_records, categories=None):
    """Print one positive and one negative sentence for each annotation category."""
    if categories is None:
        categories = sorted(
            category_to_records,
            key=lambda key: (-len(category_to_records[key]), key),
        )

    for category in categories:
        records = category_to_records.get(category, [])
        if not records:
            print(f"\n{category}: no examples found")
            continue

        positive = next(
            (
                record
                for record in records
                if record["gold_label"] in POSITIVE_SENTIMENT_LABELS
            ),
            None,
        )
        negative = next(
            (
                record
                for record in records
                if record["gold_label"] in NEGATIVE_SENTIMENT_LABELS
            ),
            None,
        )

        print(f"\n{category}")
        print(
            "Positive, "
            f"{format_challenge_sentence(positive, 'No positive example found')}"
        )
        print(
            "Negative, "
            f"{format_challenge_sentence(negative, 'No negative example found')}"
        )


def run_sentiment_challenge_inspection(dataset_path: str | Path | None = None):
    """Load the challenge dataset, plot its category heatmap, and print examples."""
    records, category_to_records = import_sentiment_challenge_dataset(
        dataset_path,
        verbose=False,
    )
    plot_sentiment_challenge_category_heatmap(
        category_to_records,
        SENTIMENT_CHALLENGE_LABELS,
    )
    print_sentiment_challenge_examples(category_to_records)
    return records, category_to_records


def load_hu_liu_opinion_lexicon():
    """Load the Hu & Liu Opinion Lexicon through NLTK."""
    return import_hu_liu_dataset()


def is_plain_word(word: str) -> bool:
    """Keep only alphabetic words so punctuation and phrases do not enter the analysis."""
    return re.fullmatch(r"[A-Za-z]+", word) is not None


def single_token_for_running_text(word: str, tokenizer):
    """Return the token id if the word is represented by one token in running text."""
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(token_ids) != 1:
        return None
    return token_ids[0]


def build_sentiment_word_records(tokenizer):
    """Create Hu & Liu word records that are represented by exactly one model token."""
    positive_lexicon, negative_lexicon = load_hu_liu_opinion_lexicon()
    sentiment_words = []

    for sentiment, lexicon_words in [
        ("positive", positive_lexicon),
        ("negative", negative_lexicon),
    ]:
        for word in lexicon_words:
            word = word.lower()
            if not is_plain_word(word):
                continue

            token_id = single_token_for_running_text(word, tokenizer)
            if token_id is None:
                continue

            sentiment_words.append(
                {
                    "word": word,
                    "sentiment": sentiment,
                    "token_id": int(token_id),
                    "token": tokenizer.decode([token_id]),
                }
            )

    if len(sentiment_words) < 2:
        raise ValueError("Not enough one-token Hu & Liu opinion words found.")

    return sentiment_words


def export_sentiment_words_json(items, sentiment: str, output_path: Path) -> None:
    """Write one-token sentiment words together with source and filtering metadata."""
    payload = {
        "source": {
            "name": "Hu & Liu Opinion Lexicon",
            "nltk_corpus": "nltk.corpus.opinion_lexicon",
            "description": (
                "Positive and negative opinion word lists introduced by Minqing Hu "
                "and Bing Liu for opinion mining / sentiment analysis."
            ),
        },
        "filtering_method": {
            "plain_word_filter": "Keep only entries matching the regex [A-Za-z]+.",
            "token_filter": (
                "Tokenize each word as running text by prepending one leading space, "
                "then keep only words where tokenizer.encode(' ' + word, "
                "add_special_tokens=False) returns exactly one token id."
            ),
            "model_tokenizer": MODEL_NAME,
            "sentiment_label": sentiment,
        },
        "count": len(items),
        "words": [
            {
                "word": item["word"],
                "token": item["token"],
                "token_id": int(item["token_id"]),
                "sentiment": item["sentiment"],
            }
            for item in sorted(items, key=lambda item: item["word"])
        ],
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare_hu_liu_sentiment_words(tokenizer, embedding_matrix):
    """Build sentiment records, export JSON files, and return vectors for plotting."""
    sentiment_words = build_sentiment_word_records(tokenizer)
    positive_words = [item for item in sentiment_words if item["sentiment"] == "positive"]
    negative_words = [item for item in sentiment_words if item["sentiment"] == "negative"]

    export_sentiment_words_json(positive_words, "positive", POSITIVE_WORDS_PATH)
    export_sentiment_words_json(negative_words, "negative", NEGATIVE_WORDS_PATH)

    token_ids = torch.tensor([item["token_id"] for item in sentiment_words], dtype=torch.long)
    vectors = embedding_matrix[token_ids].float().numpy()

    print(f"\nSaved JSON: {POSITIVE_WORDS_PATH}")
    print(f"Saved JSON: {NEGATIVE_WORDS_PATH}")
    print(f"\nHu & Liu words kept:       {len(sentiment_words)}")
    print(f"Positive one-token words:  {len(positive_words)}")
    print(f"Negative one-token words:  {len(negative_words)}")

    return sentiment_words, positive_words, negative_words, vectors


def normalized_word_key(word: str) -> str:
    """Normalize only for lookup; the original word is kept for printing."""
    return "".join(word.split()).lower()


def build_hu_liu_lookup(positive_words, negative_words):
    lookup = {}
    for item in positive_words + negative_words:
        lookup[normalized_word_key(item["word"])] = {
            "word": item["word"],
            "sentiment": item["sentiment"],
        }
    return lookup
