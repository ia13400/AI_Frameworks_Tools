import json

import torch
import torch.nn.functional as F

from config import (
    ARITHMETIC_EXPERIMENTS,
    FIELD_PAIRS_PATH,
    NEIGHBOR_TARGET_WORDS,
)
from lexicon_utils import (
    build_hu_liu_lookup,
    normalized_word_key,
    prepare_hu_liu_sentiment_words,
    single_token_for_running_text,
)
from model_utils import section
from plotting_utils import (
    plot_embedding_norms_by_field_boxes,
    plot_field_pca_panels,
    plot_global_pca,
    plot_pca_histogram,
    plot_sentiment_projection,
)


def run_positive_negative_comparison(model, tokenizer):
    section(6, "Positiv vs Negativ Vergleich")
    embedding_matrix = model.gpt_neox.embed_in.weight.detach().cpu()
    sentiment_words, positive_words, negative_words, vectors = prepare_hu_liu_sentiment_words(
        tokenizer,
        embedding_matrix,
    )

    pca = plot_global_pca(sentiment_words, positive_words, negative_words, vectors)
    plot_pca_histogram(positive_words, negative_words)

    if FIELD_PAIRS_PATH.exists():
        pairs_by_field = json.loads(FIELD_PAIRS_PATH.read_text(encoding="utf-8"))
        plot_field_pca_panels(pairs_by_field, embedding_matrix, pca)
    else:
        print(f"Skipping field PCA panels: {FIELD_PAIRS_PATH} not found.")

    print("\nExample one-token positive words:")
    print(", ".join(item["word"] for item in positive_words[:5]))
    print("\nExample one-token negative words:")
    print(", ".join(item["word"] for item in negative_words[:5]))

    return {
        "embedding_matrix": embedding_matrix,
        "sentiment_words": sentiment_words,
        "positive_words": positive_words,
        "negative_words": negative_words,
    }


def run_embedding_norm_analysis(embedding_matrix):
    if not FIELD_PAIRS_PATH.exists():
        print(f"Skipping field norm plots: {FIELD_PAIRS_PATH} not found.")
        return

    pairs_by_field = json.loads(FIELD_PAIRS_PATH.read_text(encoding="utf-8"))

    plot_embedding_norms_by_field_boxes(
        pairs_by_field["fields"],
        embedding_matrix,
        "sentiment_embedding_norms_all_fields.png",
    )


def hu_liu_lookup_from_token_text(token_text, hu_liu_lookup):
    normalized = normalized_word_key(token_text)
    if normalized in hu_liu_lookup:
        item = hu_liu_lookup[normalized]
        return f"HuLiu:{item['sentiment']}: {item['word']}"
    return ""


def nearest_neighbors_general_and_hu_liu(
    target_word,
    tokenizer,
    embedding_matrix,
    positive_words,
    negative_words,
    k=5,
):
    token_id = single_token_for_running_text(target_word, tokenizer)
    if token_id is None:
        raise ValueError(f"{target_word!r} is not represented by exactly one token.")

    hu_liu_lookup = build_hu_liu_lookup(positive_words, negative_words)
    target_vector = embedding_matrix[token_id].float()
    normalized_target = F.normalize(target_vector, dim=0)
    normalized_vocab = F.normalize(embedding_matrix.float(), dim=1)
    similarities = torch.mv(normalized_vocab, normalized_target)
    top_scores, top_ids = torch.topk(similarities, k + 1)

    hu_liu_ids = torch.tensor(
        [item["token_id"] for item in positive_words + negative_words],
        dtype=torch.long,
    )
    hu_liu_vectors = embedding_matrix[hu_liu_ids].float()
    hu_liu_scores = torch.mv(F.normalize(hu_liu_vectors, dim=1), normalized_target)
    top_hu_scores, top_hu_positions = torch.topk(hu_liu_scores, k + 1)

    print(f"\nCosine neighbors for {target_word!r}")
    print("-" * 88)
    print("General vocabulary neighbors:")
    shown = 0
    for score, neighbor_id in zip(top_scores, top_ids):
        if int(neighbor_id) == int(token_id):
            continue
        token_text = tokenizer.decode([int(neighbor_id)])
        mark = hu_liu_lookup_from_token_text(token_text, hu_liu_lookup)
        print(f"{token_text!r:<20} token_id={int(neighbor_id):>6} similarity={float(score):+.4f} {mark}")
        shown += 1
        if shown >= k:
            break

    print("\nHu & Liu dictionary neighbors:")
    shown = 0
    all_hu_words = positive_words + negative_words
    for score, position in zip(top_hu_scores, top_hu_positions):
        item = all_hu_words[int(position)]
        if int(item["token_id"]) == int(token_id):
            continue
        print(
            f"{item['word']:<20} sentiment={item['sentiment']:<8} "
            f"token_id={int(item['token_id']):>6} similarity={float(score):+.4f}"
        )
        shown += 1
        if shown >= k:
            break


def single_token_embedding_for_word(word, tokenizer, embedding_matrix):
    token_id = single_token_for_running_text(word, tokenizer)
    if token_id is None:
        raise ValueError(
            f"{word!r} must be represented by exactly one token when encoded as running text."
        )
    return token_id, embedding_matrix[token_id].float()


def embedding_arithmetic_neighbors(expression, tokenizer, embedding_matrix, positive_words, negative_words, top_k=5):
    result_vector = torch.zeros(embedding_matrix.shape[1], dtype=torch.float32)
    terms = expression.split()
    sign = 1

    for term in terms:
        if term == "+":
            sign = 1
            continue
        if term == "-":
            sign = -1
            continue
        _, vector = single_token_embedding_for_word(term, tokenizer, embedding_matrix)
        result_vector += sign * vector
        sign = 1

    normalized_result = F.normalize(result_vector, dim=0)
    normalized_vocab = F.normalize(embedding_matrix.float(), dim=1)
    similarities = torch.mv(normalized_vocab, normalized_result)
    top_scores, top_ids = torch.topk(similarities, top_k)
    hu_liu_lookup = build_hu_liu_lookup(positive_words, negative_words)

    print(f"\nEmbedding arithmetic: {expression}")
    print("-" * 88)
    for score, token_id in zip(top_scores, top_ids):
        token_text = tokenizer.decode([int(token_id)])
        mark = hu_liu_lookup_from_token_text(token_text, hu_liu_lookup)
        print(f"{token_text!r:<20} token_id={int(token_id):>6} similarity={float(score):+.4f} {mark}")


def run_cosine_neighbor_analysis(
    target_words,
    tokenizer,
    embedding_matrix,
    positive_words,
    negative_words,
):
    for target_word in target_words:
        nearest_neighbors_general_and_hu_liu(
            target_word,
            tokenizer,
            embedding_matrix,
            positive_words,
            negative_words,
        )


def run_embedding_arithmetic_analysis(
    arithmetic_experiments,
    tokenizer,
    embedding_matrix,
    positive_words,
    negative_words,
):
    for expression in arithmetic_experiments:
        embedding_arithmetic_neighbors(
            expression,
            tokenizer,
            embedding_matrix,
            positive_words,
            negative_words,
        )


def build_projection_rows(fields, embedding_matrix, sentiment_direction):
    rows = []
    for field_index, field_data in enumerate(fields):
        for pair in field_data["pairs"]:
            for sentiment in ["positive", "negative"]:
                item = pair[sentiment]
                vector = embedding_matrix[item["token_id"]]
                projection_score = torch.dot(vector, sentiment_direction).item()
                rows.append(
                    {
                        "field": field_data["field"],
                        "field_label": field_data["field_label"],
                        "field_index": field_index,
                        "pair_id": pair["id"],
                        "word": item["word"],
                        "token": item["token"],
                        "token_id": item["token_id"],
                        "sentiment": item["sentiment"],
                        "projection_score": float(projection_score),
                    }
                )
    return rows


def run_sentiment_direction_projection(tokenizer, embedding_matrix):
    if not FIELD_PAIRS_PATH.exists():
        print(f"Skipping sentiment direction plot: {FIELD_PAIRS_PATH} not found.")
        return

    good_id, good_vector = single_token_embedding_for_word("good", tokenizer, embedding_matrix)
    bad_id, bad_vector = single_token_embedding_for_word("bad", tokenizer, embedding_matrix)
    sentiment_direction = F.normalize(good_vector - bad_vector, dim=0)
    sentiment_pairs_by_field = json.loads(FIELD_PAIRS_PATH.read_text(encoding="utf-8"))

    print("\nProjection onto sentiment direction: embedding('good') - embedding('bad')")
    print(f"good token_id={good_id}, bad token_id={bad_id}")

    all_fields = sentiment_pairs_by_field["fields"]
    all_rows = build_projection_rows(all_fields, embedding_matrix, sentiment_direction)
    print("-" * 96)

    plot_sentiment_projection(
        all_rows,
        [field["field_label"] for field in all_fields],
        "sentiment_direction_projection_all_fields.png",
        "All Hu & Liu word-pair projections onto embedding('good') - embedding('bad')",
        (15, 16),
    )


def run_token_embedding_visualization(tokenizer, sentiment_state):
    section(7, "Token-Embeddings visualisieren")
    embedding_matrix = sentiment_state["embedding_matrix"]
    positive_words = sentiment_state["positive_words"]
    negative_words = sentiment_state["negative_words"]

    run_embedding_norm_analysis(embedding_matrix)
    run_cosine_neighbor_analysis(
        NEIGHBOR_TARGET_WORDS,
        tokenizer,
        embedding_matrix,
        positive_words,
        negative_words,
    )
    run_embedding_arithmetic_analysis(
        ARITHMETIC_EXPERIMENTS,
        tokenizer,
        embedding_matrix,
        positive_words,
        negative_words,
    )
    run_sentiment_direction_projection(tokenizer, embedding_matrix)
