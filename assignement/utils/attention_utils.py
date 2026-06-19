import numpy as np
import torch

from config import (
    ATTENTION_CAD_SENTIMENT_HEATMAP_PATH,
    ATTENTION_HEAD_GRID_PATH,
    ATTENTION_INDUCTION_HEATMAP_PATH,
    ATTENTION_SUBJECT_HEATMAP_PATH,
    CAD_SENTIMENT_TRAIN_PAIRED_PATH,
)
from lexicon_utils import normalized_word_key, prepare_hu_liu_lookup_state
from logit_lens_utils import load_cad_sentiment_prompt_pairs
from plotting_utils import plot_attention_head_grid, plot_layer_head_heatmap


def run_attention_forward_pass(model, tokenizer, device, prompt):
    """Tokenize one prompt, run the model with attentions, and print tensor shapes."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tokens = [tokenizer.decode([token_id]) for token_id in inputs["input_ids"][0]]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    print(f"Prompt tokens ({len(tokens)}):")
    for index, token in enumerate(tokens):
        print(f"{index:>2}: {token!r}")

    print("\nAttention tensors:")
    print(f"Number of attention matrices: {len(attentions)}")
    print(f"outputs.attentions is a tuple with {len(attentions)} entries.")
    print("One tuple entry is returned per transformer layer.")
    print(f"Single head attention matrix size: {tuple(attentions[0][0, 0].shape)}")
    for layer_index, attention in enumerate(attentions):
        print(f"Layer {layer_index}: {tuple(attention.shape)}")

    return {
        "inputs": inputs,
        "tokens": tokens,
        "attentions": attentions,
    }


def plot_selected_attention_heads(
    attentions,
    tokens,
    prompt,
    head_specs=None,
    filename_or_path=ATTENTION_HEAD_GRID_PATH,
):
    """Plot a grid of representative attention heads and save it as a PNG."""
    if head_specs is None:
        max_layer = len(attentions) - 1
        max_head = attentions[0].shape[1] - 1
        requested_specs = [(0, 0), (5, 1), (10, 2), (15, 3), (20, 4), (23, 5)]
        head_specs = [
            (min(layer, max_layer), min(head, max_head))
            for layer, head in requested_specs
        ]

    return plot_attention_head_grid(
        attentions,
        tokens,
        head_specs,
        prompt,
        filename_or_path=filename_or_path,
    )


def find_token_position(tokens, target_text):
    """Find a token by matching either exact or stripped tokenizer text."""
    matches = [
        index
        for index, token in enumerate(tokens)
        if token == target_text or token.strip() == target_text
    ]
    if not matches:
        raise ValueError(f"Could not find token {target_text!r} in tokens: {tokens}")
    return matches[0]


def attention_to_source_token_matrix(attentions, source_position, query_position=-1):
    """Return layer/head attention from one query position to one source token."""
    layer_count = len(attentions)
    head_count = attentions[0].shape[1]
    attention_values = np.zeros((layer_count, head_count), dtype=float)

    for layer_index, layer_attention in enumerate(attentions):
        for head_index in range(head_count):
            attention_values[layer_index, head_index] = float(
                layer_attention[0, head_index, query_position, source_position]
            )

    return attention_values


def sentiment_token_positions(tokens, hu_liu_lookup):
    """Find decoded prompt tokens that match the filtered Hu & Liu lookup."""
    positions = []
    for index, token in enumerate(tokens):
        key = normalized_word_key(token)
        if key not in hu_liu_lookup:
            continue
        positions.append(
            {
                "position": index,
                "token": token,
                "word": hu_liu_lookup[key]["word"],
                "sentiment": hu_liu_lookup[key]["sentiment"],
            }
        )
    return positions


def attention_to_source_positions_matrix(attentions, source_positions, query_position=-1):
    """Average layer/head attention from one query token to several source positions."""
    source_matrices = [
        attention_to_source_token_matrix(
            attentions,
            source_position,
            query_position=query_position,
        )
        for source_position in source_positions
    ]
    return np.mean(source_matrices, axis=0)


def cad_prompt_records_from_pairs(prompt_pairs):
    """Flatten CAD positive/negative prompt pairs into prompt-level records."""
    records = []
    for pair in prompt_pairs:
        records.append(
            {
                "id": f"{pair['id']}_positive",
                "pair_id": pair["id"],
                "sentiment": "positive",
                "prompt": pair["positive"],
            }
        )
        records.append(
            {
                "id": f"{pair['id']}_negative",
                "pair_id": pair["id"],
                "sentiment": "negative",
                "prompt": pair["negative"],
            }
        )
    return records


def rank_layer_heads(values, top_k=10):
    """Rank layer/head cells by descending score."""
    values = np.asarray(values, dtype=float)
    flat_indices = np.argsort(values.ravel())[::-1][:top_k]
    ranked_heads = []
    for rank, flat_index in enumerate(flat_indices, start=1):
        layer, head = np.unravel_index(flat_index, values.shape)
        ranked_heads.append(
            {
                "rank": rank,
                "layer": int(layer),
                "head": int(head),
                "score": float(values[layer, head]),
            }
        )
    return ranked_heads


def print_ranked_heads(ranked_heads, score_label="Score"):
    """Print a compact ranking table for attention heads."""
    print(f"{'Rank':<6} {'Layer':<7} {'Head':<6} {score_label}")
    print("-" * 38)
    for item in ranked_heads:
        print(
            f"{item['rank']:<6} "
            f"{item['layer']:<7} "
            f"{item['head']:<6} "
            f"{item['score']:.6f}"
        )


def analyze_subject_attention(
    attentions,
    tokens,
    prompt=None,
    subject_token="France",
    top_k=10,
    filename_or_path=ATTENTION_SUBJECT_HEATMAP_PATH,
):
    """Measure how much the final token attends to a subject token across heads."""
    subject_position = find_token_position(tokens, subject_token)
    attention_values = attention_to_source_token_matrix(attentions, subject_position)
    ranked_heads = rank_layer_heads(attention_values, top_k=top_k)
    score_label = f"Attention weight to {subject_token!r}"

    print(f"Subject token {subject_token!r} found at position {subject_position}.")
    print_ranked_heads(ranked_heads, score_label=score_label)

    plot_layer_head_heatmap(
        attention_values,
        f"Final-token attention to {subject_token!r}",
        "Attention weight",
        filename_or_path,
        top_heads=ranked_heads[:top_k],
        cmap="Blues",
        prompt_text=prompt,
    )
    return {
        "subject_position": subject_position,
        "attention_values": attention_values,
        "ranked_heads": ranked_heads,
    }


def analyze_cad_hu_liu_attention_heads(
    model,
    tokenizer,
    device,
    dataset_path=CAD_SENTIMENT_TRAIN_PAIRED_PATH,
    max_prompts=None,
    top_k=10,
    filename_or_path=ATTENTION_CAD_SENTIMENT_HEATMAP_PATH,
):
    """Average final-token attention to Hu & Liu sentiment tokens across CAD prompts."""
    sentiment_state = prepare_hu_liu_lookup_state(tokenizer)
    prompt_pairs = load_cad_sentiment_prompt_pairs(dataset_path, verbose=False)
    prompt_records = cad_prompt_records_from_pairs(prompt_pairs)
    if max_prompts is not None:
        prompt_records = prompt_records[:max_prompts]

    if not prompt_records:
        raise ValueError("No CAD prompts were available for attention analysis.")

    prompt_attention_matrices = []
    prompt_results = []
    skipped_without_sentiment_tokens = 0

    model.eval()
    for index, record in enumerate(prompt_records, start=1):
        print(f"\rCAD prompt {index}/{len(prompt_records)} handled", end="", flush=True)
        inputs = tokenizer(
            record["prompt"],
            return_tensors="pt",
            truncation=True,
        ).to(device)
        tokens = [
            tokenizer.decode([token_id])
            for token_id in inputs["input_ids"][0]
        ]
        matched_tokens = sentiment_token_positions(
            tokens,
            sentiment_state["hu_liu_lookup"],
        )
        if not matched_tokens:
            skipped_without_sentiment_tokens += 1
            continue

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attention_values = attention_to_source_positions_matrix(
            outputs.attentions,
            [item["position"] for item in matched_tokens],
        )
        prompt_attention_matrices.append(attention_values)
        prompt_results.append(
            {
                **record,
                "matched_hu_liu_tokens": matched_tokens,
                "attention_values": attention_values,
            }
        )

    print()
    if not prompt_attention_matrices:
        raise ValueError("No CAD prompts contained Hu & Liu sentiment tokens after tokenization.")

    stacked_attention = np.stack(prompt_attention_matrices, axis=0)
    mean_attention = stacked_attention.mean(axis=0)
    std_attention = stacked_attention.std(axis=0)
    ranked_heads = rank_layer_heads(mean_attention, top_k=top_k)
    for item in ranked_heads:
        item["std"] = float(std_attention[item["layer"], item["head"]])

    print(
        "CAD Hu & Liu attention prompts used:",
        len(prompt_results),
        f"(skipped without Hu & Liu token: {skipped_without_sentiment_tokens})",
    )
    print_ranked_heads(
        ranked_heads,
        score_label="Mean attention weight",
    )

    plot_layer_head_heatmap(
        mean_attention,
        (
            "CAD Dataset: Average Final-Token Attention to "
            f"Hu & Liu Sentiment Tokens (n={len(prompt_results)} prompts)"
        ),
        "Average attention weight",
        filename_or_path,
        top_heads=ranked_heads[:top_k],
        cmap="Blues",
    )
    return {
        "mean_attention": mean_attention,
        "std_attention": std_attention,
        "ranked_heads": ranked_heads,
        "prompt_results": prompt_results,
        "prompt_count": len(prompt_results),
        "skipped_without_sentiment_tokens": skipped_without_sentiment_tokens,
    }


def induction_score(attention_matrix, offsets=(1, 2)):
    """Measure attention concentrated on previous-token diagonals."""
    matrix = np.asarray(attention_matrix, dtype=float)
    diagonal_scores = []
    for offset in offsets:
        diagonal = np.diag(matrix, k=-offset)
        if diagonal.size:
            diagonal_scores.append(float(diagonal.mean()))
    return float(np.mean(diagonal_scores)) if diagonal_scores else 0.0


def compute_induction_scores(attentions, offsets=(1, 2)):
    """Compute the induction-style diagonal score for every layer/head."""
    layer_count = len(attentions)
    head_count = attentions[0].shape[1]
    scores = np.zeros((layer_count, head_count), dtype=float)

    for layer_index, layer_attention in enumerate(attentions):
        for head_index in range(head_count):
            attention_matrix = layer_attention[0, head_index].detach().float().cpu().numpy()
            scores[layer_index, head_index] = induction_score(attention_matrix, offsets=offsets)

    return scores


def analyze_induction_heads(
    attentions,
    prompt=None,
    top_k=10,
    filename_or_path=ATTENTION_INDUCTION_HEATMAP_PATH,
):
    """Rank and plot heads by their previous-token-diagonal attention score."""
    induction_scores = compute_induction_scores(attentions)
    ranked_heads = rank_layer_heads(induction_scores, top_k=top_k)

    print_ranked_heads(ranked_heads, score_label="Induction score")
    best_head = ranked_heads[0]
    print(
        "Highest induction score:",
        f"layer {best_head['layer']}, head {best_head['head']}",
        f"({best_head['score']:.6f})",
    )

    plot_layer_head_heatmap(
        induction_scores,
        "Induction-style previous-token attention by layer/head",
        "Induction score",
        filename_or_path,
        top_heads=ranked_heads[:5],
        cmap="Purples",
        prompt_text=prompt,
    )
    return {
        "induction_scores": induction_scores,
        "ranked_heads": ranked_heads,
    }
