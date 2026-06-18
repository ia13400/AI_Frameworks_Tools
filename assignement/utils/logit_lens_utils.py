import torch

from config import (
    LOGIT_LENS_NEGATIVE_PROMPT_LOGIT_SCORE_PATH,
    LOGIT_LENS_NEGATIVE_SENTIMENT_MASS_PATH,
    LOGIT_LENS_NEGATIVE_HEATMAP_PATH,
    LOGIT_LENS_POSITIVE_PROMPT_LOGIT_SCORE_PATH,
    LOGIT_LENS_POSITIVE_SENTIMENT_MASS_PATH,
    LOGIT_LENS_POSITIVE_HEATMAP_PATH,
)
from lexicon_utils import (
    classify_top_tokens_by_hu_liu,
    prepare_hu_liu_lookup_state,
)
from plotting_utils import (
    plot_logit_lens_sentiment_logit_scores,
    plot_logit_lens_sentiment_probability_mass,
    plot_logit_lens_topk_heatmap,
)


def prepare_hu_liu_sentiment_state(tokenizer):
    """Build filtered Hu & Liu sentiment records once for Logit Lens reuse."""
    sentiment_state = prepare_hu_liu_lookup_state(tokenizer)
    print(
        "Hu & Liu one-token sentiment state:",
        f"{len(sentiment_state['positive_words'])} positive words,",
        f"{len(sentiment_state['negative_words'])} negative words",
    )
    return sentiment_state


def run_forward_pass_with_hidden_states(model, tokenizer, device, prompt: str):
    """Tokenize one prompt and run the model while keeping all hidden states."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states


def run_prompt_pair_forward_passes(model, tokenizer, device, prompt_pair):
    """Run positive and negative prompts and return their hidden states."""
    positive_hidden_states = run_forward_pass_with_hidden_states(
        model,
        tokenizer,
        device,
        prompt_pair["positive"],
    )
    negative_hidden_states = run_forward_pass_with_hidden_states(
        model,
        tokenizer,
        device,
        prompt_pair["negative"],
    )
    print_hidden_state_summary(positive_hidden_states, label="Positive prompt")
    print_hidden_state_summary(negative_hidden_states, label="Negative prompt")
    return positive_hidden_states, negative_hidden_states


def print_hidden_state_summary(hidden_states, label="Prompt"):
    """Print the number and shape of hidden states returned by the model."""
    print(f"\n{label}")
    print("Number of hidden states:", len(hidden_states))
    print("Expected: model layers + 1 embedding state")
    for index, hidden_state in enumerate(hidden_states):
        print(f"Hidden state {index}: {tuple(hidden_state.shape)}")


def unembedding_matrix(model):
    """Return the unembedding matrix used by Pythia-style causal language models."""
    matrix = model.embed_out.weight
    print("Unembedding matrix shape:", tuple(matrix.shape))
    return matrix


def logit_lens_top_tokens(hidden_state, model, tokenizer, top_k: int = 5):
    """Project one hidden state through the unembedding matrix and return top tokens."""
    last_token_hidden_state = hidden_state[0, -1, :]
    logits = last_token_hidden_state @ model.embed_out.weight.T
    if torch.isnan(logits).any():
        raise ValueError("NaN detected in logits. Check model dtype/device.")

    probabilities = torch.softmax(logits.float(), dim=-1)
    top_probabilities, top_token_ids = torch.topk(probabilities, top_k)
    return [
        {
            "rank": rank,
            "token": tokenizer.decode([int(token_id)]),
            "token_id": int(token_id),
            "probability": float(probability),
        }
        for rank, (token_id, probability) in enumerate(
            zip(top_token_ids, top_probabilities),
            start=1,
        )
    ]


def logit_lens_all_layers(hidden_states, model, tokenizer, top_k: int = 5):
    """Run logit lens for each layer's hidden state."""
    return [
        logit_lens_top_tokens(hidden_state, model, tokenizer, top_k=top_k)
        for hidden_state in hidden_states
    ]


def print_top_tokens_for_final_layer(prompt: str, layer_results, sentiment_state):
    """Print final-layer top tokens and mark Hu & Liu sentiment matches."""
    predicted_sentiment, sentiment_scores, matched_tokens = classify_top_tokens_by_hu_liu(
        layer_results[-1],
        sentiment_state["hu_liu_lookup"],
    )
    matches_by_token = {
        match["token"]: match
        for match in matched_tokens
    }

    print(f"Prompt: {prompt}")
    print("-" * 60)
    for item in layer_results[-1]:
        match = matches_by_token.get(item["token"])
        hu_liu_text = ""
        if match is not None:
            hu_liu_text = (
                f" | <-- Hu & Liu: {match['hu_liu_word']}"
                f" ({match['sentiment']})"
            )
        print(
            f"{item['rank']}. Token: {item['token']!r:<15} | "
            f"Probability: {item['probability']:.4f}"
            f"{hu_liu_text}"
        )
    print(
        "Hu & Liu top-token sentiment:",
        predicted_sentiment,
        f"(positive mass={sentiment_scores['positive']:.4f},",
        f"negative mass={sentiment_scores['negative']:.4f})",
    )


def hu_liu_marker(top_token_item, sentiment_state):
    """Return a compact marker when one token matches Hu & Liu."""
    if sentiment_state is None:
        return ""

    _, _, matched_tokens = classify_top_tokens_by_hu_liu(
        [top_token_item],
        sentiment_state["hu_liu_lookup"],
    )
    if not matched_tokens:
        return ""

    match = matched_tokens[0]
    return f" <-- Hu & Liu: {match['hu_liu_word']} ({match['sentiment']})"


def print_layer_top1_table(positive_layer_results, negative_layer_results, sentiment_state=None):
    """Print a compact positive-vs-negative top-1 table for all layers."""
    print(
        f"{'Layer':<8} | "
        f"{'Top-1 Positive':<45} | "
        f"{'P(Positive)':<11} | "
        f"{'Top-1 Negative':<45} | "
        f"{'P(Negative)':<11}"
    )
    print("-" * 142)
    for layer_index, (positive_layer, negative_layer) in enumerate(
        zip(positive_layer_results, negative_layer_results)
    ):
        positive_top = positive_layer[0]
        negative_top = negative_layer[0]
        positive_text = f"{positive_top['token']!r}{hu_liu_marker(positive_top, sentiment_state)}"
        negative_text = f"{negative_top['token']!r}{hu_liu_marker(negative_top, sentiment_state)}"
        print(
            f"{layer_index:<8} | "
            f"{positive_text:<45} | "
            f"{positive_top['probability']:<11.4f} | "
            f"{negative_text:<45} | "
            f"{negative_top['probability']:<11.4f}"
        )


def first_hu_liu_sentiment_token(layer_results, sentiment_state, sentiment: str, top_n: int = 1):
    """Find the first layer where any top-n token appears in Hu & Liu."""
    for layer_index, results in enumerate(layer_results):
        _, _, matched_tokens = classify_top_tokens_by_hu_liu(
            results[:top_n],
            sentiment_state["hu_liu_lookup"],
        )
        for match in matched_tokens:
            if match["sentiment"] == sentiment:
                return {
                    "layer": layer_index,
                    "token": match["token"],
                    "word": match["hu_liu_word"],
                    "sentiment": match["sentiment"],
                    "probability": match["probability"],
                }
    return None


def print_first_hu_liu_sentiment_token(layer_results, sentiment_state, sentiment: str, top_n: int = 1):
    """Print the first layer where a Hu & Liu sentiment token appears."""
    finding = first_hu_liu_sentiment_token(
        layer_results,
        sentiment_state,
        sentiment=sentiment,
        top_n=top_n,
    )
    if finding is None:
        print(f"No Hu & Liu {sentiment} token appears in the Top-{top_n} of any layer.")
        return None

    print(
        f"First Hu & Liu {sentiment} token in Top-{top_n}: "
        f"layer={finding['layer']}, "
        f"token={finding['token']!r}, "
        f"word={finding['word']}, "
        f"probability={finding['probability']:.4f}"
    )
    return finding


def hu_liu_sentiment_probability_mass_from_top_tokens(
    layer_results,
    sentiment_state,
    sentiment: str,
    top_n: int = 5,
):
    """Sum probability mass of matching Hu & Liu sentiment tokens in each layer's top-n."""
    probability_mass = []
    first_finding = None

    for layer_index, results in enumerate(layer_results):
        _, _, matched_tokens = classify_top_tokens_by_hu_liu(
            results[:top_n],
            sentiment_state["hu_liu_lookup"],
        )
        sentiment_matches = [
            match
            for match in matched_tokens
            if match["sentiment"] == sentiment
        ]
        probability_mass.append(
            sum(match["probability"] for match in sentiment_matches)
        )

        if first_finding is None and sentiment_matches:
            first_finding = {
                "layer": layer_index,
                "matches": sentiment_matches,
                "probability_mass": probability_mass[-1],
            }

    return probability_mass, first_finding


def print_topn_sentiment_finding(label: str, sentiment: str, top_n: int, finding):
    """Print the first layer where a top-n Hu & Liu sentiment match appears."""
    if finding is None:
        print(f"{label}: no Hu & Liu {sentiment} token appears in the Top-{top_n}.")
        return

    match_text = ", ".join(
        (
            f"{match['token']!r} -> {match['hu_liu_word']} "
            f"({match['probability']:.4f})"
        )
        for match in finding["matches"]
    )
    print(
        f"{label}: first Hu & Liu {sentiment} Top-{top_n} match "
        f"in layer {finding['layer']} "
        f"(mass={finding['probability_mass']:.4f}): {match_text}"
    )


def analyze_hu_liu_sentiment_probability_mass(
    positive_layer_results,
    negative_layer_results,
    sentiment_state,
    prompt_pair,
    top_n: int = 5,
    positive_filename_or_path=LOGIT_LENS_POSITIVE_SENTIMENT_MASS_PATH,
    negative_filename_or_path=LOGIT_LENS_NEGATIVE_SENTIMENT_MASS_PATH,
):
    """Create positive-word and negative-word Top-N mass analyses for a prompt pair."""
    results = {}
    plot_paths = {
        "positive": positive_filename_or_path,
        "negative": negative_filename_or_path,
    }

    for sentiment in ["positive", "negative"]:
        positive_probs, positive_finding = hu_liu_sentiment_probability_mass_from_top_tokens(
            positive_layer_results,
            sentiment_state,
            sentiment=sentiment,
            top_n=top_n,
        )
        negative_probs, negative_finding = hu_liu_sentiment_probability_mass_from_top_tokens(
            negative_layer_results,
            sentiment_state,
            sentiment=sentiment,
            top_n=top_n,
        )

        print_topn_sentiment_finding(
            "Positive prompt",
            sentiment,
            top_n,
            positive_finding,
        )
        print_topn_sentiment_finding(
            "Negative prompt",
            sentiment,
            top_n,
            negative_finding,
        )
        print(f"Positive prompt {sentiment} Hu & Liu Top-{top_n} mass:", positive_probs)
        print(f"Negative prompt {sentiment} Hu & Liu Top-{top_n} mass:", negative_probs)

        plot_logit_lens_sentiment_probability_mass(
            positive_probs,
            negative_probs,
            f"Top-{top_n} {sentiment} Hu & Liu",
            prompt_pair["positive"],
            prompt_pair["negative"],
            filename_or_path=plot_paths[sentiment],
            positive_label=f"Positive prompt: {sentiment} Hu & Liu Top-{top_n} mass",
            negative_label=f"Negative prompt: {sentiment} Hu & Liu Top-{top_n} mass",
            y_label=f"{sentiment.capitalize()} Hu & Liu probability mass in Top-{top_n}",
            title=(
                f"Logit Lens: {sentiment.capitalize()} Hu & Liu "
                f"Mass in Top-{top_n} Tokens"
            ),
        )
        results[sentiment] = {
            "positive_prompt_mass": positive_probs,
            "negative_prompt_mass": negative_probs,
            "positive_prompt_first_match": positive_finding,
            "negative_prompt_first_match": negative_finding,
        }

    return results


def hu_liu_logit_scores_per_layer(hidden_states, model, sentiment_state):
    """Calculate summed positive and negative Hu & Liu logits for every layer."""
    positive_token_ids = torch.tensor(
        [item["token_id"] for item in sentiment_state["positive_words"]],
        dtype=torch.long,
        device=model.embed_out.weight.device,
    )
    negative_token_ids = torch.tensor(
        [item["token_id"] for item in sentiment_state["negative_words"]],
        dtype=torch.long,
        device=model.embed_out.weight.device,
    )

    positive_scores = []
    negative_scores = []
    logit_differences = []

    for hidden_state in hidden_states:
        last_token_hidden_state = hidden_state[0, -1, :]
        logits = last_token_hidden_state @ model.embed_out.weight.T
        if torch.isnan(logits).any():
            raise ValueError("NaN detected in logits. Check model dtype/device.")

        positive_score = float(logits[positive_token_ids].float().sum())
        negative_score = float(logits[negative_token_ids].float().sum())
        positive_scores.append(positive_score)
        negative_scores.append(negative_score)
        logit_differences.append(positive_score - negative_score)

    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "logit_differences": logit_differences,
    }


def print_logit_score_summary(label: str, score_result):
    """Print the first layer where the positive/negative logit score wins."""
    differences = score_result["logit_differences"]
    first_positive_layer = next(
        (index for index, value in enumerate(differences) if value > 0),
        None,
    )
    first_negative_layer = next(
        (index for index, value in enumerate(differences) if value < 0),
        None,
    )
    final_difference = differences[-1]

    print(f"{label}: final logit difference (positive - negative) = {final_difference:.4f}")
    if first_positive_layer is None:
        print(f"{label}: positive score is never above negative score.")
    else:
        print(f"{label}: positive score first exceeds negative score in layer {first_positive_layer}.")

    if first_negative_layer is None:
        print(f"{label}: negative score is never above positive score.")
    else:
        print(f"{label}: negative score first exceeds positive score in layer {first_negative_layer}.")


def analyze_hu_liu_logit_sentiment_scores(
    positive_hidden_states,
    negative_hidden_states,
    model,
    sentiment_state,
    prompt_pair,
    positive_filename_or_path=LOGIT_LENS_POSITIVE_PROMPT_LOGIT_SCORE_PATH,
    negative_filename_or_path=LOGIT_LENS_NEGATIVE_PROMPT_LOGIT_SCORE_PATH,
):
    """Plot Hu & Liu positive/negative logit scores for both prompt directions."""
    positive_prompt_scores = hu_liu_logit_scores_per_layer(
        positive_hidden_states,
        model,
        sentiment_state,
    )
    negative_prompt_scores = hu_liu_logit_scores_per_layer(
        negative_hidden_states,
        model,
        sentiment_state,
    )

    print_logit_score_summary("Positive prompt", positive_prompt_scores)
    print_logit_score_summary("Negative prompt", negative_prompt_scores)

    plot_logit_lens_sentiment_logit_scores(
        positive_prompt_scores["positive_scores"],
        positive_prompt_scores["negative_scores"],
        "Positive prompt",
        prompt_pair["positive"],
        filename_or_path=positive_filename_or_path,
    )
    plot_logit_lens_sentiment_logit_scores(
        negative_prompt_scores["positive_scores"],
        negative_prompt_scores["negative_scores"],
        "Negative prompt",
        prompt_pair["negative"],
        filename_or_path=negative_filename_or_path,
    )

    return {
        "positive_prompt": positive_prompt_scores,
        "negative_prompt": negative_prompt_scores,
    }


def save_logit_lens_heatmaps(positive_layer_results, negative_layer_results):
    """Save positive and negative top-k logit-lens heatmaps."""
    plot_logit_lens_topk_heatmap(
        positive_layer_results,
        "Logit Lens Heatmap: Top-5 Tokens by Layer for Positive Prompt",
        filename_or_path=LOGIT_LENS_POSITIVE_HEATMAP_PATH,
        top_k=5,
    )
    plot_logit_lens_topk_heatmap(
        negative_layer_results,
        "Logit Lens Heatmap: Top-5 Tokens by Layer for Negative Prompt",
        filename_or_path=LOGIT_LENS_NEGATIVE_HEATMAP_PATH,
        top_k=5,
    )
