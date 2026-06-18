import torch

from config import (
    LOGIT_LENS_NEGATIVE_HEATMAP_PATH,
    LOGIT_LENS_POSITIVE_HEATMAP_PATH,
    LOGIT_LENS_TARGET_PROBABILITY_PATH,
)
from lexicon_utils import (
    classify_top_tokens_by_hu_liu,
    prepare_hu_liu_lookup_state,
)
from plotting_utils import (
    plot_logit_lens_target_probability,
    plot_logit_lens_topk_heatmap,
)


def select_hu_liu_target_records(
    tokenizer=None,
    sentiment_state=None,
    positive_word: str = "great",
    negative_word: str = "terrible",
):
    """Select positive and negative target tokens from filtered Hu & Liu records."""
    if sentiment_state is None:
        if tokenizer is None:
            raise ValueError("Provide either tokenizer or sentiment_state.")
        sentiment_state = prepare_hu_liu_sentiment_state(tokenizer)

    words_by_sentiment = {
        "positive": sentiment_state["positive_words"],
        "negative": sentiment_state["negative_words"],
    }

    def select_word(sentiment, preferred_word):
        preferred_word = preferred_word.lower()
        selected = next(
            (
                item
                for item in words_by_sentiment[sentiment]
                if item["word"] == preferred_word
            ),
            None,
        )
        if selected is None:
            selected = sorted(
                words_by_sentiment[sentiment],
                key=lambda item: item["word"],
            )[0]
        return selected

    target_records = {
        "positive": select_word("positive", positive_word),
        "negative": select_word("negative", negative_word),
    }
    print(
        "Hu & Liu target tokens:",
        f"positive={target_records['positive']['token']!r} "
        f"({target_records['positive']['word']}),",
        f"negative={target_records['negative']['token']!r} "
        f"({target_records['negative']['word']})",
    )
    return target_records


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


def print_layer_top1_table(positive_layer_results, negative_layer_results):
    """Print a compact positive-vs-negative top-1 table for all layers."""
    print(
        f"{'Layer':<8} | "
        f"{'Top-1 Positive':<15} | "
        f"{'P(Positive)':<11} | "
        f"{'Top-1 Negative':<15} | "
        f"{'P(Negative)':<11}"
    )
    print("-" * 82)
    for layer_index, (positive_layer, negative_layer) in enumerate(
        zip(positive_layer_results, negative_layer_results)
    ):
        positive_top = positive_layer[0]
        negative_top = negative_layer[0]
        print(
            f"{layer_index:<8} | "
            f"{positive_top['token']!r:<15} | "
            f"{positive_top['probability']:<11.4f} | "
            f"{negative_top['token']!r:<15} | "
            f"{negative_top['probability']:<11.4f}"
        )


def first_layer_with_token(layer_results, target_token: str, top_n: int):
    """Return the first layer where target_token appears in the top-n tokens."""
    for layer_index, results in enumerate(layer_results):
        tokens = [item["token"] for item in results[:top_n]]
        if target_token in tokens:
            return layer_index
    return None


def print_target_token_findings(layer_results, target_token: str):
    """Print when a target token first appears as top-1 and top-5."""
    top1_layer = first_layer_with_token(layer_results, target_token, top_n=1)
    top5_layer = first_layer_with_token(layer_results, target_token, top_n=5)

    if top1_layer is None:
        print(f"{target_token!r} is never Top-1.")
    else:
        print(f"{target_token!r} is first Top-1 in layer {top1_layer}.")

    if top5_layer is None:
        print(f"{target_token!r} is never in the Top-5.")
    else:
        print(f"{target_token!r} first appears in the Top-5 in layer {top5_layer}.")


def token_probability_per_layer(hidden_states, model, token_id: int):
    """Compute one token's logit-lens probability for every layer."""
    probabilities = []
    for hidden_state in hidden_states:
        last_token_hidden_state = hidden_state[0, -1, :]
        logits = last_token_hidden_state @ model.embed_out.weight.T
        if torch.isnan(logits).any():
            raise ValueError("NaN detected in logits. Check model dtype/device.")
        probabilities.append(float(torch.softmax(logits.float(), dim=-1)[token_id]))
    return probabilities


def analyze_target_token_probability(
    positive_hidden_states,
    negative_hidden_states,
    model,
    tokenizer,
    prompt_pair,
    target_token: str,
):
    """Compute and plot a target token's probability over layers."""
    target_token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
    print("Target token:", repr(target_token))
    print("Target token ID:", target_token_id)
    print("Decoded check:", tokenizer.decode([target_token_id]))

    positive_probs = token_probability_per_layer(positive_hidden_states, model, target_token_id)
    negative_probs = token_probability_per_layer(
        negative_hidden_states,
        model,
        target_token_id,
    )
    print("Positive prompt probabilities:", positive_probs)
    print("Negative prompt probabilities:", negative_probs)

    plot_logit_lens_target_probability(
        positive_probs,
        negative_probs,
        target_token,
        prompt_pair["positive"],
        prompt_pair["negative"],
        filename_or_path=LOGIT_LENS_TARGET_PROBABILITY_PATH,
    )
    return positive_probs, negative_probs


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
