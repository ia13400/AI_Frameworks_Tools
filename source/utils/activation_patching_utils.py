import contextlib
import io

import numpy as np
import torch

from config import (
    ACTIVATION_PATCHING_CAD_POSITIVE_MASS_PATH,
    ACTIVATION_PATCHING_HEAD_HEATMAP_PATH,
    ACTIVATION_PATCHING_HEATMAP_PATH,
    CAD_SENTIMENT_TRAIN_PAIRED_PATH,
)
from logit_lens_utils import (
    load_cad_sentiment_prompt_pairs,
    prepare_hu_liu_sentiment_state,
)
from plotting_utils import (
    plot_activation_head_patching_heatmap,
    plot_activation_patching_heatmap,
    plot_cad_activation_patching_positive_mass_effect,
)


def hidden_state_from_layer_output(output):
    """Return the hidden-state tensor from a transformer layer output."""
    return output[0] if isinstance(output, tuple) else output


def replace_hidden_state_in_output(output, hidden_state):
    """Return a layer output with the hidden-state tensor replaced."""
    if isinstance(output, tuple):
        return (hidden_state,) + output[1:]
    return hidden_state


def prepare_activation_patching_prompts(
    model,
    tokenizer,
    device,
    clean_prompt,
    corrupted_prompt,
    correct_answer,
):
    """Tokenize clean/corrupted prompts and measure baseline answer logits."""
    correct_token_ids = tokenizer.encode(correct_answer, add_special_tokens=False)
    if len(correct_token_ids) != 1:
        raise ValueError(
            f"Expected one answer token for {correct_answer!r}, "
            f"but got {len(correct_token_ids)} tokens: {correct_token_ids}"
        )
    correct_tok_id = int(correct_token_ids[0])

    clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(device)
    corrupted_inputs = tokenizer(corrupted_prompt, return_tensors="pt").to(device)
    if clean_inputs["input_ids"].shape != corrupted_inputs["input_ids"].shape:
        raise ValueError(
            "Clean and corrupted prompts must have the same tokenized shape. "
            f"Clean={tuple(clean_inputs['input_ids'].shape)}, "
            f"corrupted={tuple(corrupted_inputs['input_ids'].shape)}"
        )

    clean_tokens = [
        tokenizer.decode([token_id])
        for token_id in clean_inputs["input_ids"][0]
    ]
    corrupted_tokens = [
        tokenizer.decode([token_id])
        for token_id in corrupted_inputs["input_ids"][0]
    ]

    print("Clean tokens:")
    for index, token in enumerate(clean_tokens):
        print(f"{index:>2}: {token!r}")
    print("\nCorrupted tokens:")
    for index, token in enumerate(corrupted_tokens):
        print(f"{index:>2}: {token!r}")

    model.eval()
    with torch.no_grad():
        clean_outputs = model(**clean_inputs)
        corrupted_outputs = model(**corrupted_inputs)

    clean_logits = clean_outputs.logits[0, -1, :]
    corrupted_logits = corrupted_outputs.logits[0, -1, :]
    clean_probability = float(torch.softmax(clean_logits.float(), dim=-1)[correct_tok_id])
    corrupted_probability = float(torch.softmax(corrupted_logits.float(), dim=-1)[correct_tok_id])
    clean_logit = float(clean_logits[correct_tok_id])
    corrupted_logit = float(corrupted_logits[correct_tok_id])

    print(f"\nCorrect answer token: {correct_answer!r} -> id {correct_tok_id}")
    print(f'P({correct_answer!r} | clean)     = {clean_probability:.6f}')
    print(f'P({correct_answer!r} | corrupted) = {corrupted_probability:.6f}')
    print(f"Probability difference = {clean_probability - corrupted_probability:.6f}")
    print(f"Clean logit            = {clean_logit:.6f}")
    print(f"Corrupted logit        = {corrupted_logit:.6f}")

    return {
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "correct_answer": correct_answer,
        "correct_tok_id": correct_tok_id,
        "clean_inputs": clean_inputs,
        "corrupted_inputs": corrupted_inputs,
        "clean_tokens": clean_tokens,
        "corrupted_tokens": corrupted_tokens,
        "seq_len": int(clean_inputs["input_ids"].shape[1]),
        "n_layers": int(model.config.num_hidden_layers),
        "clean_logit": clean_logit,
        "corrupted_logit": corrupted_logit,
        "clean_probability": clean_probability,
        "corrupted_probability": corrupted_probability,
    }


def cache_clean_layer_activations(model, clean_inputs, verbose=True):
    """Run the clean prompt and cache its hidden states from every transformer layer."""
    clean_cache = {}
    hooks = []

    def make_hook(layer_index):
        def hook_fn(module, inputs, output):
            clean_cache[layer_index] = hidden_state_from_layer_output(output).detach().clone()
            return output

        return hook_fn

    for layer_index, layer in enumerate(model.gpt_neox.layers):
        hooks.append(layer.register_forward_hook(make_hook(layer_index)))

    model.eval()
    try:
        with torch.no_grad():
            model(**clean_inputs)
    finally:
        for hook in hooks:
            hook.remove()

    if verbose:
        print(f"Cached clean activations for {len(clean_cache)} transformer layers.")
    return clean_cache


def patch_layer_position_and_measure(
    model,
    corrupted_inputs,
    clean_cache,
    patch_layer,
    patch_pos,
    correct_tok_id,
):
    """Run the corrupted prompt, patch in one clean activation, and return the answer logit."""
    target_layer = int(patch_layer)
    target_pos = int(patch_pos)

    def hook_fn(module, inputs, output):
        hidden_state = hidden_state_from_layer_output(output)
        modified = hidden_state.clone()
        # Replace the corrupted-run activation at this layer/position with the
        # activation cached earlier from the clean prompt.
        modified[0, target_pos, :] = clean_cache[target_layer][0, target_pos, :]
        return replace_hidden_state_in_output(output, modified)

    hook = model.gpt_neox.layers[target_layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            outputs = model(**corrupted_inputs)
    finally:
        hook.remove()

    return float(outputs.logits[0, -1, correct_tok_id])


def hu_liu_positive_token_ids(sentiment_state, device):
    """Return unique one-token Hu & Liu positive-token ids on the model device."""
    token_ids = sorted(
        {
            int(item["token_id"])
            for item in sentiment_state["positive_words"]
        }
    )
    if not token_ids:
        raise ValueError("No one-token positive Hu & Liu words are available.")

    return torch.tensor(token_ids, dtype=torch.long, device=device)


def positive_hu_liu_probability_from_logits(logits, positive_token_ids):
    """Calculate full softmax probability mass assigned to positive Hu & Liu tokens."""
    probabilities = torch.softmax(logits.float(), dim=-1)
    return float(probabilities.index_select(0, positive_token_ids).sum().item())


def next_token_positive_hu_liu_probability(model, inputs, positive_token_ids):
    """Run one prompt and measure P(any positive Hu & Liu token) at the final position."""
    with torch.no_grad():
        outputs = model(**inputs)
    return positive_hu_liu_probability_from_logits(
        outputs.logits[0, -1, :],
        positive_token_ids,
    )


def patch_layer_last_position_and_measure_positive_mass(
    model,
    negative_inputs,
    positive_cache,
    patch_layer,
    positive_token_ids,
):
    """Patch one positive last-token activation into the negative prompt and score it."""
    target_layer = int(patch_layer)

    def hook_fn(module, inputs, output):
        hidden_state = hidden_state_from_layer_output(output)
        modified = hidden_state.clone()
        # The CAD prompts can have different tokenized lengths. Patching only the
        # final position lets each prompt keep its own context length.
        modified[0, -1, :] = positive_cache[target_layer][0, -1, :]
        return replace_hidden_state_in_output(output, modified)

    hook = model.gpt_neox.layers[target_layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            outputs = model(**negative_inputs)
    finally:
        hook.remove()

    return positive_hu_liu_probability_from_logits(
        outputs.logits[0, -1, :],
        positive_token_ids,
    )


def run_cad_pair_last_position_positive_mass_patching(
    model,
    tokenizer,
    device,
    prompt_pair,
    positive_token_ids,
):
    """Patch the positive prompt's final-position layer activations into its negative pair."""
    positive_inputs = tokenizer(prompt_pair["positive"], return_tensors="pt").to(device)
    negative_inputs = tokenizer(prompt_pair["negative"], return_tensors="pt").to(device)

    positive_probability = next_token_positive_hu_liu_probability(
        model,
        positive_inputs,
        positive_token_ids,
    )
    negative_probability = next_token_positive_hu_liu_probability(
        model,
        negative_inputs,
        positive_token_ids,
    )
    positive_cache = cache_clean_layer_activations(model, positive_inputs, verbose=False)

    patched_probabilities = []
    for layer_index in range(int(model.config.num_hidden_layers)):
        patched_probability = patch_layer_last_position_and_measure_positive_mass(
            model,
            negative_inputs,
            positive_cache,
            layer_index,
            positive_token_ids,
        )
        patched_probabilities.append(patched_probability)

    effects = [
        patched_probability - negative_probability
        for patched_probability in patched_probabilities
    ]
    recovery_denominator = positive_probability - negative_probability
    if abs(recovery_denominator) < 1e-12:
        normalized_effects = [0.0 for _ in effects]
    else:
        normalized_effects = [
            effect / recovery_denominator
            for effect in effects
        ]

    return {
        "id": prompt_pair["id"],
        "positive_prompt": prompt_pair["positive"],
        "negative_prompt": prompt_pair["negative"],
        "positive_probability": positive_probability,
        "negative_probability": negative_probability,
        "patched_probabilities": patched_probabilities,
        "effects": effects,
        "normalized_effects": normalized_effects,
    }


def run_activation_patching_grid(model, patching_state, clean_cache):
    """Patch every layer/position pair and compute normalized recovery effects."""
    n_layers = patching_state["n_layers"]
    seq_len = patching_state["seq_len"]
    if len(clean_cache) != n_layers:
        raise ValueError(
            "clean_cache must contain activations from the clean prompt for every layer."
        )
    results = np.zeros((n_layers, seq_len), dtype=float)

    for layer_index in range(n_layers):
        print(f"\rActivation patching layer {layer_index + 1}/{n_layers}", end="", flush=True)
        for position_index in range(seq_len):
            # The model run below uses corrupted_inputs; clean_cache supplies only
            # the replacement activation for the selected layer/position.
            results[layer_index, position_index] = patch_layer_position_and_measure(
                model,
                patching_state["corrupted_inputs"],
                clean_cache,
                layer_index,
                position_index,
                patching_state["correct_tok_id"],
            )
    print()

    denominator = patching_state["clean_logit"] - patching_state["corrupted_logit"]
    if abs(denominator) < 1e-8:
        raise ValueError("Clean and corrupted answer logits are too close to normalize.")
    normalized = (results - patching_state["corrupted_logit"]) / denominator
    return {
        "patched_logits": results,
        "normalized_effects": normalized,
    }


def aggregate_cad_last_position_positive_mass_patching(
    model,
    tokenizer,
    device,
    sentiment_state,
    prompt_pairs,
    print_progress=True,
):
    """Run last-position CAD patching and aggregate layer effects across pairs."""
    positive_token_ids = hu_liu_positive_token_ids(sentiment_state, device)
    pair_results = []
    effect_curves = []
    normalized_effect_curves = []
    total_pairs = len(prompt_pairs)

    for index, prompt_pair in enumerate(prompt_pairs, start=1):
        if print_progress:
            print(f"\rCAD activation patching pair {index}/{total_pairs}", end="", flush=True)

        pair_result = run_cad_pair_last_position_positive_mass_patching(
            model,
            tokenizer,
            device,
            prompt_pair,
            positive_token_ids,
        )
        pair_results.append(pair_result)
        effect_curves.append(pair_result["effects"])
        normalized_effect_curves.append(pair_result["normalized_effects"])

    if print_progress:
        print()

    if not effect_curves:
        raise ValueError("No CAD prompt pairs were available for activation patching.")

    effects = torch.tensor(effect_curves, dtype=torch.float32)
    normalized_effects = torch.tensor(normalized_effect_curves, dtype=torch.float32)
    mean_effect = effects.mean(dim=0).tolist()
    std_effect = effects.std(dim=0, unbiased=False).tolist()
    mean_normalized_effect = normalized_effects.mean(dim=0).tolist()
    std_normalized_effect = normalized_effects.std(dim=0, unbiased=False).tolist()
    best_layer = int(torch.argmax(torch.tensor(mean_effect)).item())

    return {
        "pair_results": pair_results,
        "mean_effect": mean_effect,
        "std_effect": std_effect,
        "mean_normalized_effect": mean_normalized_effect,
        "std_normalized_effect": std_normalized_effect,
        "best_layer": best_layer,
        "best_layer_mean_effect": mean_effect[best_layer],
        "pair_count": len(pair_results),
        "positive_token_count": int(len(positive_token_ids)),
    }


def run_cad_activation_patching_positive_mass(
    model,
    tokenizer,
    device,
    sentiment_state=None,
    dataset_path=CAD_SENTIMENT_TRAIN_PAIRED_PATH,
    max=None,
    filename_or_path=ACTIVATION_PATCHING_CAD_POSITIVE_MASS_PATH,
):
    """Run CAD last-position activation patching and save the aggregate effect plot."""
    if sentiment_state is None:
        with contextlib.redirect_stdout(io.StringIO()):
            sentiment_state = prepare_hu_liu_sentiment_state(tokenizer)

    prompt_pairs = load_cad_sentiment_prompt_pairs(dataset_path, verbose=False)
    if max is not None:
        prompt_pairs = prompt_pairs[:max]

    aggregation = aggregate_cad_last_position_positive_mass_patching(
        model,
        tokenizer,
        device,
        sentiment_state,
        prompt_pairs,
        print_progress=True,
    )
    plot_cad_activation_patching_positive_mass_effect(
        aggregation["mean_effect"],
        aggregation["std_effect"],
        aggregation["pair_count"],
        filename_or_path=filename_or_path,
        verbose=False,
    )
    return aggregation


def plot_activation_patching_results(
    patching_result,
    patching_state,
    filename_or_path=ACTIVATION_PATCHING_HEATMAP_PATH,
):
    """Save the layer/position activation-patching visualization."""
    return plot_activation_patching_heatmap(
        patching_result["normalized_effects"],
        patching_state["corrupted_tokens"],
        filename_or_path=filename_or_path,
    )


def top_activation_patching_points(patching_result, patching_state, top_k=10):
    """Return and print the strongest layer/position patching effects."""
    normalized = patching_result["normalized_effects"]
    seq_len = normalized.shape[1]
    ranked_indices = np.argsort(normalized.ravel())[::-1][:top_k]
    rows = []

    print(
        f"{'Rank':<6} {'Layer':<7} {'Position':<9} "
        f"{'Corrupted token':<18} {'Clean patch token':<18} Effect"
    )
    print("-" * 82)
    for rank, flat_index in enumerate(ranked_indices, start=1):
        layer_index, position_index = divmod(int(flat_index), seq_len)
        row = {
            "rank": rank,
            "layer": layer_index,
            "position": position_index,
            "token": patching_state["corrupted_tokens"][position_index],
            "corrupted_token": patching_state["corrupted_tokens"][position_index],
            "clean_patch_token": patching_state["clean_tokens"][position_index],
            "normalized_effect": float(normalized[layer_index, position_index]),
        }
        rows.append(row)
        print(
            f"{rank:<6} "
            f"{layer_index:<7} "
            f"{position_index:<9} "
            f"{row['corrupted_token']!r:<18} "
            f"{row['clean_patch_token']!r:<18} "
            f"{row['normalized_effect']:.6f}"
        )

    best = rows[0]
    print(
        "\nStrongest patch point:",
        f"layer={best['layer']},",
        f"position={best['position']},",
        f"corrupted_token={best['corrupted_token']!r},",
        f"clean_patch_token={best['clean_patch_token']!r},",
        f"effect={best['normalized_effect']:.6f}",
    )
    return rows


def cache_clean_attention_outputs(model, clean_inputs):
    """Cache clean attention-module outputs for every layer."""
    attention_cache = {}
    hooks = []

    def make_hook(layer_index):
        def hook_fn(module, inputs, output):
            attention_cache[layer_index] = hidden_state_from_layer_output(output).detach().clone()
            return output

        return hook_fn

    for layer_index, layer in enumerate(model.gpt_neox.layers):
        hooks.append(layer.attention.register_forward_hook(make_hook(layer_index)))

    try:
        with torch.no_grad():
            model(**clean_inputs)
    finally:
        for hook in hooks:
            hook.remove()

    return attention_cache


def patch_attention_head_and_measure(
    model,
    corrupted_inputs,
    attention_cache,
    patch_layer,
    patch_head,
    patch_pos,
    correct_tok_id,
):
    """Patch one attention-head slice at one position and return the answer logit."""
    target_layer = int(patch_layer)
    target_head = int(patch_head)
    head_count = int(model.config.num_attention_heads)
    hidden_size = int(model.config.hidden_size)
    head_dim = hidden_size // head_count
    start_index = target_head * head_dim
    end_index = start_index + head_dim

    def hook_fn(module, inputs, output):
        attention_output = hidden_state_from_layer_output(output)
        modified = attention_output.clone()
        modified[0, patch_pos, start_index:end_index] = attention_cache[
            target_layer
        ][0, patch_pos, start_index:end_index]
        return replace_hidden_state_in_output(output, modified)

    hook = model.gpt_neox.layers[target_layer].attention.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            outputs = model(**corrupted_inputs)
    finally:
        hook.remove()

    return float(outputs.logits[0, -1, correct_tok_id])


def run_head_level_patching(
    model,
    patching_state,
    patch_pos=-1,
    filename_or_path=ACTIVATION_PATCHING_HEAD_HEATMAP_PATH,
):
    """Patch each attention head at one token position and save a head heatmap."""
    attention_cache = cache_clean_attention_outputs(model, patching_state["clean_inputs"])
    n_layers = patching_state["n_layers"]
    n_heads = int(model.config.num_attention_heads)
    results = np.zeros((n_layers, n_heads), dtype=float)
    resolved_pos = patch_pos if patch_pos >= 0 else patching_state["seq_len"] + patch_pos
    if not 0 <= resolved_pos < patching_state["seq_len"]:
        raise ValueError(
            f"patch_pos={patch_pos} resolves to {resolved_pos}, "
            f"but valid positions are 0 to {patching_state['seq_len'] - 1}."
        )
    corrupted_token = patching_state["corrupted_tokens"][resolved_pos]
    clean_patch_token = patching_state["clean_tokens"][resolved_pos]

    print(
        "Head-level patching position:",
        f"position={resolved_pos},",
        f"corrupted_token={corrupted_token!r},",
        f"clean_patch_token={clean_patch_token!r}",
    )

    for layer_index in range(n_layers):
        print(f"\rHead patching layer {layer_index + 1}/{n_layers}", end="", flush=True)
        for head_index in range(n_heads):
            results[layer_index, head_index] = patch_attention_head_and_measure(
                model,
                patching_state["corrupted_inputs"],
                attention_cache,
                layer_index,
                head_index,
                resolved_pos,
                patching_state["correct_tok_id"],
            )
    print()

    denominator = patching_state["clean_logit"] - patching_state["corrupted_logit"]
    normalized = (results - patching_state["corrupted_logit"]) / denominator
    plot_activation_head_patching_heatmap(
        normalized,
        filename_or_path=filename_or_path,
        patch_position=resolved_pos,
        corrupted_token=corrupted_token,
        clean_patch_token=clean_patch_token,
    )
    return {
        "patched_logits": results,
        "normalized_effects": normalized,
        "patch_position": resolved_pos,
        "corrupted_token": corrupted_token,
        "clean_patch_token": clean_patch_token,
    }


def print_activation_patching_reflection(top_points, patching_state):
    """Answer the notebook reflection questions in German."""
    best = top_points[0]
    print("1. Activation Patching ist kausaler als Attention-Analyse, weil es aktiv in den Modelllauf eingreift. Attention zeigt nur, wohin Information fließt; Patching testet, ob eine Aktivierung tatsächlich für die Ausgabe benötigt wird.")
    print(
        "2. Der stärkste Patching-Effekt liegt in "
        f"Schicht {best['layer']} an Position {best['position']} "
        f"({best['token']!r}). Das deutet darauf hin, dass diese Aktivierung "
        "für die Wiederherstellung der korrekten Antwort besonders wichtig ist."
    )
    print("3. Probing und Patching müssen nicht dieselbe Schicht hervorheben. Probing zeigt lineare Dekodierbarkeit, Patching zeigt kausalen Einfluss auf die konkrete Ausgabe. Eine Diskrepanz bedeutet, dass Information vorhanden sein kann, ohne direkt genutzt zu werden.")
    print("4. Activation Patching ist rechenintensiv, hängt stark vom gewählten Clean/Corrupted-Prompt-Paar ab und kann durch Superposition schwer interpretierbar sein.")
    print("5. Für Head-Level-Patching patcht man nur den Aktivierungsanteil eines einzelnen Attention Heads. Dadurch kann man prüfen, welche Heads kausal zur Wiederherstellung der Antwort beitragen.")
