import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA

from config import (
    LOGIT_LENS_CAD_LOGIT_DIFFERENCE_AGGREGATE_PATH,
    FILTERED_SENTIMENT_CHALLENGE_HEATMAP_PATH,
    LOGIT_LENS_NEGATIVE_HEATMAP_PATH,
    LOGIT_LENS_PROMPT_LOGIT_DIFFERENCE_PATH,
    LOGIT_LENS_PROMPT_LOGIT_SCORES_PATH,
    LOGIT_LENS_POSITIVE_HEATMAP_PATH,
    LOGIT_LENS_TARGET_PROBABILITY_PATH,
    OUTPUT_PNG_DIR,
    SENTIMENT_CHALLENGE_CATEGORY_ACCURACY_PATH,
    SENTIMENT_CHALLENGE_CATEGORY_CONFUSION_PATH,
    SENTIMENT_CHALLENGE_HEATMAP_PATH,
)
from output_utils import save_figure_if_changed


matplotlib.rcParams["figure.dpi"] = 100


def set_even_layer_xticks(ax, layer_count):
    """Label model layers on even numbers; layer 0 is the embedding state."""
    ticks = np.arange(2, layer_count, 2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])


def plot_sentiment_challenge_category_heatmap(
    category_to_records,
    sentiment_labels,
    filename_or_path=SENTIMENT_CHALLENGE_HEATMAP_PATH,
    sentiment_key="gold_label",
    title="Sentiment Challenge Dataset: Categories by Sentiment Level",
):
    """Save a heatmap of challenge annotation categories by gold sentiment label."""
    output_path = (
        OUTPUT_PNG_DIR / filename_or_path
        if isinstance(filename_or_path, str)
        else filename_or_path
    )
    categories = sorted(
        category_to_records,
        key=lambda key: (-len(category_to_records[key]), key),
    )
    sentiment_ids = list(sentiment_labels)
    heatmap_counts = [
        [
            sum(
                record[sentiment_key] == sentiment_id
                for record in category_to_records[category]
            )
            for sentiment_id in sentiment_ids
        ]
        for category in categories
    ]
    sentiment_names = [
        (
            sentiment_labels[sentiment_id]
            if str(sentiment_id) == str(sentiment_labels[sentiment_id])
            else f"{sentiment_id}: {sentiment_labels[sentiment_id]}"
        )
        for sentiment_id in sentiment_ids
    ]

    height = max(9, len(categories) * 0.58)
    fig, ax = plt.subplots(figsize=(12.5, height))
    sns.heatmap(
        heatmap_counts,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar_kws={"label": "Number of sentences"},
        xticklabels=sentiment_names,
        yticklabels=categories,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Annotation category")
    x_rotation = 0 if len(sentiment_names) <= 2 else 30
    ax.tick_params(axis="x", rotation=x_rotation)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    save_figure_if_changed(fig, output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return heatmap_counts


def plot_filtered_sentiment_challenge_category_heatmap(
    category_to_records,
    filename_or_path=FILTERED_SENTIMENT_CHALLENGE_HEATMAP_PATH,
):
    """Save the binary positive/negative heatmap for the filtered challenge data."""
    return plot_sentiment_challenge_category_heatmap(
        category_to_records,
        {"negative": "negative", "positive": "positive"},
        filename_or_path=filename_or_path,
        sentiment_key="sentiment",
        title="Filtered Sentiment Challenge Dataset: Categories by Binary Sentiment",
    )


def plot_sentiment_challenge_category_confusion_matrices(
    category_confusion_counts,
    filename_or_path=SENTIMENT_CHALLENGE_CATEGORY_CONFUSION_PATH,
    prompt_suffix=None,
):
    """Save one confusion matrix per challenge annotation category."""
    output_path = (
        OUTPUT_PNG_DIR / filename_or_path
        if isinstance(filename_or_path, str)
        else filename_or_path
    )
    categories = list(category_confusion_counts)
    true_labels = ["negative", "positive"]
    predicted_labels = ["negative", "positive", "no sentiment"]

    n_cols = 3
    n_rows = int(np.ceil(len(categories) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(17, max(5, n_rows * 3.25)),
    )
    axes = np.asarray(axes).reshape(-1)

    max_count = max(
        max(max(row) for row in matrix)
        for matrix in category_confusion_counts.values()
    )

    for ax, category in zip(axes, categories):
        matrix = category_confusion_counts[category]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar=False,
            vmin=0,
            vmax=max_count,
            xticklabels=predicted_labels,
            yticklabels=true_labels,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        ax.set_title(category, fontsize=10.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=25, labelsize=8.5)
        ax.tick_params(axis="y", rotation=0, labelsize=8.5)

    for ax in axes[len(categories):]:
        ax.axis("off")

    title = "Sentiment Challenge: Hu & Liu Top-10 Token Sentiment by Category"
    if prompt_suffix is not None:
        title += f"\nPrompt suffix: {prompt_suffix!r}"
    fig.suptitle(title, fontsize=16, y=0.996)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    save_figure_if_changed(fig, output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_sentiment_challenge_category_accuracy(
    category_accuracy_rows,
    filename_or_path=SENTIMENT_CHALLENGE_CATEGORY_ACCURACY_PATH,
    prompt_suffix=None,
):
    """Save a stacked bar chart with correct, wrong, and no-sentiment rates."""
    output_path = (
        OUTPUT_PNG_DIR / filename_or_path
        if isinstance(filename_or_path, str)
        else filename_or_path
    )
    rows = sorted(category_accuracy_rows, key=lambda row: (row["accuracy"], row["category"]))
    categories = [row["category"] for row in rows]
    y_positions = np.arange(len(rows))
    correct_rates = np.array([row["accuracy"] * 100 for row in rows])
    wrong_rates = np.array([row["wrong_sentiment_rate"] * 100 for row in rows])
    no_sentiment_rates = np.array([row["no_sentiment_rate"] * 100 for row in rows])

    fig, ax = plt.subplots(figsize=(15.5, max(8, len(rows) * 0.5)))
    error_total_rates = wrong_rates + no_sentiment_rates
    right_bar_edge = 125
    wrong_left = right_bar_edge - error_total_rates
    no_sentiment_left = right_bar_edge - no_sentiment_rates

    ax.barh(
        y_positions,
        correct_rates,
        color="#2E8B57",
        alpha=0.86,
        label="correct sentiment",
    )
    ax.barh(
        y_positions,
        wrong_rates,
        left=wrong_left,
        color="#B22222",
        alpha=0.82,
        label="wrong sentiment",
    )
    ax.barh(
        y_positions,
        no_sentiment_rates,
        left=no_sentiment_left,
        color="#8C8C8C",
        alpha=0.8,
        label="no sentiment",
    )

    for y_pos, row, correct_value in zip(y_positions, rows, correct_rates):
        correct_label_x = correct_value - 3 if correct_value >= 90 else correct_value + 1.2
        ax.text(
            correct_label_x,
            y_pos,
            f"{correct_value:.1f}% (n={row['total']})",
            va="center",
            ha="left" if correct_value < 90 else "right",
            fontsize=9,
        )
        if correct_value < 100:
            ax.vlines(
                correct_value,
                y_pos - 0.4,
                y_pos + 0.4,
                color="white",
                linewidth=2.0,
                zorder=4,
            )

    for y_pos, wrong_value, no_sentiment_value in zip(
        y_positions,
        wrong_rates,
        no_sentiment_rates,
    ):
        if wrong_value >= 3:
            x_position = right_bar_edge - error_total_rates[y_pos] + wrong_value / 2
            ax.text(
                x_position,
                y_pos,
                f"wrong {wrong_value:.1f}%",
                va="center",
                ha="center",
                fontsize=8.5,
                color="white" if wrong_value >= 12 else "#202020",
            )

        if no_sentiment_value >= 3:
            x_position = right_bar_edge - no_sentiment_value / 2
            ax.text(
                x_position,
                y_pos,
                f"no {no_sentiment_value:.1f}%",
                va="center",
                ha="center",
                fontsize=8.5,
                color="white" if no_sentiment_value >= 12 else "#202020",
            )

    ax.axvline(50, color="#404040", linewidth=1, linestyle="--")
    ax.axvline(100, color="#B0B0B0", linewidth=0.8)
    ax.set_xlim(0, right_bar_edge)
    ax.set_xticks([0, 25, 50, 75, 100, 125])
    ax.set_xticklabels(["0", "25", "50", "75", "100", "0"])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(categories)
    ax.set_xlabel("Correct share (%) on the left; error share (%) on the right")
    ax.set_ylabel("Annotation category")
    title = "Sentiment Challenge: Correct vs Error Type by Category"
    if prompt_suffix is not None:
        title += f"\nPrompt suffix: {prompt_suffix!r}"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower right", frameon=True)

    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(y_positions)
    ax_right.set_yticklabels(categories, fontsize=9)
    ax_right.set_ylabel("Annotation category")

    plt.tight_layout()
    save_figure_if_changed(fig, output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_logit_lens_sentiment_probability_mass(
    positive_probs,
    negative_probs,
    sentiment_label,
    positive_prompt,
    negative_prompt,
    filename_or_path=LOGIT_LENS_TARGET_PROBABILITY_PATH,
    positive_label=None,
    negative_label=None,
    y_label=None,
    title=None,
):
    """Save a line chart of Hu & Liu sentiment-token probability mass by layer."""
    output_path = (
        OUTPUT_PNG_DIR / filename_or_path
        if isinstance(filename_or_path, str)
        else filename_or_path
    )
    layers = np.arange(len(positive_probs))

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    positive_label = positive_label or f"Positive: {positive_prompt}"
    negative_label = negative_label or f"Negative: {negative_prompt}"
    y_label = y_label or f"P(any {sentiment_label} Hu & Liu token)"
    title = title or f"Logit Lens: Probability Mass of {sentiment_label} Hu & Liu Tokens"

    ax.plot(layers, positive_probs, marker="o", label=positive_label)
    ax.plot(layers, negative_probs, marker="o", label=negative_label)
    ax.set_xlabel("Layer index (0 = embedding)")
    set_even_layer_xticks(ax, len(positive_probs))
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8.5)

    plt.tight_layout()
    save_figure_if_changed(fig, output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_logit_lens_prompt_sentiment_logit_scores(
    positive_prompt_scores,
    negative_prompt_scores,
    positive_prompt_text,
    negative_prompt_text,
    filename_or_path=LOGIT_LENS_PROMPT_LOGIT_SCORES_PATH,
):
    """Save side-by-side positive/negative Hu & Liu logit scores for both prompts."""
    output_path = (
        OUTPUT_PNG_DIR / filename_or_path
        if isinstance(filename_or_path, str)
        else filename_or_path
    )
    plot_specs = [
        ("Positive prompt", positive_prompt_text, positive_prompt_scores),
        ("Negative prompt", negative_prompt_text, negative_prompt_scores),
    ]
    layer_count = len(positive_prompt_scores["positive_scores"])
    layers = np.arange(layer_count)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharey=True)
    for ax, (prompt_label, prompt_text, scores) in zip(axes, plot_specs):
        positive_scores = np.asarray(scores["positive_scores"], dtype=float)
        negative_scores = np.asarray(scores["negative_scores"], dtype=float)
        ax.plot(layers, positive_scores, marker="o", color="#2E8B57", label="Positive Hu & Liu logit score")
        ax.plot(layers, negative_scores, marker="o", color="#B22222", label="Negative Hu & Liu logit score")
        ax.fill_between(
            layers,
            positive_scores,
            negative_scores,
            where=positive_scores >= negative_scores,
            color="#2E8B57",
            alpha=0.18,
            interpolate=True,
            label="positive > negative",
        )
        ax.fill_between(
            layers,
            positive_scores,
            negative_scores,
            where=positive_scores < negative_scores,
            color="#B22222",
            alpha=0.16,
            interpolate=True,
            label="negative > positive",
        )
        ax.axhline(0, color="#707070", linewidth=0.8, alpha=0.7)
        set_even_layer_xticks(ax, layer_count)
        ax.set_xlabel("Layer index (0 = embedding)")
        ax.set_title(prompt_label)
        ax.text(
            0.01,
            0.02,
            f"Prompt: {prompt_text}",
            transform=ax.transAxes,
            fontsize=8.0,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#D0D0D0"},
        )
        ax.grid(True, alpha=0.28)

    axes[0].set_ylabel("Sum of Hu & Liu logits")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Logit Lens Sentiment Scores by Layer", fontsize=14, y=0.985)
    fig.legend(
        handles,
        labels,
        fontsize=8.5,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.91),
    )

    plt.tight_layout(rect=(0, 0, 1, 0.875))
    save_figure_if_changed(fig, output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_logit_lens_prompt_logit_differences(
    positive_prompt_difference,
    negative_prompt_difference,
    positive_prompt_text,
    negative_prompt_text,
    filename_or_path=LOGIT_LENS_PROMPT_LOGIT_DIFFERENCE_PATH,
):
    """Save both prompts' Hu & Liu logit-difference curves in one image."""
    output_path = (
        OUTPUT_PNG_DIR / filename_or_path
        if isinstance(filename_or_path, str)
        else filename_or_path
    )
    layers = np.arange(len(positive_prompt_difference))
    positive_prompt_difference = np.asarray(positive_prompt_difference, dtype=float)
    negative_prompt_difference = np.asarray(negative_prompt_difference, dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.plot(
        layers,
        positive_prompt_difference,
        marker="o",
        color="#2E8B57",
        linewidth=2.0,
        label="Positive prompt logit difference",
    )
    ax.plot(
        layers,
        negative_prompt_difference,
        marker="s",
        color="#B22222",
        linewidth=2.0,
        label="Negative prompt logit difference",
    )
    ax.fill_between(
        layers,
        positive_prompt_difference,
        negative_prompt_difference,
        color="#7B68EE",
        alpha=0.18,
        interpolate=True,
        label="Difference between prompt curves",
    )
    ax.axhline(0, color="#707070", linewidth=0.8, alpha=0.7)
    set_even_layer_xticks(ax, len(layers))
    ax.set_xlabel("Layer index (0 = embedding)")
    ax.set_ylabel("Logit difference (positive - negative)")
    ax.set_title("Hu & Liu logit difference by layer")
    ax.text(
        0.01,
        0.02,
        f"Positive prompt: {positive_prompt_text}\nNegative prompt: {negative_prompt_text}",
        transform=ax.transAxes,
        fontsize=8.3,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "#D0D0D0"},
    )
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8.5, loc="best")

    plt.tight_layout()
    save_figure_if_changed(fig, output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_cad_logit_difference_aggregate(
    mean_curve,
    std_curve,
    pair_count,
    filename_or_path=LOGIT_LENS_CAD_LOGIT_DIFFERENCE_AGGREGATE_PATH,
):
    """Save mean CAD logit-difference separation with standard deviation band."""
    output_path = (
        OUTPUT_PNG_DIR / filename_or_path
        if isinstance(filename_or_path, str)
        else filename_or_path
    )
    mean_curve = np.asarray(mean_curve, dtype=float)
    std_curve = np.asarray(std_curve, dtype=float)
    layers = np.arange(len(mean_curve))

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(
        layers,
        mean_curve,
        color="#3B5BA9",
        marker="o",
        linewidth=2.0,
        label="Mean logit-difference separation",
    )
    ax.fill_between(
        layers,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color="#3B5BA9",
        alpha=0.18,
        label="±1 standard deviation",
    )
    ax.axhline(0, color="#707070", linewidth=0.8, alpha=0.7)
    set_even_layer_xticks(ax, len(mean_curve))
    ax.set_xlabel("Layer index (0 = embedding)")
    ax.set_ylabel("Positive-prompt diff minus negative-prompt diff")
    ax.set_title(f"CAD Sentiment Pairs: Mean Hu & Liu Logit-Difference Separation (n={pair_count})")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8.5, loc="best")

    plt.tight_layout()
    save_figure_if_changed(fig, output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_logit_lens_topk_heatmap(
    layer_results,
    title,
    filename_or_path=LOGIT_LENS_POSITIVE_HEATMAP_PATH,
    top_k=5,
):
    """Save a heatmap of top-k logit-lens tokens for each layer."""
    output_path = (
        OUTPUT_PNG_DIR / filename_or_path
        if isinstance(filename_or_path, str)
        else filename_or_path
    )
    token_matrix = [
        [item["token"] for item in layer[:top_k]]
        for layer in layer_results
    ]
    prob_matrix = np.array(
        [
            [item["probability"] for item in layer[:top_k]]
            for layer in layer_results
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(11, max(8, len(layer_results) * 0.45)))
    image = ax.imshow(prob_matrix, cmap="Blues", aspect="auto")
    ax.set_xlabel("Top-k rank")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"Top-{rank}" for rank in range(1, top_k + 1)])
    ax.set_yticks(range(len(layer_results)))
    ax.set_yticklabels(range(len(layer_results)))

    max_probability = prob_matrix.max() if prob_matrix.size else 0.0
    for row_index in range(prob_matrix.shape[0]):
        for col_index in range(prob_matrix.shape[1]):
            token_text = token_matrix[row_index][col_index].replace("\n", "\\n")
            text_color = "white" if prob_matrix[row_index, col_index] > max_probability * 0.55 else "#1F1F1F"
            ax.text(
                col_index,
                row_index,
                f"{token_text!r}\n{prob_matrix[row_index, col_index]:.3f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
            )

    fig.colorbar(image, ax=ax, label="Probability")
    plt.tight_layout()
    save_figure_if_changed(fig, output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def representative_labels(group, n=8):
    edge_ranked = sorted(
        group,
        key=lambda item: abs(item["x"]) + abs(item["y"]),
        reverse=True,
    )
    return edge_ranked[:n]


def plot_global_pca(sentiment_words, positive_words, negative_words, vectors):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    for item, (x_coord, y_coord) in zip(sentiment_words, coords):
        item["x"] = float(x_coord)
        item["y"] = float(y_coord)

    print(
        "PCA explained variance:    "
        f"PC1={pca.explained_variance_ratio_[0]:.2%}, "
        f"PC2={pca.explained_variance_ratio_[1]:.2%}"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    for group, color, label in [
        (positive_words, "#2E8B57", "positive Hu & Liu words"),
        (negative_words, "#B22222", "negative Hu & Liu words"),
    ]:
        ax.scatter(
            [item["x"] for item in group],
            [item["y"] for item in group],
            c=color,
            label=label,
            alpha=0.65,
            s=38,
            edgecolors="white",
            linewidths=0.4,
        )

    words_to_label = representative_labels(positive_words) + representative_labels(negative_words)
    for label_index, item in enumerate(words_to_label):
        x_offset = 5 if label_index % 2 == 0 else -5
        y_offset = 5 if label_index % 3 else -9
        ax.annotate(
            item["word"],
            (item["x"], item["y"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left" if x_offset > 0 else "right",
            fontsize=7.5,
            alpha=0.9,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.72},
        )

    ax.axhline(0, color="#D0D0D0", linewidth=0.8)
    ax.axvline(0, color="#D0D0D0", linewidth=0.8)
    ax.set_title("Hu & Liu opinion words in Pythia-410M embedding space")
    ax.set_xlabel("PCA dimension 1")
    ax.set_ylabel("PCA dimension 2")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    save_figure_if_changed(fig, OUTPUT_PNG_DIR / "hu_liu_opinion_embeddings.png", dpi=120)
    plt.close()

    return pca


def plot_pca_histogram(positive_words, negative_words):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        [item["x"] for item in positive_words],
        bins=40,
        alpha=0.32,
        color="#2E8B57",
        edgecolor="#1F5F3C",
        linewidth=0.9,
        label="positive Hu & Liu words",
    )
    ax.hist(
        [item["x"] for item in negative_words],
        bins=40,
        alpha=0.32,
        color="#B22222",
        edgecolor="#7A1616",
        linewidth=0.9,
        label="negative Hu & Liu words",
    )
    ax.axvline(0, color="#404040", linewidth=0.8)
    ax.set_title("Distribution of opinion words along PCA dimension 1")
    ax.set_xlabel("PCA dimension 1")
    ax.set_ylabel("Number of words")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    save_figure_if_changed(fig, OUTPUT_PNG_DIR / "hu_liu_opinion_histogram.png", dpi=120)
    plt.close()


def plot_field_pca_panels(pairs_by_field, embedding_matrix, pca):
    field_count = len(pairs_by_field["fields"])
    n_cols = 2
    n_rows = int(np.ceil(field_count / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 4.2 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)
    field_colors = {"positive": "#2E8B57", "negative": "#B22222"}

    for ax, field_data in zip(axes, pairs_by_field["fields"]):
        field_items = []
        for pair in field_data["pairs"]:
            field_items.append({**pair["positive"], "pair_id": pair["id"]})
            field_items.append({**pair["negative"], "pair_id": pair["id"]})

        field_token_ids = torch.tensor([item["token_id"] for item in field_items], dtype=torch.long)
        field_vectors = embedding_matrix[field_token_ids].float().numpy()
        field_coords = pca.transform(field_vectors)

        for item, (x_coord, y_coord) in zip(field_items, field_coords):
            item["x"] = float(x_coord)
            item["y"] = float(y_coord)

        items_by_pair = {}
        for item in field_items:
            items_by_pair.setdefault(item["pair_id"], {})[item["sentiment"]] = item

        for pair_items in items_by_pair.values():
            if "positive" in pair_items and "negative" in pair_items:
                ax.plot(
                    [pair_items["positive"]["x"], pair_items["negative"]["x"]],
                    [pair_items["positive"]["y"], pair_items["negative"]["y"]],
                    color="#B8B8B8",
                    linewidth=0.8,
                    alpha=0.75,
                    zorder=1,
                )

        for sentiment, marker in [("positive", "o"), ("negative", "X")]:
            group = [item for item in field_items if item["sentiment"] == sentiment]
            ax.scatter(
                [item["x"] for item in group],
                [item["y"] for item in group],
                c=field_colors[sentiment],
                marker=marker,
                s=72,
                edgecolors="white",
                linewidths=0.8,
                label=sentiment,
                zorder=3,
            )

        for label_index, item in enumerate(field_items):
            x_offset = 5 if label_index % 2 == 0 else -5
            y_offset = 7 if item["sentiment"] == "positive" else -11
            ax.annotate(
                item["word"],
                (item["x"], item["y"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="left" if x_offset > 0 else "right",
                va="bottom" if item["sentiment"] == "positive" else "top",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.78},
            )

        ax.set_title(field_data["field_label"], fontsize=11)
        ax.axhline(0, color="#D0D0D0", linewidth=0.7)
        ax.axvline(0, color="#D0D0D0", linewidth=0.7)
        ax.grid(alpha=0.18)

    for ax in axes[field_count:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Hu & Liu word-pair embeddings by field", fontsize=15, y=0.992)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    fig.supxlabel("PCA dimension 1")
    fig.supylabel("PCA dimension 2")
    plt.tight_layout(rect=(0, 0, 1, 0.935))
    save_figure_if_changed(fig, OUTPUT_PNG_DIR / "hu_liu_field_embedding_panels.png", dpi=130)
    plt.close()


def plot_embedding_norms(items, embedding_matrix, filename, title):
    rows = []
    for item in items:
        vector = embedding_matrix[item["token_id"]]
        rows.append(
            {
                "word": item["word"],
                "sentiment": item["sentiment"],
                "norm": float(torch.linalg.vector_norm(vector).item()),
            }
        )

    positive_rows = sorted(
        [row for row in rows if row["sentiment"] == "positive"],
        key=lambda row: row["norm"],
    )
    negative_rows = sorted(
        [row for row in rows if row["sentiment"] == "negative"],
        key=lambda row: row["norm"],
    )
    ordered_rows = positive_rows + negative_rows
    colors = [
        "#2E8B57" if row["sentiment"] == "positive" else "#B22222"
        for row in ordered_rows
    ]

    min_norm = min(row["norm"] for row in ordered_rows)
    max_norm = max(row["norm"] for row in ordered_rows)
    padding = max((max_norm - min_norm) * 0.08, 0.01)

    fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(ordered_rows))))
    ax.barh([row["word"] for row in ordered_rows], [row["norm"] for row in ordered_rows], color=colors, alpha=0.82)
    ax.set_xlim(min_norm - padding, max_norm + padding)
    ax.set_xlabel("L2 norm")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)

    if positive_rows and negative_rows:
        ax.axhline(len(positive_rows) - 0.5, color="#404040", linewidth=1.3)

    plt.tight_layout()
    save_figure_if_changed(fig, OUTPUT_PNG_DIR / filename, dpi=130)
    plt.close()


def plot_embedding_norms_by_field_boxes(fields, embedding_matrix, filename):
    field_count = len(fields)
    n_cols = 2
    n_rows = int(np.ceil(field_count / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.2 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    color_by_sentiment = {"positive": "#2E8B57", "negative": "#B22222"}

    all_rows_by_field = []
    all_norms = []
    for field in fields:
        rows = []
        for pair in field["pairs"]:
            for sentiment in ["positive", "negative"]:
                item = pair[sentiment]
                vector = embedding_matrix[item["token_id"]]
                norm = float(torch.linalg.vector_norm(vector).item())
                rows.append(
                    {
                        "word": item["word"],
                        "sentiment": item["sentiment"],
                        "norm": norm,
                    }
                )
                all_norms.append(norm)
        rows = sorted(rows, key=lambda row: (row["sentiment"] != "positive", row["norm"]))
        all_rows_by_field.append((field, rows))

    min_norm = min(all_norms)
    max_norm = max(all_norms)
    padding = max((max_norm - min_norm) * 0.08, 0.01)

    for ax, (field, rows) in zip(axes, all_rows_by_field):
        colors = [color_by_sentiment[row["sentiment"]] for row in rows]
        labels = [
            f"{row['word']} +" if row["sentiment"] == "positive" else f"{row['word']} -"
            for row in rows
        ]
        ax.barh(labels, [row["norm"] for row in rows], color=colors, alpha=0.82)
        ax.set_xlim(min_norm - padding, max_norm + padding)
        ax.set_title(field["field_label"], fontsize=10.5)
        ax.grid(axis="x", alpha=0.22)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_facecolor("#FBFBFB")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#6F6F6F")
            spine.set_linewidth(1.1)

    for ax in axes[field_count:]:
        ax.axis("off")

    fig.suptitle("L2 norms of Hu & Liu sentiment word embeddings by field", fontsize=15, y=0.995)
    fig.supxlabel("L2 norm")
    plt.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure_if_changed(fig, OUTPUT_PNG_DIR / filename, dpi=130)
    plt.close()


def plot_sentiment_projection(rows, field_labels, filename, title, figsize):
    positive_lanes = [-0.24, -0.18, -0.12, -0.06]
    negative_lanes = [0.06, 0.12, 0.18, 0.24]
    marker_by_sentiment = {"positive": "o", "negative": "X"}
    color_by_sentiment = {"positive": "#2E8B57", "negative": "#B22222"}

    for field_index in range(len(field_labels)):
        for sentiment, lanes in [("positive", positive_lanes), ("negative", negative_lanes)]:
            field_group = sorted(
                [
                    row for row in rows
                    if row["field_index"] == field_index and row["sentiment"] == sentiment
                ],
                key=lambda item: item["projection_score"],
            )
            for lane_index, row in enumerate(field_group):
                row["plot_y"] = field_index + lanes[lane_index % len(lanes)]

    fig, ax = plt.subplots(figsize=figsize)
    for sentiment in ["positive", "negative"]:
        group = [row for row in rows if row["sentiment"] == sentiment]
        ax.scatter(
            [row["projection_score"] for row in group],
            [row["plot_y"] for row in group],
            c=color_by_sentiment[sentiment],
            marker=marker_by_sentiment[sentiment],
            s=64 if len(field_labels) <= 3 else 58,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.9,
            label=sentiment,
            zorder=3,
        )

    for row in sorted(rows, key=lambda item: (item["field_index"], item["projection_score"])):
        x_offset = 5 if row["sentiment"] == "positive" else -5
        ax.annotate(
            row["word"],
            (row["projection_score"], row["plot_y"]),
            xytext=(x_offset, 0),
            textcoords="offset points",
            ha="left" if x_offset > 0 else "right",
            va="center",
            fontsize=7.0 if len(field_labels) <= 3 else 6.4,
            bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.7},
        )

    ax.axvline(0, color="#404040", linewidth=1)
    for y_coord in range(len(field_labels)):
        ax.axhline(y_coord, color="#D8D8D8", linewidth=0.75, alpha=0.75, zorder=0)
    for y_coord in np.arange(0.5, len(field_labels) - 0.5, 1.0):
        ax.axhline(y_coord, color="#6F6F6F", linewidth=1.6, alpha=0.85, zorder=0)

    ax.set_yticks(range(len(field_labels)))
    ax.set_yticklabels(field_labels)
    ax.set_ylim(-0.32, len(field_labels) - 0.68)
    ax.invert_yaxis()
    ax.set_xlabel("Projection score onto normalized sentiment direction")
    ax.set_title(title)
    ax.legend(
        frameon=True,
        loc="lower right",
        facecolor="white",
        edgecolor="#808080",
        framealpha=0.95,
    )
    ax.grid(axis="x", alpha=0.22)
    plt.tight_layout()
    save_figure_if_changed(fig, OUTPUT_PNG_DIR / filename, dpi=130)
    plt.close()


def plot_logistic_regression_probabilities(probability_rows):
    fig, ax = plt.subplots(figsize=(11, max(8, 0.22 * len(probability_rows))))
    colors = [
        "#2E8B57" if row["sentiment"] == "positive" else "#B22222"
        for row in probability_rows
    ]
    ax.barh(
        [row["word"] for row in probability_rows],
        [row["probability_positive"] for row in probability_rows],
        color=colors,
        alpha=0.82,
    )
    ax.axvline(0.5, color="#404040", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability of positive sentiment")
    ax.set_title("Linear probe positive-class probabilities for field-pair words")
    ax.grid(axis="x", alpha=0.22)
    plt.tight_layout()
    save_figure_if_changed(
        fig,
        OUTPUT_PNG_DIR / "logistic_regression_field_word_probabilities.png",
        dpi=130,
    )
    plt.close()

def plot_logistic_regression_sentiment_axis(
    probability_rows,
    title="Logistic Regression Sentiment Axis",
    max_words_per_side=25,
):
    """
    Visualize the sentiment axis learned by the logistic-regression probe.

    Words are ordered by their predicted probability of belonging
    to the positive sentiment class.

    Positive words:
        probability -> 1

    Negative words:
        probability -> 0

    Misclassified words are highlighted with a red border.
    """

    probability_rows = sorted(
        probability_rows,
        key=lambda row: row["probability_positive"]
    )

    boundary_rows = sorted(
        probability_rows,
        key=lambda row: abs(
            row["probability_positive"] - 0.5
        )
    )

    selected_rows = boundary_rows[:max_words_per_side]

    words = [row["word"] for row in selected_rows]

    probabilities = [
        row["probability_positive"]
        for row in selected_rows
    ]

    colors = [
        "tab:orange"
        if row["sentiment"] == "positive"
        else "tab:blue"
        for row in selected_rows
    ]

    fig = plt.figure(
        figsize=(12, max(8, len(selected_rows) * 0.25))
    )

    bars = plt.barh(
        words,
        probabilities,
        color=colors,
    )

    for bar, row in zip(bars, selected_rows):
        if not row["is_correct"]:
            bar.set_edgecolor("red")
            bar.set_linewidth(2.5)

    plt.axvline(
        0.5,
        linestyle="--",
        linewidth=1.5,
        label="Decision Boundary",
    )

    plt.xlabel("Predicted Probability of Positive Sentiment")
    plt.ylabel("Word")
    plt.title(title)

    plt.xlim(0, 1)
    plt.legend()

    plt.tight_layout()
    save_figure_if_changed(
        fig,
        OUTPUT_PNG_DIR / "logistic_regression_sentiment_axis.png",
        dpi=130,
    )
    plt.close()
