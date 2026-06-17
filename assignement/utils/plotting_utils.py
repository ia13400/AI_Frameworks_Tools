import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from config import OUTPUT_PNG_DIR


matplotlib.rcParams["figure.dpi"] = 100


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
    plt.savefig(OUTPUT_PNG_DIR / "hu_liu_opinion_embeddings.png", dpi=120)
    plt.close()
    print("\nSaved plot: hu_liu_opinion_embeddings.png")

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
    plt.savefig(OUTPUT_PNG_DIR / "hu_liu_opinion_histogram.png", dpi=120)
    plt.close()
    print("Saved plot: hu_liu_opinion_histogram.png")


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
    plt.savefig(OUTPUT_PNG_DIR / "hu_liu_field_embedding_panels.png", dpi=130)
    plt.close()
    print("Saved plot: hu_liu_field_embedding_panels.png")


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
    plt.savefig(OUTPUT_PNG_DIR / filename, dpi=130)
    plt.close()
    print(f"Saved plot: {filename}")


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
    plt.savefig(OUTPUT_PNG_DIR / filename, dpi=130)
    plt.close()
    print(f"Saved plot: {filename}")


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
    plt.savefig(OUTPUT_PNG_DIR / filename, dpi=130)
    plt.close()
    print(f"Saved plot: {filename}")


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
    plt.savefig(OUTPUT_PNG_DIR / "logistic_regression_field_word_probabilities.png", dpi=130)
    plt.close()
    print("Saved plot: logistic_regression_field_word_probabilities.png")

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

    plt.figure(
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
    plt.savefig(OUTPUT_PNG_DIR / "logistic_regression_sentiment_axis.png", dpi=130)
    plt.close()
    print("Saved plot: logistic_regression_sentiment_axis.png")