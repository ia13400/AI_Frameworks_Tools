import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import FLAT_PAIRS_PATH, PROBE_SPLIT_PATH
from model_utils import section
from plotting_utils import (
    plot_logistic_regression_probabilities,
    plot_logistic_regression_sentiment_axis,
)


PROBE_SPLIT_STRATEGY = "stratified_all_hu_liu_one_token_words_v1"


def split_word_record(item):
    return {
        "word": item["word"],
        "token": item["token"],
        "token_id": int(item["token_id"]),
        "sentiment": item["sentiment"],
    }


def load_or_create_probe_split(positive_words, negative_words):
    all_probe_words = sorted(
        positive_words + negative_words,
        key=lambda item: (item["sentiment"], item["word"], item["token_id"]),
    )

    use_saved_split = False
    if PROBE_SPLIT_PATH.exists():
        split_payload = json.loads(PROBE_SPLIT_PATH.read_text(encoding="utf-8"))
        use_saved_split = split_payload.get("split_strategy") == PROBE_SPLIT_STRATEGY

    if use_saved_split:
        train_words = split_payload["train"]
        test_words = split_payload["test"]
        print(f"\nLoaded fixed logistic-regression split from {PROBE_SPLIT_PATH}")
    else:
        if PROBE_SPLIT_PATH.exists():
            print(
                "\nExisting logistic-regression split uses an older strategy; "
                "regenerating it with all one-token Hu & Liu words."
            )

        y_all = np.array([1 if item["sentiment"] == "positive" else 0 for item in all_probe_words])
        split_train_indices, split_test_indices = train_test_split(
            np.arange(len(all_probe_words)),
            test_size=0.2,
            stratify=y_all,
            random_state=42,
        )
        train_words = [split_word_record(all_probe_words[index]) for index in split_train_indices]
        test_words = [split_word_record(all_probe_words[index]) for index in split_test_indices]
        split_payload = {
            "source": "Hu & Liu one-token sentiment words",
            "split_strategy": PROBE_SPLIT_STRATEGY,
            "random_state": 42,
            "test_size": 0.2,
            "stratify": "sentiment label",
            "label_mapping": {"negative": 0, "positive": 1},
            "all_one_token_word_counts": {
                "positive": int((y_all == 1).sum()),
                "negative": int((y_all == 0).sum()),
                "total": int(len(y_all)),
            },
            "train": train_words,
            "test": test_words,
        }
        PROBE_SPLIT_PATH.write_text(
            json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved fixed logistic-regression split to {PROBE_SPLIT_PATH}")

    probe_words = train_words + test_words
    train_indices = np.arange(len(train_words))
    test_indices = np.arange(len(train_words), len(probe_words))
    return probe_words, train_indices, test_indices


def print_probe_evaluation(split_name, y_true, y_pred):
    print(f"\n--- Logistic Regression Linear Probe Evaluation: {split_name} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 score : {f1_score(y_true, y_pred):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))


def print_misclassified_words(split_name, word_records, y_true, y_pred, y_prob):
    misclassified = [
        {
            "word": item["word"],
            "true": "positive" if true_label == 1 else "negative",
            "predicted": "positive" if predicted_label == 1 else "negative",
            "probability_positive": float(probability),
        }
        for item, true_label, predicted_label, probability in zip(
            word_records,
            y_true,
            y_pred,
            y_prob,
        )
        if true_label != predicted_label
    ]

    print(f"\nMisclassified {split_name} words:")
    print("-" * 72)
    if misclassified:
        for row in sorted(misclassified, key=lambda item: item["word"]):
            print(
                f"{row['word']:<18} "
                f"true={row['true']:<8} "
                f"predicted={row['predicted']:<8} "
                f"p_positive={row['probability_positive']:.4f}"
            )
    else:
        print("None")


def build_logistic_regression_sentiment_axis_data(
    classifier,
    scaler,
    embedding_matrix,
    probe_words,
):
    """
    Prepare visualization data for the sentiment-axis plot.
    """

    token_ids = torch.tensor(
        [item["token_id"] for item in probe_words],
        dtype=torch.long,
    )

    X = embedding_matrix[token_ids].float().numpy()
    X_scaled = scaler.transform(X)

    probabilities = classifier.predict_proba(X_scaled)[:, 1]
    predictions = classifier.predict(X_scaled)

    rows = []

    for item, probability, prediction in zip(
        probe_words,
        probabilities,
        predictions,
    ):
        true_label = (
            1
            if item["sentiment"] == "positive"
            else 0
        )

        rows.append(
            {
                "word": item["word"],
                "sentiment": item["sentiment"],
                "probability_positive": float(probability),
                "predicted_label": int(prediction),
                "true_label": true_label,
                "is_correct": int(prediction) == true_label,
            }
        )

    return rows

def cosine_similarity_between_lr_and_good_bad(
    classifier,
    scaler,
    embedding_matrix,
    probe_words,
    good_word="good",
    bad_word="bad",
):
    """
    Compare the sentiment direction learned by the logistic-regression probe
    with a manually defined sentiment direction.

    The manual sentiment direction is constructed as:

        embedding(good) - embedding(bad)

    and represents a simple positive-versus-negative axis in embedding space.

    The logistic-regression coefficient vector represents the sentiment
    direction learned automatically from all Hu & Liu sentiment words.

    A high cosine similarity indicates that both approaches capture a
    similar sentiment structure in the embedding space.
    """

    def find_token_id(word, sentiment):
        """
        Retrieve the token ID of a sentiment word from the probe dataset.
        """
        matches = [
            item
            for item in probe_words
            if item["word"] == word and item["sentiment"] == sentiment
        ]

        if not matches:
            raise ValueError(
                f"Could not find word={word!r} with sentiment={sentiment!r}"
            )

        return int(matches[0]["token_id"])

    # Retrieve embeddings for the sentiment anchor words.
    good_id = find_token_id(good_word, "positive")
    bad_id = find_token_id(bad_word, "negative")

    good_vec = embedding_matrix[good_id].float().numpy()
    bad_vec = embedding_matrix[bad_id].float().numpy()

    # The classifier was trained on standardized embeddings.
    # Therefore, the sentiment direction must be computed in the same
    # feature space to ensure a meaningful comparison.
    good_vec_scaled = scaler.transform(good_vec.reshape(1, -1))[0]
    bad_vec_scaled = scaler.transform(bad_vec.reshape(1, -1))[0]

    # Manually defined sentiment direction.
    good_bad_direction = good_vec_scaled - bad_vec_scaled

    # Logistic-regression weight vector.
    # This vector represents the direction that best separates
    # positive and negative sentiment words.
    lr_direction = classifier.coef_[0]

    # Measure directional similarity between both sentiment axes.
    cosine_similarity = np.dot(
        lr_direction,
        good_bad_direction,
    ) / (
        np.linalg.norm(lr_direction)
        * np.linalg.norm(good_bad_direction)
    )

    print("\n--- Sentiment Direction Comparison ---")
    print(
        f"Cosine similarity between "
        f"LR direction and ({good_word} - {bad_word}): "
        f"{cosine_similarity:.4f}"
    )

    if cosine_similarity > 0.8:
        print(
            "Interpretation: Strong agreement between the manually "
            "defined and automatically learned sentiment directions."
        )
    elif cosine_similarity > 0.5:
        print(
            "Interpretation: Moderate agreement between the two "
            "sentiment directions."
        )
    elif cosine_similarity > 0:
        print(
            "Interpretation: Weak agreement. The classifier captures "
            "additional sentiment information beyond the simple "
            f"{good_word}-{bad_word} axis."
        )
    else:
        print(
            "Interpretation: No agreement between the manually defined "
            "and learned sentiment directions."
        )

    return cosine_similarity

def run_logistic_regression_probe(sentiment_state):
    section(8, "Logistic Regression Sentiment Linear Probe")

    # Logistic regression is used as a linear probe. It tests whether a simple
    # linear decision boundary can separate sentiment labels in embedding space.
    embedding_matrix = sentiment_state["embedding_matrix"]
    positive_words = sentiment_state["positive_words"]
    negative_words = sentiment_state["negative_words"]

    probe_words, train_indices, test_indices = load_or_create_probe_split(
        positive_words,
        negative_words,
    )

    X = embedding_matrix[
        torch.tensor([item["token_id"] for item in probe_words], dtype=torch.long)
    ].float().numpy()
    y = np.array([1 if item["sentiment"] == "positive" else 0 for item in probe_words])

    print(f"\nX.shape = {X.shape}")
    print(f"y.shape = {y.shape}")
    print(f"Positive labels: {int(y.sum())}")
    print(f"Negative labels: {int((y == 0).sum())}")
    print(f"Train examples: {len(train_indices)}")
    print(f"Test examples : {len(test_indices)}")

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    print(
        "Train labels  : "
        f"positive={int(y_train.sum())}, negative={int((y_train == 0).sum())}"
    )
    print(
        "Test labels   : "
        f"positive={int(y_test.sum())}, negative={int((y_test == 0).sum())}"
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = LogisticRegression(max_iter=2000, random_state=42)
    classifier.fit(X_train_scaled, y_train)

    y_test_pred = classifier.predict(X_test_scaled)
    y_test_prob = classifier.predict_proba(X_test_scaled)[:, 1]
    y_train_pred = classifier.predict(X_train_scaled)
    y_train_prob = classifier.predict_proba(X_train_scaled)[:, 1]

    print_probe_evaluation("test set", y_test, y_test_pred)
    test_word_records = [probe_words[index] for index in test_indices]
    print_misclassified_words("test", test_word_records, y_test, y_test_pred, y_test_prob)
    print_probe_evaluation("train set", y_train, y_train_pred)
    train_word_records = [probe_words[index] for index in train_indices]


    # ------------------------------------------------------------------
    # Compare the automatically learned sentiment direction of the
    # logistic-regression probe with the manually defined sentiment
    # direction (good - bad).
    #
    # A high cosine similarity suggests that the classifier has learned
    # a sentiment axis that is consistent with the manually chosen
    # sentiment anchors.
    # ------------------------------------------------------------------
    cosine_similarity_between_lr_and_good_bad(
        classifier=classifier,
        scaler=scaler,
        embedding_matrix=embedding_matrix,
        probe_words=probe_words,
    )

    sentiment_axis_rows = (
    build_logistic_regression_sentiment_axis_data(
        classifier=classifier,
        scaler=scaler,
        embedding_matrix=embedding_matrix,
        probe_words=probe_words,
    )
    )

    plot_logistic_regression_sentiment_axis(
    sentiment_axis_rows,
    title="Logistic Regression Sentiment Axis",
    max_words_per_side=50,)

    return classifier, scaler
