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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CAD_SENTIMENT_TRAIN_PAIRED_PATH,
    FLAT_PAIRS_PATH,
    LINEAR_PROBE_CAD_ACCURACY_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    PROBE_SPLIT_PATH,
)
from logit_lens_utils import load_cad_sentiment_prompt_pairs
from output_utils import write_json_if_changed
from plotting_utils import (
    plot_linear_probe_layer_accuracy,
    plot_logistic_regression_probabilities,
    plot_logistic_regression_sentiment_axis,
)


PROBE_SPLIT_STRATEGY = "stratified_all_hu_liu_one_token_words_v1"


def cad_prompt_records_from_pairs(prompt_pairs):
    """Flatten CAD positive/negative pairs into labeled prompt records."""
    records = []
    for pair in prompt_pairs:
        records.append(
            {
                "id": f"{pair['id']}_positive",
                "pair_id": pair["id"],
                "prompt": pair["positive"],
                "sentiment": "positive",
                "label": 1,
            }
        )
        records.append(
            {
                "id": f"{pair['id']}_negative",
                "pair_id": pair["id"],
                "prompt": pair["negative"],
                "sentiment": "negative",
                "label": 0,
            }
        )
    return records


def load_cad_linear_probe_records(
    dataset_path=CAD_SENTIMENT_TRAIN_PAIRED_PATH,
    max_prompts=None,
):
    """Load CAD sentiment prompts and return records with binary labels."""
    prompt_pairs = load_cad_sentiment_prompt_pairs(dataset_path, verbose=False)
    prompt_records = cad_prompt_records_from_pairs(prompt_pairs)
    if max_prompts is not None:
        prompt_records = prompt_records[:max_prompts]

    labels = np.array([record["label"] for record in prompt_records], dtype=int)
    if len(prompt_records) == 0:
        raise ValueError("No CAD prompts were available for linear probing.")
    if len(np.unique(labels)) < 2:
        raise ValueError("CAD linear probing requires both positive and negative prompts.")

    print(
        "CAD linear probing records:",
        f"{len(prompt_records)} prompts",
        f"({int(labels.sum())} positive, {int((labels == 0).sum())} negative)",
    )
    return prompt_records, labels


def extract_all_layer_activations(
    model,
    tokenizer,
    device,
    prompts,
    batch_size=4,
):
    """Collect last-token activations from every transformer layer using hooks."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layer_count = len(model.gpt_neox.layers)
    activations_by_layer = [[] for _ in range(layer_count)]
    current_attention_mask = None
    hooks = []

    def make_hook(layer_index):
        def hook_fn(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            last_valid_positions = current_attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(
                hidden_states.shape[0],
                device=hidden_states.device,
            )
            selected_activations = hidden_states[
                batch_indices,
                last_valid_positions,
                :,
            ]
            activations_by_layer[layer_index].append(
                selected_activations.detach().float().cpu().numpy()
            )

        return hook_fn

    for layer_index, layer in enumerate(model.gpt_neox.layers):
        hooks.append(layer.register_forward_hook(make_hook(layer_index)))

    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                end = start + len(batch_prompts)
                print(f"\rCAD prompt {end}/{len(prompts)} handled", end="", flush=True)
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                current_attention_mask = inputs["attention_mask"]
                model(**inputs)
    finally:
        for hook in hooks:
            hook.remove()
        print()

    return {
        layer_index: np.concatenate(layer_activations, axis=0)
        for layer_index, layer_activations in enumerate(activations_by_layer)
    }


def train_probe_per_layer(activations, labels, n_layers=None):
    """Train one logistic-regression probe per layer with stratified CV."""
    labels = np.asarray(labels, dtype=int)
    if n_layers is None:
        n_layers = len(activations)

    min_class_count = int(min(np.bincount(labels)))
    cv_folds = min(5, min_class_count)
    if cv_folds < 2:
        raise ValueError("At least two examples per class are required for cross-validation.")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracies = np.zeros(n_layers, dtype=float)
    std_devs = np.zeros(n_layers, dtype=float)

    for layer_index in range(n_layers):
        X = np.asarray(activations[layer_index], dtype=float)
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        )
        cv_scores = cross_val_score(
            probe,
            X,
            labels,
            cv=cv,
            scoring="accuracy",
        )
        accuracies[layer_index] = float(cv_scores.mean())
        std_devs[layer_index] = float(cv_scores.std())

    return accuracies, std_devs


def print_linear_probe_results_table(accuracies, std_devs):
    """Print layer-wise probe results as a compact table."""
    best_layer = int(np.argmax(accuracies))
    print(f"{'Layer':<8} {'CV accuracy':<14} {'Std':<10}")
    print("-" * 34)
    for layer_index, (accuracy, std_dev) in enumerate(zip(accuracies, std_devs)):
        marker = " <-- best" if layer_index == best_layer else ""
        print(f"{layer_index:<8} {accuracy:<14.4f} {std_dev:<10.4f}{marker}")
    print(
        f"\nBest layer: {best_layer} "
        f"(accuracy={accuracies[best_layer]:.4f}, std={std_devs[best_layer]:.4f})"
    )


def run_cad_linear_probe_analysis(
    model,
    tokenizer,
    device,
    dataset_path=CAD_SENTIMENT_TRAIN_PAIRED_PATH,
    max_prompts=None,
    batch_size=4,
    filename_or_path=LINEAR_PROBE_CAD_ACCURACY_PATH,
):
    """Run hook-based CAD sentiment probing and save the layer accuracy plot."""
    prompt_records, labels = load_cad_linear_probe_records(
        dataset_path=dataset_path,
        max_prompts=max_prompts,
    )
    prompts = [record["prompt"] for record in prompt_records]
    activations = extract_all_layer_activations(
        model,
        tokenizer,
        device,
        prompts,
        batch_size=batch_size,
    )
    accuracies, std_devs = train_probe_per_layer(
        activations,
        labels,
        n_layers=len(model.gpt_neox.layers),
    )
    print_linear_probe_results_table(accuracies, std_devs)
    plot_path = plot_linear_probe_layer_accuracy(
        accuracies,
        std_devs,
        sample_count=len(prompt_records),
        filename_or_path=filename_or_path,
    )
    return {
        "prompt_records": prompt_records,
        "labels": labels,
        "activations": activations,
        "accuracies": accuracies,
        "std_devs": std_devs,
        "best_layer": int(np.argmax(accuracies)),
        "best_accuracy": float(np.max(accuracies)),
        "plot_path": plot_path,
    }


def log_linear_probe_with_mlflow(probe_result, mlflow_module=None):
    """Log CAD linear-probe metrics to MLflow, or print a table if unavailable."""
    if mlflow_module is None:
        print("MLflow is not installed; keeping the printed layer table as tracking output.")
        print_linear_probe_results_table(
            probe_result["accuracies"],
            probe_result["std_devs"],
        )
        return False

    mlflow_module.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow_module.set_experiment("linear_probing_pythia410m")
    with mlflow_module.start_run(run_name="cad_sentiment_probe"):
        mlflow_module.log_param("model", MODEL_NAME)
        mlflow_module.log_param("n_examples", len(probe_result["prompt_records"]))
        for layer_index, accuracy in enumerate(probe_result["accuracies"]):
            mlflow_module.log_metric(
                f"layer_{layer_index:02d}_accuracy",
                float(accuracy),
                step=layer_index,
            )
            mlflow_module.log_metric(
                f"layer_{layer_index:02d}_std",
                float(probe_result["std_devs"][layer_index]),
                step=layer_index,
            )
        mlflow_module.log_metric("best_accuracy", probe_result["best_accuracy"])
        mlflow_module.log_metric("best_layer", probe_result["best_layer"])
    print(f"Logged CAD linear-probe run to MLflow at {MLFLOW_TRACKING_URI}.")
    return True


def print_linear_probe_reflection(probe_result):
    """Answer the notebook reflection questions in German."""
    best_layer = int(probe_result["best_layer"])
    best_accuracy = float(probe_result["best_accuracy"])
    print("1. Die Probe erreicht ihr Maximum in Schicht "
          f"{best_layer} mit einer Genauigkeit von {best_accuracy:.3f}.")
    print("2. Eine Genauigkeit von etwa 50% entspricht Zufallsniveau: "
          "die Sentiment-Information ist linear kaum dekodierbar. Eine Genauigkeit "
          "von etwa 90% bedeutet, dass positive und negative CAD-Prompts in dieser "
          "Schicht fast linear trennbar sind.")
    print("3. Probing zeigt eine Korrelation: Information ist in den Aktivierungen "
          "dekodierbar. Activation Patching ist kausaler, weil Aktivierungen "
          "gezielt ausgetauscht werden und danach gemessen wird, ob sich das "
          "Modellverhalten ändert.")
    print("4. StandardScaler ist wichtig, weil logistische Regression empfindlich "
          "auf unterschiedliche Feature-Skalen reagiert. Standardisierung macht "
          "die Dimensionen vergleichbarer und stabilisiert das Training.")
    print("5. Ein nicht-linearer Probe wie ein MLP könnte höhere Genauigkeit erreichen, "
          "ist aber schwerer interpretierbar. Er kann selbst komplexe Muster lernen, "
          "sodass unklarer wird, ob die Information einfach im Modell vorhanden ist "
          "oder erst vom Probe konstruiert wurde.")


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
        write_json_if_changed(PROBE_SPLIT_PATH, split_payload)
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
