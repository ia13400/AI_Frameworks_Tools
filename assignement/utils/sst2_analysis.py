import numpy as np

from model_utils import section


def run_sst2_inspection():
    section(9, "Stanford Sentiment Treebank SST-2")

    print(
        "\nSST-2 is a sentence-level sentiment dataset. Each example contains a "
        "sentence and a binary label: 0 = negative, 1 = positive."
    )

    try:
        from datasets import load_dataset

        sst2 = None
        dataset_errors = []
        for dataset_name in ["nyu-mll/glue", "glue"]:
            try:
                sst2 = load_dataset(dataset_name, "sst2")
                print(f"\nLoaded SST-2 from Hugging Face dataset: {dataset_name}")
                break
            except Exception as dataset_exc:
                dataset_errors.append(
                    f"{dataset_name}: {type(dataset_exc).__name__}: {dataset_exc}"
                )

        if sst2 is None:
            raise RuntimeError("; ".join(dataset_errors))

        print("\nSST-2 splits:")
        for split_name, split_data in sst2.items():
            print(f"- {split_name:<10} examples={len(split_data)}")

        print("\nSST-2 features:")
        print(sst2["train"].features)

        print("\nExample rows from the training split:")
        for row in sst2["train"].select(range(8)):
            label_name = "positive" if row["label"] == 1 else "negative"
            print(
                f"label={row['label']} ({label_name:<8}) "
                f"idx={row['idx']:<6} sentence={row['sentence']}"
            )

        train_labels = np.array(sst2["train"]["label"])
        validation_labels = np.array(sst2["validation"]["label"])

        print("\nLabel distribution:")
        print(
            "train      : "
            f"positive={int((train_labels == 1).sum())}, "
            f"negative={int((train_labels == 0).sum())}"
        )
        print(
            "validation : "
            f"positive={int((validation_labels == 1).sum())}, "
            f"negative={int((validation_labels == 0).sum())}"
        )

        print("\nHow SST-2 can be used in this project:")
        print(
            "1. Compare word-level Hu & Liu sentiment with sentence-level SST-2 "
            "sentiment."
        )
        print(
            "2. Build sentence embeddings from Pythia input-token embeddings, for "
            "example by averaging token embeddings for each sentence."
        )
        print(
            "3. Train the same logistic-regression linear probe on SST-2 sentence "
            "embeddings and compare it with the Hu & Liu word-level probe."
        )
        print(
            "4. Analyze mistakes: SST-2 contains context, negation, and phrasing, "
            "while Hu & Liu contains isolated sentiment words."
        )

    except Exception as exc:
        print("\nCould not load SST-2 with Hugging Face datasets.")
        print(f"Reason: {type(exc).__name__}: {exc}")
        print(
            "Install/update dependencies with `uv sync`, then run the script again. "
            "The first successful run may also need internet access to download SST-2."
        )
