import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from config import MODEL_NAME, OUTPUT_PNG_DIR
from transformers import AutoModelForCausalLM, AutoTokenizer


matplotlib.rcParams["figure.dpi"] = 100


def section(num: int, chapter_name: str) -> None:
    """Print a visible chapter separator."""
    width = 72
    print("\n" + "=" * width)
    print(f"  Kapitel {num}: {chapter_name}")
    print("=" * width)


def setup_environment() -> torch.device:
    section(1, "Installation & Imports")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch-Version : {torch.__version__}")
    print(f"Gerät           : {device}")
    return device


def load_model_and_tokenizer(device: torch.device):
    section(2, "Modell und Tokenizer laden")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    print("\nModell-Architektur:")
    print(model)
    return model, tokenizer


def print_architecture_overview(model) -> None:
    section(3, "Architekturübersicht")
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    print("\n--- Architektur-Parameter ---")
    print(f"Modelltyp              : {model.config.model_type}")
    print(f"Versteckte Größe       : {model.config.hidden_size}")
    print(f"Schichten              : {model.config.num_hidden_layers}")
    print(f"Attention-Köpfe        : {model.config.num_attention_heads}")
    print(f"Kopfdimension          : {head_dim}")
    print(f"Vokabulargröße         : {model.config.vocab_size}")
    print(f"Max. Positionen        : {model.config.max_position_embeddings}")


def count_params(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def analyze_parameters(model) -> None:
    section(4, "Parameter-Zählung")

    total_params = count_params(model)
    token_embedding_params = count_params(model.gpt_neox.embed_in)
    token_unembedding_params = count_params(model.embed_out)
    attention_params = sum(
        count_params(model.gpt_neox.layers[index].attention)
        for index in range(model.config.num_hidden_layers)
    )
    mlp_params = sum(
        count_params(model.gpt_neox.layers[index].mlp)
        for index in range(model.config.num_hidden_layers)
    )

    print(f"Total parameters: {total_params:,}")
    print(
        f"Token Embedding Params:   {token_embedding_params:,}  "
        f"({token_embedding_params / total_params * 100:.2f}%)"
    )
    print(
        f"Token Unembedding Params: {token_unembedding_params:,}  "
        f"({token_unembedding_params / total_params * 100:.2f}%)"
    )
    print(f"\nEmbed_in  data_ptr:  {model.gpt_neox.embed_in.weight.data_ptr()}")
    print(f"Embed_out data_ptr:  {model.embed_out.weight.data_ptr()}")
    print(
        "Embedding and Unembedding weights are tied?",
        model.gpt_neox.embed_in.weight.data_ptr() == model.embed_out.weight.data_ptr(),
    )
    

    print("\nFirst layer parameter breakdown:")
    for name, parameter in model.gpt_neox.layers[0].named_parameters():
        print(f"  {name:40s} {parameter.numel():,}")

    labels = ["Attention\n(x24 layers)", "MLP\n(x24 layers)", "Embedding", "Unembedding"]
    values = [attention_params, mlp_params, token_embedding_params, token_unembedding_params]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bars = ax1.bar(labels, [value / 1e6 for value in values], color=colors)
    ax1.set_ylabel("Parameters (millions)")
    ax1.set_title("Parameter Count by Component")
    for bar, value in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value / 1e6:.1f}M",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax2.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
    ax2.set_title("Parameter Distribution")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_DIR / "parameterverteilung.png", dpi=120)
    plt.close()

    print(f"\n{'Component':<25} {'Params':>12}   {'Share':>6}")
    print("-" * 48)
    for label, value in zip(labels, values):
        name = label.replace("\n", " ")
        print(f"{name:<25} {value:>12,}   {value / total_params * 100:>5.1f}%")


def top_k_predictions(prompt: str, model, tokenizer, device: torch.device, k: int = 5) -> None:
    """Print the top-k next-token predictions for one prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    next_token_logits = outputs.logits[0, -1, :]
    probabilities = torch.softmax(next_token_logits, dim=-1)
    top_probabilities, top_indices = torch.topk(probabilities, k)

    print(f"\nPrompt: {prompt}")
    print("-" * 60)

    for rank, (token_id, probability) in enumerate(
        zip(top_indices, top_probabilities),
        start=1,
    ):
        token = tokenizer.decode(token_id)
        print(
            f"{rank}. Token: {repr(token):15s} | "
            f"Token-ID: {token_id.item():5d} | "
            f"Wahrscheinlichkeit: {probability.item():.4f}"
        )


def run_top_k_predictions(prompts, model, tokenizer, device: torch.device, k: int = 5) -> None:
    section(5, "Top-K Vorhersagen")
    for prompt in prompts:
        top_k_predictions(prompt, model, tokenizer, device, k=k)
