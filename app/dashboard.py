"""sentiment analysis.

Team 4 · AI Frameworks & Tools
Starten:  streamlit run app.py

"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="Interpretability Dashboard", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    # cache_resource: Modell wird EINMAL geladen, nicht bei jedem Rerun (Stolperstein!)
    MODEL_NAME = "EleutherAI/pythia-410m"
   
    # Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Modell laden
    model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",   # wichtig für Attention-Ausgabe
    torch_dtype=torch.float32
    ).to(device)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

st.title("Team 4 : Sentiment Analysis Dashboard")
st.caption(
    "Vorlage fuer euer Projekt-Dashboard: Verhalten, Attention und Patching in einer App. "
)

st.sidebar.header("Einstellungen")
prompt = st.sidebar.text_input("Prompt", "The movie was good. The sentiment is")
top_k = st.sidebar.slider("Top-k Vorhersagen", 3, 15, 8)

tab_pred, tab_attn, tab_patch = st.tabs(["Logit Lens", "Attention", "Patching"])


@st.cache_data
def run_forward(text: str):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs, output_attentions=True, output_hidden_states=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

    probs = torch.softmax(out.logits[0, -1], dim=-1).cpu()

    if not out.attentions or out.attentions[0] is None:
        raise RuntimeError(
            "Keine Attention-Gewichte erhalten. Das Modell muss mit "
            "attn_implementation='eager' geladen werden."
        )

    attentions = torch.stack([
        att.cpu() for att in out.attentions
    ]).squeeze(1)
    hidden_states = out.hidden_states

    return tokens, probs, attentions , hidden_states

tokens, probs, attentions, hidden_states = run_forward(prompt)

with tab_pred:
    st.subheader("Logit Lens")
    layer_num = st.slider(
        "Layer",
        0,
        model.config.num_hidden_layers,
        model.config.num_hidden_layers
    )
    with torch.no_grad():
        hidden = hidden_states[layer_num][0, -1]
        lm_head = model.embed_out.weight
        logits = hidden @ lm_head.T
        layer_probs = torch.softmax(logits, dim=-1)
        values, indices = torch.topk(layer_probs, top_k)

    df = pd.DataFrame({
        "Token": [
            repr(tokenizer.decode([i.item()]))
            for i in indices
        ],
        "Wahrscheinlichkeit": values.cpu().numpy(),
    })
    fig = px.bar(
        df,
        x="Wahrscheinlichkeit",
        y="Token",
        orientation="h"
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)

with tab_attn:
    st.subheader("Attention-Muster (live berechnet)")
    col1, col2 = st.columns(2)
    layer = col1.slider("Layer", 0, model.config.num_hidden_layers- 1, 5)
    head = col2.slider("Head", 0,model.config.num_attention_heads- 1, 0)
    fig = px.imshow(
    attentions[layer, head].numpy(),
    x=tokens,
    y=tokens,
    labels={
        "x": "Key (wird beachtet)",
        "y": "Query (beachtet)",
        "color": "Attention"
    },
    color_continuous_scale="Blues",
    title=f"Layer {layer}, Head {head}",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("Erinnerung aus Block 4: Attention zeigt, wo Information fliessen KOENNTE -- kein kausaler Nachweis.")

with tab_patch:
    st.subheader("Activation Patching: Recovery Scores")
    candidates = [
        Path(__file__).parent.parent / "patching_results.npz",
        Path("../notebooks/patching_results.npz"),
    ]
    npz_path = next((p for p in candidates if p.exists()), None)
    if npz_path is None:
        st.warning(
            "Keine patching_results.npz gefunden. Fuehrt zuerst das Demo-Notebook "
            "01_Patching_mit_Hooks_Demo.ipynb aus -- es speichert die Ergebnisse fuer dieses Tab."
        )
    else:
        data = np.load(npz_path, allow_pickle=True)
        recovery = data["recovery"]
        patch_tokens = list(data["tokens"])
        fig = px.imshow(
            recovery,
            x=patch_tokens,
            labels={"x": "Token-Position", "y": "Layer", "color": "Recovery"},
            color_continuous_scale="RdBu_r",
            zmin=-0.2, zmax=1.0,
            title="Recovery Score (1.0 = Clean-Verhalten wiederhergestellt)",
        )
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("P(Target | clean)", f"{float(data['p_clean']):.4f}")
        c2.metric("P(Target | corrupted)", f"{float(data['p_corrupted']):.4f}")
        c3.metric("Max. Recovery", f"{recovery.max():.2f}")
        st.success("Das ist euer kausales Kernergebnis -- dieses Tab gehoert in jedes Projekt-Dashboard.")
