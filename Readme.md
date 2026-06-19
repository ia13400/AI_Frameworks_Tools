# AI Frameworks & Tools — Mechanistische Interpretierbarkeit von Sentiment in Pythia-410M

Semesterprojekt im Modul **AI Frameworks & Tools**. Untersucht wird, wo und wie das offene Sprachmodell [Pythia-410M](https://huggingface.co/EleutherAI/pythia-410m) (EleutherAI) Sentiment intern repräsentiert — mit Methoden der mechanistischen Interpretierbarkeit (Residual-Stream-Analyse, Logit Lens, Attention-Analyse, lineares Probing, Activation Patching).

## Inhaltsverzeichnis

- [Team](#team)
- [Repository-Struktur](#repository-struktur)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Datengrundlage](#datengrundlage)
- [Wissenschaftliche Arbeit](#wissenschaftliche-arbeit)
- [Lizenz](#lizenz)

## Team

| Rolle | Name |
|---|---|
| Betreuer | Dr. Sigurd Schacht |
| Autor | Alireza Roozitalab |
| Autor | Daniel Durst-Claus |
| Autor | Islam Abdalla |

## Repository-Struktur

```text
AI_Frameworks_Tools/
├── Paper2/                  # Finale wissenschaftliche Arbeit (LaTeX, aktueller Stand)
├── Paper/                   # Frühere Fassung der Arbeit
├── assignement/              # Analyse-Pipeline des Projekts
│   ├── notebooks/             # 01–05: Model Inspection, Logit Lens, Attention, Probing, Patching
│   ├── utils/                  # Python-Module hinter den Notebooks (Modell, Lexikon, Plots, ...)
│   ├── inputs/                  # Rohdaten: Hu-&-Liu-Lexikon, CAD-Paare, Sentiment-Challenge-Sätze
│   └── output/                   # Generierte Abbildungen sowie JSON-/TXT-Reports
├── notebooks/                # Frühere Explorations-Notebooks (inkl. optionalem ONNX-Export)
├── app/                       # Streamlit-Dashboard zur interaktiven Exploration
├── data/                       # Zusätzlicher Tweet-Datensatz für die frühen Explorations-Notebooks
├── pyproject.toml / uv.lock   # Python-Umgebung (verwaltet mit uv)
└── Readme.md
```

## Voraussetzungen

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/) als Paketmanager
- Optional: CUDA-fähige GPU (PyTorch wird über den `pytorch-cu124`-Index installiert)
- Für den Build der wissenschaftlichen Arbeit zusätzlich `texlive-full`, `latexmk`, `biber` — siehe [Paper2/readme.md](Paper2/readme.md)

## Installation

```bash
git clone git@github.com:ia13400/AI_Frameworks_Tools.git
cd AI_Frameworks_Tools
uv sync
```

## Verwendung

### Analyse-Notebooks

Die eigentliche Untersuchung liegt in `assignement/notebooks/` und folgt der Reihenfolge der fünf Werkzeuge (Korrelation → Kausalität):

```bash
uv run jupyter notebook assignement/notebooks
```

| Notebook | Inhalt |
|---|---|
| `01_Model_Inspection.ipynb` | Architektur und Parameterverteilung von Pythia-410M |
| `02_Logit_Lens.ipynb` | Schichtweise Entstehung von Sentiment im Residual Stream |
| `03_Attention_Analysis.ipynb` | Aufmerksamkeit der Attention Heads auf sentimenttragende Wörter |
| `04_Linear_Probing.ipynb` | Lineare Trennbarkeit von Sentiment im Embedding-Raum |
| `05_Activation_Patching.ipynb` | Kausale Überprüfung einzelner Schichten/Köpfe |

### Dashboard

```bash
uv run streamlit run app/dashboard.py
```

Interaktive Exploration von Logit Lens, Attention und Patching für selbst gewählte Prompts.

## Datengrundlage

| Datensatz | Quelle | Verwendung |
|---|---|---|
| Hu-&-Liu Opinion Lexicon | Hu, M., & Liu, B. (2004). *Mining and Summarizing Customer Reviews*. KDD. [Paper](https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf) · [Datensatz](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html) | Identifikation sentimenttragender Token; Grundlage für Embedding-Analysen, lineares Probing und logitbasierte Sentiment-Scores |
| Sentiment Challenge Dataset | Barnes, J., Øvrelid, L., & Velldal, E. (2019). *Sentiment Analysis Is Not Solved! Assessing and Probing Sentiment Classification*. Proceedings of the 2019 ACL Workshop BlackboxNLP, S. 12–23. | Verhaltensbasierte Bewertung anhand sprachlich schwieriger Fälle (Negation, Ironie, Idiome, ...) |
| Counterfactually Augmented Data (CAD) | Kaushik, D., Hovy, E., & Lipton, Z. C. (2020). *Learning the Difference that Makes a Difference with Counterfactually-Augmented Data*. ICLR. [Paper](https://openreview.net/pdf?id=Sklgs0NFvr) · [GitHub](https://github.com/acmi-lab/counterfactually-augmented-data) | Aggregierte Logit-Lens- und Attention-Analyse über kontextgleiche positiv/negativ-Paare |

> Der zusätzliche Tweet-Datensatz in `data/` (`prompts.csv`, `annotations.csv`) wurde nur in den frühen Explorations-Notebooks (`notebooks/`) verwendet und fließt nicht in die finale Arbeit ein.

## Wissenschaftliche Arbeit

Die aktuelle Fassung liegt in [`Paper2/`](Paper2/) (kompiliertes [`main.pdf`](Paper2/main.pdf)), eine frühere Fassung in [`Paper/`](Paper/). Build-Anleitung (WSL2, LaTeX, Glossar, Bibliographie) siehe [`Paper2/readme.md`](Paper2/readme.md).

## Lizenz

Akademisches Hochschulprojekt ohne Lizenzvergabe. Nutzung der Datensätze unterliegt den jeweiligen Lizenzen der Originalquellen (siehe Tabelle oben).
