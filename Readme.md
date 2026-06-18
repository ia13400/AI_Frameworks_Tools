# Referenzen und Datensätze

## Hu & Liu Opinion Lexicon

Hu, M., & Liu, B. (2004). *Mining and Summarizing Customer Reviews*. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD).

- Paper Link: https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf
- Datensatz Link: https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
- NLTK Distribution: https://www.nltk.org/api/nltk.corpus.reader.opinion_lexicon.html
- Verwendung: Das Hu-&-Liu-Opinion-Lexikon enthält Listen positiver und negativer Wörter. In dieser Arbeit wird es verwendet, um sentimentbezogene Tokens zu identifizieren und daraus Logit-basierte Sentiment-Scores über die Transformer-Schichten von Pythia-410M zu berechnen.

---

## Sentiment Challenge Dataset

Sharma, A., & Mittal, A. (2019). *Compositionality and Sentiment in Sentiment Analysis*. Proceedings of the 10th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis (WASSA).

- Paper Link: https://aclanthology.org/W19-4802/
- PDF Link: https://aclanthology.org/W19-4802.pdf
- GitHub Link: https://github.com/EleutherAI/sentiment-challenge
- Verwendung: Der Sentiment Challenge Datensatz wurde entwickelt, um die Fähigkeit von Sprachmodellen zur Verarbeitung sentimentbezogener Informationen systematisch zu untersuchen. Der Datensatz enthält gezielt konstruierte Beispiele mit positiven und negativen Sentiment-Ausprägungen sowie sprachlichen Herausforderungen wie Kompositionalität, Negation und Kontextabhängigkeit. In dieser Arbeit wird der Datensatz verwendet, um die Entwicklung sentimentbezogener Repräsentationen innerhalb der Transformer-Schichten von Pythia-410M zu analysieren.

---

## Counterfactually Augmented Data (CAD)

Kaushik, D., Hovy, E., & Lipton, Z. C. (2020). *Learning the Difference that Makes a Difference with Counterfactually-Augmented Data*. International Conference on Learning Representations (ICLR).

- Paper Link: https://openreview.net/pdf?id=Sklgs0NFvr
- GitHub Link: https://github.com/acmi-lab/counterfactually-augmented-data
- Verwendung: Der Datensatz enthält originale und gegenfaktische Sentiment-Beispiele, die sich nur durch minimale Änderungen sentimenttragender Wörter unterscheiden. Dadurch eignet er sich besonders für die Untersuchung von Sentiment-Repräsentationen, da positive und negative Beispiele nahezu identischen Kontext besitzen. In dieser Arbeit wird der Datensatz verwendet, um die Entwicklung sentimentbezogener Repräsentationen mittels Logit Lens sowie die kausale Bedeutung einzelner Modellkomponenten durch Activation Patching zu analysieren.
