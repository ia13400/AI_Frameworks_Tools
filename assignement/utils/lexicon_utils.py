import json
import re
from pathlib import Path

import nltk
import torch

from config import INPUT_DIR, MODEL_NAME, NEGATIVE_WORDS_PATH, POSITIVE_WORDS_PATH


NLTK_DATA_DIR = INPUT_DIR / "nltk_data"
OPINION_LEXICON_DIR = NLTK_DATA_DIR / "corpora" / "opinion_lexicon"


def read_hu_liu_word_file(filename: str):
    """Read one Hu & Liu word-list file and ignore comments/blank lines."""
    path = OPINION_LEXICON_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Hu & Liu word file not found: {path}")

    words = []
    for line in path.read_text(encoding="ISO-8859-1").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        words.append(line)
    return words


def import_hu_liu_dataset():
    """Ensure the Hu & Liu Opinion Lexicon is available locally and load it."""
    nltk_data_path = str(NLTK_DATA_DIR.resolve())
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)

    positive_path = OPINION_LEXICON_DIR / "positive-words.txt"
    negative_path = OPINION_LEXICON_DIR / "negative-words.txt"

    if not positive_path.exists() or not negative_path.exists():
        NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Hu & Liu Opinion Lexicon to: {NLTK_DATA_DIR}")
        nltk.download("opinion_lexicon", download_dir=nltk_data_path, quiet=True)
    else:
        print(f"Hu & Liu Opinion Lexicon already available in: {NLTK_DATA_DIR}")

    positive = read_hu_liu_word_file("positive-words.txt")
    negative = read_hu_liu_word_file("negative-words.txt")

    print(f"Positive lexicon words: {len(positive)}")
    print(f"Negative lexicon words: {len(negative)}")
    return positive, negative


def load_hu_liu_opinion_lexicon():
    """Load the Hu & Liu Opinion Lexicon through NLTK."""
    return import_hu_liu_dataset()


def is_plain_word(word: str) -> bool:
    """Keep only alphabetic words so punctuation and phrases do not enter the analysis."""
    return re.fullmatch(r"[A-Za-z]+", word) is not None


def single_token_for_running_text(word: str, tokenizer):
    """Return the token id if the word is represented by one token in running text."""
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(token_ids) != 1:
        return None
    return token_ids[0]


def build_sentiment_word_records(tokenizer):
    """Create Hu & Liu word records that are represented by exactly one model token."""
    positive_lexicon, negative_lexicon = load_hu_liu_opinion_lexicon()
    sentiment_words = []

    for sentiment, lexicon_words in [
        ("positive", positive_lexicon),
        ("negative", negative_lexicon),
    ]:
        for word in lexicon_words:
            word = word.lower()
            if not is_plain_word(word):
                continue

            token_id = single_token_for_running_text(word, tokenizer)
            if token_id is None:
                continue

            sentiment_words.append(
                {
                    "word": word,
                    "sentiment": sentiment,
                    "token_id": int(token_id),
                    "token": tokenizer.decode([token_id]),
                }
            )

    if len(sentiment_words) < 2:
        raise ValueError("Not enough one-token Hu & Liu opinion words found.")

    return sentiment_words


def export_sentiment_words_json(items, sentiment: str, output_path: Path) -> None:
    """Write one-token sentiment words together with source and filtering metadata."""
    payload = {
        "source": {
            "name": "Hu & Liu Opinion Lexicon",
            "nltk_corpus": "nltk.corpus.opinion_lexicon",
            "description": (
                "Positive and negative opinion word lists introduced by Minqing Hu "
                "and Bing Liu for opinion mining / sentiment analysis."
            ),
        },
        "filtering_method": {
            "plain_word_filter": "Keep only entries matching the regex [A-Za-z]+.",
            "token_filter": (
                "Tokenize each word as running text by prepending one leading space, "
                "then keep only words where tokenizer.encode(' ' + word, "
                "add_special_tokens=False) returns exactly one token id."
            ),
            "model_tokenizer": MODEL_NAME,
            "sentiment_label": sentiment,
        },
        "count": len(items),
        "words": [
            {
                "word": item["word"],
                "token": item["token"],
                "token_id": int(item["token_id"]),
                "sentiment": item["sentiment"],
            }
            for item in sorted(items, key=lambda item: item["word"])
        ],
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare_hu_liu_sentiment_words(tokenizer, embedding_matrix):
    """Build sentiment records, export JSON files, and return vectors for plotting."""
    sentiment_words = build_sentiment_word_records(tokenizer)
    positive_words = [item for item in sentiment_words if item["sentiment"] == "positive"]
    negative_words = [item for item in sentiment_words if item["sentiment"] == "negative"]

    export_sentiment_words_json(positive_words, "positive", POSITIVE_WORDS_PATH)
    export_sentiment_words_json(negative_words, "negative", NEGATIVE_WORDS_PATH)

    token_ids = torch.tensor([item["token_id"] for item in sentiment_words], dtype=torch.long)
    vectors = embedding_matrix[token_ids].float().numpy()

    print(f"\nSaved JSON: {POSITIVE_WORDS_PATH}")
    print(f"Saved JSON: {NEGATIVE_WORDS_PATH}")
    print(f"\nHu & Liu words kept:       {len(sentiment_words)}")
    print(f"Positive one-token words:  {len(positive_words)}")
    print(f"Negative one-token words:  {len(negative_words)}")

    return sentiment_words, positive_words, negative_words, vectors


def normalized_word_key(word: str) -> str:
    """Normalize only for lookup; the original word is kept for printing."""
    return "".join(word.split()).lower()


def build_hu_liu_lookup(positive_words, negative_words):
    lookup = {}
    for item in positive_words + negative_words:
        lookup[normalized_word_key(item["word"])] = {
            "word": item["word"],
            "sentiment": item["sentiment"],
        }
    return lookup
