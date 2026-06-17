from pathlib import Path


MODEL_NAME = "EleutherAI/pythia-410m"
UTILS_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = UTILS_DIR.parent
PROJECT_ROOT = ASSIGNMENT_DIR.parent
INPUT_DIR = ASSIGNMENT_DIR / "inputs"
OUTPUT_DIR = ASSIGNMENT_DIR / "output"
OUTPUT_PNG_DIR = OUTPUT_DIR / "png"
OUTPUT_JSON_DIR = OUTPUT_DIR / "json"

OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

POSITIVE_WORDS_PATH = OUTPUT_JSON_DIR / "hu_liu_positive_one_token_words.json"
NEGATIVE_WORDS_PATH = OUTPUT_JSON_DIR / "hu_liu_negative_one_token_words.json"
FIELD_PAIRS_PATH = INPUT_DIR / "hu_liu_sentiment_word_pairs_by_field.json"
FLAT_PAIRS_PATH = INPUT_DIR / "hu_liu_sentiment_word_pairs_flat.json"
SENTIMENT_PROMPTS_PATH = INPUT_DIR / "sentiment_prompts.json"
PROMPTS_PATH = INPUT_DIR / "prompts.json"
PROBE_SPLIT_PATH = OUTPUT_JSON_DIR / "hu_liu_logistic_regression_split.json"

EXAMPLE_PROMPTS = [
    "The food was delicious and the service was",
    "The food was disgusting and the service was",
    "I love this movie and I feel very",
    "I hate this movie and I feel very",
]

NEIGHBOR_TARGET_WORDS = ["good", "bad", "delicious", "disgusting"]

ARITHMETIC_EXPERIMENTS = [
    "good + excellent - bad",
    "terrible + sad - great",
    "excellent - good",
    "terrible - bad",
    "popular - unpopular",
]
