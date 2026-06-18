from pathlib import Path


MODEL_NAME = "EleutherAI/pythia-410m"
UTILS_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = UTILS_DIR.parent
PROJECT_ROOT = ASSIGNMENT_DIR.parent
INPUT_DIR = ASSIGNMENT_DIR / "inputs"
OUTPUT_DIR = ASSIGNMENT_DIR / "output"
OUTPUT_PNG_DIR = OUTPUT_DIR / "png"
OUTPUT_JSON_DIR = OUTPUT_DIR / "json"
OUTPUT_TXT_DIR = OUTPUT_DIR / "txt"

OUTPUT_PNG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TXT_DIR.mkdir(parents=True, exist_ok=True)

POSITIVE_WORDS_PATH = OUTPUT_JSON_DIR / "hu_liu_positive_one_token_words.json"
NEGATIVE_WORDS_PATH = OUTPUT_JSON_DIR / "hu_liu_negative_one_token_words.json"
FIELD_PAIRS_PATH = INPUT_DIR / "hu_liu_sentiment_word_pairs_by_field.json"
FLAT_PAIRS_PATH = INPUT_DIR / "hu_liu_sentiment_word_pairs_flat.json"
PROMPTS_PATH = INPUT_DIR / "prompts.json"
PROBE_SPLIT_PATH = OUTPUT_JSON_DIR / "hu_liu_logistic_regression_split.json"
SENTIMENT_CHALLENGE_DIR = INPUT_DIR / "assessing_and_probing_sentiment"
SENTIMENT_CHALLENGE_ANNOTATED_PATH = SENTIMENT_CHALLENGE_DIR / "annotated.txt"
CAD_SENTIMENT_DIR = INPUT_DIR / "cad_dataset"
CAD_SENTIMENT_TRAIN_PAIRED_PATH = CAD_SENTIMENT_DIR / "train_paired.tsv"
SENTIMENT_CHALLENGE_HEATMAP_PATH = OUTPUT_PNG_DIR / "sentiment_challenge_category_heatmap.png"
FILTERED_SENTIMENT_CHALLENGE_HEATMAP_PATH = (
    OUTPUT_PNG_DIR / "sentiment_challenge_filtered_category_heatmap.png"
)
SENTIMENT_CHALLENGE_TOP_TOKENS_PATH = (
    OUTPUT_JSON_DIR / "sentiment_challenge_prompt_top10_tokens.json"
)
SENTIMENT_CHALLENGE_TOP_TOKENS_REPORT_PATH = (
    OUTPUT_TXT_DIR / "sentiment_challenge_prompt_top10_tokens_report.txt"
)
SENTIMENT_CHALLENGE_CATEGORY_CONFUSION_PATH = (
    OUTPUT_PNG_DIR / "sentiment_challenge_category_confusion_matrices.png"
)
SENTIMENT_CHALLENGE_CATEGORY_ACCURACY_PATH = (
    OUTPUT_PNG_DIR / "sentiment_challenge_category_accuracy.png"
)
LOGIT_LENS_TARGET_PROBABILITY_PATH = OUTPUT_PNG_DIR / "logit_lens_target_probability.png"
LOGIT_LENS_POSITIVE_SENTIMENT_MASS_PATH = (
    OUTPUT_PNG_DIR / "logit_lens_positive_hu_liu_topk_mass.png"
)
LOGIT_LENS_NEGATIVE_SENTIMENT_MASS_PATH = (
    OUTPUT_PNG_DIR / "logit_lens_negative_hu_liu_topk_mass.png"
)
LOGIT_LENS_PROMPT_LOGIT_SCORES_PATH = (
    OUTPUT_PNG_DIR / "logit_lens_prompt_hu_liu_logit_scores.png"
)
LOGIT_LENS_PROMPT_LOGIT_DIFFERENCE_PATH = (
    OUTPUT_PNG_DIR / "logit_lens_prompt_hu_liu_logit_difference.png"
)
LOGIT_LENS_CAD_LOGIT_DIFFERENCE_AGGREGATE_PATH = (
    OUTPUT_PNG_DIR / "logit_lens_cad_logit_difference_aggregate.png"
)
LOGIT_LENS_POSITIVE_HEATMAP_PATH = OUTPUT_PNG_DIR / "logit_lens_positive_topk_heatmap.png"
LOGIT_LENS_NEGATIVE_HEATMAP_PATH = OUTPUT_PNG_DIR / "logit_lens_negative_topk_heatmap.png"

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
