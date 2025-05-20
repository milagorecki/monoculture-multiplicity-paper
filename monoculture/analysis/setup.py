from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier

RESULTS_ROOT_DIR = Path("./results/")
RESULTS_CSV_SAME_PROMPT = RESULTS_ROOT_DIR / "overview_results_by_prompt_style.csv"
RESULTS_CSV_VARY_PROMPT = RESULTS_ROOT_DIR / "overview_results_variations.csv"
FIGURES_ROOT_DIR = RESULTS_ROOT_DIR / "figures/"


ACS_TASKS = (
    "ACSIncome",
    "ACSEmployment",
    "ACSMobility",
    "ACSTravelTime",
    "ACSPublicCoverage",
)

TABLESHIFT_TASKS = (
    "BRFSS_Diabetes",
    "BRFSS_Blood_Pressure",
)

LLM_MODELS = [
    # Google Gemma2 models
    "google/gemma-2b",
    "google/gemma-1.1-2b-it",
    "google/gemma-7b",
    "google/gemma-1.1-7b-it",
    #
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b",
    "google/gemma-2-27b-it",
    #
    # Meta Llama3 models
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    #
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    #
    "meta-llama/Meta-Llama-3.2-1B",
    "meta-llama/Meta-Llama-3.2-1B-Instruct",
    "meta-llama/Meta-Llama-3.2-3B",
    "meta-llama/Meta-Llama-3.2-3B-Instruct",
    #
    "meta-llama/Meta-Llama-3.3-70B-Instruct",
    #
    # Mistral AI models
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mistralai/Mistral-Small-24B-Base-2501",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    #
    # Yi models
    "01-ai/Yi-6B",
    "01-ai/Yi-6B-Chat",
    "01-ai/Yi-34B",
    "01-ai/Yi-34B-Chat",
    #
    # Qwen2 models
    # "Qwen/Qwen2-1.5B",
    # "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-72B",
    "Qwen/Qwen2-72B-Instruct",
    #
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-72B-Instruct",
    #
    # OLMo models
    "allenai/OLMo-1B-0724-hf",
    "allenai/OLMo-1B-hf",
    "allenai/OLMo-7B-0724-hf",
    "allenai/OLMo-7B-hf",
    #
    "allenai/OLMo-7B-Instruct-hf",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-7B-Instruct",
    # GPT models
    "gpt-4.1",
    "gpt-3.5-turbo-0125",
]

model_families_coarse = sorted(
    [
        "Gemma",
        "Llama",
        "Mistral",
        "OLMo",
        "Qwen",
        "Yi",
    ]
)  # "GPT",
model_families = sorted(
    [
        "Gemma 2",
        "Gemma",
        # "GPT",
        # "Llama-3.3",
        "Llama 3.2",
        "Llama 3.1",
        "Llama 3",
        "Mistral",
        "OLMo",
        "Qwen",
        "Yi",
    ]
)

BASELINES = {
    "Constant": DummyClassifier(strategy="prior"),
    "LogisticRegression": LogisticRegression(),
    "GBM": HistGradientBoostingClassifier(),
    "XGBoost": XGBClassifier(),
    "NN": MLPClassifier(),
}
BASELINE_RESULTS_PATH = Path("./results/baselines")


# --------------------------------------------------
# Prompting Changes
# --------------------------------------------------
num_shots = [0, 10]

formats = ["bullet", "text", "comma"]  # , "textbullet"
connectors = ["is", "=", ":"]
granularities = ["original", "low"]
feature_order = [
    "AGEP,COW,SCHL,MAR,OCCP,POBP,RELP,WKHP,SEX,RAC1P",  # original
    "RAC1P,WKHP,AGEP,SCHL,MAR,SEX,RELP,POBP,COW,OCCP",
    "WKHP,OCCP,RAC1P,MAR,AGEP,RELP,SCHL,POBP,COW,SEX",
    "AGEP,SCHL,OCCP,MAR,COW,WKHP,RAC1P,RELP,SEX,POBP",
    "RAC1P,SEX,WKHP,RELP,POBP,OCCP,MAR,SCHL,COW,AGEP",  # reversed
]
map_feature_order_to_short = dict(
    zip(feature_order, ["default", "rand 1", "rand 2", "rand 3", " reversed"])
)
map_short_to_feature_order = {v: k for k, v in map_feature_order_to_short.items()}
variations = {
    "feature_order": list(map(lambda o: map_feature_order_to_short[o], feature_order)),
    "format": formats,
    "connector": connectors,
    "granularity": granularities,
}
