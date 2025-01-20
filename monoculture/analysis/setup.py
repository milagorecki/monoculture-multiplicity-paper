# defining some global variables or info
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier

ACS_TASKS = (  # all ACS prediction tasks
    "ACSIncome",
    "ACSEmployment",
    # "ACSMobility",
    "ACSTravelTime",
    "ACSPublicCoverage",
)

TABLESHIFT_TASKS = (  # all table shift tasks
    "BRFSS_Diabetes",
    #"BRFSS_Blood_Pressure"
)

LLM_MODELS = [  # LLMs to evaluate
    # Google Gemma2 models
    "google/gemma-2b",
    "google/gemma-1.1-2b-it",
    "google/gemma-7b",
    "google/gemma-1.1-7b-it",
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b",
    "google/gemma-2-27b-it",
    # Meta Llama3 models
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.2-1B",
    "meta-llama/Meta-Llama-3.2-1B-Instruct",
    "meta-llama/Meta-Llama-3.2-3B",
    "meta-llama/Meta-Llama-3.2-3B-Instruct",
    # Mistral AI models
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    # Yi models
    "01-ai/Yi-34B",
    "01-ai/Yi-34B-Chat",
    "01-ai/Yi-6B-Chat",
    # Qwen2 models
    "Qwen/Qwen2-1.5B",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-72B",
    "Qwen/Qwen2-72B-Instruct",
    # Tabula
    "mlfoundations/tabula-8b",
    # OLMo
    "allenai/OLMo-1B-0724-hf",
    "allenai/OLMo-1B-hf",
    "allenai/OLMo-7B-0724-hf",
    "allenai/OLMo-7B-hf",
    "allenai/OLMo-7B-Instruct-hf",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-7B-Instruct",
]

# Baselines
baselines = {
    "Constant": DummyClassifier(strategy="prior"),
    "LR": LogisticRegression(),
    "GBM": HistGradientBoostingClassifier(),
    "XGBoost": XGBClassifier(),
}
BASELINE_RESULTS_PATH = Path("./results/baselines")


# What changes were made to the prompts
prompt_styles = ["bullet", "text"]
prompt_connectors = ["is", ":", "="]
prompt_connectors_extended = ["is", ":", "=", "text"]
shots = [0, 3, 5]
