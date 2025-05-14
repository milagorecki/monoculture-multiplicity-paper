import os
import re
from .setup import (
    ACS_TASKS,
    TABLESHIFT_TASKS,
    model_families,
    variations,
    map_feature_order_to_short,
)  # LLM_MODELS, BASELINE_RESULTS_PATH,
import folktexts
from folktexts._io import load_json  # , save_json
from folktexts.llm_utils import get_model_size_B
from folktexts.acs import ACSTaskMetadata, ACSDataset
from folktexts.ts import TableshiftBRFSSTaskMetadata, TableshiftBRFSSDataset

import torch
import pandas as pd
from pathlib import Path
import logging

# --------------------------------------
# Utils
# --------------------------------------


def model_to_key(str: str):
    return str.replace("/", "--")


def key_to_model(str: str):
    return str.replace("--", "/")


def prettify_model_name(model_hf_name: str) -> str:
    """Get prettified version of the given model name."""
    dct = {
        # Google Gemma models
        "google/gemma-1.1-2b-it": "Gemma 2B (it)",
        "google/gemma-1.1-7b-it": "Gemma 7B (it)",
        "google/gemma-2b": "Gemma 2B",
        "google/gemma-7b": "Gemma 7B",
        "google/gemma-2-9b": "Gemma 2 9B",
        "google/gemma-2-9b-it": "Gemma 2 9B (it)",
        "google/gemma-2-27b": "Gemma 2 27B",
        "google/gemma-2-27b-it": "Gemma 2 27B (it)",
        # Meta Llama models
        "meta-llama/Meta-Llama-3-70B": "Llama 3 70B",
        "meta-llama/Meta-Llama-3-70B-Instruct": "Llama 3 70B (it)",
        "meta-llama/Meta-Llama-3-8B": "Llama 3 8B",
        "meta-llama/Meta-Llama-3-8B-Instruct": "Llama 3 8B (it)",
        "meta-llama/Meta-Llama-3.1-8B": "Llama 3.1 8B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B (it)",
        "meta-llama/Meta-Llama-3.1-70B": "Llama 3.1 70B",
        "meta-llama/Meta-Llama-3.1-70B-Instruct": "Llama 3.1 70B (it)",
        "meta-llama/Meta-Llama-3.2-1B": "Llama 3.2 1B",
        "meta-llama/Meta-Llama-3.2-1B-Instruct": "Llama 3.2 1B (it)",
        "meta-llama/Meta-Llama-3.2-3B": "Llama 3.2 3B",
        "meta-llama/Meta-Llama-3.2-3B-Instruct": "Llama 3.2 3B (it)",
        "meta-llama/Meta-Llama-3.3-70B-Instruct": "Llama 3.3 70B (it)",
        # Mistral AI models
        "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B (it)",
        "mistralai/Mistral-7B-v0.1": "Mistral 7B",
        "mistralai/Mixtral-8x22B-Instruct-v0.1": "Mixtral 8x22B (it)",
        "mistralai/Mixtral-8x22B-v0.1": "Mixtral 8x22B",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B (it)",
        "mistralai/Mixtral-8x7B-v0.1": "Mixtral 8x7B",
        "mistralai/Mistral-Small-24B-Base-2501": "Mistral Small 24B",
        "mistralai/Mistral-Small-24B-Instruct-2501": "Mistral Small 24B (it)",
        # Yi models
        "01-ai/Yi-34B": "Yi 34B",
        "01-ai/Yi-34B-Chat": "Yi 34B (chat)",
        "01-ai/Yi-6B-Chat": "Yi 6B (chat)",
        "01-ai/Yi-6B": "Yi 6B",
        "01-ai/Yi-1.5-6B": "Yi 1.5 6B",
        # Qwen2 models
        "Qwen/Qwen2-1.5B": "Qwen 2 1.5B",
        "Qwen/Qwen2-1.5B-Instruct": "Qwen 2 1.5B (it)",
        "Qwen/Qwen2-7B": "Qwen 2 7B",
        "Qwen/Qwen2-7B-Instruct": "Qwen 2 7B (it)",
        "Qwen/Qwen2-72B": "Qwen 2 72B",
        "Qwen/Qwen2-72B-Instruct": "Qwen 2 72B (it)",
        "Qwen/Qwen2.5-7B": "Qwen 2.5 7B",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B (it)",
        "Qwen/Qwen2.5-72B": "Qwen 2.5 72B",
        "Qwen/Qwen2.5-72B-Instruct": "Qwen 2.5 72B (it)",
        # Tabula
        "mlfoundations/tabula-8b": "Tabula 8B",
        # Olmo
        "allenai/OLMo-1B-0724-hf": "OLMo 1B 0724",
        "allenai/OLMo-1B-hf": "OLMo 1B",
        "allenai/OLMo-7B-0724-hf": "OLMo 7B 0724",
        "allenai/OLMo-7B-hf": "OLMo 7B",
        "allenai/OLMo-7B-Instruct-hf": "OLMo 7B (it)",
        "allenai/OLMo-2-1124-7B": "OLMo 2 7B",
        "allenai/OLMo-2-1124-7B-Instruct": "OLMo 2 7B (it)",
    }

    if model_hf_name in dct:
        return dct[model_hf_name]
    else:
        print(f"Couldn't find prettified name for {model_hf_name}.")
        return model_hf_name


def is_instruction_tuned(model_hf_name: str) -> bool:
    """Indicator if a model is instruction tuned (solely inferred from the model name).

    Args:
        model_name (str): name of the model

    Returns:
        bool: model is instruction-finetuned
    """
    indicators = ["Instruct", "it", "Chat"]
    return any(ind in model_hf_name for ind in indicators)


def get_size(model_key):
    # just a wrapper to facilitate imports
    return get_model_size_B(model_key)


def get_size_and_it(model_key, eps=0.001):
    return get_model_size_B(model_key) + eps * int(is_instruction_tuned(model_key))


def sort_by_size(models: list):
    return sorted(models, key=get_size_and_it)


def sort_by_size_and_family(models: list, factor=1000):
    # factor to ensure model families are well separated
    model_family_to_key = {k: v for v, k in enumerate(model_families, start=1)}

    def sort_key(model_key):
        return get_size_and_it(model_key) + factor * model_family_to_key.get(
            next(
                (mf for mf in model_families if mf.lower() in model_key.lower()), None
            ),
            -1,
        )

    return sorted(models, key=sort_key)


def create_result_df(
    root_dir: str | Path,
    subfolders: list,
    tasks: list = ACS_TASKS,
    save_path: str | Path = None,
    # add_baselines: list = [],
) -> pd.DataFrame:
    """
    Creates a pandas DataFrame with relevant metadata and paths from result files.

    Parameters
    ----------
    root_dir : str or Path
        Root directory where results are gathered from.
    tasks : list, optional
        List of tasks to include in the DataFrame (default is ACS_TASKS).
    save_path : str or Path, optional
        If provided, the DataFrame will be saved to this path.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
            - task: str, task name
            - model: str, model name
            - is_inst: int, whether the model is instruction-finetuned
            - bench_hash: str, hash of the benchmarking result
            - num_shots: int, number of shots used for few-shot prompting
            - prompt_format: str, format used for prompts if specified (if None: bullet)
            - prompt_connector: str, connector used for prompts if specified (if None: is)
            - prompt_granularity: str, low/original granularity used for the prompt if specified (if None: orginal)
            - prompt_feature_order: str/list, order in which features are presented in prompt (if None: orginal order)
            - eval_results_path: str, path to the evaluation results file
            - predictions_path: str, path to the predictions file
    """
    assert (
        isinstance(subfolders, list)
        or isinstance(subfolders, tuple)
        or isinstance(subfolders, str)
    )
    assert isinstance(tasks, list) or isinstance(tasks, tuple) or isinstance(tasks, str)
    if isinstance(subfolders, str):
        subfolders = [subfolders]
    if isinstance(tasks, str):
        tasks = [tasks]

    results_all_tasks = []
    for task in tasks:
        for folder in subfolders:
            # file name pattern
            pattern_json = r"^results.bench-(?P<hash>\d+)[.]json$"
            # find results files
            bench_results_files = find_files(
                (Path(root_dir) / folder), pattern_json, dir_pattern=task
            )
            for file_path in bench_results_files:
                model_name = (
                    Path(file_path)
                    .parent.parent.name.replace("model-", "")
                    .replace(f"_task-{task}", "")
                )  # extract model_name from model folder
                bench_hash = Path(file_path).parent.name.split("_bench-")[1]
                parsed_results = parse_results_dict(load_json(file_path))

                # save relevant metadata and path in df
                # "Prompting style is backward compatible for older results json files"
                prompt_format = next(
                    (
                        parsed_results[key]
                        for key in (
                            "config_prompt_variation_format",
                            "config_prompt_style_format",
                            "config_prompt_style",
                        )
                        if key in parsed_results
                    ),
                    "bullet",  # or raise error
                )
                prompt_connector = next(
                    (
                        parsed_results[key]
                        for key in (
                            "config_prompt_variation_connector",
                            "config_prompt_style_connector",
                            "config_prompt_connector",
                        )
                        if key in parsed_results
                    ),
                    "is",
                )
                prompt_granularity = parsed_results.get(
                    "config_prompt_variation_granularity", "original"
                )
                prompt_feature_order = map_feature_order_to_short.get(
                    parsed_results.get("config_prompt_variation_order"), "default"
                )
                threshold = get_metric(file_path, metric="threshold")
                accuracy = get_metric(file_path, metric="accuracy")

                res = [
                    task,
                    model_name,
                    int(is_instruction_tuned(model_name)),
                    int(
                        bool(parsed_results.get("threshold_fitted_on"))
                    ),  # (bool), if True fitted on 500 data points
                    threshold,
                    accuracy,
                    bench_hash,
                    (
                        parsed_results["config_few_shot"]
                        if parsed_results.get("config_few_shot")
                        else 0
                    ),
                    prompt_format,
                    prompt_connector,
                    prompt_granularity,
                    prompt_feature_order,
                    file_path,
                    parsed_results["predictions_path"][
                        parsed_results["predictions_path"].find("results/") :
                    ],
                ]
                results_all_tasks.append(res)
                if (
                    bool(parsed_results.get("threshold_fitted_on"))
                    and parsed_results.get("threshold") == 0.5
                ):  # fitting did overwrite the non-fitted results (hash unchanged)
                    tmp = res.copy()
                    tmp[3] = 0
                    results_all_tasks.append(tmp)

    df = pd.DataFrame(
        results_all_tasks,
        columns=[
            "task",
            "model",
            "is_inst",
            "threshold_fitted",
            "threshold",
            "accuracy",
            "bench_hash",
            "num_shots",
            "prompt_format",
            "prompt_connector",
            "prompt_granularity",
            "prompt_feature_order",
            "eval_results_path",
            "predictions_path",
        ],
    )
    print("Shape of df: ", df.shape)
    # some checks

    assert set(df["prompt_format"].unique()).issubset(
        set(variations["format"])
    ), f"{set(df['prompt_format'].unique())}, {set(variations['format'])}"
    assert set(df["prompt_connector"].unique()).issubset(
        set(variations["connector"])
    ), f"{set(df['prompt_connector'].unique())}, {set(variations['connector'])}"
    assert set(df["prompt_granularity"].unique()).issubset(
        set(variations["granularity"])
    ), f"{set(df['prompt_granularity'].unique())}, {set(variations['granularity'])}"
    assert set(df["prompt_feature_order"].unique()).issubset(
        set(variations["feature_order"])
    ), f"{set(df['prompt_feature_order'].unique())}, {set(variations['feature_order'])}"
    if save_path:
        print(f"Saving dataframe to {save_path}")
        df.to_csv(save_path, index=False)
    return df


def infer_treshold_fitted(file_path):
    logging.warning("deprecated, threshold fitting is now documented in results")
    print("Inferring whether threshold was fitted from file structure, prone to error.")
    # infer whether the treshold was fitted on training examples based on
    # - whether there are test_predictions in the same folder
    # - if so, check if there is another folder
    # e.g. BRFSS_Blood_Pressure_full_seed-42_hash-3784204307.test_predictions.csv
    pattern_csv = r".*\.test_predictions\.csv$"
    csv_file = list(find_files(Path(file_path).parent, pattern=pattern_csv))
    # print("Infer if thrshold was fitted", Path(file_path).parent, csv_file)
    if len(csv_file) > 0:
        # check if there is another result folder
        print("found prediction file, but need to check for other folder")
    else:
        return len(csv_file) == 0


def find_files(root_folder, pattern, dir_pattern=""):
    # Compile the regular expression pattern
    regex = re.compile(pattern)

    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if dir_pattern in dirpath:
            for filename in filenames:
                if regex.match(filename):
                    # If the filename matches the pattern, add it to the list
                    yield os.path.join(dirpath, filename)


def parse_model_name(name: str) -> str:
    name = name[name.find("--") + 2 :]
    return name


def get_base_name(name):
    name = re.sub(r"-(Instruct|Chat|it|1\.1)$", "", name, count=1)
    name = re.sub(r"-v0\.2$", "-v0.1", name, count=1)
    return name


model_col = "config_model_name"
# model_col = "model_name"
feature_subset_col = "config_feature_subset"
population_subset_col = "config_population_filter"
predictions_path_col = "predictions_path"

uses_all_features_col = "uses_all_features"
uses_all_samples_col = "uses_all_samples"


def parse_results_dict(dct) -> dict:
    """Parses results dict; brings all information to the top-level."""
    dct = dct.copy()
    dct.pop("plots", None)
    config = dct.pop("config", {})
    for key, val in config.items():
        if isinstance(config[key], dict):
            style_specs = config[key]  # config.pop(key, {})
            for subkey, subval in style_specs.items():
                dct[f"config_{key}_{subkey}"] = subval
        else:
            dct[f"config_{key}"] = val

    # Parse model name
    dct[model_col] = parse_model_name(dct[model_col])
    dct[uses_all_features_col] = dct[feature_subset_col] is None
    if dct[feature_subset_col] is None:
        dct[feature_subset_col] = "full"

    dct[uses_all_samples_col] = dct[population_subset_col] is None

    dct["base_name"] = get_base_name(dct[model_col])
    dct["is_inst"] = dct["base_name"] != dct[model_col]

    for key, val in dct.items():
        if isinstance(val, dict):
            print(key, val)
    assert not any(isinstance(val, dict) for val in dct.values()), dct
    return dct


def load_risk_scores(csv_path: str | Path) -> pd.DataFrame:
    """
    Loads risk scores from a csv file, removes the 'label' column, and renames
    the 'risk_score' column to the model name.

    Args:
        csv_path (str | Path): The file path to the csv containing model predictions.

    Returns:
        pd.DataFrame: A DataFrame with risk scores, where the 'risk_score' column
                      is renamed to the model name.
    """
    # load risk scores, change column to <model_name>
    risk_score = (
        pd.read_csv(csv_path, index_col=0)
        .drop("label", axis=1)
        .rename(
            columns={
                "risk_score": re.search(r"model-(.+?)_task-", csv_path)
                .group(1)
                .split("/")[1]
            }
        )
    )
    return risk_score


def get_predictions(
    csv_path: str | Path,
    eval_json_path: str | Path = None,
) -> pd.DataFrame:
    """
    Loads risk scores and binarize usiing binarization threshold from
    evaluation results. Returns a DataFrame.

    Args:
        csv_path (str | Path): The file path to the CSV containing risk scores.
        eval_json_path (str | Path): If stored elsewhere, file path to the json containing evals including the threshold.

    Returns:
        pd.DataFrame: A DataFrame with binarized predictions.
    """
    risk_scores = load_risk_scores(csv_path)
    # binarize using threshold from respective eval results
    if not eval_json_path:
        bench_hash = Path(csv_path).parent.as_posix().split("bench-")[1]
        threshold = load_json(
            Path(csv_path).parent / f"results.bench-{bench_hash}.json"
        ).get("threshold")
    else:
        threshold = load_json(Path(eval_json_path)).get("threshold")
    if threshold is None:
        logging.warning(f"Threshold not found for {csv_path}, defaulting to 0.5")
        threshold = 0.5
    return risk_scores.map(lambda x: int(x >= threshold))


def get_metric(json_path: str | Path, metric: str = "accuracy"):
    """
    Retrieves the specified metric from a benchmark results file.

    Args:
        json_path (str | Path): Path to the JSON file containing evaluation results.
        metric (str, optional): The name of the metric to retrieve. Defaults to "accuracy".

    Returns:
        The value of the specified metric from the evaluation results.
    """
    evals = load_json(json_path)
    return evals[metric]


def get_metrics(json_path: str | Path, metrics: list[str]):
    """
    Retrieves multiple metrics from a benchmark results file.

    Args:
        json_path (str | Path): Path to the JSON file containing evaluation results.
        metrics (list[str]): A list of metric names to retrieve.

    Returns:
        A dictionary with the metric names as keys and their respective values as values.
    """
    evals = load_json(json_path)
    return {metric: evals[metric] for metric in metrics}


def cumulative_sum(tensor: torch.Tensor, end=None):
    if end == "left":
        # from right to left (total at xmin)
        return torch.flip(torch.cumsum(torch.flip(tensor, dims=[0]), dim=0), dims=[0])
    else:
        # from left to right (total at xmax)
        return torch.cumsum(tensor, dim=0)


def truncate(num: float, digits: int = 6) -> float:
    return round(num - 10**-digits / 2, digits)


def binarize_using_threshold(col, evals: dict):
    m = col.name
    return col > evals[m]["threshold"]


def load_task_data(tasks: str | list, data_dir: Path | str):
    if isinstance(tasks, str):
        tasks = [tasks]
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    data = {}
    for task in tasks:
        print(task)
        logging.info(f"Loading data for task {task}.")
        if task in ACS_TASKS:
            acs_task = ACSTaskMetadata.get_task(task)
            acs_dataset_configs = (
                folktexts.benchmark.Benchmark.ACS_DATASET_CONFIGS.copy()
            )
            acs_dataset = ACSDataset.make_from_task(
                task=acs_task, cache_dir=data_dir, **acs_dataset_configs
            )
            X_test, y_test = acs_dataset.get_data_split("test")
            data[task] = (X_test, y_test)
        else:
            brfss_task = TableshiftBRFSSTaskMetadata.get_task(task)
            dataset_configs = (
                folktexts.benchmark.Benchmark.TABLESHIFT_DATASET_CONFIGS.copy()
            )
            ts_dataset = TableshiftBRFSSDataset.make_from_task(
                task=brfss_task, cache_dir=data_dir, **dataset_configs
            )
            X_test, y_test = ts_dataset.get_data_split("test")
            data[task] = (X_test, y_test)
    return data


def load_model_outputs_same_prompt(
    df: pd.DataFrame,
    tasks: list[str] = ACS_TASKS + TABLESHIFT_TASKS,
    return_risk_scores: bool = True,
):
    outputs = {}
    for task in tasks:
        task_df = df[df["task"] == task]
        # get available models
        models = task_df["model"].unique().tolist()
        models.sort(key=lambda m: get_model_size_B(m) + int(is_instruction_tuned(m)))
        outputs_per_task = []
        for m in models:
            data_m = task_df[task_df["model"] == m]
            if data_m.shape[0] != 1:
                logging.warning(
                    f"Expected 1 row for model {m}, but found {data_m.shape[0]}. Skipping this model.\n {data_m}"
                )
                continue
            if return_risk_scores:
                outputs_per_task.append(
                    load_risk_scores(data_m.iloc[0]["predictions_path"])
                )
            else:
                outputs_per_task.append(
                    get_predictions(
                        csv_path=data_m.iloc[0]["predictions_path"],
                        eval_json_path=data_m.iloc[0]["eval_results_path"],
                    )
                )
        logging.debug(task, len(outputs_per_task))
        if len(outputs_per_task) > 0:
            outputs[task] = pd.concat(outputs_per_task, axis=1)
    return outputs
