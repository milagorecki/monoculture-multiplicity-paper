from monoculture.analysis.setup import BASELINE_RESULTS_PATH
from folktexts._io import load_json
import pandas as pd


def load_baselines(baselines: dict, tasks: list, rerun: bool = False) -> tuple:
    baseline_results_all_tasks = {}
    baseline_risk_scores_all_tasks = {}
    for task_name in tasks:
        print(f"Loading baselines for {task_name}.")
        results = {}
        risk_scores = []
        for clf_name, clf in baselines.items():
            clf_path = BASELINE_RESULTS_PATH / f"{clf_name}_task-{task_name}"
            if (clf_path).exists() and not rerun:
                print(f"- {clf_name}: Load predictions from '{clf_path}'.")
                scores = pd.read_csv(
                    clf_path / f"{task_name}.test_predictions.csv", index_col=0
                )
                prediction_eval = load_json(
                    path=clf_path / f"{task_name}-results.bench.json"
                )
            else:
                print(f"Skipping {clf_name}")
                prediction_eval = {}
                scores = pd.Series()
            results[clf_name] = prediction_eval
            risk_scores.append(scores)

        baseline_results_all_tasks[task_name] = results
        baseline_risk_scores_all_tasks[task_name] = pd.concat(risk_scores, axis=1)

    return baseline_risk_scores_all_tasks, baseline_results_all_tasks
