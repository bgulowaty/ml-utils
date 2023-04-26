import mlflow
from joblib import Parallel, delayed
from mlflow import MlflowClient
from tqdm import tqdm

def create_runs_for_params(experiment_id, params, client = MlflowClient(), n_jobs=-1):
    def create_run_and_log_params(params):
        run = client.create_run(experiment_id=experiment_id)
        run_id = run.info.run_id

        for key, val in params.items():
            client.log_param(run_id, key, val)

        return run_id

    run_ids = Parallel(n_jobs=n_jobs, backend='threading')(delayed(create_run_and_log_params)(param) for param in tqdm(params))

    return run_ids

def get_runs(experiment_id, additional_query=None):
    all_runs = mlflow.search_runs(experiment_ids=experiment_id)

    if additional_query:
        return all_runs.query(additional_query)
    else:
        return all_runs

def get_unfinished_runs(experiment_id, additional_query=None):
    all_runs = get_runs(experiment_id, additional_query)

    return all_runs.query('`end_time`.isnull() or `status` == "RUNNING"', engine='python')

def get_unfinished_run_ids(experiment_id, additional_query=None):
    return get_unfinished_runs(experiment_id, additional_query)['run_id'].tolist()