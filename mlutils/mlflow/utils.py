import mlflow
from box import Box
from joblib import Parallel, delayed
from mlflow import MlflowClient
from tqdm import tqdm
from loguru import logger

def create_runs_for_params(params, experiment_id = None, experiment_name = None, client = MlflowClient(), n_jobs=-1):
    if experiment_id is None:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
            logger.info("Experiment not found, creating new. Id={}", experiment_id)
        else:
            experiment_id = experiment.experiment_id
            logger.info("Experiment found. Id={}", experiment_id)


    def create_run_and_log_params(params):
        client = MlflowClient()
        run = client.create_run(experiment_id=experiment_id)
        run_id = run.info.run_id

        for key, val in params.items():
            client.log_param(run_id, key, val)

        return run_id

    run_ids = Parallel(n_jobs=n_jobs)(delayed(create_run_and_log_params)(param) for param in tqdm(params))

    return run_ids

def experiment_name_to_id(experiment_name, client = MlflowClient()):
    return client.get_experiment_by_name(experiment_name).experiment_id

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

def get_run_params(run_id, client = MlflowClient()):
    return Box(client.get_run(run_id).data.params)