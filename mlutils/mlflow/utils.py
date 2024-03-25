import traceback

import mlflow
from box import Box
from joblib import Parallel, delayed
from mlflow import MlflowClient
from tqdm import tqdm
from loguru import logger
from copy import deepcopy

def terminate_run(run_id, client = MlflowClient()):
    client.set_terminated(run_id, "FINISHED")

def finish_run_and_print_exception(run_id, exception, client = MlflowClient(), logger = logger):
    logger.error(exception)
    tb = traceback.format_exc()
    logger.error(tb)
    client.set_tag(run_id, "exception", traceback.format_exc())
    client.set_terminated(run_id, "FAILED")

def log_2d_metrics(array, names, run_id, mlflow_client = MlflowClient()):
    assert len(names) == array.shape[1]
    for step, metrics in enumerate(array):
        for metric_name, value in zip(names, metrics):
            mlflow_client.log_metric(run_id=run_id, key=metric_name, value=value, step=step)

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
        thread_client = deepcopy(client)
        run = thread_client.create_run(experiment_id=experiment_id)
        run_id = run.info.run_id

        for key, val in params.items():
            thread_client.log_param(run_id, key, val)

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