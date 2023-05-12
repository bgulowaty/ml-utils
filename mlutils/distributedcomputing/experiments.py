from joblib import Parallel, delayed
from box import Box
from loguru import logger

from mlutils.distributedcomputing.job_scheduling import create_experiment_notebook, run_experiments_in_slurm


def run_experiments(run_ids, experiment_function, backend='joblib', **params):
    backend_params = Box(params)
    logger.info(backend_params)

    if backend == 'joblib':
        return Parallel(n_jobs = backend_params.n_jobs if 'n_jobs' in backend_params else -1)(
            delayed(experiment_function)(run_id) for run_id in run_ids
        )
    elif backend == 'plain':
        return [experiment_function(run_id) for run_id in run_ids]

    elif backend == "slurm":
        logger.warning("Using SLURM backend. Make sure you execute this function not from experiment notebook!")

        assert "notebook_path" in params
        assert type(experiment_function) == str

        notebook_run_id_param = params.get("notebook_run_id_param", "EXPERIMENT_INSTANCE_ID")

        path_to_notebook = create_experiment_notebook(
            notebook_name=params['notebook_path'],
            experiment_function=experiment_function,
            instance_id_param_name=notebook_run_id_param
        )

        return run_experiments_in_slurm(
            run_ids=run_ids,
            notebook_path=str(path_to_notebook),
            output_dir_path = params.get('output_dir_path', None),
            papermill_path = params.get('papermill_path', None),
            script_path = params.get('script_path', None),
            notebook_run_id_param=notebook_run_id_param
        )

    raise NotImplementedError(f"{backend} not supported")
