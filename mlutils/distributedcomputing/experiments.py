from joblib import Parallel, delayed
from box import Box
from loguru import logger

from mlutils.distributedcomputing.job_scheduling import create_experiment_notebook, run_experiments_in_slurm, \
    create_slurm_script, run_in_batches, run_in_papermill


def run_experiments(run_ids, experiment_function, backend='joblib', **params):
    backend_params = Box(params)
    logger.info(backend_params)

    if backend == 'joblib':
        return Parallel(n_jobs = backend_params.n_jobs if 'n_jobs' in backend_params else -1)(
            delayed(experiment_function)(run_id) for run_id in run_ids
        )
    elif backend == 'plain':
        return [experiment_function(run_id) for run_id in run_ids]

    elif backend == 'papermill':
        logger.warning("Using Papermill backend. Make sure you execute this function not from experiment notebook!")

        assert "notebook_path" in params
        assert type(experiment_function) == str

        notebook_run_id_param = params.get("notebook_run_id_param", "EXPERIMENT_INSTANCE_ID")

        path_to_notebook = create_experiment_notebook(
            notebook_name=params['notebook_path'],
            experiment_function=experiment_function,
            instance_id_param_name=notebook_run_id_param
        )

        logger.info("Notebook path={}", path_to_notebook)

        return run_in_papermill(run_ids,
                                notebook_path=path_to_notebook,
                                n_jobs=params.get("n_jobs", -1),
                                notebook_run_id_param=notebook_run_id_param,
                                papermill_path=params.get("papermill_path", None),
                                output_dir_path=params.get("output_dir_path", None),
                                execution_timeout=params.get("execution_timeout", 600))

    elif backend == "slurm":
        logger.warning("Using SLURM backend. Make sure you execute this function not from experiment notebook!")

        assert "notebook_path" in params
        assert type(experiment_function) == str

        notebook_run_id_param = params.get("notebook_run_id_param", "EXPERIMENT_INSTANCE_ID")

        should_run_in_batches = params.get("run_in_batches", False)
        path_to_notebook = create_experiment_notebook(
            notebook_name=params['notebook_path'],
            experiment_function=experiment_function,
            instance_id_param_name=notebook_run_id_param
        )

        logger.info("Notebook path={}", path_to_notebook)

        slurm_script_path = create_slurm_script(params.get('slurm_arguments', {}))

        if should_run_in_batches:
            return run_in_batches(
                run_ids=run_ids,
                batch_size=params.get("batch_size", 250),
                sleep_interval=params.get("sleep_interval", 25),
                username=params.get("username", "bogul"),
                command = lambda single_batch: run_experiments_in_slurm(
                    run_ids=single_batch,
                    notebook_path=str(path_to_notebook),
                    output_dir_path = params.get('output_dir_path', None),
                    papermill_path = params.get('papermill_path', None),
                    script_path = slurm_script_path,
                    notebook_run_id_param=notebook_run_id_param,
                )
            )

        return run_experiments_in_slurm(
            run_ids=run_ids,
            notebook_path=str(path_to_notebook),
            output_dir_path = params.get('output_dir_path', None),
            papermill_path = params.get('papermill_path', None),
            script_path = slurm_script_path,
            notebook_run_id_param=notebook_run_id_param,
        )

    raise NotImplementedError(f"{backend} not supported")
