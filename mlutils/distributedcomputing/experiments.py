from joblib import Parallel, delayed
from box import Box

def run_experiments(run_ids, experiment_function, backend='joblib', **params):
    backend_params = Box(params)
    print(backend_params)

    if backend == 'joblib':
        return Parallel(n_jobs = backend_params.n_jobs if 'n_jobs' in backend_params else -1)(
            delayed(experiment_function)(run_id) for run_id in run_ids
        )
    elif backend == 'plain':
        return [experiment_function(run_id) for run_id in run_ids]

    raise NotImplementedError(f"{backend} not supported")
