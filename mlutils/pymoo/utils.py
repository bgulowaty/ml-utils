import ray


@ray.remote
def execute_in_ray(f, x):
    return f(x)

class RayParallelization:

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, f, X):
        results = [execute_in_ray.remote(f, x) for x in X]

        return ray.get(results)


    def __getstate__(self):
        state = self.__dict__.copy()
        return state

class ExecutorParallelization:

    def __init__(self, executor) -> None:
        super().__init__()
        self.executor = executor

    def __call__(self, f, X):
        jobs = [self.executor.submit(f, x) for x in X]
        return [job.result() for job in jobs]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("executor", None) # is not serializable
        return state