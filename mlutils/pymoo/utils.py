from __future__ import annotations

from pathlib import Path

import ray
from typing import Literal, Any, Callable, Iterable, Generator
import joblib


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

class JoblibParallelization:
    def __init__(
            self,
            n_jobs: int = -1,
            backend: Literal["loky", "threading", "multiprocessing"] = "loky",
            return_as: Literal["list", "generator"] = "list",
            verbose: int = 0,
            timeout: float | None = None,
            pre_dispatch: str | int = "2 * n_jobs",
            batch_size: int | Literal["auto"] = "auto",
            temp_folder: str | Path | None = None,
            max_nbytes: int | str | None = "1M",
            mmap_mode: Literal["r+", "r", "w+", "c"] | None = "r",
            prefer: Literal["processes", "threads"] | None = None,
            require: Literal["sharedmem"] | None = None,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        self.n_jobs = n_jobs
        self.backend = backend
        self.return_as = return_as
        self.verbose = verbose
        self.timeout = timeout
        self.pre_dispatch = pre_dispatch
        self.batch_size = batch_size
        self.temp_folder = temp_folder
        self.max_nbytes = max_nbytes
        self.mmap_mode = mmap_mode
        self.prefer = prefer
        self.require = require
        super().__init__()

    def __call__(
            self,
            f: Callable[..., Any],
            X: Iterable[Any],
    ) -> list[Any] | Generator[Any, Any, None]:
        with joblib.Parallel(
                n_jobs=self.n_jobs,
                backend=self.backend,
                return_as=self.return_as,
                verbose=self.verbose,
                timeout=self.timeout,
                pre_dispatch=self.pre_dispatch,
                batch_size=self.batch_size,
                temp_folder=self.temp_folder,
                max_nbytes=self.max_nbytes,
                mmap_mode=self.mmap_mode,
                prefer=self.prefer,
                require=self.require,
        ) as parallel:
            return parallel(joblib.delayed(f)(x) for x in X)