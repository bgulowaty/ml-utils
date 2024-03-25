from time import perf_counter
from loguru import logger as log
class catchtime:
    def __init__(self, name):
        self._name = name

    def __enter__(self):
        log.info("Starting time measurement {}", self._name)
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'Time: {self.time:.3f} seconds {self._name}'
        log.info(self.readout)
