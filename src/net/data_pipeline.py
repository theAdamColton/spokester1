"""
DataPipeline similar to Webdataset's FluidInterface
"""

import copy
import random
import multiprocessing
import threading
import queue
from typing import Callable, Iterable, Literal

import torch


def get_collation_fn(mode="cat"):
    def collation_fn(samples):
        assert isinstance(samples, list)
        assert isinstance(samples[0], dict)

        keys = samples[0].keys()
        result = {}

        for k in keys:
            batch = [sample[k] for sample in samples]
            if isinstance(batch[0], list):
                batch = [x for y in batch for x in y]
            elif isinstance(batch[0], torch.Tensor):
                if mode == "cat":
                    batch = torch.cat(batch, 0)
                elif mode == "stack":
                    batch = torch.stack(batch, 0)
                else:
                    raise ValueError(mode)
            else:
                batch = list(batch)
            result[k] = batch

        return result

    return collation_fn


def _parallel_worker_fn(pipeline, result_queue, stop_event):
    """
    The target function for each worker (thread or process).
    It creates its own iterator from the provided pipeline
    and puts items into the result_queue.
    """
    worker_source = iter(pipeline)
    try:
        while not stop_event.is_set():
            try:
                # This is the primary blocking call
                item = next(worker_source)
            except StopIteration:
                # The source for this worker is exhausted
                break

            # Try to put the item, but check the stop_event frequently
            while not stop_event.is_set():
                try:
                    # queue.Full is the base exception for both
                    # threading.Queue and multiprocessing.Queue
                    result_queue.put(item, timeout=0.1)
                    break  # Item was successfully put
                except queue.Full:
                    continue  # Queue is full, retry

    except Exception as e:
        # If a worker fails, put the exception in the queue
        # so the main thread can re-raise it.
        if not stop_event.is_set():
            # Use a special tuple to signal an error
            result_queue.put(("__error__", e))

    # Worker (thread or process) will exit here


class DataPipeline:
    def __init__(self, source):
        super().__init__()

        self.source = source
        self._stages = []

    def map_columns(self, **kwargs):
        def _map(row):
            for key, fn in kwargs.items():
                row[key] = fn(row[key])
            return row

        return self.map(_map)

    def map(self, fn: Callable[[dict], dict]):
        result = copy.deepcopy(self)
        result._stages.append(("map", fn))
        return result

    def compose(self, fn: Callable[[Iterable], Iterable]):
        result = copy.deepcopy(self)
        result._stages.append(("compose", fn))
        return result

    def batched(
        self, batch_size: int, collate_fn=get_collation_fn("stack"), partial=False
    ):
        def _batched(source):
            batch = []
            for sample in source:
                batch.append(sample)
                if len(batch) >= batch_size:
                    yield collate_fn(batch)
                    batch = []

            if partial and len(batch) > 0:
                yield collate_fn(batch)

        return self.compose(_batched)

    def load_parallel(
        self,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        mode: Literal["thread", "process"] = "thread",
    ):
        if num_workers == 0:
            return self

        # Select the correct concurrency modules based on the mode
        if mode == "thread":
            PoolWorker = threading.Thread
            StopEvent = threading.Event
            ResultQueue = queue.Queue
        elif mode == "process":
            mp_context = multiprocessing.get_context("spawn")
            PoolWorker = mp_context.Process
            StopEvent = mp_context.Event
            ResultQueue = mp_context.Queue
        else:
            raise ValueError(mode)

        def _load_parallel(_):
            # Create a queue to collect results from workers
            result_queue = ResultQueue(maxsize=num_workers * prefetch_factor)
            stop_event = StopEvent()

            workers = []
            for i in range(num_workers):
                # We must deepcopy the pipeline for *each* worker
                # so they have an independent state and iterator source.
                worker_pipeline = copy.deepcopy(self)

                # Create the worker (Thread or Process)
                worker = PoolWorker(
                    target=_parallel_worker_fn,
                    name=f"datapipe_worker_{i:02}",
                    args=(worker_pipeline, result_queue, stop_event),
                    daemon=True,
                )
                worker.start()
                workers.append(worker)

            # More robust main loop to handle worker exit.
            # A worker exiting (is_alive() == False) is a
            # normal event when it runs out of data.
            try:
                active_workers = list(workers)
                while active_workers:
                    try:
                        # Get an item from the queue
                        item = result_queue.get(timeout=0.1)

                        if isinstance(item, tuple) and item[0] == "__error__":
                            # It's an exception from a worker
                            stop_event.set()  # Signal all workers to stop
                            raise item[1]  # Re-raise the exception

                        yield item

                    except queue.Empty:
                        # Queue is empty, check on worker health
                        active_workers = [w for w in workers if w.is_alive()]
                        if not active_workers:
                            # All workers are dead, exit the loop
                            break

                # All workers are dead, but the queue might have stragglers.
                # Drain the rest of the queue.
                while True:
                    try:
                        item = result_queue.get_nowait()
                        if isinstance(item, tuple) and item[0] == "__error__":
                            raise item[1]
                        yield item
                    except queue.Empty:
                        break  # Queue is fully drained

            finally:
                # Signal all workers to stop (for a graceful shutdown)
                # This cleans up any workers stuck waiting on result_queue.put()
                stop_event.set()

                # We do not explicitly .join() daemon threads/processes,
                # as they will be terminated when the main process exits.

        return self.compose(_load_parallel)

    def shuffle(self, size: int = 500, seed=None):
        if size <= 0:
            return self
        return self.compose(Shuffle(size, seed))

    def __iter__(self):
        source = iter(self.source)

        for stage_type, stage_fn in self._stages:
            if stage_type == "map":
                source = map(stage_fn, source)
            elif stage_type == "compose":
                source = stage_fn(source)
            else:
                raise ValueError(stage_type)

        return source


class RngMixin:
    _rng: random.Random | None = None

    def set_seed(self, seed):
        if seed is not None:
            self._rng = random.Random(seed)

    @property
    def rng(self):
        if self._rng is None:
            return random
        return self._rng


class Shuffle(RngMixin):
    def __init__(self, size=500, seed=None):
        self.size = size
        self.set_seed(seed)

    def __call__(self, source):
        buffer = []
        for row in source:
            buffer.append(row)
            while len(buffer) >= self.size:
                i = self.rng.randint(0, len(buffer) - 1)
                yield buffer.pop(i)
