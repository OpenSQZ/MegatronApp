# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

from dataclasses import dataclass
from functools import wraps
import json
import time
import torch
import os
import threading
import queue
from typing import Any, Dict, List, Optional, Tuple

from megatron.core import parallel_state
import torch.distributed


def _save_traces_to_disk_thread(work_queue: queue.Queue, trace_dir: str):
    """
    Worker thread that pulls trace data from a queue and saves it to disk.

    This runs in a separate thread to avoid blocking the main training loop.
    It performs append-style writes to JSON files, ensuring that trace files
    are always valid JSON.

    Args:
        work_queue: The queue to get trace data from.
        trace_dir: The directory on rank 0 where trace files will be saved.
    """
    if not os.path.exists(trace_dir):
        try:
            os.makedirs(trace_dir, exist_ok=True)
        except OSError as e:
            print(f"Rank 0: Error creating trace directory {trace_dir}: {e}", flush=True)
            return

    while True:
        try:
            payloads = work_queue.get()
            if payloads is None:  # Sentinel for termination
                work_queue.task_done()
                break

            for filename, new_records in payloads:
                if not new_records:
                    continue

                filepath = os.path.join(trace_dir, filename)
                existing_records = []
                try:
                    if os.path.exists(filepath):
                        with open(filepath, 'r') as f:
                            content = f.read()
                            if content:  # Avoid error on empty file
                                existing_records = json.loads(content)
                        if not isinstance(existing_records, list):
                            print(f"Warning: Trace file {filepath} appears to be corrupted. Overwriting.", flush=True)
                            existing_records = []
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not read or parse existing trace file {filepath}. Overwriting. Error: {e}", flush=True)
                    existing_records = []

                existing_records.extend(new_records)

                with open(filepath, 'w') as f:
                    json.dump(existing_records, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

            work_queue.task_done()

        except Exception as e:
            print(f"Error in trace saving thread: {e}", flush=True)
            if 'work_queue' in locals() and isinstance(work_queue, queue.Queue):
                 work_queue.task_done()


class _TracerScope:
    def __init__(
        self,
        tracer: "Tracer",
        name: Optional[str],
        in_attrs: Dict[str, Any],
        out_attrs: Dict[str, Any],
    ) -> None:
        self.tracer = tracer
        self.name = name
        self.in_attrs = in_attrs
        self.out_attrs = out_attrs

    def __enter__(self) -> None:
        self.tracer._push_scope(self)
        if self.name is not None:
            self.tracer._tick(self.name, "B", {})

    def __exit__(self, type, value, traceback) -> None:
        if self.name is not None:
            self.tracer._tick(self.name, "E", self.out_attrs)
        self.tracer._pop_scope()

    def get(self, q: str) -> Optional[Any]:
        """Get from in_attrs."""
        return self.in_attrs.get(q)

    def set(self, q: str, v: Any) -> bool:
        """Set to out_attrs, if this is required."""
        if q in self.out_attrs and self.out_attrs[q] is not None:
            print(f"Already set q: {q}, v: {v}")
        if q in self.out_attrs and self.out_attrs[q] is None:
            self.out_attrs[q] = v
            return True
        else:
            return False


@dataclass
class _Pending:
    name: str
    phase: str
    event: Any
    attrs: Dict[str, Any]


class Tracer:
    """Global tracer to record and print timestamp during training process"""

    def __init__(self) -> None:
        self._records: List[Any] = []
        self._cur: Optional[int] = None
        self._pending_pad_before: Optional[int] = None
        self._pendings: Optional[List[_Pending]] = None
        self._scopes: List[_TracerScope] = []
        self.iter = 0
        self.global_args = None
        self.interval: int = 1000
        self.continuous_trace_iters: int = 1

        self._work_queue: Optional[queue.Queue] = None
        self._save_thread: Optional[threading.Thread] = None

    def _initialize_save_thread(self):
        """Initializes and starts the background saver thread on rank 0."""
        assert torch.distributed.get_rank() == 0
        if self._save_thread is not None:
            return

        assert self.global_args is not None, "Tracer's global_args has not been set"
        trace_dir = self.global_args.trace_dir

        # Clean up existing trace files in the directory
        if os.path.exists(trace_dir):
            try:
                for filename in os.listdir(trace_dir):
                    if filename.endswith('.json'):
                        os.remove(os.path.join(trace_dir, filename))
            except OSError as e:
                print(f"Warning: Could not clean up trace directory {trace_dir}: {e}", flush=True)

        self._work_queue = queue.Queue()
        self._save_thread = threading.Thread(
            target=_save_traces_to_disk_thread,
            args=(self._work_queue, trace_dir),
            daemon=True,
        )
        self._save_thread.start()

    def iteration_begin(self, iteration: int) -> None:
        """Start tracing an iteration. Note that this performs synchronization."""
        self.iter = iteration

        # if self.global_args and self.global_args.trace:
        #     self._initialize_save_thread()

        if self.is_tracing_active():
            if self._pendings is None:  # Start of a tracing window
                self._pending_pad_before = self._calibrate()
                self._pendings = []
            # Mark the beginning of the iteration
            self._add_cuda_event("iteration", "B", {})
        else:
            self._pendings = None

    def _calibrate(self) -> int:
        """Reset the clock and get delta."""
        cur = time.time_ns()
        if self._cur is None:
            delta = 0
        else:
            delta = cur - self._cur
        self._cur = cur
        return delta

    def _add_record(self, attrs: Dict[str, Any]) -> None:
        self._records.append(attrs)

    def _last_record(self) -> Dict[str, Any]:
        return self._records[-1]

    def _add_pending(self, pending: _Pending) -> None:
        if self._pendings is not None:
            self._pendings.append(pending)

    def _add_cuda_event(self, name: str, phase: str, attrs: Dict[str, Any]) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record() # type: ignore
        pending = _Pending(name, phase, event, attrs) # type: ignore
        self._add_pending(pending)

    def is_tracing(self) -> bool:
        return self._pendings is not None

    def _process_pending_scope(
        self, ref_ts: int, ref_event: torch.cuda.Event, i: int
    ) -> int:
        """Process the pending scopes.
        ref must be a "B".
        Args:
            ref_ts: reference timestamp.
            ref_event: reference event.
            i: index of the pending scope to be processed.
        Returns:
            The next index to process.
        """
        assert self._pendings is not None
        data_parallel_rank = parallel_state.get_data_parallel_rank()
        pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        tensor_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
        device = torch.cuda.current_device()
        global_rank = torch.distributed.get_rank()

        while i < len(self._pendings):
            pending = self._pendings[i]
            elapsed = int(ref_event.elapsed_time(pending.event) * 1e6)
            rel_ts = ref_ts + elapsed
            chrome_event = {
                **pending.attrs,
                "name": pending.name,
                "ph": pending.phase,
                "rel_ts": rel_ts,
                "dp_rk": data_parallel_rank,
                "pp_rk": pipeline_parallel_rank,
                "tp_rk": tensor_parallel_rank,
                "dev": device,
                "g_rk": global_rank,
            }
            self._add_record(chrome_event)
            i += 1
            if pending.phase == "B":
                # Nested scope
                i = self._process_pending_scope(rel_ts, pending.event, i)
            elif pending.phase == "E":
                # End of this scope
                if "data" in pending.attrs:
                    last = self._last_record()
                    if pending.attrs["data"] is None:
                        last["bandwidth"] = None
                    else:
                        # 1 Gb = 2 ** 30 b = 2 ** 27 B
                        gb = pending.attrs["data"] / (2**27)
                        secs = elapsed / 1e9
                        bandwidth = gb / secs  # Gbps
                        last["bandwidth"] = bandwidth
                return i
        assert i == len(self._pendings), "Mismatched scopes"
        return i

    def iteration_end(self) -> None:
        """End tracing an iteration. Note that this performs synchronization."""
        if self.is_tracing_active() and self._pendings is not None:
            # Mark the end of the iteration
            self._add_cuda_event("iteration", "E", {})
            # Wait for all events to finish
            torch.cuda.synchronize()
            # Get wall clock duration for this iteration
            wall_duration = self._calibrate()

            self._add_record({
                "name": "iteration",
                "ph": "B",
                "pad_before": self._pending_pad_before,
            })
            if not self._pendings:
                return

            iteration_begin_event = self._pendings[0].event
            # We cannot know the absolute timestamp of the first event, so we set it to 0.
            self._process_pending_scope(0, iteration_begin_event, 1)
            end = self._last_record()
            end["duration_wall"] = wall_duration
            end["duration_cuda"] = end["rel_ts"]

        if self.should_log_this_iter():
            self.log()

    def _tick(self, name: str, phase: str, attrs: Dict[str, Any]) -> None:
        if self.is_tracing():
            self._add_cuda_event(name, phase, attrs)

    def tick(self, name: str, **attrs: Any) -> None:
        """Record an event."""
        self._tick(name, "i", attrs)

    def scope(
        self,
        name: Optional[str],
        *args,
        ctx: Optional[Dict[str, Any]] = None,
        slots: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> _TracerScope:
        """Create a scope of code.
        Args:
            name: Name of the scope. If None, the scope is not timed.
            ctx: Parameters to be passed to the scope.
            kwargs: Items to be recorded. If an item is None, it should be filled by some inner scope.
            slots: Parameters that are passed to the scope and must be filled. (They go to both ctx and kwargs.)
        """
        assert len(args) == 0, "Positional arguments are not supported"
        if ctx is None:
            ctx = {}
        if slots is None:
            slots = []
        for slot in slots:
            ctx[slot] = True
            kwargs[slot] = None
        return _TracerScope(self, name=name, in_attrs=ctx, out_attrs=kwargs)

    # def scoped(self, func):
    #     """Decorator to time a function."""
    #     @wraps(func)
    #     def wrapper(*args, **kwargs):
    #         with self.scope(func.__name__):
    #             return func(*args, **kwargs)
    #     return wrapper

    def scoped(
        self,
        name: Optional[str] = None,
        ctx: Optional[Dict[str, Any]] = None,
        slots: Optional[List[str]] = None,
        **kwargs0: Any,
    ):
        if ctx is None:
            ctx = {}
        if slots is None:
            slots = []
        def decorator(func):
            from .training import get_args
            args = get_args()
            # if we are not tracing, just return the function
            if args is None or not args.trace:
                return func
            @wraps(func)
            def wrapper(*args, **kwargs):
                if name is None:
                    with self.scope(func.__name__, ctx=ctx, slots=slots, **kwargs0):
                        return func(*args, **kwargs)
                else:
                    with self.scope(name, ctx=ctx, slots=slots, **kwargs0):
                        return func(*args, **kwargs)
            return wrapper
        return decorator

    def _push_scope(self, scope) -> None:
        self._scopes.append(scope)

    def _pop_scope(self) -> None:
        self._scopes.pop()

    def get(self, q: str) -> Optional[Any]:
        """Query parameter from scopes."""
        for scope in reversed(self._scopes):
            v = scope.get(q)
            if v is not None:
                return v
        return None

    def set(self, q: str, v: Any) -> None:
        """Set parameter to the nearest requiring scope."""
        for scope in reversed(self._scopes):
            if scope.set(q, v):
                return
        # for scope in reversed(self._scopes):
        #     print(f"scope: '{scope.name}', q: '{q}', v: '{v}', in_attrs: {scope.in_attrs}, out_attrs: {scope.out_attrs}")
        assert False, f"Cannot find a requiring scope for '{q}'"

    def set_group(self, group: torch.distributed.ProcessGroup | List[int]) -> None:
        # get ranks in the group
        if isinstance(group, torch.distributed.ProcessGroup):
            ranks = torch.distributed.get_process_group_ranks(group)
        else:
            ranks = group
        cur_rk = torch.distributed.get_rank()
        # print(f"cur_rk: {cur_rk}, ranks: {ranks}")
        assert cur_rk is not None and cur_rk in ranks
        ranks.remove(cur_rk)
        self.set("group", ranks)

    def log(self):
        """
        Gathers trace records from all ranks, and on rank 0,
        puts the data onto a queue to be written to disk by a separate thread.
        """
        if not self.is_tracing_active():
            return

        # Ensure save thread is initialized on rank 0
        if torch.distributed.get_rank() == 0 and self._save_thread is None:
            self._initialize_save_thread()

        # Each rank constructs its own filename and payload
        dp_rank = parallel_state.get_data_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        filename = f"benchmark-data-{dp_rank}-pipeline-{pp_rank}-tensor-{tp_rank}.json"
        payload = (filename, self._records)

        # Everyone needs to participate in the gather call.
        # Gathers a list of payloads from all ranks.
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered_payloads = [None] * world_size
            torch.distributed.gather_object(
                payload,
                gathered_payloads if torch.distributed.get_rank() == 0 else None,
                dst=0,
            )
        else:
            gathered_payloads = [payload]


        # Rank 0 handles the logging
        if torch.distributed.get_rank() == 0:
            assert self._work_queue is not None, "Work queue is not initialized on rank 0"
            # Filter out empty records and put valid payloads onto the queue
            # A payload is a (filename, records_list) tuple.
            payloads_to_save = [p for p in gathered_payloads if p and p[1]]
            if payloads_to_save:
                self._work_queue.put(payloads_to_save)

        # Clear records after sending them for logging
        self._records = []

    def shutdown(self):
        """
        Signal the saver thread to shut down and wait for it to finish.
        """
        if self.global_args and self.global_args.trace:
            # Barrier to ensure all ranks have finished their work before shutdown
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if self._save_thread is not None and self._save_thread.is_alive():
                if self._work_queue is not None:
                    # Send sentinel to the thread and wait for it to process everything
                    self._work_queue.put(None)
                    self._work_queue.join()
                self._save_thread.join(timeout=10)
                if self._save_thread.is_alive():
                    print("Warning: Trace saving thread did not terminate gracefully.", flush=True)
                self._save_thread = None
                self._work_queue = None

    def should_log_this_iter(self) -> bool:
        """Checks if we should log at the end of this iteration."""
        args = self.global_args
        if not args or not args.trace or not self.is_tracing_active():
            return False
        
        # self.iter is 1-based.
        idx = (self.iter - 1) % self.interval
        return idx == self.continuous_trace_iters - 1

    def is_tracing_active(self) -> bool:
        """Checks if we are in a tracing interval."""
        args = self.global_args
        if args is None:
            return False
        if not args.trace or self.interval is None or self.continuous_trace_iters is None:
            return False

        # self.iter is 1-based, but the argument is 0-based.
        idx = (self.iter - 1) % self.interval
        return 0 <= idx < self.continuous_trace_iters


tracers = Tracer()


def get_tracer() -> Tracer:
    return tracers


def get_tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.nelement() * tensor.element_size()
