from dataclasses import dataclass
from functools import wraps
import json
import time
import torch
from typing import Any, Dict, List, Optional


class _TracerScope:
    def __init__(self, tracer: "Tracer", name: Optional[str], in_attrs: Dict[str, Any], out_attrs: Dict[str, Any]) -> None:
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
        if q in self.out_attrs and self.out_attrs[q] is None:
            self.out_attrs[q] = v
            return True
        else:
            return False


@dataclass
class _Pending:
    name: str
    phase: str
    event: torch.cuda.Event
    attrs: Dict[str, Any]


class Tracer:
    """Global tracer to record and print timestamp during training process"""

    def __init__(self) -> None:
        self._records: List[Any] = []
        self._cur: int = None
        self._pending_pad_before: int = None
        self._pendings: List[_Pending] = None
        self._scopes: List[_TracerScope] = []

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
        self._pendings.append(pending)

    def _add_cuda_event(self, name: str, phase: str, attrs: Dict[str, Any]) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        pending = _Pending(name, phase, event, attrs)
        self._add_pending(pending)

    def iteration_begin(self) -> None:
        """Start tracing an iteration. Note that this performs synchronization."""
        pad_before = self._calibrate()
        self._pending_pad_before = pad_before
        self._pendings = []
        # Mark the beginning of the iteration
        self._add_cuda_event("iteration", "B", {})

    def is_tracing(self) -> bool:
        return self._pendings is not None

    def _process_pending_scope(self, ref_ts: int, ref_event: torch.cuda.Event, i: int) -> int:
        """Process the pending scopes.
        ref must be a "B".
        Args:
            ref_ts: reference timestamp.
            ref_event: reference event.
            i: index of the pending scope to be processed.
        Returns:
            The next index to process.
        """
        while i < len(self._pendings):
            pending = self._pendings[i]
            elapsed = int(ref_event.elapsed_time(pending.event) * 1e6)
            rel_ts = ref_ts + elapsed
            chrome_event = {
                **pending.attrs,
                "name": pending.name,
                "ph": pending.phase,
                "rel_ts": rel_ts,
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
                        gb = pending.attrs["data"] / (2 ** 27)
                        secs = elapsed / 1e9
                        bandwidth = gb / secs # Gbps
                        last["bandwidth"] = bandwidth
                return i
        assert i == len(self._pendings), "Mismatched scopes"
        return i

    def iteration_end(self) -> None:
        """End tracing an iteration. Note that this performs synchronization."""
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
        iteration_begin_event = self._pendings[0].event
        # We cannot know the absolute timestamp of the first event, so we set it to 0.
        self._process_pending_scope(0, iteration_begin_event, 1)
        end = self._last_record()
        end["duration_wall"] = wall_duration
        end["duration_cuda"] = end["rel_ts"]

        self._pending_pad_before = None
        self._pendings = None

    def _tick(self, name: str, phase: str, attrs: Dict[str, Any]) -> None:
        if self.is_tracing():
            self._add_cuda_event(name, phase, attrs)

    def tick(self, name: str, **attrs: Any) -> None:
        """Record an event."""
        self._tick(name, "i", attrs)

    def scope(self, name: Optional[str], *args, ctx: Dict[str, Any] = {}, slots: List[str] = [], **kwargs: Any) -> _TracerScope:
        """Create a scope of code.
        Args:
            name: Name of the scope. If None, the scope is not timed.
            ctx: Parameters to be passed to the scope.
            kwargs: Items to be recorded. If an item is None, it should be filled by some inner scope.
            slots: Parameters that are passed to the scope and must be filled. (They go to both ctx and kwargs.)
        """
        assert len(args) == 0, "Positional arguments are not supported"
        for slot in slots:
            ctx[slot] = True
            kwargs[slot] = None
        return _TracerScope(self, name=name, in_attrs=ctx, out_attrs=kwargs)

    def scoped(self, func):
        """Decorator to time a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.scope(func.__name__):
                return func(*args, **kwargs)
        return wrapper

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
        assert False, f"Cannot find a requiring scope for {q}"

    def log(self, filename) -> None:
        with open(filename, "w", newline="") as file:
            json.dump(self._records, file, indent=2)


tracers = Tracer()


def get_tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.nelement() * tensor.element_size()
