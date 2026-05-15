import torch
import megatron.virtual_tensor_parallel_communication as dist
from megatron.core import parallel_state
import threading
import copy
import time
import contextlib
from dataclasses import dataclass
import operator
import os


def _prod(shape):
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


class _RequestGroup:

    def __init__(self, reqs):
        self.reqs = reqs or []
        self.waited = False

    def wait(self):
        if self.waited:
            return
        for req in self.reqs:
            req.wait()
        self.waited = True


@dataclass(frozen=True)
class _PoolKey:
    kind: str
    dtype: torch.dtype
    shapes: tuple


class _BufferLease:

    def __init__(self, pool, key, buffers):
        self.pool = pool
        self.key = key
        self.buffers = buffers
        self.released = False

    def release(self):
        if self.released:
            return
        self.pool.release(self)
        self.released = True


class ActivationConnectionPool:

    def __init__(self):
        self.free = {}
        self.lock = threading.Lock()

    def acquire(self, kind, shapes, dtype):
        normalized_shapes = tuple(tuple(int(dim) for dim in shape) for shape in shapes)
        key = _PoolKey(kind=kind, dtype=dtype, shapes=normalized_shapes)
        with self.lock:
            buffers = self.free.get(key)
            if buffers:
                leased = buffers.pop()
                if not buffers:
                    self.free.pop(key, None)
                return _BufferLease(self, key, leased)

        leased = [
            torch.empty(shape, device=torch.cuda.current_device(), dtype=dtype)
            for shape in normalized_shapes
        ]
        return _BufferLease(self, key, leased)

    def release(self, lease):
        with self.lock:
            self.free.setdefault(lease.key, []).append(lease.buffers)


class PersistentActivationTransport:

    def __init__(self):
        self.pool = ActivationConnectionPool()

    def acquire_send_buffers(self, shapes, dtype):
        return self.pool.acquire('send', shapes, dtype)

    def acquire_recv_buffers(self, shapes, dtype):
        return self.pool.acquire('recv', shapes, dtype)


Transport = PersistentActivationTransport()

ACTIVATION_CHANNEL_CHECKPOINT = "checkpoint"
ACTIVATION_CHANNEL_LINEAR = "linear"
ACTIVATION_CHANNELS = (ACTIVATION_CHANNEL_CHECKPOINT, ACTIVATION_CHANNEL_LINEAR)
_ACTIVATION_STORE_DEBUG_ENABLED = None
_ACTIVATION_TRANSPORT_PROFILE_ENABLED = None
_LINEAR_LOAD_ORDER_STATE = threading.local()
_MAX_INFLIGHT_SENDS = None


def _is_activation_store_debug_enabled():
    global _ACTIVATION_STORE_DEBUG_ENABLED
    if _ACTIVATION_STORE_DEBUG_ENABLED is not None:
        return _ACTIVATION_STORE_DEBUG_ENABLED

    enabled = False
    try:
        from megatron.training.global_vars import get_args
        args = get_args()
        enabled = bool(getattr(args, "activation_store_debug", False))
    except Exception:
        enabled = False

    _ACTIVATION_STORE_DEBUG_ENABLED = enabled
    return enabled


def _is_activation_transport_profile_enabled():
    global _ACTIVATION_TRANSPORT_PROFILE_ENABLED
    if _ACTIVATION_TRANSPORT_PROFILE_ENABLED is not None:
        return _ACTIVATION_TRANSPORT_PROFILE_ENABLED

    enabled = False
    try:
        from megatron.training.global_vars import get_args
        args = get_args()
        enabled = bool(getattr(args, "activation_transport_profile", False))
    except Exception:
        enabled = False

    _ACTIVATION_TRANSPORT_PROFILE_ENABLED = enabled
    return enabled


def _max_inflight_sends():
    global _MAX_INFLIGHT_SENDS
    if _MAX_INFLIGHT_SENDS is None:
        try:
            _MAX_INFLIGHT_SENDS = max(1, int(os.getenv("ACTS_MAX_INFLIGHT_SENDS", "8")))
        except Exception:
            _MAX_INFLIGHT_SENDS = 8
    return _MAX_INFLIGHT_SENDS


def _write_into_log(message):
    if not _is_activation_store_debug_enabled():
        return
    try:
        dist.write_into_log(message)
    except Exception:
        pass


def _write_trace_into_log(message):
    try:
        dist.write_into_log(message)
    except Exception:
        pass


def _checkpoint_seq_debug_enabled():
    return os.getenv("ACTS_CHECKPOINT_SEQ_DEBUG", "0") == "1"


def _disable_async_transport() -> bool:
    return os.getenv("ACTS_DISABLE_ASYNC_COMM", "0") == "1"


def _format_profile_value(value):
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def write_transport_profile(event, **metrics):
    if not _is_activation_transport_profile_enabled():
        return
    parts = [f"event={event}"]
    for key, value in metrics.items():
        parts.append(f"{key}={_format_profile_value(value)}")
    try:
        dist.write_into_log("acts prof " + " ".join(parts))
    except Exception:
        pass


def _activation_payload_bytes(activation):
    if activation is None:
        return 0
    if isinstance(activation, tuple):
        total = 0
        for element in activation:
            total += _activation_payload_bytes(element)
        return total
    return activation.numel() * activation.element_size()


def _as_concrete_int(value, name: str) -> int:
    """Convert scalar-like metadata to a concrete Python int for tensor shapes."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"{name} must be scalar, got numel={value.numel()}")
        # Move to CPU and extract eagerly to avoid symbolic shape values.
        value = value.detach().to(device="cpu", dtype=torch.int64).item()
    try:
        concrete = operator.index(value)
    except Exception:
        concrete = int(value)
    return int(concrete)


def _as_tensor_list(maybe_list):
    if isinstance(maybe_list, list):
        return maybe_list
    return [maybe_list]


def _pick_metadata_slot(meta_tensors, name: str, min_value: int = 0, max_value: int = 1_000_000):
    tensors = _as_tensor_list(meta_tensors)
    candidates = []
    for idx, tensor in enumerate(tensors):
        try:
            val = _as_concrete_int(tensor.view(-1)[0], f"{name}[{idx}]")
        except Exception:
            continue
        if min_value <= val <= max_value:
            candidates.append((idx, val))
    if not candidates:
        preview = []
        for idx, tensor in enumerate(tensors):
            try:
                preview.append((idx, _as_concrete_int(tensor.view(-1)[0], f"{name}[{idx}]")))
            except Exception:
                preview.append((idx, "err"))
        raise RuntimeError(
            f"Failed to pick valid metadata slot for {name}; candidates={preview}"
        )
    return candidates[0]


def _normalize_shape_list(shape_payload):
    """Return shape metadata as a list[tuple[int,...]] regardless of wire form."""
    if isinstance(shape_payload, torch.Tensor):
        # Single range shape received as one 1-D tensor.
        return [tuple(shape_payload.tolist())]
    normalized = []
    for item in shape_payload:
        if isinstance(item, torch.Tensor):
            normalized.append(tuple(item.tolist()))
        else:
            normalized.append(tuple(item))
    return normalized

class ActivationSet:

    def __init__(self, Last = None):
        self.activations = []
        self.activations_reqs = []
        self.dp_dim = 0
        self.total_payload_bytes = 0
        self.payload_shapes = []
        self.recv_buffer_lease = None
        self.inflight_send_leases = []
        self._view_plan = None
        if Last is not None:
            self.id = Last.id
            self.num = Last.num
            self.shapes = Last.shapes
            self.first_comm = Last.first_comm
            self.payload_shapes = Last.payload_shapes
            self._view_plan = Last._view_plan
            self.send_seq = Last.send_seq
            self.recv_seq = Last.recv_seq
        else:
            self.id = []
            self.num = 0
            self.shapes = []
            self.first_comm = True
            self.send_seq = 0
            self.recv_seq = 0

    def store_activation(self, activation):
        self.num += 1
        self.total_payload_bytes += _activation_payload_bytes(activation)
        if isinstance(activation, tuple):
            for element in activation:
                self.activations.append(element)
                if element is None:
                    self.id.append(-self.num)
                else:
                    # print('store', element.shape)
                    self.id.append(self.num)
        else:
            self.activations.append(activation)
            # print('store', activation.shape)
            if activation is None:
                self.id.append(-self.num)
            else:
                self.id.append(self.num)
        if _checkpoint_seq_debug_enabled():
            head_id = self.id[0] if len(self.id) > 0 else None
            tail_id = self.id[-1] if len(self.id) > 0 else None
            _write_into_log(
                "acts checkpoint_seq store "
                f"num={self.num} qlen={len(self.activations)} "
                f"id_head={head_id} id_tail={tail_id}"
            )

    def _activation_matches_expected_last_dim(self, activation, expected_last_dim):
        if expected_last_dim is None:
            return True
        if activation is None:
            return False

        # Non-tuple activation payloads are stored as lists of per-range shards.
        if isinstance(activation, list) and len(activation) > 0 and activation[0] is not None:
            return activation[0].shape[-1] == expected_last_dim

        # Tuple payloads are lists/None per tuple element; accept if any tensor element matches.
        if isinstance(activation, tuple):
            for element in activation:
                if isinstance(element, list) and len(element) > 0 and element[0] is not None:
                    if element[0].shape[-1] == expected_last_dim:
                        return True
            return False

        return False

    def load_activation(self, from_end=False, expected_last_dim=None, channel=None):
        load_start = time.time()
        trace_enabled = os.getenv("ACTS_LOAD_TRACE", "0") == "1"
        load_id = -1
        if trace_enabled:
            try:
                load_id = int(time.time() * 1000000)
            except Exception:
                load_id = -1
        if len(self.activations) == 0:
            raise RuntimeError("Activation queue is empty while trying to load activation.")

        chosen_index = None
        if expected_last_dim is None:
            if from_end:
                chosen_index = len(self.activations) - 1
                reqs = self.activations_reqs.pop()
                activation = self.activations.pop()
            else:
                chosen_index = 0
                reqs = self.activations_reqs.pop(0)
                activation = self.activations.pop(0)
        else:
            if from_end:
                candidate_indices = range(len(self.activations) - 1, -1, -1)
            else:
                candidate_indices = range(len(self.activations))

            for idx in candidate_indices:
                if self._activation_matches_expected_last_dim(
                    self.activations[idx], expected_last_dim
                ):
                    chosen_index = idx
                    break

            if chosen_index is None:
                raise RuntimeError(
                    f"No activation matches expected_last_dim={expected_last_dim}. "
                    f"queue_len={len(self.activations)}"
                )

            reqs = self.activations_reqs.pop(chosen_index)
            activation = self.activations.pop(chosen_index)
        if trace_enabled:
            _write_trace_into_log(
                "acts_load_trace "
                f"phase=pick channel={channel} load_id={load_id} "
                f"chosen_index={chosen_index} reqs={len(reqs)} qlen_after_pop={len(self.activations)} "
                f"from_end={int(bool(from_end))} expected_last_dim={expected_last_dim} ts={time.time():.6f}"
            )
        if _checkpoint_seq_debug_enabled():
            head_id = self.id[0] if len(self.id) > 0 else None
            tail_id = self.id[-1] if len(self.id) > 0 else None
            _write_into_log(
                "acts checkpoint_seq load_pick "
                f"chosen_index={chosen_index} from_end={int(bool(from_end))} "
                f"expected_last_dim={expected_last_dim} "
                f"id_head={head_id} id_tail={tail_id} "
                f"reqs={len(reqs)} qlen_after_pop={len(self.activations)}"
            )
        wait_start = time.time()
        for req in reqs:
            req.wait()
        wait_end = time.time()
        if trace_enabled:
            _write_trace_into_log(
                "acts_load_trace "
                f"phase=req_wait_done channel={channel} load_id={load_id} "
                f"wait_s={wait_end - wait_start:.6f} reqs={len(reqs)} ts={time.time():.6f}"
            )
        if len(reqs) > 0:
            _write_into_log(f"acts load_wait {wait_end - wait_start}")
        write_transport_profile(
            "load_activation",
            channel=channel if channel is not None else "unknown",
            queue_depth=len(self.activations) + 1,
            reqs=len(reqs),
            wait_s=wait_end - wait_start,
            from_end=int(bool(from_end)),
            expected_last_dim=expected_last_dim if expected_last_dim is not None else -1,
            dt_s=time.time() - load_start,
        )
        if isinstance(activation, tuple):
            cat_start = time.time()
            res = []
            for element in activation:
                if element is None:
                    res.append(None)
                else:
                    res.append(torch.cat(element, dim=self.dp_dim))
            if trace_enabled:
                _write_trace_into_log(
                    "acts_load_trace "
                    f"phase=cat_done channel={channel} load_id={load_id} "
                    f"cat_s={time.time() - cat_start:.6f} ts={time.time():.6f}"
                )
        # if isinstance(activation, tuple):
        #     for element in activation:
        #         print('load', element.shape)
            if trace_enabled:
                _write_trace_into_log(
                    "acts_load_trace "
                    f"phase=return channel={channel} load_id={load_id} "
                    f"total_s={time.time() - load_start:.6f} ts={time.time():.6f}"
                )
            return tuple(res)
        else:
            cat_start = time.time()
            out = torch.cat(activation, dim=self.dp_dim)
            if trace_enabled:
                _write_trace_into_log(
                    "acts_load_trace "
                    f"phase=cat_done channel={channel} load_id={load_id} "
                    f"cat_s={time.time() - cat_start:.6f} ts={time.time():.6f}"
                )
                _write_trace_into_log(
                    "acts_load_trace "
                    f"phase=return channel={channel} load_id={load_id} "
                    f"total_s={time.time() - load_start:.6f} ts={time.time():.6f}"
                )
            return out

    def _get_payload_shapes(self):
        if self.payload_shapes:
            return self.payload_shapes

        ranges = parallel_state.get_forward_backward_parallel_ranges()
        payload_sizes = [0 for _ in ranges]
        for i in range(self.num):
            if self.id[i] <= 0:
                continue
            for range_idx, shape in enumerate(self.shapes[i]):
                payload_sizes[range_idx] += _prod(shape)
        self.payload_shapes = [(payload_size,) for payload_size in payload_sizes]
        return self.payload_shapes

    def _pack_activation_payload(self, dp_dim):
        ranges = parallel_state.get_forward_backward_parallel_ranges()
        tensors = [activation for activation in self.activations if activation is not None]
        if not tensors:
            return None

        # Compute payload sizes using shape arithmetic to avoid creating many temporary views.
        payload_shapes = []
        for start, end in ranges:
            length = end - start
            total_numel = 0
            for activation in tensors:
                total_numel += (activation.numel() // activation.size(dp_dim)) * length
            payload_shapes.append((total_numel,))
        self.payload_shapes = payload_shapes

        lease = Transport.acquire_send_buffers(payload_shapes, tensors[0].dtype)
        # Pack one fused buffer per destination range to reduce Python overhead and
        # avoid many tiny copy kernels from nested loops.
        for range_idx, (start, end) in enumerate(ranges):
            length = end - start
            flat_shards = []
            for activation in tensors:
                shard = torch.narrow(activation, dim=dp_dim, start=start, length=length)
                if not shard.is_contiguous():
                    shard = shard.contiguous()
                flat_shards.append(shard.view(-1))

            if len(flat_shards) == 1:
                lease.buffers[range_idx].copy_(flat_shards[0], non_blocking=True)
            else:
                torch.cat(flat_shards, out=lease.buffers[range_idx])

        return lease

    def _build_activation_views(self, reqs):
        if self._view_plan is None:
            offsets = [0 for _ in self.payload_shapes]
            group = []
            view_plan = []
            for i in range(self.num):
                if self.id[i] > 0:
                    part = []
                    for range_idx, shape in enumerate(self.shapes[i]):
                        numel = _prod(shape)
                        start = offsets[range_idx]
                        end = start + numel
                        part.append((range_idx, start, end, shape))
                        offsets[range_idx] = end
                    group.append(part)
                else:
                    group.append(None)

                if i == self.num - 1 or abs(self.id[i]) != abs(self.id[i + 1]):
                    view_plan.append(tuple(group))
                    group = []
            self._view_plan = tuple(view_plan)

        request_group = _RequestGroup(reqs)
        self.activations = []
        self.activations_reqs = []
        buffers = self.recv_buffer_lease.buffers if self.recv_buffer_lease is not None else None

        for group in self._view_plan:
            if len(group) == 1:
                part = group[0]
                if part is None:
                    self.activations.append(None)
                    self.activations_reqs.append([])
                else:
                    if buffers is None:
                        raise RuntimeError("Missing receive buffers for non-empty activation payload.")
                    element = [
                        buffers[range_idx][start:end].view(shape)
                        for range_idx, start, end, shape in part
                    ]
                    self.activations.append(element)
                    self.activations_reqs.append([request_group])
                continue

            tuple_parts = []
            has_tensor = False
            for part in group:
                if part is None:
                    tuple_parts.append(None)
                else:
                    has_tensor = True
                    if buffers is None:
                        raise RuntimeError("Missing receive buffers for non-empty activation payload.")
                    tuple_parts.append(
                        [
                            buffers[range_idx][start:end].view(shape)
                            for range_idx, start, end, shape in part
                        ]
                    )
            self.activations.append(tuple(tuple_parts))
            self.activations_reqs.append([request_group] if has_tensor else [])

    def release_buffers(self):
        if self.recv_buffer_lease is not None:
            self.recv_buffer_lease.release()
            self.recv_buffer_lease = None

    def _retire_finished_send_leases(self):
        if not self.inflight_send_leases:
            return
        # Never poll Work.is_completed() here. In some NCCL fault states that
        # path can segfault. Instead, retire leases based on a CUDA event
        # recorded after enqueue on the same stream.
        remaining = []
        for reqs, lease, done_event in self.inflight_send_leases:
            done = done_event.query() if done_event is not None else False
            if done:
                lease.release()
            else:
                remaining.append((reqs, lease, done_event))
        self.inflight_send_leases = remaining

    def _chunk_tensor_views(self, buffers, chunk_bytes, max_chunks=None):
        if not buffers:
            return []
        element_size = buffers[0].element_size()
        chunk_elems = max(1, int(chunk_bytes // element_size))
        chunk_groups = []
        for buffer in buffers:
            flat = buffer.view(-1)
            total = flat.numel()
            if max_chunks is not None and max_chunks > 0:
                # Favor fewer larger chunks to reduce torch.distributed call overhead.
                chunk_elems = max(chunk_elems, (total + max_chunks - 1) // max_chunks)
            start = 0
            while start < total:
                length = min(chunk_elems, total - start)
                chunk_groups.append([flat.narrow(0, start, length)])
                start += length
        return chunk_groups

    def wait_for_inflight_sends(self, channel=None):
        if not self.inflight_send_leases:
            return
        wait_start = time.time()
        inflight_before = len(self.inflight_send_leases)
        for reqs, lease, _ in self.inflight_send_leases:
            for req in reqs:
                req.wait()
            lease.release()
        self.inflight_send_leases = []
        write_transport_profile(
            "send_wait_inflight",
            channel=channel if channel is not None else "unknown",
            inflight_before=inflight_before,
            dt_s=time.time() - wait_start,
        )
    
    def send_coresponding_activations(self, dp_dim, config, channel=None):
        from megatron.core.pipeline_parallel import p2p_communication
        self._retire_finished_send_leases()
        self.send_seq += 1
        _write_into_log(
            f"acts send_call_begin channel={channel} seq={self.send_seq} first_comm={int(bool(self.first_comm))} num={len(self.activations)}"
        )
        # Bound outstanding async sends to avoid NCCL/P2P queue backpressure
        # when linear-channel payloads are large. Unbounded in-flight sends can
        # make subsequent enqueue calls block for long periods.
        if len(self.inflight_send_leases) > _max_inflight_sends():
            self.wait_for_inflight_sends(channel=channel)
        total_start = time.time()
        write_transport_profile(
            "send_begin",
            channel=channel if channel is not None else "unknown",
            num=len(self.activations),
            payload_bytes=self.total_payload_bytes,
            inflight=len(self.inflight_send_leases),
            first_comm=int(bool(self.first_comm)),
        )
        if self.first_comm:
            meta_start = time.time()
            _write_into_log(
                f"acts send_meta_begin channel={channel} seq={self.send_seq} num={len(self.activations)}"
            )
            num = torch.tensor(
                [len(self.activations)],
                device=torch.cuda.current_device(),
                dtype=torch.int64,
            )
            # print('gogogo', num, dist.get_rank())
            # print('send done', dist.get_rank())
            p2p_communication.send_corresponding_forward(num, config, bypass_controller=True)
            p2p_communication.send_corresponding_forward(
                torch.tensor(self.id, device=torch.cuda.current_device(), dtype=torch.int64),
                config,
                bypass_controller=True,
            )

            ranges = parallel_state.get_forward_backward_parallel_ranges()
            for activation in self.activations:
                if activation is not None:
                    shape = []
                    for r in ranges:
                        ele = list(activation.shape)
                        ele[dp_dim] = r[1]-r[0]
                        shape.append(
                            torch.tensor(
                                ele,
                                device=torch.cuda.current_device(),
                                dtype=torch.int64,
                            )
                        )
                    # print(ranges, shape)
                    p2p_communication.send_corresponding_forward(
                        torch.tensor(
                            [len(shape[0])],
                            device=torch.cuda.current_device(),
                            dtype=torch.int64,
                        ),
                        config,
                        bypass_controller=True,
                    )
                    p2p_communication.send_corresponding_forward(shape, config, bypass_controller=True)

            self.first_comm = False
            meta_end = time.time()
            _write_into_log(
                f"acts send_meta_end channel={channel} seq={self.send_seq} dt={meta_end - meta_start}"
            )
            _write_into_log(f"acts send_meta {meta_end - meta_start}")
            write_transport_profile(
                "send_meta",
                channel=channel if channel is not None else "unknown",
                dt_s=meta_end - meta_start,
                num=len(self.activations),
            )

        # print('send done', dist.get_rank())
        pack_start = time.time()
        send_lease = self._pack_activation_payload(dp_dim)
        pack_end = time.time()
        _write_into_log(f"acts send_pack {pack_end - pack_start}")
        write_transport_profile(
            "send_pack",
            channel=channel if channel is not None else "unknown",
            dt_s=pack_end - pack_start,
            payload_bytes=self.total_payload_bytes,
            num=len(self.activations),
        )
        if send_lease is not None:
            enqueue_start = time.time()
            reqs = []
            if channel == ACTIVATION_CHANNEL_LINEAR:
                # Use coarse chunking only for very large payloads.
                # Fine-grained chunking increases call overhead and hurts throughput.
                if self.total_payload_bytes > 512 * 1024 * 1024:
                    chunk_groups = self._chunk_tensor_views(
                        send_lease.buffers,
                        chunk_bytes=256 * 1024 * 1024,
                        max_chunks=2,
                    )
                    for chunk_group in chunk_groups:
                        chunk_reqs = p2p_communication.send_corresponding_forward(
                            chunk_group,
                            config,
                            bypass_controller=True,
                            wait_on_reqs=False,
                        )
                        if chunk_reqs:
                            reqs.extend(chunk_reqs)
                else:
                    reqs = p2p_communication.send_corresponding_forward(
                        send_lease.buffers,
                        config,
                        bypass_controller=True,
                        wait_on_reqs=False,
                    )
            else:
                reqs = p2p_communication.send_corresponding_forward(
                    send_lease.buffers,
                    config,
                    bypass_controller=True,
                    wait_on_reqs=False,
                )
            enqueue_end = time.time()
            _write_into_log(f"acts send_enqueue {enqueue_end - enqueue_start}")
            req_count = len(reqs) if reqs else 0
            write_transport_profile(
                "send_enqueue",
                channel=channel if channel is not None else "unknown",
                dt_s=enqueue_end - enqueue_start,
                payload_bytes=self.total_payload_bytes,
                reqs=req_count,
            )
            if reqs:
                if _disable_async_transport():
                    for req in reqs:
                        req.wait()
                    send_lease.release()
                else:
                    done_event = torch.cuda.Event(enable_timing=False)
                    done_event.record(torch.cuda.current_stream())
                    self.inflight_send_leases.append((reqs, send_lease, done_event))
            else:
                send_lease.release()
        total_end = time.time()
        _write_into_log(f"acts send_total {total_end - total_start}")
        write_transport_profile(
            "send_total",
            channel=channel if channel is not None else "unknown",
            dt_s=total_end - total_start,
            payload_bytes=self.total_payload_bytes,
            inflight=len(self.inflight_send_leases),
        )

        end_time = time.time()
        _write_into_log(f"pure p2p {end_time - pack_end}")
        _write_into_log(
            f"acts send_call_end channel={channel} seq={self.send_seq} inflight={len(self.inflight_send_leases)}"
        )

        # print('send done', dist.get_rank())

    def recv_coresponding_activations(self, dp_dim, config, channel=None):
        from megatron.core.pipeline_parallel import p2p_communication
        # print('recv done', dist.get_rank())
        self.recv_seq += 1
        _write_into_log(
            f"acts recv_call_begin channel={channel} seq={self.recv_seq} first_comm={int(bool(self.first_comm))} num={self.num}"
        )
        total_start = time.time()
        write_transport_profile(
            "recv_begin",
            channel=channel if channel is not None else "unknown",
            first_comm=int(bool(self.first_comm)),
            num=self.num,
            queue_pending=len(self.activations),
        )
        if self.first_comm:
            meta_start = time.time()
            _write_into_log(
                f"acts recv_meta_begin channel={channel} seq={self.recv_seq}"
            )
            num = torch.empty(1, device=torch.cuda.current_device(), dtype=torch.int64)
            num_all = p2p_communication.recv_corresponding_forward(
                num.shape,
                config,
                dtype=torch.int64,
                bypass_controller=True,
            )
            slot_idx, num_val = _pick_metadata_slot(num_all, "num", min_value=0, max_value=1_000_000)
            id = torch.empty((num_val,), device=torch.cuda.current_device(), dtype=torch.int64)
            id_all = p2p_communication.recv_corresponding_forward(
                id.shape,
                config,
                dtype=torch.int64,
                bypass_controller=True,
            )
            id = _as_tensor_list(id_all)[slot_idx]
            shapes = []
            for i in range(num_val):
                if _as_concrete_int(id[i], "id[i]") > 0:
                    ndim = torch.empty(1, device=torch.cuda.current_device(), dtype=torch.int64)
                    ndim_all = p2p_communication.recv_corresponding_forward(
                        ndim.shape,
                        config,
                        dtype=torch.int64,
                        bypass_controller=True,
                    )
                    ndim = _as_tensor_list(ndim_all)[slot_idx]
                    ndim_val = _as_concrete_int(ndim, "ndim")
                    shape = torch.empty((ndim_val,), device=torch.cuda.current_device(), dtype=torch.int64)
                    shape_all = p2p_communication.recv_corresponding_forward(
                        shape.shape,
                        config,
                        dtype=torch.int64,
                        bypass_controller=True,
                    )
                    shape = _as_tensor_list(shape_all)[slot_idx]
                    shapes.append(_normalize_shape_list(shape))
                else:
                    shapes.append(0)
            self.num = num_val
            self.id = id
            self.shapes = shapes
            self.payload_shapes = []
            self._get_payload_shapes()
            self._view_plan = None
            self.first_comm = False
            meta_end = time.time()
            _write_into_log(
                f"acts recv_meta_end channel={channel} seq={self.recv_seq} dt={meta_end - meta_start} num={self.num}"
            )
            _write_into_log(f"acts recv_meta {meta_end - meta_start}")
            write_transport_profile(
                "recv_meta",
                channel=channel if channel is not None else "unknown",
                dt_s=meta_end - meta_start,
                num=self.num,
            )

        # print('recv done', dist.get_rank())
        setup_start = time.time()
        self.release_buffers()
        reqs = []
        if any(shape[0] > 0 for shape in self.payload_shapes):
            acquire_start = time.time()
            self.recv_buffer_lease = Transport.acquire_recv_buffers(
                self.payload_shapes,
                config.pipeline_dtype,
            )
            acquire_end = time.time()
            _write_into_log(f"acts recv_acquire {acquire_end - acquire_start}")
            dtype_element_size = torch.empty((), dtype=config.pipeline_dtype).element_size()
            write_transport_profile(
                "recv_acquire",
                channel=channel if channel is not None else "unknown",
                dt_s=acquire_end - acquire_start,
                payload_bytes=sum(shape[0] for shape in self.payload_shapes) * dtype_element_size,
            )
            enqueue_start = time.time()
            if channel == ACTIVATION_CHANNEL_LINEAR:
                total_payload_bytes = (
                    sum(shape[0] for shape in self.payload_shapes)
                    * torch.empty((), dtype=config.pipeline_dtype).element_size()
                )
                if total_payload_bytes > 512 * 1024 * 1024:
                    chunk_groups = self._chunk_tensor_views(
                        self.recv_buffer_lease.buffers,
                        chunk_bytes=256 * 1024 * 1024,
                        max_chunks=2,
                    )
                    reqs = []
                    for chunk_group in chunk_groups:
                        _, chunk_reqs = p2p_communication.recv_corresponding_forward_async_into(
                            chunk_group,
                            config,
                            dtype=config.pipeline_dtype,
                            bypass_controller=True,
                        )
                        if chunk_reqs:
                            reqs.extend(chunk_reqs)
                else:
                    _, reqs = p2p_communication.recv_corresponding_forward_async_into(
                        self.recv_buffer_lease.buffers,
                        config,
                        dtype=config.pipeline_dtype,
                        bypass_controller=True,
                    )
            else:
                _, reqs = p2p_communication.recv_corresponding_forward_async_into(
                    self.recv_buffer_lease.buffers,
                    config,
                    dtype=config.pipeline_dtype,
                    bypass_controller=True,
                )
            enqueue_end = time.time()
            _write_into_log(f"acts recv_enqueue {enqueue_end - enqueue_start}")
            write_transport_profile(
                "recv_enqueue",
                channel=channel if channel is not None else "unknown",
                dt_s=enqueue_end - enqueue_start,
                reqs=len(reqs),
            )
            if reqs and _disable_async_transport():
                for req in reqs:
                    req.wait()
                reqs = []
        setup_end = time.time()
        _write_into_log(f"acts recv_setup {setup_end - setup_start}")
        write_transport_profile(
            "recv_setup",
            channel=channel if channel is not None else "unknown",
            dt_s=setup_end - setup_start,
        )

        view_start = time.time()
        self._build_activation_views(reqs)
        view_end = time.time()
        _write_into_log(f"acts recv_build_views {view_end - view_start}")
        write_transport_profile(
            "recv_build_views",
            channel=channel if channel is not None else "unknown",
            dt_s=view_end - view_start,
            num=self.num,
        )
        total_end = time.time()
        _write_into_log(f"acts recv_total {total_end - total_start}")
        write_transport_profile(
            "recv_total",
            channel=channel if channel is not None else "unknown",
            dt_s=total_end - total_start,
            num=self.num,
        )

        _write_into_log(f"pure p2p {view_end - setup_end}")
        _write_into_log(
            f"acts recv_call_end channel={channel} seq={self.recv_seq} num={self.num}"
        )

        self.dp_dim = dp_dim

    def reset(self, ):
        self.activations = []
        self.activations_reqs = []
        self.id = []
        self.num = 0
        self.total_payload_bytes = 0
        self.shapes = []
        self.payload_shapes = []
        self._view_plan = None
        self.release_buffers()
        self._retire_finished_send_leases()

TensorStore = None
TensorStoreSets = None
LastStore = None


def _log_debug(message):
    _write_into_log(message)


def _summarize_queue_state(activation_set):
    return (
        f"num={activation_set.num} ids={len(activation_set.id)} "
        f"acts={len(activation_set.activations)} reqs={len(activation_set.activations_reqs)}"
    )


def _current_store_index():
    idx = dist.get_thread_index()
    # Some execution paths can run on threads that are not registered in
    # virtual_tensor_parallel_communication (idx == -1). Normalize to 0 so
    # store/send/reset consistently use the same queue.
    if idx is None or idx < 0:
        _log_debug(f"acts debug normalize_thread_index raw={idx} mapped=0")
        return 0
    if TensorStore is not None and idx >= len(TensorStore):
        mapped = idx % len(TensorStore)
        _log_debug(
            f"acts debug normalize_thread_index raw={idx} mapped={mapped} stores={len(TensorStore)}"
        )
        return mapped
    return idx


def _new_channel_store(last_store=None):
    if last_store is None:
        return {channel: ActivationSet() for channel in ACTIVATION_CHANNELS}
    return {channel: ActivationSet(last_store[channel]) for channel in ACTIVATION_CHANNELS}


def _validate_channel(channel):
    if channel not in ACTIVATION_CHANNELS:
        raise ValueError(
            f"Unknown activation channel '{channel}'. Expected one of {ACTIVATION_CHANNELS}."
        )


def _linear_reverse_load_enabled():
    return bool(getattr(_LINEAR_LOAD_ORDER_STATE, "reverse", False))


@contextlib.contextmanager
def linear_activation_reverse_loading(enabled=True):
    prev = bool(getattr(_LINEAR_LOAD_ORDER_STATE, "reverse", False))
    _LINEAR_LOAD_ORDER_STATE.reverse = bool(enabled)
    try:
        yield
    finally:
        _LINEAR_LOAD_ORDER_STATE.reverse = prev

def init_sets():
    global TensorStore
    global TensorStoreSets
    global LastStore
    if parallel_state.is_forward_stage():
        # Preserve metadata across iterations to avoid repeated send_meta handshakes.
        if TensorStore is None or len(TensorStore) != dist.num_threads:
            TensorStore = [_new_channel_store() for _ in range(dist.num_threads)]
        else:
            for thread_store in TensorStore:
                for channel in ACTIVATION_CHANNELS:
                    thread_store[channel].reset()
    else:
        # Start each iteration with an empty queue, but keep LastStore metadata
        # so recv side can skip repeated recv_meta handshakes.
        if TensorStoreSets is None:
            TensorStoreSets = []
        else:
            while len(TensorStoreSets) > 0:
                for channel in ACTIVATION_CHANNELS:
                    TensorStoreSets[0][channel].release_buffers()
                TensorStoreSets.pop(0)

        if LastStore is not None:
            for channel in ACTIVATION_CHANNELS:
                last_channel_store = LastStore[channel]
                last_channel_store.activations = []
                last_channel_store.activations_reqs = []
                last_channel_store.release_buffers()
    # print(dist.get_rank(), TensorStore)

def store_activation(activation, channel=ACTIVATION_CHANNEL_CHECKPOINT):
    global TensorStore
    _validate_channel(channel)
    # print(dist.get_rank(), TensorStore)
    store_index = _current_store_index()
    TensorStore[store_index][channel].store_activation(activation)
    state = _summarize_queue_state(TensorStore[store_index][channel])
    _log_debug(
        f"acts debug store channel={channel} store_index={store_index} {state}"
    )

def load_activation(channel=ACTIVATION_CHANNEL_CHECKPOINT, expected_last_dim=None):
    global TensorStoreSets
    _validate_channel(channel)
    trace_enabled = os.getenv("ACTS_LOAD_TRACE", "0") == "1"
    from_end = channel == ACTIVATION_CHANNEL_LINEAR and _linear_reverse_load_enabled()
    queue_depth = len(TensorStoreSets)
    if trace_enabled:
        try:
            dist.write_into_log(
                "acts_load_trace "
                f"phase=begin channel={channel} set_depth={queue_depth} ts={time.time():.6f}"
            )
        except Exception:
            pass
    state_before = _summarize_queue_state(TensorStoreSets[0][channel])
    _log_debug(
        f"acts debug load_begin channel={channel} set_depth={queue_depth} {state_before}"
    )
    output = TensorStoreSets[0][channel].load_activation(
        from_end=from_end,
        expected_last_dim=expected_last_dim,
        channel=channel,
    )
    state_after = _summarize_queue_state(TensorStoreSets[0][channel])
    _log_debug(
        f"acts debug load_end channel={channel} set_depth={queue_depth} {state_after}"
    )
    if trace_enabled:
        try:
            dist.write_into_log(
                "acts_load_trace "
                f"phase=end channel={channel} set_depth={queue_depth} ts={time.time():.6f}"
            )
        except Exception:
            pass
    return output

def send_activations(config):
    global TensorStore
    # print('sending')
    store_index = _current_store_index()
    thread_store = TensorStore[store_index]
    # Prioritize checkpoint channel (small/latency-sensitive) before linear channel.
    channel_order = (ACTIVATION_CHANNEL_CHECKPOINT, ACTIVATION_CHANNEL_LINEAR)
    for channel in channel_order:
        _log_debug(
            f"acts debug send_begin channel={channel} store_index={store_index} "
            f"{_summarize_queue_state(thread_store[channel])}"
        )
        thread_store[channel].send_coresponding_activations(1, config, channel=channel)
        _log_debug(
            f"acts debug send_done channel={channel} store_index={store_index} "
            f"{_summarize_queue_state(thread_store[channel])}"
        )
        thread_store[channel].reset()
        _log_debug(
            f"acts debug send_reset channel={channel} store_index={store_index} "
            f"{_summarize_queue_state(thread_store[channel])}"
        )

def recv_activations(config):
    global TensorStoreSets
    global LastStore
    if LastStore is None:
        current_store = _new_channel_store()
    else:
        current_store = _new_channel_store(LastStore)
    # print('recving')
    channel_order = (ACTIVATION_CHANNEL_CHECKPOINT, ACTIVATION_CHANNEL_LINEAR)
    for channel in channel_order:
        _log_debug(
            f"acts debug recv_begin channel={channel} set_depth={len(TensorStoreSets)} "
            f"{_summarize_queue_state(current_store[channel])}"
        )
        current_store[channel].recv_coresponding_activations(1, config, channel=channel)
        _log_debug(
            f"acts debug recv_done channel={channel} set_depth={len(TensorStoreSets)} "
            f"{_summarize_queue_state(current_store[channel])}"
        )
    TensorStoreSets.append(current_store)
    _log_debug(f"acts debug recv_append new_set_depth={len(TensorStoreSets)}")
    write_transport_profile("recv_append", queue_depth=len(TensorStoreSets))

    LastStore = current_store

def next_set():
    global TensorStoreSets
    if len(TensorStoreSets) == 0:
        _log_debug("acts debug next_set called with empty TensorStoreSets")
        return
    not_ready_channels = []
    for channel in ACTIVATION_CHANNELS:
        _log_debug(
            f"acts debug next_set_release channel={channel} "
            f"{_summarize_queue_state(TensorStoreSets[0][channel])}"
        )
        if (
            len(TensorStoreSets[0][channel].activations) > 0
            or len(TensorStoreSets[0][channel].activations_reqs) > 0
        ):
            not_ready_channels.append(channel)
    if len(not_ready_channels) > 0:
        _log_debug(f"acts debug next_set_defer channels={not_ready_channels}")
        write_transport_profile(
            "next_set_defer",
            channels=",".join(not_ready_channels),
            queue_depth=len(TensorStoreSets),
        )
        return
    for channel in ACTIVATION_CHANNELS:
        TensorStoreSets[0][channel].release_buffers()
    TensorStoreSets.pop(0)
    _log_debug(f"acts debug next_set_done new_set_depth={len(TensorStoreSets)}")
    write_transport_profile("next_set_done", queue_depth=len(TensorStoreSets))

##########################################################################################################################################

import queue
barrier = None

def activation_sender(registry_queue, comm_stream, receiver, device):
    # print(f"Rank {rank} [Thread]: Listener started. Waiting for requests...")
    
    # Pre-allocate a buffer for receiving the Tensor ID (Control Message)
    torch.cuda.set_device(device)
    request_buffer = torch.zeros(1, dtype=torch.long, device=device)
    current_registry = None

    with torch.cuda.stream(comm_stream):
        while True:
            # 1. Blocking Receive for the Control Message (The ID)
            # We use the default stream for this small control message for simplicity,
            # as it is a blocking synchronization point anyway.
            # print('sending1', request_buffer, receiver)
            if current_registry is None:
                current_registry = registry_queue.get()  # blocking
                if current_registry is None:
                    # Final termination signal
                    break

            req = dist.recv_with_virtual_rank(request_buffer, src=receiver)
            # print('sending2')
            # req.wait()
            # torch.cuda.current_stream().synchronize()
            # print('sending3', request_buffer, receiver)
            
            req_id = request_buffer.item()
            
            # Check for termination signal
            if req_id == -1:
                current_registry = None
                barrier.wait()
                # print(f"Rank {rank} [Thread]: Received termination signal.")
                continue

            # print(f"Rank {rank} [Thread]: Received request for Tensor ID {req_id}")

            # 2. Look up tensor safely
            if req_id >= len(current_registry) or req_id < 0:
                print(f"Rank {rank} [Thread]: ID {req_id} not found!")
                continue
            
            # Get tensor and the event recording when it was computed
            tensor_data = current_registry[req_id]
            tensor_data = tensor_data.to(device)
            print(tensor_data.shape, req_id)
            # print(tensor_data.shape, tensor_data.device, torch.cuda.current_device())

            # 3. Send the Data Asynchronously on Comm Stream
            # Ensure the computation on the default stream is done before sending
            # comm_stream.wait_event(ready_event)
            
            
            # print(f"Rank {rank} [Thread]: Sending tensor {req_id} on Comm Stream...")
            req = dist.send_with_virtual_rank(tensor_data, dst=receiver)
            # req.wait()
            # torch.cuda.current_stream().synchronize()
            # Note: isend is async, so the thread loops back immediately 
            # while the GPU hardware handles the transfer.

def start_activations_sender(config):
    receiver = parallel_state.get_forward_backward_parallel_dual_rank()[0][1]
    comm_stream = torch.cuda.Stream()
    # for tensor in tensor_registry:
    #     print(type(tensor))
    # activation_sender(tensor_registry, comm_stream, receiver, torch.cuda.current_device(), is_last)
    # print('finish sending')
    global barrier
    if dist.get_thread_index() == 0:
        barrier = threading.Barrier(dist.num_threads, timeout=None)
    registry_queue = queue.Queue()
    listener_thread = threading.Thread(
        target=activation_sender,
        args=(registry_queue, comm_stream, receiver, torch.cuda.current_device())
    )
    listener_thread.start()
    return registry_queue

def add_activations(registry_queue, tensor_registry):
    registry_queue.put(tensor_registry)

def receive_activation(sender, tensor, tensor_id, tensor_shape, tensor_dtype):
    # if tensor is not None:
    #     print('receiving', tensor_shape, tensor.shape)
    # torch.cuda.synchronize()
    # print('send start', tensor_id)
    id_sender = torch.zeros(1, dtype=torch.long, device=torch.cuda.current_device())
    id_sender[0] = tensor_id
    # print('send', dist.get_rank(),id_sender, sender, torch.cuda.current_device())
    req = dist.send_with_virtual_rank(id_sender, dst=sender)
    # req.wait()
    # print('send', id_sender)
    if tensor_id == -1:
        return
    recv_buffer = torch.zeros(tensor_shape, dtype = tensor_dtype, device=torch.cuda.current_device())
    req = dist.recv_with_virtual_rank(recv_buffer, src=sender)
    # print('receiving', tensor.shape, recv_buffer.shape)
    # req.wait()
    # torch.cuda.synchronize()
    with torch.no_grad():
        tensor.data = recv_buffer
    # tensor.data.copy_(recv_buffer.data)
    # print('received', tensor_shape)
