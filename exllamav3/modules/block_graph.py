"""Bounded per-block HIP graph capture for ROCm decode.

Opt-in via EXL3_BLOCK_GRAPH=1 (default off; deployment defaults unchanged). Scope: one graph
per (device, layer instance, (B, Q), recurrent-history mode,
cache generation) slot for non-QSA GDN+MoE TransformerBlocks. QSA layers, the PLE layer, the
epilogue (mixer + lm_head), TP modes, prefill, and the MTP draft head decline and stay eager.

Contract:
- The block's forward mutates the fp32 stream stack in place, so the graph uses a static
  input buffer (copy-in before replay, clone out after). Downstream semantics are unchanged.
- Recurrent slot ids change as data, not addresses: a static device copy of the recurrent
  slots tensor is refreshed before every replay; conv/recurrent state tensors are
  cache-owned and stable for the cache lifetime.
- Warmups are completed logical calls whose outputs are used; capture records without
  executing; the capture call then replays once, committing exactly one state transition.
- Graphs on a device share one private pool. Replays are sequential on the decode stream in
  invariant block order, which is the documented safe sharing pattern.
- A failed capture declines the slot; by default the error is raised, since a capture error
  may mean captured-region work already ran. A failed
  replay disables the slot and raises: replay failure is not automatically safe to retry
  eagerly.
"""

from __future__ import annotations

import os
from collections import Counter

import torch

from .block_sparse_mlp import BlockSparseMLP
from .gated_delta_net import GatedDeltaNet

_env_int = lambda name, default: int(os.environ.get(name, default) or default)

BLOCK_GRAPH_ENABLED = os.environ.get("EXL3_BLOCK_GRAPH", "0") == "1"
BLOCK_GRAPH_WARMUPS = max(1, _env_int("EXL3_BLOCK_GRAPH_WARMUPS", 3))
BLOCK_GRAPH_MAX_SLOTS = max(1, _env_int("EXL3_BLOCK_GRAPH_MAX_SLOTS", 4))

MLP_GRAPH_ENABLED = os.environ.get("EXL3_MLP_GRAPH", "0") == "1"
MLP_GRAPH_WARMUPS = max(1, _env_int("EXL3_MLP_GRAPH_WARMUPS", 3))
MLP_GRAPH_MAX_SLOTS = max(1, _env_int("EXL3_MLP_GRAPH_MAX_SLOTS", 8))
MLP_GRAPH_MAX_ROWS = _env_int("EXL3_MLP_GRAPH_MAX_ROWS", 32)
MLP_GRAPH_DEBUG = os.environ.get("EXL3_MLP_GRAPH_DEBUG", "0") == "1"

# Rows above the dense GEMV cap reconstruct+hgemm; still captureable, but the first bounded
# implementation keeps the served decode envelope. Revisit after (B,Q) family measurements.
BLOCK_GRAPH_MAX_ROWS = _env_int("EXL3_BLOCK_GRAPH_MAX_ROWS", 16)
BLOCK_GRAPH_DEBUG = os.environ.get("EXL3_BLOCK_GRAPH_DEBUG", "0") == "1"
_devices_env = os.environ.get("EXL3_BLOCK_GRAPH_DEVICES")
BLOCK_GRAPH_DEVICES = ({int(d) for d in _devices_env.split(",") if d.strip()}
                       if _devices_env else None)
BLOCK_GRAPH_MAX_TOTAL = _env_int("EXL3_BLOCK_GRAPH_MAX_TOTAL", 0)  # 0 = unlimited

# One shared private pool per device across every captured block graph. Captures and replays
# are strictly sequential on the decode stream in invariant fwd-module order, which is the
# sharing order torch's graph pools are documented for (same pattern as batch capture).
# EXL3_BLOCK_GRAPH_SHARED_POOL=0 gives every graph its own pool (isolation/debugging).
BLOCK_GRAPH_SHARED_POOL = os.environ.get("EXL3_BLOCK_GRAPH_SHARED_POOL", "1") == "1"
# torch.cuda.graph lazily creates ONE class-global default capture stream on the first device
# used (torch/cuda/graphs.py default_capture_stream) and reuses it for captures on every other
# device, so multi-device capture must pass an explicit per-device stream. Set
# EXL3_BLOCK_GRAPH_EXPLICIT_STREAM=0 to reproduce the cross-device stream trap for diagnosis.
BLOCK_GRAPH_EXPLICIT_STREAM = os.environ.get("EXL3_BLOCK_GRAPH_EXPLICIT_STREAM", "1") == "1"
_capture_streams: dict[int, object] = {}
# Capture-time failures can mean escaped execution mutated state (Python side effects run
# during capture; wrongly-streamed kernels may run for real). Fail closed by default; the old
# decline-to-eager behavior stays available for diagnosis via =eager.
BLOCK_GRAPH_CAPTURE_FALLBACK = os.environ.get("EXL3_BLOCK_GRAPH_CAPTURE_FALLBACK", "closed")
_shared_pools: dict[int, object] = {}
_registry: list["BlockGraphRunner"] = []
_total_captures = [0]


def _capture_stream_for(device_index: int):
    """One retained capture stream per device, passed explicitly to torch.cuda.graph."""
    stream = _capture_streams.get(device_index)
    if stream is None:
        stream = _capture_streams[device_index] = torch.cuda.Stream(device=device_index)
    return stream


def _pool_for(device_index: int):
    if not BLOCK_GRAPH_SHARED_POOL:
        return None
    pool = _shared_pools.get(device_index)
    if pool is None:
        pool = _shared_pools[device_index] = torch.cuda.graph_pool_handle()
    return pool


class _BlockGraphSlot:
    __slots__ = ("graph", "x_in", "slots_dev", "last_used", "slots_src")

    def __init__(self, graph, x_in, slots_dev, stamp):
        self.graph = graph
        self.x_in = x_in
        self.slots_dev = slots_dev
        self.last_used = stamp
        self.slots_src = None


class BlockGraphRunner:
    """Owns the graph slots of one TransformerBlock. Attached lazily on first eligible decode
    call; all decline paths return None and the caller falls through to the eager forward."""

    def __init__(self, block):
        self.block = block
        self.slots: dict[tuple, _BlockGraphSlot] = {}
        self.warmups_left: dict[tuple, int] = {}
        self.disabled: set[tuple] = set()
        self.stamp = 0
        self.stats = {"captures": 0, "replays": 0, "warmups": 0, "declines": Counter(),
                      "capture_errors": Counter()}

    # -- eligibility -----------------------------------------------------------

    def slot_key(self, x: torch.Tensor, params: dict):
        """Specialization key, or None when this call must not be graphed."""
        block = self.block
        attn = block.attn
        mlp = block.mlp
        if not torch.version.hip:
            return None
        if params.get("prefill") or params.get("reconstruct") or \
                params.get("activate_all_experts") or params.get("autosplit_measure") or \
                params.get("quant_preserve") is not None or "capture" in params:
            self.stats["declines"]["call_mode"] += 1
            return None
        if not isinstance(attn, GatedDeltaNet) or attn.bc is not None:
            self.stats["declines"]["attn_type"] += 1
            return None
        if getattr(attn, "qsa_indexer", None) is not None:
            self.stats["declines"]["qsa"] += 1
            return None
        if not isinstance(mlp, BlockSparseMLP) or \
                not (getattr(mlp, "support_hip_grouped", False) or
                     getattr(mlp, "support_hip_prefill", False)):
            self.stats["declines"]["mlp_route"] += 1
            return None
        if getattr(mlp, "tp_mode", None) is not None or getattr(attn, "tp_reduce", False) or \
                getattr(mlp, "tp_reduce", False):
            self.stats["declines"]["tp"] += 1
            return None
        if x.device != self.block.device or x.device.type != "cuda":
            self.stats["declines"]["device"] += 1
            return None
        if x.dtype != torch.float32 or x.dim() != 4 or not x.is_contiguous():
            self.stats["declines"]["io"] += 1
            return None
        bsz, seqlen = x.shape[0], x.shape[1]
        rows = bsz * seqlen
        if not (1 <= rows <= BLOCK_GRAPH_MAX_ROWS):
            self.stats["declines"]["rows"] += 1
            return None
        num_v_heads = getattr(attn, "num_v_heads", None)
        if num_v_heads is not None and seqlen >= num_v_heads:
            # The chunk-rule path consumes slot ids host-side at capture time, so the
            # replay-time device-side slots refresh cannot retarget it. Capture stays on
            # the recurrent-kernel path, which reads slot ids from device memory.
            self.stats["declines"]["chunk_path"] += 1
            return None
        if BLOCK_GRAPH_DEVICES is not None and x.device.index not in BLOCK_GRAPH_DEVICES:
            self.stats["declines"]["device_filter"] += 1
            return None
        if os.environ.get("EXL3_MOE_SYNC_FREE_COUNT", "1") == "0":
            # torch.bincount in the generic MoE branch issues a capture-rejected H2D even when
            # the gfx12 prefill route is eligible; capture requires the sync-free histogram.
            self.stats["declines"]["sync_free_count_off"] += 1
            return None
        rsg = params.get("recurrent_states")
        if not rsg or rsg[0].exported:
            self.stats["declines"]["recurrent_state"] += 1
            return None
        recurrent_slots = params.get("recurrent_slots")
        if recurrent_slots is None or recurrent_slots.shape[0] != bsz:
            self.stats["declines"]["recurrent_slots"] += 1
            return None
        history = bool(params.get("recurrent_history", False))
        return (x.device.index, bsz, seqlen, history, id(rsg[0].cache),
                params.get("layer_instance", 0),
                tuple(x.shape), str(x.dtype))

    # -- capture / replay ------------------------------------------------------

    def maybe_forward(self, x: torch.Tensor, params: dict):
        key = self.slot_key(x, params)
        if key is None:
            return None
        if key in self.disabled:
            self.stats["declines"]["disabled_slot"] += 1
            return None

        warmups = self.warmups_left.get(key)
        if warmups is None:
            self.warmups_left[key] = BLOCK_GRAPH_WARMUPS
            self.stats["warmups"] += 1
            return None  # completed logical call, output used downstream
        if warmups > 1:
            self.warmups_left[key] = warmups - 1
            self.stats["warmups"] += 1
            return None

        slot = self.slots.get(key)
        if slot is None:
            if BLOCK_GRAPH_MAX_TOTAL and _total_captures[0] >= BLOCK_GRAPH_MAX_TOTAL:
                self.stats["declines"]["total_cap"] += 1
                return None
            slot = self._capture(key, x, params)
            if slot is None:
                return None
        return self._replay(slot, x, params)

    def _capture(self, key, x: torch.Tensor, params: dict):
        dev = x.device
        # Capture must not start while other devices have work in flight: the capture begins
        # on the block's device while the same logical step's earlier-device kernels may still
        # be queued, and hipGraph capture on ROCm has no cross-device ordering with them.
        for d in range(torch.cuda.device_count()):
            if torch.cuda.memory_allocated(d) > 0:
                torch.cuda.synchronize(d)
        torch.cuda.synchronize(dev)
        x_in = torch.empty_like(x)
        live_slots = params["recurrent_slots"]
        slots_cpu = live_slots.clone()  # stable CPU source; its device copy is slot-static
        slots_dev = get_static_device_copy(slots_cpu, dev)
        params_cap = dict(params)
        params_cap["recurrent_slots"] = slots_cpu
        graph = torch.cuda.CUDAGraph()
        try:
            self._capturing = True
            pool = _pool_for(dev.index)
            # Capture and replay must run under the block's device context: torch captures on
            # the current device's side stream, and kernels the extension launches into other
            # devices' streams during capture are not recorded.
            #
            # torch.cuda.graph keeps ONE class-global default capture stream (created on the
            # first device used) and would otherwise reuse it for captures on every other
            # device, recording this device's kernels onto the wrong device's stream. Always
            # pass an explicit per-device stream.
            with torch.cuda.device(dev):
                if BLOCK_GRAPH_EXPLICIT_STREAM:
                    capture_stream = _capture_stream_for(dev.index)
                else:
                    capture_stream = None  # reproduces the cross-device stream trap
                mode = os.environ.get("EXL3_BLOCK_GRAPH_CAPTURE_MODE", "thread_local")
                graph_kwargs = {"capture_error_mode": mode}
                if pool is not None:
                    graph_kwargs["pool"] = pool
                if capture_stream is not None:
                    graph_kwargs["stream"] = capture_stream
                with torch.cuda.graph(graph, **graph_kwargs):
                    if capture_stream is not None:
                        assert capture_stream.device.index == dev.index, \
                            f"capture stream on device {capture_stream.device.index}, " \
                            f"expected {dev.index}"
                        assert torch.cuda.current_stream(dev) == capture_stream
                        assert torch.cuda.is_current_stream_capturing()
                    self.block.forward(x_in, params_cap)
        except Exception as e:
            self.disabled.add(key)
            self.stats["captures"] += 1
            self.stats["capture_failed"] = self.stats.get("capture_failed", 0) + 1
            self.stats["declines"][f"capture_error:{type(e).__name__}"] += 1
            self.stats["capture_error_msg"] = repr(e)[:500]
            if BLOCK_GRAPH_CAPTURE_FALLBACK == "eager":
                # DIAGNOSTIC ONLY: a capture error may mean captured-region work already ran
                # for real (wrong-stream escapes); re-running eagerly can double-advance state.
                return None
            raise
        finally:
            self._capturing = False

        # LRU slot replacement: evict only after the device is quiescent.
        self.stamp += 1
        self._evict_to_limit(dev)
        self.slots[key] = _BlockGraphSlot(graph, x_in, slots_dev, self.stamp)
        self.stats["captures"] += 1
        _total_captures[0] += 1
        if BLOCK_GRAPH_DEBUG:
            print(f"[blockgraph] captured {self.block.key} key={key[1:4]}", flush=True)
        return self.slots[key]

    def _evict_to_limit(self, dev):
        while len(self.slots) >= BLOCK_GRAPH_MAX_SLOTS:
            oldest = min(self.slots.values(), key = lambda s: s.last_used)
            evict_key = next(k for k, v in self.slots.items() if v is oldest)
            if dev is not None:
                torch.cuda.synchronize(dev)
            del self.slots[evict_key]
            self.stats["evictions"] = self.stats.get("evictions", 0) + 1

    def _replay(self, slot: _BlockGraphSlot, x: torch.Tensor, params: dict):
        self.stamp += 1
        slot.last_used = self.stamp
        try:
            slot.x_in.copy_(x)
            # Slot ids are immutable per slot-tuple: the live CPU tensor is _static_dev_cache
            # with a persistent per-device copy. Refresh the graph's fixed-address slots buffer
            # with a stream-ordered D2D copy only when the source object changes (job slot
            # reassignment); same-tuple continuations upload nothing (a blocking pageable H2D
            # here would drain the queue once per captured block). The held reference makes
            # object identity exact: a recycled allocation address cannot alias the source.
            live = params["recurrent_slots"]
            if live is not slot.slots_src:
                from ..util.tensor import get_for_device
                live_dev = get_for_device(params, "recurrent_slots", x.device)
                slot.slots_dev.copy_(live_dev)
                slot.slots_src = live
            with torch.cuda.device(x.device):
                slot.graph.replay()
        except Exception as e:
            # Launch-time failure: disable the slot. The step is not silently retried eagerly;
            # the exception propagates to the caller's normal failure handling.
            self.disabled.add(self._key_of(slot))
            self.stats["declines"][f"replay_error:{type(e).__name__}"] += 1
            raise
        self.stats["replays"] += 1
        if BLOCK_GRAPH_DEBUG:
            print(f"[blockgraph] replay {self.block.key}", flush=True)
        return slot.x_in.clone()

    def _key_of(self, slot):
        return next(k for k, v in self.slots.items() if v is slot)

    _capturing = False


def get_static_device_copy(cpu_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Persistent per-device copy of a small CPU control tensor (same convention as
    _static_dev_cache in util/tensor.get_for_device): the address is stable, contents are
    updated by the caller before each replay. The source tensor is flagged _static_dev_cache
    so get_for_device resolves to this copy during capture instead of issuing an unpinned H2D
    copy node (rejected during stream capture)."""
    cpu_tensor._static_dev_cache = True
    cache = cpu_tensor.__dict__.get("_static_dev_copies")
    if cache is None:
        cache = cpu_tensor._static_dev_copies = {}
    dv = cache.get(device)
    if dv is None:
        dv = cpu_tensor.to(device)
        cache[device] = dv
    return dv


def maybe_graph_forward(block, x: torch.Tensor, params: dict):
    """Entry hook for TransformerBlock.forward. Returns a tensor on the graphed path, or None
    to decline (caller then runs the eager forward)."""
    if not BLOCK_GRAPH_ENABLED:
        return None  # dynamic check: the flag may be toggled mid-process for qualification
    runner = getattr(block, "block_graph_runner", None)
    if runner is None:
        if getattr(block, "_block_graph_checked", False):
            return None
        block._block_graph_checked = True
        if not isinstance(block.mlp, BlockSparseMLP) or not isinstance(block.attn, GatedDeltaNet):
            return None  # cheap structural pre-filter; full eligibility checked per call
        runner = block.block_graph_runner = BlockGraphRunner(block)
        _registry.append(runner)
    if runner._capturing:
        return None
    return runner.maybe_forward(x, params)


def global_stats():
    out = {"runners": len(_registry), "captures": 0, "replays": 0, "warmups": 0,
           "evictions": 0, "declines": Counter(), "capture_failed": 0}
    for r in _registry:
        out["captures"] += r.stats["captures"]
        out["replays"] += r.stats["replays"]
        out["warmups"] += r.stats["warmups"]
        out["evictions"] += r.stats.get("evictions", 0)
        out["capture_failed"] += r.stats.get("capture_failed", 0)
        out["declines"].update(r.stats["declines"])
    return out

def purge():
    """Release every live block graph (qualification harness teardown: models are reloaded
    between runs and stale graphs would otherwise hold their pools alive)."""
    for r in _registry:
        if r.slots:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            r.slots.clear()
        b = r.block
        if b is not None:
            for attr in ("block_graph_runner", "_block_graph_checked"):
                if hasattr(b, attr):
                    try:
                        delattr(b, attr)
                    except Exception:
                        pass
    _registry.clear()
    _shared_pools.clear()
    _total_captures[0] = 0


# ============================================================================
# Per-MLP graph capture (dense GatedMLP decode path)
#
# Scope: GatedMLP blocks whose decode shape takes the split gate/up + down GEMV path
# (no fused multi_gu, no TP reduce, no BC_GatedMLP internal graph). The eager MLP path
# is a chain of raw kernel launches (ext gemv chunks, aten activation) with no host
# syncs and no state mutation: replay is output-identical (verified bit-exact against
# eager on layers 1-4 of the Qwen3.5-27B GDN hybrid at rows=18). The attention/GDN
# internals of a block are NOT naive-capture safe (pointer-patched replay graphs inside
# a captured region bake stale pointers), so the capture deliberately stops at the MLP
# boundary.
#
# Opt-in via EXL3_MLP_GRAPH=1 (default off). Slots key on (device, shape, dtype,
# out_dtype) + the eligibility predicate below, so prefill (rows > cap) and BC-eligible
# small shapes stay eager and the captured region is branch-stable.
# ============================================================================

_mlp_registry: list["MLPGraphRunner"] = []


class _MLPGraphSlot:
    __slots__ = ("graph", "x_in", "y_out", "last_used")

    def __init__(self, graph, x_in, y_out, stamp):
        self.graph = graph
        self.x_in = x_in
        self.y_out = y_out
        self.last_used = stamp


class MLPGraphRunner:
    """Owns the graph slots of one GatedMLP. Attached lazily on first eligible decode call;
    all decline paths return None and the caller falls through to the eager forward."""

    def __init__(self, mlp):
        self.mlp = mlp
        self.slots: dict[tuple, _MLPGraphSlot] = {}
        self.warmups_left: dict[tuple, int] = {}
        self.disabled: set[tuple] = set()
        self.stamp = 0
        self._capturing = False
        self.stats = {"captures": 0, "replays": 0, "warmups": 0, "declines": Counter()}

    # -- eligibility -----------------------------------------------------------

    def slot_key(self, x: torch.Tensor, params: dict, out_dtype):
        """Specialization key, or None when this call must not be graphed."""
        mlp = self.mlp
        if x.device.type != "cuda" or not x.is_contiguous():
            return None
        if x.dtype not in (torch.half, torch.bfloat16):
            return None
        if "reconstruct" in params or "ovr" in params or params.get("prefill"):
            return None
        bsz, q_len, dim = x.shape
        rows = bsz * q_len
        if not (dim == mlp.hidden_size and 1 <= rows <= MLP_GRAPH_MAX_ROWS):
            return None
        # Single-slice eager split path only: fused multi_gu and the bsz<=8 BC_GatedMLP
        # internal graph take different branch arms whose capture safety is unproven.
        if mlp.num_slices != 1 or mlp.tp_reduce:
            return None
        _max_bszn = 8  # must match MAX_BSZN in mlp.py / block_sparse_mlp.py / libtorch/mlp.h
        if mlp.bc is not None and rows <= _max_bszn:
            return None
        if mlp.multi_gu[0] is not None and rows <= 32:
            return None
        if out_dtype not in (None, mlp.out_dtype):
            return None
        return (x.device.index, tuple(x.shape), str(x.dtype))

    # -- dispatch --------------------------------------------------------------

    def maybe_forward(self, x: torch.Tensor, params: dict, out_dtype):
        key = self.slot_key(x, params, out_dtype)
        if key is None or key in self.disabled or self._capturing:
            return None
        slot = self.slots.get(key)
        if slot is not None:
            return self._replay(slot, x)
        left = self.warmups_left.setdefault(key, MLP_GRAPH_WARMUPS)
        if left > 0:
            self.warmups_left[key] = left - 1
            self.stats["warmups"] += 1
            return None
        return self._capture(key, x, params, out_dtype)

    def _evict_to_limit(self, dev):
        while len(self.slots) >= MLP_GRAPH_MAX_SLOTS:
            oldest_key = min(self.slots, key = lambda k: self.slots[k].last_used)
            if dev is not None:
                torch.cuda.synchronize(dev)
            del self.slots[oldest_key]
            self.stats["evictions"] = self.stats.get("evictions", 0) + 1

    def _capture(self, key, x: torch.Tensor, params: dict, out_dtype):
        dev = x.device
        self._capturing = True
        try:
            torch.cuda.synchronize(dev)
            x_in = torch.empty_like(x)
            x_in.copy_(x)
            graph = torch.cuda.CUDAGraph()
            graph_kwargs = {"capture_error_mode":
                            os.environ.get("EXL3_MLP_GRAPH_CAPTURE_MODE", "thread_local")}
            pool = _pool_for(dev.index)
            if pool is not None:
                graph_kwargs["pool"] = pool
            graph_kwargs["stream"] = _capture_stream_for(dev.index)
            with torch.cuda.device(dev):
                with torch.cuda.graph(graph, **graph_kwargs):
                    y_cap = self.mlp.forward(x_in, params, out_dtype)
            torch.cuda.synchronize(dev)
        except Exception as e:
            self.disabled.add(key)
            self.stats["captures"] += 1
            self.stats["capture_failed"] = self.stats.get("capture_failed", 0) + 1
            self.stats["declines"][f"capture_error:{type(e).__name__}"] += 1
            raise
        finally:
            self._capturing = False
        # y_cap's storage is uninitialized at capture exit (capture does not execute). Store it
        # as the slot output directly: every replay rewrites it, and _replay clones only after
        # its replay has run, so callers never see capture-time garbage.
        self.stamp += 1
        self._evict_to_limit(dev)
        self.slots[key] = _MLPGraphSlot(graph, x_in, y_cap, self.stamp)
        self.stats["captures"] += 1
        if MLP_GRAPH_DEBUG:
            print(f"[mlpgraph] captured {self.mlp.key} key={key[1:]}", flush=True)
        return self._replay(self.slots[key], x)

    def _replay(self, slot: _MLPGraphSlot, x: torch.Tensor):
        key = next(k for k, v in self.slots.items() if v is slot)
        self.stamp += 1
        slot.last_used = self.stamp
        try:
            slot.x_in.copy_(x)
            with torch.cuda.device(x.device):
                slot.graph.replay()
        except Exception as e:
            self.disabled.add(key)
            self.stats["declines"][f"replay_error:{type(e).__name__}"] += 1
            raise
        self.stats["replays"] += 1
        return slot.y_out.clone()

    _capturing = False


def maybe_mlp_graph_forward(mlp, x: torch.Tensor, params: dict, out_dtype):
    """Entry hook for GatedMLP.forward. Returns a graphed output tensor, or None to decline."""
    if not MLP_GRAPH_ENABLED:
        return None  # dynamic check: the flag may be toggled mid-process for qualification
    runner = getattr(mlp, "mlp_graph_runner", None)
    if runner is None:
        if getattr(mlp, "_mlp_graph_checked", False):
            return None
        mlp._mlp_graph_checked = True
        if mlp.__class__.__name__ != "GatedMLP":
            return None  # cheap structural pre-filter; full eligibility checked per call
        runner = mlp.mlp_graph_runner = MLPGraphRunner(mlp)
        _mlp_registry.append(runner)
    return runner.maybe_forward(x, params, out_dtype)


def global_mlp_stats():
    out = {"runners": len(_mlp_registry), "captures": 0, "replays": 0, "warmups": 0,
           "evictions": 0, "declines": Counter(), "capture_failed": 0}
    for r in _mlp_registry:
        for k in ("captures", "replays", "warmups", "evictions", "capture_failed"):
            out[k] += r.stats.get(k, 0)
        out["declines"].update(r.stats["declines"])
    return out


def release_all_mlp_graphs():
    """Release every live MLP graph (qualification harness teardown)."""
    for r in _mlp_registry:
        if r.slots:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            r.slots.clear()
        m = r.mlp
        if m is not None:
            for attr in ("mlp_graph_runner", "_mlp_graph_checked"):
                if hasattr(m, attr):
                    try:
                        delattr(m, attr)
                    except Exception:
                        pass
    _mlp_registry.clear()
