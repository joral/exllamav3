"""
Skinny GEMM for decode/verify-sized dense projections: y[M, N] = x[M, K] @ W[N, K]^T with M <= 64, weights streamed from
DRAM at close to peak bandwidth. Weights are bf16, or e4m3 (FP8 weight-only, W8A16) with one fp32 scale per output
channel applied after the fp32 accumulation. Activations stay bf16; accumulation fp32.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _skinny_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, sxm, swn, swk, sym,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KSPLIT: tl.constexpr, W8: tl.constexpr,
                   ATOMIC: tl.constexpr, PARTIAL: tl.constexpr = False):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < M
    mn = offs_n < N
    kper = tl.cdiv(K, KSPLIT * BK) * BK
    k_lo = pid_k * kper
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(k_lo, tl.minimum(k_lo + kper, K), BK):
        offs_k = k0 + tl.arange(0, BK)
        mk = offs_k < K
        x = tl.load(x_ptr + offs_m[:, None] * sxm + offs_k[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk, mask=mn[:, None] & mk[None, :], other=0.0)
        if W8:
            w = w.to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w))
    if W8:
        sc = tl.load(s_ptr + offs_n, mask=mn, other=0.0)
        acc = acc * sc[None, :]
    yp = y_ptr + offs_m[:, None] * sym + offs_n[None, :]
    if PARTIAL:      # deterministic split-K: one fp32 partial per k-slice, summed by the consumer in fixed order
        yp += pid_k * M * sym
    if ATOMIC:
        tl.atomic_add(yp, acc, mask=mm[:, None] & mn[None, :])
    else:
        tl.store(yp, acc.to(y_ptr.dtype.element_ty), mask=mm[:, None] & mn[None, :])


def _bm(M):
    return 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128


# (BN, BK, KSPLIT, num_warps, num_stages) chosen per (M bucket, N, K) by serving/gemm_bench.py; fallback below
CONFIGS = {   # serving/gemm_tune.py on one RTX PRO 6000 at 300 W (2026-11-15); kept only where >= 3% faster than the fallback
    (16, 3276, 4608, False): (32, 128, 1, 4, 5),
    (16, 4608, 1536, False): (64, 128, 1, 4, 4),
    (16, 4608, 2048, True): (64, 256, 1, 8, 5),
    (16, 4608, 4608, True): (32, 256, 1, 4, 3),
    (32, 264, 4608, False): (16, 256, 1, 8, 3),
    (32, 2048, 4608, True): (16, 128, 1, 4, 4),
    (32, 4608, 2048, True): (64, 128, 1, 4, 4),
    (32, 4608, 4608, True): (64, 128, 1, 4, 3),
    (64, 264, 4608, False): (16, 256, 1, 4, 3),
    (64, 2048, 4608, True): (16, 128, 1, 8, 5),
    (64, 3788, 4608, False): (32, 128, 1, 8, 4),
    (64, 4608, 2048, True): (32, 128, 1, 4, 3),
    (64, 4608, 4608, True): (128, 128, 4, 8, 3),
    (128, 264, 4608, False): (16, 64, 8, 8, 4),
    (128, 2048, 4608, True): (32, 64, 4, 4, 3),
    (128, 3276, 4608, False): (32, 128, 1, 4, 3),
    (128, 3788, 4608, False): (32, 64, 1, 4, 5),
    (128, 4608, 1536, False): (32, 64, 1, 4, 5),
    (128, 4608, 2048, True): (32, 64, 1, 8, 3),
    (128, 4608, 4608, True): (128, 64, 4, 8, 3),
}


def pick(M, N, K, w8):
    key = (_bm(M), N, K, bool(w8))
    if key in CONFIGS:
        return CONFIGS[key]
    if M > 64:
        return (64, 64, 1, 8, 3)
    return (64, 128, 1, 4, 4)


def skinny_mm(x, w, scale=None, out=None, out_dtype=torch.bfloat16, cfg=None):
    """x [M, K] bf16 (row stride arbitrary, unit column stride), w [N, K] bf16 or float8_e4m3fn, scale [N] fp32 (fp8 only).
    Returns / writes y [M, N] in out_dtype (split-K configs accumulate into a zeroed fp32 buffer)."""
    M, K = x.shape
    N = w.shape[0]
    assert M <= 128 and x.stride(1) == 1
    w8 = w.dtype == torch.float8_e4m3fn
    BN, BK, KS, nw, ns = cfg or pick(M, N, K, w8)
    atomic = KS > 1
    if atomic:
        y = torch.zeros((M, N), dtype=torch.float32, device=x.device)
    else:
        y = out if out is not None else torch.empty((M, N), dtype=out_dtype, device=x.device)
    grid = (triton.cdiv(N, BN), KS)
    _skinny_kernel[grid](x, w, scale if w8 else w, y, M, N, K, x.stride(0), w.stride(0), w.stride(1), y.stride(0),
                         BM=_bm(M), BN=BN, BK=BK, KSPLIT=KS, W8=w8, ATOMIC=atomic, num_warps=nw, num_stages=ns)
    if atomic and out_dtype != torch.float32:
        y = y.to(out_dtype)
    if atomic and out is not None:
        out.copy_(y); y = out
    return y


def quant_fp8_rows(w):
    """[N, K] bf16 -> (e4m3 [N, K], fp32 per-row scale [N]); scale = absmax / 448"""
    wf = w.float()
    s = (wf.abs().amax(1) / 448.0).clamp_min(1e-12)
    q = (wf / s[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.contiguous(), s.contiguous()


@triton.jit
def _route_topk_kernel(z_ptr, b_ptr, ids_ptr, w_ptr, E, T, scale, TOPK: tl.constexpr, EP: tl.constexpr, KS: tl.constexpr = 1):
    """Inkling sigmoid router on fp32 logits z [KS, T, E] (split-K partials, summed in order): top-k of sigmoid(z) + bias,
    weights = normalized sigmoid(z) * scale"""
    t = tl.program_id(0)
    offs = tl.arange(0, EP)
    m = offs < E
    z = tl.zeros([EP], dtype=tl.float32)
    for k in tl.static_range(KS):
        z += tl.load(z_ptr + (k * T + t) * E + offs, mask=m, other=0.0)
    sc = tl.sigmoid(z)
    v = tl.where(m, sc + tl.load(b_ptr + offs, mask=m, other=0.0), -float("inf"))
    offs_k = tl.arange(0, TOPK)
    sel = tl.zeros([TOPK], dtype=tl.int32)
    ws = tl.zeros([TOPK], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        i = tl.argmax(v, 0)
        wi = tl.sum(tl.where(offs == i, sc, 0.0), 0)
        sel = tl.where(offs_k == k, i, sel)
        ws = tl.where(offs_k == k, wi, ws)
        v = tl.where(offs == i, -float("inf"), v)
    ws = ws / (tl.sum(ws, 0) + 1e-20) * scale
    tl.store(ids_ptr + t * TOPK + offs_k, sel)
    tl.store(w_ptr + t * TOPK + offs_k, ws)


def route_sigmoid_topk(x, w_ek, bias, topk, scale, ks=8, bn=16, bk=128):
    """x [T, K] fp16, w_ek [E, K] router weight (contiguous), bias [E] fp32. Deterministic split-K logits + fused top-k.
    Returns (ids int32 [T, topk], weights fp32 [T, topk])"""
    T, K = x.shape
    E = w_ek.shape[0]
    assert T <= 64 and w_ek.stride(1) == 1 and x.stride(1) == 1
    z = torch.empty((ks, T, E), dtype=torch.float32, device=x.device)
    _skinny_kernel[(triton.cdiv(E, bn), ks)](x, w_ek, w_ek, z, T, E, K, x.stride(0), w_ek.stride(0), 1, E,
                                             BM=_bm(T), BN=bn, BK=bk, KSPLIT=ks, W8=False, ATOMIC=False, PARTIAL=True,
                                             num_warps=4, num_stages=3)
    ids = torch.empty((T, topk), dtype=torch.int32, device=x.device)
    wts = torch.empty((T, topk), dtype=torch.float32, device=x.device)
    _route_topk_kernel[(T,)](z, bias, ids, wts, E, T, float(scale), TOPK=topk, EP=triton.next_power_of_2(E), KS=ks, num_warps=4)
    return ids, wts


def route_sigmoid_topk_tiled(x, w_ek, bias, topk, scale):
    """route_sigmoid_topk for any T in 64-row tiles (prefill-sized batches): same per-row math as the decode router"""
    T = x.shape[0]
    if T <= 64:
        return route_sigmoid_topk(x, w_ek, bias, topk, scale)
    parts = [route_sigmoid_topk(x[s:s + 64], w_ek, bias, topk, scale) for s in range(0, T, 64)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
