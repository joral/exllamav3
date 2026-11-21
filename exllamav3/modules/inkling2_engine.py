"""
Rank-local batched Inkling-2 engine: state (KV pages, SWA rings, conv rings), layer math over a token batch, tensor
parallel attention (heads) + expert parallel MoE. One instance per GPU; ranks synchronize through torch.distributed
(NCCL) all-reduces. See inkling2.py for the reference math and inkling2_ops.py for the kernels.
"""
from __future__ import annotations
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import inkling2_ops as ops
from . import inkling2 as I2
from .inkling2_gemm import skinny_mm
from ..architecture.inkling2 import Inkling2Model, Inkling2Layer

RING_EXTRA = 64            # SWA ring headroom for speculative tokens beyond the window
# The SWA ring must also hold every row of one multi-row call (prefill chunk) beyond the window: all rows' K/V are written
# before any row's attention reads, so with window + 64 a 512-row chunk overwrote entries its own early rows still read
# (KV16 KLD 0.01623 at chunk 512 vs 0.01295 at chunk 64). max_rows = the largest rows-per-sequence of any call.
MAX_ROWS_DEFAULT = 512


class Batch:
    """Token batch descriptor (all int32 device tensors of length T; rows of a sequence contiguous, ascending)"""

    def __init__(self, slot, pos, nprev, multi: bool, nsplit_g: int = 1, nsplit_l: int = 4, nafter = None):
        self.slot, self.pos, self.nprev = slot, pos, nprev
        self.single = nafter is None
        if nafter is None:      # single sequence in the batch
            nafter = (slot.shape[0] - 1 - nprev).to(torch.int32)
        self.nafter = nafter
        self.T = slot.shape[0]
        self.multi = multi              # more than one row per sequence (prefill chunk / speculative verify)
        self.nsplit_g, self.nsplit_l = nsplit_g, nsplit_l
        self.seqs = None                # optional host list [(first row, rows, first position, slot)] (prefill attention plan)

    def host_seqs(self):
        """sequences of the batch as host ints (one device sync unless the caller set .seqs)"""
        if self.seqs is None:
            starts = torch.nonzero(self.nprev == 0).flatten()
            st = torch.stack([starts.int(), (self.nafter[starts] + 1).int(), self.pos[starts].int(), self.slot[starts].int()], 1).tolist()
            self.seqs = [tuple(x) for x in st]
        return self.seqs

    def sub_seqs(self, r0, r1):
        """host_seqs() restricted to batch rows [r0, r1), with row indices relative to r0"""
        out = []
        for (t0, n, p0, sl) in self.host_seqs():
            a, b = max(t0, r0), min(t0 + n, r1)
            if a < b:
                out.append((a - r0, b - a, p0 + (a - t0), sl))
        return out


class I2Engine:
    def __init__(self, model: Inkling2Model, device, rank: int, world: int, max_slots: int, num_pages: int,
                 page_size: int = 64, bits_g: int = 4, bits_l: int = 8, max_pages_per_seq: int = 8192, embed_on_rank: int = 0,
                 kv_rot: bool = False, kv_ng: int = 1, kv_q16: bool = False, max_rows: int = MAX_ROWS_DEFAULT):
        """kv_rot / kv_ng / kv_q16 (global-layer KV only): randomized-Hadamard rotation of K/V/Q along head_dim before
        quantizing, scale groups per 128-vector (fp16 scales), 16-level symmetric int4 grid"""
        self.model, self.cfg = model, model.config.i2
        self.device, self.rank, self.world = device, rank, world
        self.max_slots, self.num_pages, self.ps = max_slots, num_pages, page_size
        self.bits_g, self.bits_l = bits_g, bits_l
        self.kv_rot = ops.make_kv_rotation(device) if kv_rot else None
        self.kv_ng, self.kv_q16 = kv_ng, kv_q16
        self.layers = model.layers
        self.embed = model.embed
        # vocab-parallel embedding: this rank keeps rows [rank*n, (rank+1)*n); lookups are summed over ranks (exact: one
        # nonzero contribution per token). Frees (world-1)/world of the 1.85 GB bf16 table per GPU
        V = model.embed.weight.shape[0]
        assert V % world == 0
        self.embed_n = V // world
        if world > 1:
            w = model.embed.weight[rank * self.embed_n:(rank + 1) * self.embed_n].clone()
            model.embed.weight = None
            torch.cuda.empty_cache()
            self.embed_w = w
        else:
            self.embed_w = model.embed.weight
        self.head = model.head
        c = self.cfg
        self.Hl, self.Hkl = c.heads // world, c.kv_heads // world
        self.G = self.Hl // self.Hkl
        self.ring = c.window + max(RING_EXTRA, max_rows)
        self.max_rows = max_rows
        dev = device
        D = c.hidden
        self.blocks = None
        dhp = lambda bits: 64 if bits == 4 else 128
        ct = lambda bits: torch.bfloat16 if bits == 16 else torch.int8 if bits == 8 else torch.uint8
        self.kv = {}
        def _alloc_layer_state(i):
            if c.is_global(i):
                shape = (num_pages, page_size, self.Hkl, dhp(bits_g))
                bits = bits_g
            else:
                shape = (max_slots, self.ring, self.Hkl, dhp(bits_l))
                bits = bits_l
            kq = torch.zeros(shape, dtype = ct(bits), device = dev)
            vq = torch.zeros(shape, dtype = ct(bits), device = dev)
            ks = vs = None
            if bits != 16:
                sshape = shape[:-1] + ((kv_ng,) if c.is_global(i) and kv_ng > 1 else ())
                ks = torch.zeros(sshape, dtype = torch.half, device = dev)
                vs = torch.zeros(sshape, dtype = torch.half, device = dev)
            self.kv[i] = (kq, vq, ks, vs)
        for L in self.layers:
            _alloc_layer_state(L.idx)
        # loop passes: private K/V and conv-ring copies per (core layer, pass); pass 0 shares the physical layer's
        # state, passes 1.. get slots c.cache_vidx(i, p)
        self.has_loop = getattr(c, "loop_passes", 1) > 1
        if self.has_loop:
            for p in range(1, c.loop_passes):
                for i in range(c.loop_start, c.loop_end):
                    _alloc_layer_state(c.cache_vidx(i, p))
        z = lambda ch: torch.zeros((max_slots, ops.RC, ch), dtype = torch.bfloat16, device = dev)
        self.rings = {}
        for L in self.layers:
            self.rings[L.idx] = (z(self.Hkl * 128), z(self.Hkl * 128), z(D), z(D))
        if self.has_loop:
            for p in range(1, c.loop_passes):
                for i in range(c.loop_start, c.loop_end):
                    self.rings[c.cache_vidx(i, p)] = (z(self.Hkl * 128), z(self.Hkl * 128), z(D), z(D))
        self.page_table = torch.zeros((max_slots, max_pages_per_seq), dtype = torch.int32, device = dev)
        self.group = None
        self.moe_chunk_max = int(__import__('os').environ.get('I2_MOE_CHUNK_MAX', 192))
        # dtype of the two per-layer all-reduces (attention o_proj partials, routed-expert partials): the sums are
        # rounded to bf16 by the consuming sconv anyway; fp16 halves the PCIe bytes vs fp32 with 3 more mantissa bits than bf16
        # Reference (HF) stores every stage output in bf16; default matches it. fp16/fp32 remain
        # available for experiments via I2_AR_DTYPE
        self.ar_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[__import__('os').environ.get('I2_AR_DTYPE', 'bf16')]
        self.skinny = __import__('os').environ.get('I2_SKINNY', '1') == '1'      # Triton skinny GEMM for T <= skinny_max projections
        self.skinny_max = int(__import__('os').environ.get('I2_SKINNY_MAX', 64))
        # FP8 weight-only attention projections (e4m3, per-output-row scale; I2_ATTN_FP8=1): halves the dense bytes every
        # decode step reads and frees their memory for KV. Decode / verify (<= 128 rows) run the FP8 skinny GEMM; bigger
        # calls dequantize per call (same rounded weights in both paths)
        self.attn_fp8 = __import__('os').environ.get('I2_ATTN_FP8', '0') == '1'
        if self.attn_fp8:
            from .inkling2_gemm import quant_fp8_rows
            for L in self.layers:
                for k in ("wq_cat", "wk_cat", "wv_cat", "wqkv", "wo"):
                    w = getattr(L, k, None)
                    if w is None: continue
                    q8, s8 = quant_fp8_rows(w)
                    setattr(L, k + "_q8", q8); setattr(L, k + "_s8", s8); setattr(L, k, None)
                    del w
            torch.cuda.empty_cache()
        self.rel_fused = __import__('os').environ.get('I2_REL_FUSED', '1') == '1'   # relative bias inside the attention kernel
        # query-blocked global attention for multi-row eager batches (prefill / mixed steps) of at least this many rows
        self.attn_prefill_min = int(__import__('os').environ.get('I2_ATTN_PREFILL_MIN', 65))
        self.attn_bq = int(__import__('os').environ.get('I2_ATTN_BQ', 8))
        # SM120 CUDA prefill attention (inkling2_pattn) for both layer types on multi-row eager batches of at least
        # pattn_min rows; needs the served KV formats (global 4-bit rot/q16/ng=4, SWA int8, 64-row pages)
        self.pattn, self.pattn_min = None, int(__import__('os').environ.get('I2_PATTN_MIN', 65))
        if (__import__('os').environ.get('I2_PATTN', '1') == '1' and bits_g == 4 and kv_rot and kv_q16 and kv_ng == 4
                and bits_l == 8 and page_size == 64 and self.ring % 64 == 0 and self.Hl == 6 * self.Hkl):
            try:
                from .inkling2_pattn import PrefillAttn
                self.pattn = PrefillAttn(device)
            except Exception as e:
                print(f"inkling2_pattn: build failed ({e}); using the Triton prefill attention", flush = True)
        # PCIe P2P recursive-doubling all-reduce (inkling2_ar) for messages up to peer_ar_max bytes, NCCL above (prefill).
        # Graph-captured 4-GPU fp16 [T, 4608]: T=5 6.8 us vs NCCL 15.3, T=12 10.0 vs 20.2, T=48 26 vs 45 (serving/t_ar.py)
        self.peer_ar, self.peer_ar_max = None, int(__import__('os').environ.get('I2_PEER_AR_MAX', 640 << 10))
        if world in (2, 4) and __import__('os').environ.get('I2_PEER_AR', '1') == '1':
            from .inkling2_ar import PeerAllReduce
            self.peer_ar = PeerAllReduce.create(rank, world, 2 * self.peer_ar_max)
        # two-shot (reduce-scatter + all-gather, per-GPU barrier between rounds) for larger messages up to peer_big_max
        # (prefill / mixed [<= 512, 4608] fp16): T=96 45 us vs NCCL 77, T=512 185 vs 191
        self.peer_big, self.peer_big_max = None, int(__import__('os').environ.get('I2_PEER_BIG_MAX', 5 << 20))
        if self.peer_ar is not None and world == 4 and self.peer_big_max > 0:
            self.peer_big = PeerAllReduce.create(rank, world, self.peer_big_max // 2)
        # token-sharded prefill (prefill-comm, I2_PC_SHARD=1): single-sequence multi-row calls of pc_min..pc_tmax rows run
        # with every rank owning a quarter of the rows (forward_sharded); copy-engine ring all-gathers / reduce-scatters
        self.pc = None
        self.pc_min = int(__import__('os').environ.get('I2_PC_MIN', 256))
        self.pc_tmax = int(__import__('os').environ.get('I2_PC_TMAX', 4096))
        if world == 4 and __import__('os').environ.get('I2_PC_SHARD', '0') == '1':
            from .inkling2_pc import PcComm
            Tm, qm = self.pc_tmax, -(-self.pc_tmax // world) + 3
            slab = 4 * Tm * D * 2 + 4 * int(__import__('os').environ.get('I2_DRAFT_RINGD', 1088)) * D * 2 + Tm * D * 2 + Tm * D // 16 + Tm * 8 * 8 + 3 * 2 * (world - 1) * qm * D * 2 + (8 << 20)    # 3 reduce-scatters (ra, rm, emb)
            # two prefill micro-batches in flight (I2_PC_MB=1): lane 1 gets its own regions for half-size chunks
            self.pc_mb = __import__('os').environ.get('I2_PC_MB', '0') == '1'
            Th = int(__import__('os').environ.get('I2_PC_MB_ROWS', Tm // 2))
            if self.pc_mb:
                slab += 5 * Th * D * 2 + Th * 8 * 8 + 3 * 2 * (world - 1) * qm * D * 2 + (8 << 20)
            self.pc = PcComm.create(rank, world, slab, rs_rows = qm)
            if self.pc is not None:       # regions in the same order on every rank (same slab offsets)
                self._pcx = {3: self.pc.region("x3", (3, Tm, D), torch.bfloat16), 1: self.pc.region("x1", (1, Tm, D), torch.bfloat16)}
                # expert-input packet rows on the wire: fp16 (exact) or int8 / fp8 codes + fp16 scale per 32 (I2_PC_PK)
                self.pc_pk = __import__('os').environ.get('I2_PC_PK', 'fp16')
                if self.pc_pk == "fp16":
                    self._pkx = self.pc.region("pkx", (Tm, D), torch.float16)
                else:
                    self._pkq = self.pc.region("pkq", (Tm, D), torch.uint8)
                    self._pks = self.pc.region("pks", (Tm, D // 32), torch.float16)
                    self._pkx = torch.empty((Tm, D), dtype = torch.float16, device = dev)     # dequantized, local
                self._pki = self.pc.region("pki", (Tm, c.topk), torch.int32)
                self._pkw = self.pc.region("pkw", (Tm, c.topk), torch.float32)
                self._pcred = [torch.empty((qm, D), dtype = self.ar_dtype, device = dev) for _ in range(2)]
                if self.pc_mb and self.pc_pk == "fp16":
                    self.pc_th = Th
                    self._pcx1 = {3: self.pc.region("x3_1", (3, Th, D), torch.bfloat16), 1: self.pc.region("x1_1", (1, Th, D), torch.bfloat16)}
                    self._pkx1 = self.pc.region("pkx_1", (Th, D), torch.float16)
                    self._pki1 = self.pc.region("pki_1", (Th, c.topk), torch.int32)
                    self._pkw1 = self.pc.region("pkw_1", (Th, c.topk), torch.float32)
                    self._pcred1 = [torch.empty((qm, D), dtype = self.ar_dtype, device = dev) for _ in range(2)]
                    self._mb_streams = [torch.cuda.Stream(device = dev), torch.cuda.Stream(device = dev)]
                    self._mb_ev = [[torch.cuda.Event() for _ in range(2)] for _ in range(c.layers)]
                    if self.pattn is not None:
                        self.pattn.add_lane()
                else:
                    self.pc_mb = False
        # adaptive verify: optional bool [T] row mask of the current call (set by the caller around forward(); graph-safe:
        # a persistent device tensor). Dead rows route to an expert of another EP group (skipped, weight 0): their routed
        # experts cost no weight bytes; every other op still computes them (never accepted, KV overwritten later)
        self.live = None
        self.b12x = None          # optional inkling2_b12x.B12xMoE (b12x fused Trellis MoE for the prepared batch sizes)
        self.moe_group = int(__import__('os').environ.get('I2_MOE_GROUP', 64))   # rows per fused coop MoE call (P0, validated on v3; 8 = old path)

    # ---- memory accounting
    def kv_bytes(self):
        tot = 0
        for t in self.kv.values():
            tot += sum(x.numel() * x.element_size() for x in t if x is not None)
        return tot

    def mmw(self, x, L, k, out_dtype = torch.bfloat16):
        """x @ (layer weight k)^T: bf16 weight, or its FP8 rows + scales (I2_ATTN_FP8)"""
        w = getattr(L, k)
        if w is not None:
            if out_dtype == torch.bfloat16:
                return self.mm(x, w)
            if self.skinny and x.shape[0] <= self.skinny_max:
                return skinny_mm(x, w, out_dtype = out_dtype)
            y = torch.mm(x, w.t(), out_dtype = torch.float32)
            return y if out_dtype == torch.float32 else y.to(out_dtype)
        q8, s8 = getattr(L, k + "_q8"), getattr(L, k + "_s8")
        if x.shape[0] <= 128:
            return skinny_mm(x, q8, s8, out_dtype = out_dtype)
        wd = (q8.float() * s8[:, None]).to(torch.bfloat16)
        y = torch.mm(x, wd.t(), out_dtype = torch.float32)
        return y if out_dtype == torch.float32 else y.to(out_dtype)

    def mm(self, x, w):
        """x [T, K] @ w[N, K]^T in bf16 (skinny Triton GEMM for decode/verify-sized T)"""
        if self.skinny and x.shape[0] <= self.skinny_max:
            return skinny_mm(x, w, out_dtype = torch.bfloat16)
        return torch.mm(x, w.t())

    # ---- collectives
    _NO_AR = __import__('os').environ.get('I2_NO_AR', '0') == '1'      # timing diagnostics only: outputs are wrong

    def allreduce(self, x):
        if self.world > 1 and not self._NO_AR:
            p = self.peer_ar
            n = x.numel() * x.element_size()
            if p is not None and n <= self.peer_ar_max and p.fits(x):
                p(x)
            elif self.peer_big is not None and n <= self.peer_big_max and self.peer_big.fits_big(x):
                self.peer_big.big(x)
            else:
                dist.all_reduce(x)
        return x

    # ---- embedding / head
    def embed_tokens(self, ids):
        c = self.cfg
        if self.world == 1:
            e = self.embed_w[ids]
        else:
            loc = ids - self.rank * self.embed_n
            ok = (loc >= 0) & (loc < self.embed_n)
            e = self.embed_w[loc.clamp(0, self.embed_n - 1)] * ok[:, None].to(self.embed_w.dtype)
            self.allreduce(e)           # exact: one nonzero term per element
        return ops_rms_norm(e, self.embed.norm, c.eps)

    def head_hidden(self, h, blocks):
        c = self.cfg
        return ops.attn_res_norm(blocks, c.layers // c.block_layers, h, self.head.out_query[None], self.head.norm, c.eps)[0]

    def head_logits(self, h, blocks):
        """h: [T, D] bf16, blocks: [8, T, D]. Returns this rank's softcapped logits shard [T, V/world] fp32"""
        c = self.cfg
        x = self.head_hidden(h, blocks)
        z = torch.mm(x, self.head.unembed.t(), out_dtype = torch.float32)
        return torch.tanh(z / c.final_cap) * c.final_cap

    def global_argmax(self, z):
        """argmax over the full (unpadded) vocabulary of vocab-sharded logits z [T, V/world]"""
        c = self.cfg
        n = z.shape[1]
        valid = min(n, c.unpadded_vocab - self.rank * n)
        if valid < n:
            z = z.clone()
            z[:, valid:] = float("-inf")
        v, i = z.max(-1)
        i = i + self.rank * n
        if self.world == 1:
            return i
        vs = torch.empty((self.world,) + v.shape, dtype = v.dtype, device = v.device)
        is_ = torch.empty((self.world,) + v.shape, dtype = i.dtype, device = v.device)
        dist.all_gather_into_tensor(vs, v.contiguous())
        dist.all_gather_into_tensor(is_, i.contiguous())
        best = vs.argmax(0)
        return is_.gather(0, best[None])[0]

    # ---- one decoder layer
    # ---- loop-pass message gates and cross-pass messages

    def _gate(self, h, m, gate):
        """Inkling2LoopMessageGate: (1 - tanh(g @ forget^T)) * h + tanh(g @ gain^T) * m, fp32.
        gate = (down [rank, 2D] bf16, gain [D, rank] fp32, forget [D, rank] fp32): the down projection of the
        concatenated unweighted-RMS-normed inputs runs in bf16, the gate projections in fp32 (HF keeps them fp32)"""
        eps = self.cfg.eps
        def un(x):
            xf = x.float()
            return xf * torch.rsqrt(xf.square().mean(-1, keepdim = True) + eps)
        down, gain, forget = gate
        gh = torch.mm(torch.cat([un(h), un(m)], -1).to(down.dtype), down.t()).float()
        return (1 - torch.tanh(gh @ forget.t())) * h.float() + torch.tanh(gh @ gain.t()) * m.float()

    def _rms_norm(self, x, w):
        """Weighted RMS norm (fp32 compute), bf16 output — matches an Inkling2LoopRMSNorm().to(model dtype)"""
        return I2.rms_norm(x, w, self.cfg.eps).to(torch.bfloat16)

    def _mix_messages(self, prev_blocks, prev_h, loop_idx):
        """Inkling2LoopMessageAttentionResidual for core layer `loop_idx` (0-based) of the current pass.
        prev_blocks: tail block states of the previous pass [n, T, D] bf16; prev_h: previous pass's final hidden
        state [T, D]. Returns messages [4, T, D] fp32 (query, key, value, mlp)."""
        eps = self.cfg.eps
        keys = torch.cat([prev_blocks.float(), prev_h.float().unsqueeze(0)], 0)      # [nk, T, D]
        keys = keys * torch.rsqrt(keys.square().mean(-1, keepdim = True) + eps)      # unweighted RMS norm
        q = self.model.msg_query[loop_idx]                                           # [4, D] fp32, not normed
        logits = torch.einsum("od,ntd->not", q, keys) * (q.shape[-1] ** -0.5)
        w = torch.softmax(logits.float(), dim = 1)                                   # softmax over the nk + 1 keys
        return torch.einsum("not,ntd->otd", w, keys)

    def _attn_input(self, L: Inkling2Layer, h, blocks, out = None, msgs = None):
        """normalized attention input stream(s) [nq, T, D] bf16 (nq = 3 for global layers: q, k, v inputs)"""
        c = self.cfg
        i = L.idx
        nb = c.num_blocks_before(i)
        if msgs is not None:
            # loop passes >= 1: unnormalized block-attention-residual mixture, per-stream message gate, then the
            # input norms (HF: gate -> input_layernorm for sliding; gate -> q/k/v_input_layernorm for global)
            if L.is_global:
                mix = ops.attn_res_norm(blocks, nb, h, L.attn_query, None, c.eps, out_f32 = True)
                return torch.stack([
                    self._rms_norm(self._gate(mix[s], msgs[s], L.gates[n]), w)
                    for s, (n, w) in enumerate((("q", L.q_norm), ("k", L.k_norm), ("v", L.v_norm)))
                ])
            mix = ops.attn_res_norm(blocks, nb, h, L.attn_query[None], None, c.eps, out_f32 = True)[0]
            return torch.stack([
                self._rms_norm(self._gate(mix, msgs[s], L.gates[n]), L.attn_norm)
                for s, n in enumerate(("q", "k", "v"))
            ])
        if L.is_global:
            return ops.attn_res_norm(blocks, nb, h, L.attn_query, L.qkv_norms, c.eps, out = out)
        if i == 0:
            return ops.attn_res_norm(blocks, 0, h, torch.zeros((1, c.hidden), device = h.device), L.attn_norm, c.eps, out = out)
        return ops.attn_res_norm(blocks, nb, h, L.attn_query[None], L.attn_norm, c.eps, out = out)

    def _oproj(self, L: Inkling2Layer, o):
        """attention output projection partial [T, D] in the all-reduce dtype"""
        if L.wo is None:            # FP8 weights (I2_ATTN_FP8)
            return self.mmw(o, L, "wo", out_dtype = self.ar_dtype)
        if self.skinny and o.shape[0] <= self.skinny_max:
            return skinny_mm(o, L.wo, out_dtype = self.ar_dtype)          # fp32 accumulation, one rounding to the AR dtype
        oa = torch.mm(o, L.wo.t(), out_dtype = torch.float32)
        return oa.to(self.ar_dtype) if self.ar_dtype != torch.float32 else oa

    def _attention(self, L: Inkling2Layer, xs, B: Batch, vidx = None):
        """projections, KV writes, attention and K/V conv-ring writes for all T rows of xs [nq, T, D]; returns o [T, Hl*128]
        vidx: state slot (physical layer index for pass 0, c.cache_vidx for later loop passes)"""
        c = self.cfg
        i, T = (L.idx if vidx is None else vidx), xs.shape[1]
        Hl, Hkl, G = self.Hl, self.Hkl, self.G
        kq, vq, ks, vs = self.kv[i]
        kring, vring, aring, mring = self.rings[i]
        bits = self.bits_g if L.is_global else self.bits_l
        dev = xs.device
        q_out = torch.empty((T, Hl, 128), dtype = torch.bfloat16, device = dev)
        rel_out = torch.empty((T, Hl, 16), dtype = torch.bfloat16, device = dev)
        write_ring = not B.multi
        swa_sub = T

        # projections
        if L.is_global:
            qb = self.mmw(xs[0], L, "wq_cat")
            kb = self.mmw(xs[1], L, "wk_cat")
            vb = self.mmw(xs[2], L, "wv_cat")
            rel_off, gate_off, sink_off = Hl * 128, Hl * 128 + Hl * 16, Hl * 128 + Hl * 16 + Hl * 128
            ops.qkv_post(qb, kb, vb, L.kc_static, L.vc_static, L.q_scale, L.rel_scale, kring, vring, B.slot, B.pos, B.nprev,
                         q_out, rel_out, kq, vq, ks, vs, self.page_table, c, True, bits, rel_off, Hkl * 128, Hkl * 128,
                         self.ring, self.ps, write_ring, Hkl, G, rot = self.kv_rot, ng = self.kv_ng, q16 = self.kv_q16)
            kraw, vraw = kb, vb
            Nq = qb
        else:
            qb = self.mmw(xs[0], L, "wqkv")
            rel_off, gate_off, sink_off = Hl * 128, Hl * 128 + Hl * 16, Hl * 128 + Hl * 16 + Hl * 128
            koff = sink_off + Hl
            kbv, vbv = qb[:, koff:], qb[:, koff + Hkl * 128:]
            swa_sub = self.max_rows if (B.multi and T > self.max_rows) else T
            if swa_sub == T:
                ops.qkv_post(qb, kbv, vbv, L.kc_w, L.vc_w, L.q_scale, L.rel_scale, kring, vring, B.slot, B.pos, B.nprev,
                             q_out, rel_out, kq, vq, ks, vs, None, c, False, bits, rel_off, 0, 0, self.ring, self.ps,
                             write_ring, Hkl, G)
            kraw, vraw = kbv, vbv
            Nq = qb
        gate = Nq[:, gate_off:gate_off + Hl * 128]
        sink = Nq[:, sink_off:sink_off + Hl]
        if self.rel_fused:
            R = None
            if not hasattr(L, "rel_proj_f32"):
                L.rel_proj_f32 = L.rel_proj.float().contiguous()
            relarg = (rel_out, L.rel_proj_f32)
        else:
            R = torch.mm(rel_out.view(T * Hl, 16).float(), L.rel_proj.float()).view(T, Hl, c.rel_extent)
            relarg = None
        o = torch.empty((T, Hl * 128), dtype = torch.bfloat16, device = dev)
        nsplit = B.nsplit_g if L.is_global else B.nsplit_l
        pm = torch.empty((T, Hl, nsplit), dtype = torch.float32, device = dev)
        pl = torch.empty((T, Hl, nsplit), dtype = torch.float32, device = dev)
        po = torch.empty((T, Hl, nsplit, 128), dtype = torch.float32, device = dev)
        gq = dict(rot = self.kv_rot, ng = self.kv_ng, q16 = self.kv_q16) if L.is_global else {}
        use_pattn = (self.pattn is not None and B.multi and T >= self.pattn_min and relarg is not None
                     and not torch.cuda.is_current_stream_capturing())
        if (not use_pattn and L.is_global and B.multi and T >= self.attn_prefill_min and relarg is not None
                and not torch.cuda.is_current_stream_capturing()):
            if getattr(B, "qblocks", None) is None:
                B.qblocks = ops.prefill_qblocks(B.nprev, B.nafter, self.attn_bq)
            gq.update(qblocks = B.qblocks, bq = self.attn_bq)
        if not L.is_global and swa_sub < T:
            # SWA K/V of a multi-row call larger than the ring headroom (window + max_rows): write and attend in sub-chunks of
            # max_rows rows, each attention before the next sub-chunk's write (that write overwrites the window entries
            # max_rows back). In-batch conv history of a sub-chunk's first rows comes from the rows before its view.
            for s0 in range(0, T, swa_sub):
                r_ = slice(s0, min(T, s0 + swa_sub))
                ops.qkv_post(qb[r_], kbv[r_], vbv[r_], L.kc_w, L.vc_w, L.q_scale, L.rel_scale, kring, vring, B.slot[r_], B.pos[r_],
                             B.nprev[r_], q_out[r_], rel_out[r_], kq, vq, ks, vs, None, c, False, bits, rel_off, 0, 0, self.ring,
                             self.ps, write_ring, Hkl, G)
                if use_pattn:
                    self.pattn.run(1, q_out[r_], rel_out[r_], L.rel_proj, sink[r_], gate[r_], kq, vq, ks, vs, None, self.ring,
                                   B.sub_seqs(r_.start, r_.stop), o[r_], lane = getattr(B, "lane", 0))
                else:
                    ops.attention(q_out[r_], R[r_] if R is not None else None, sink[r_], gate[r_], kq, vq, ks, vs, None, B.slot[r_],
                                  B.pos[r_], (pm[r_], pl[r_], po[r_]), o[r_], c, False, bits, self.ring, self.ps, nsplit, Hkl, G,
                                  rel = (rel_out[r_], relarg[1]) if relarg is not None else None)
        elif use_pattn:   # prefill-attn's SM120 CUDA kernel (both layer types)
            self.pattn.run(0 if L.is_global else 1, q_out, rel_out, L.rel_proj, sink, gate, kq, vq, ks, vs,
                           self.page_table if L.is_global else None, self.ring, B.host_seqs(), o,
                           rot = self.kv_rot if L.is_global else None, lane = getattr(B, "lane", 0))
        else:
            ops.attention(q_out, R, sink, gate, kq, vq, ks, vs, self.page_table if L.is_global else None, B.slot, B.pos,
                          (pm, pl, po), o, c, L.is_global, bits, self.ring, self.ps, nsplit, Hkl, G, rel = relarg, **gq)
        if B.multi:      # conv-ring rows after every qkv_post of this call (they read the ring before the batch)
            ops.ring_write(kraw[:, :Hkl * 128], kring, B.slot, B.pos, B.nafter)
            ops.ring_write(vraw[:, :Hkl * 128], vring, B.slot, B.pos, B.nafter)
        return o

    def layer(self, L: Inkling2Layer, h, blocks, B: Batch, vidx = None, msgs = None):
        c = self.cfg
        i, T = L.idx, h.shape[0]
        nb = c.num_blocks_before(i)
        boundary = c.is_boundary(i)
        sidx = L.idx if vidx is None else vidx          # state slot (see _attention)
        kring, vring, aring, mring = self.rings[sidx]
        write_ring = not B.multi
        o = self._attention(L, self._attn_input(L, h, blocks, msgs = msgs), B, vidx = sidx)
        oa = self._oproj(L, o)
        self.allreduce(oa)

        # post-attention conv + residual, attention residual for the MLP input
        resid = torch.empty((T, c.hidden), dtype = torch.float32, device = h.device)
        ops.sconv(oa, L.attn_sconv, aring, B.slot, B.pos, B.nprev, 0, h = h, resid = resid, boundary = boundary,
                  write_ring = write_ring)
        if B.multi:
            ops.ring_write(oa, aring, B.slot, B.pos, B.nafter)
        if boundary:
            blocks[nb].copy_(h)
        nb_m = nb + 1 if boundary else nb
        if msgs is None:
            xm = ops.attn_res_norm(blocks, nb_m, resid, L.mlp_query[None], L.mlp_norm, c.eps, out_f16 = True)[0]
        else:
            xm32 = ops.attn_res_norm(blocks, nb_m, resid, L.mlp_query[None], None, c.eps, out_f32 = True)[0]
            xm = self._rms_norm(self._gate(xm32, msgs[3], L.gates["mlp"]), L.mlp_norm).to(torch.float16)

        # routed experts (expert parallel), summed over ranks
        G = self.moe_group
        if self.b12x is not None and self.b12x.supports(T):
            moe = self.b12x.forward(L.idx, L.mlp, xm, out_dtype = self.ar_dtype, live = self.live)
        elif G < T <= self.moe_chunk_max:
            # the fused coop kernels take up to MAX_BSZN (64) rows per call: groups of `moe_group` rows
            moe = torch.cat([L.mlp.forward(xm[s:s + G], {}).view(-1, c.hidden).clone() for s in range(0, T, G)], 0)
        else:
            moe = L.mlp.forward(xm, {})
        moe = moe.view(T, c.hidden)
        if moe.dtype != self.ar_dtype:
            moe = moe.to(self.ar_dtype)
        self.allreduce(moe)
        h_new = torch.empty_like(h)
        ops.sconv(moe, L.mlp_sconv, mring, B.slot, B.pos, B.nprev, 1, resid = resid, out = h_new, write_ring = write_ring)
        if B.multi:
            ops.ring_write(moe, mring, B.slot, B.pos, B.nafter)
        return h_new

    def forward_loop(self, h, blocks, B: Batch, logit_rows = None, return_hidden = False):
        """Inkling-2-Loop forward (mirrors HF Inkling2LoopTextModel.forward): pre-loop layers, `loop_passes`
        passes over the core layers (passes >= 1 restart from the pre-loop states and receive message gates from
        the previous pass), post-loop layers, final block-attention-residual + norm + unembed"""
        c = self.cfg
        ls, le, np_ = c.loop_start, c.loop_end, c.loop_passes
        n_pre = sum(c.is_boundary(i) for i in range(ls))          # boundary layers before the core
        # pre-loop layers
        for L in self.layers[:ls]:
            h = self.layer(L, h, blocks, B)
        pre_h, pre_blocks = h.clone(), blocks.clone()
        # pass 1 (no messages)
        for L in self.layers[ls:le]:
            h = self.layer(L, h, blocks, B)
        prev_tail, prev_h = blocks[n_pre + 1:].clone(), h.clone()
        # passes 2..: restart from the pre-loop states, per-core-layer message gates from the previous pass,
        # private K/V + conv-ring slots per pass
        for p in range(1, np_):
            blocks.copy_(pre_blocks)
            h = pre_h
            for j, L in enumerate(self.layers[ls:le]):
                msgs = self._mix_messages(prev_tail, prev_h, j)
                h = self.layer(L, h, blocks, B, vidx = c.cache_vidx(L.idx, p), msgs = msgs)
            prev_tail, prev_h = blocks[n_pre + 1:].clone(), h.clone()
        # post-loop layers
        for L in self.layers[le:]:
            h = self.layer(L, h, blocks, B)
        x = self.head_hidden(h, blocks)
        if return_hidden:
            return x
        if logit_rows is not None:      # only these rows' logits (prefill needs the last row)
            x = x[logit_rows]
        z = torch.mm(x, self.head.unembed.t(), out_dtype = torch.float32)
        return torch.tanh(z / c.final_cap) * c.final_cap

    def new_blocks(self, T):
        c = self.cfg
        return torch.zeros((c.layers // c.block_layers, T, c.hidden), dtype = torch.bfloat16, device = self.device)

    TAP_LAYERS = (5, 11, 17, 23, 29, 35, 41)

    def forward(self, ids, B: Batch, stop_layer = None, taps = None, return_hidden = False, logit_rows = None):
        """ids: [T] token ids (all ranks pass the same ids). Returns this rank's logits shard [T, V/world].
        taps: optional [T, 8, D] bf16 buffer receiving the draft's target features (inputs of layers 5..41, final-norm output)"""
        # multi-row calls longer than max_rows run their SWA layers in sub-chunks (layer()); no row limit here
        if (self.pc is not None and B.multi and B.single and self.pc_min <= ids.shape[0] <= self.pc_tmax and stop_layer is None
                and hasattr(self.b12x, "run_routed") and not torch.cuda.is_current_stream_capturing()):
            if getattr(self, "pc_mb", False) and ids.shape[0] >= 2 * self.pc_min and ids.shape[0] // 2 <= self.pc_th:
                return self.forward_sharded_mb(ids, B, taps, logit_rows, return_hidden)
            return self.forward_sharded(ids, B, taps, logit_rows, return_hidden)
        h = self.embed_tokens(ids)
        blocks = self.new_blocks(h.shape[0])
        if self.has_loop:
            assert stop_layer is None and taps is None, "loop model does not support draft taps / partial runs"
            return self.forward_loop(h, blocks, B, logit_rows = logit_rows, return_hidden = return_hidden)
        for L in self.layers:
            if taps is not None and L.idx in self.TAP_LAYERS:
                taps[:, self.TAP_LAYERS.index(L.idx)].copy_(h)
            h = self.layer(L, h, blocks, B)
            if stop_layer is not None and L.idx == stop_layer:
                return h
        x = self.head_hidden(h, blocks)
        if taps is not None:
            taps[:, 7].copy_(x)
        if return_hidden:
            return x
        if logit_rows is not None:      # only these rows' logits (prefill needs the last row: skips a [T, V/4] fp32 head GEMM)
            x = x[logit_rows]
        z = torch.mm(x, self.head.unembed.t(), out_dtype = torch.float32)
        return torch.tanh(z / self.cfg.final_cap) * self.cfg.final_cap


    # ---- token-sharded prefill (prefill-comm)
    # Layer structure after Flare's token-sharded Inkling-2 decoder (NVIDIA/hyperflare-inference, recipes/inkling2/
    # decoder.py, Apache-2.0), adapted to EP2 x TP2 experts and to the engine's numerics: the output short-convs run on the
    # owner's REDUCED rows (as in the all-reduce path) with 3 halo rows carried by the reduce-scatter, instead of on
    # partial sums, so the per-rank conv rings keep holding full sums and decode is unchanged. Rank r owns rows own[r]:
    #   1. owner: attention-residual mix + norm(s) of its rows -> slab; ring all-gather (bf16, 1 or 3 streams)
    #   2. all rows: projections, KV / conv-ring writes, attention (TP heads), o-proj partials
    #   3. ring reduce-scatter of the partials (owned rows + 3 halo rows); owner: post-attention conv + residual
    #   4. owner: MLP-input mix + norm, router (top-8); ring all-gather of [fp16 rows | ids | weights]
    #   5. all rows: this rank's EP2 x TP2 expert slice; ring reduce-scatter; owner: post-MLP conv + residual
    # The last RC rows of every sublayer's partials are summed once at the end (one all-reduce) for the conv-ring writes.
    def _shard_plan(self, T):
        W = self.world
        base, rem = divmod(T, W)
        own, r0 = [], 0
        for o in range(W):
            n = base + (1 if o < rem else 0)
            own.append((r0, n)); r0 += n
        halo = [(max(0, a - 3), n + min(3, a)) for a, n in own]
        return own, halo

    def _layer_sh(self, L: Inkling2Layer, h, blocks, B: Batch, S, tails):
        c = self.cfg
        pc = self.pc
        i, T = L.idx, B.T
        nb = c.num_blocks_before(i)
        boundary = c.is_boundary(i)
        kring, vring, aring, mring = self.rings[i]
        r0, n, hoff = S.r0, S.n, S.hoff
        nq = 3 if L.is_global else 1
        xb = self._pcx[nq][:, :T]
        self._attn_input(L, h, blocks, out = xb[:, r0:r0 + n])
        pc.all_gather(f"x{nq}", xb, S.own)
        o = self._attention(L, xb, B)
        pc.release(f"x{nq}")
        oa = self._oproj(L, o)
        tails[i, 0].copy_(oa[T - S.rc:])
        red = pc.reduce_scatter("ra", oa, S.halo, self._pcred[0])
        resid = torch.empty((n, c.hidden), dtype = torch.float32, device = h.device)
        ops.sconv(red[hoff:hoff + n], L.attn_sconv, aring, S.slot, S.pos, S.nprev, 0, h = h, resid = resid, boundary = boundary,
                  write_ring = False)
        if boundary:
            blocks[nb].copy_(h)
        nb_m = nb + 1 if boundary else nb
        xm = self._pkx[r0:r0 + n]
        ops.attn_res_norm(blocks, nb_m, resid, L.mlp_query[None], L.mlp_norm, c.eps, out_f16 = True, out = xm[None])
        ids_o, w_o = self.b12x.route(L.idx, xm)          # routes from the exact fp16 rows
        self._pki[r0:r0 + n].copy_(ids_o); self._pkw[r0:r0 + n].copy_(w_o)
        px, pi, pw = self._pkx[:T], self._pki[:T], self._pkw[:T]
        if self.pc_pk == "fp16":
            pc.all_gather("pk", [px, pi, pw], S.own)
            moe = self.b12x.run_routed(L.idx, px, pi, pw, out_dtype = self.ar_dtype)
            pc.release("pk")
        else:
            from .inkling2_pc import pk_quant, pk_dequant
            fp8 = self.pc_pk == "fp8"
            pk_quant(xm, self._pkq[r0:r0 + n], self._pks[r0:r0 + n], fp8)
            pc.all_gather("pk", [self._pkq[:T], self._pks[:T], pi, pw], S.own)
            pk_dequant(self._pkq[:T], self._pks[:T], px, fp8)
            ids_a, w_a = pi.clone(), pw.clone()
            pc.release("pk")
            moe = self.b12x.run_routed(L.idx, px, ids_a, w_a, out_dtype = self.ar_dtype)
        tails[i, 1].copy_(moe[T - S.rc:])
        red = pc.reduce_scatter("rm", moe, S.halo, self._pcred[1])
        h_new = torch.empty_like(h)
        ops.sconv(red[hoff:hoff + n], L.mlp_sconv, mring, S.slot, S.pos, S.nprev, 1, resid = resid, out = h_new, write_ring = False)
        return h_new

    # ---- token-sharded prefill with two micro-batches in flight (I2_PC_MB=1)
    # The chunk's rows split into lane 0 = [0, h) and lane 1 = [h, T), each sharded over the ranks as in forward_sharded and
    # each on its own compute stream, copy stream, slab regions and kernel scratch. Lane 1 of layer L starts its attention
    # after lane 0 of layer L wrote its K/V (pages, SWA ring, K/V conv rings): it is lane 0's continuation chunk. Lane 1's
    # output short-convs need the 3 rows before it summed over ranks: its reduce-scatters carry lane 0's last 3 partial rows
    # as halo rows in front (so the original in-chunk row counts apply and the conv rings are only written at the end).
    # The copy-engine traffic of one lane then overlaps the other lane's kernels.
    def _mb_plan(self, n_, ext, a, B):
        from types import SimpleNamespace
        own_l, _ = self._shard_plan(n_)
        own = [(r + ext, m) for r, m in own_l]
        halo = [(max(0, r - 3), m + min(3, r)) for r, m in own]
        r0l, n = own_l[self.rank]
        r0 = r0l + ext
        g0 = a + r0l                                   # first owned row in chunk coordinates
        return SimpleNamespace(own = own_l, own_x = own, halo = halo, r0 = r0l, n = n, hoff = r0 - halo[self.rank][0],
                               rc = min(ops.RC, n_), slot = B.slot[g0:g0 + n], pos = B.pos[g0:g0 + n],
                               nprev = B.nprev[g0:g0 + n], ext = ext)

    def _layer_mb(self, L: Inkling2Layer, h, blocks, Bl, S, lane, tailbuf, tails):
        c = self.cfg
        pc = self.pc
        pc.set_lane(lane)
        i, T = L.idx, Bl.T
        nb = c.num_blocks_before(i)
        boundary = c.is_boundary(i)
        kring, vring, aring, mring = self.rings[i]
        r0, n, hoff, ext = S.r0, S.n, S.hoff, S.ext
        sfx = "_1" if lane else ""
        nq = 3 if L.is_global else 1
        xb = (self._pcx1 if lane else self._pcx)[nq][:, :T]
        self._attn_input(L, h, blocks, out = xb[:, r0:r0 + n])
        pc.all_gather(f"x{nq}{sfx}", xb, S.own)
        ev0, ev1 = self._mb_ev[i]
        cur = torch.cuda.current_stream()
        if lane:
            cur.wait_event(ev0)                        # lane 0's K/V of this layer are written
        o = self._attention(L, xb, Bl)
        pc.release(f"x{nq}{sfx}")
        oa = self._oproj(L, o)
        if lane:
            oa_x = torch.empty((T + ext, c.hidden), dtype = oa.dtype, device = oa.device)
            oa_x[:ext].copy_(tailbuf[i, 0]); oa_x[ext:].copy_(oa)
            tails[i, 0].copy_(oa[T - S.rc:])
        else:
            oa_x = oa
            tailbuf[i, 0].copy_(oa[T - 3:])
            ev0.record(cur)
        red = pc.reduce_scatter(f"ra{sfx}", oa_x, S.halo, (self._pcred1 if lane else self._pcred)[0])
        resid = torch.empty((n, c.hidden), dtype = torch.float32, device = h.device)
        ops.sconv(red[hoff:hoff + n], L.attn_sconv, aring, S.slot, S.pos, S.nprev, 0, h = h, resid = resid, boundary = boundary,
                  write_ring = False)
        if boundary:
            blocks[nb].copy_(h)
        nb_m = nb + 1 if boundary else nb
        pkx, pki, pkw = (self._pkx1, self._pki1, self._pkw1) if lane else (self._pkx, self._pki, self._pkw)
        xm = pkx[r0:r0 + n]
        ops.attn_res_norm(blocks, nb_m, resid, L.mlp_query[None], L.mlp_norm, c.eps, out_f16 = True, out = xm[None])
        ids_o, w_o = self.b12x.route(L.idx, xm)
        pki[r0:r0 + n].copy_(ids_o); pkw[r0:r0 + n].copy_(w_o)
        px, pi, pw = pkx[:T], pki[:T], pkw[:T]
        pc.all_gather(f"pk{sfx}", [px, pi, pw], S.own)
        moe = self.b12x.run_routed(L.idx, px, pi, pw, out_dtype = self.ar_dtype, lane = lane)
        pc.release(f"pk{sfx}")
        if lane:
            cur.wait_event(ev1)
            moe_x = torch.empty((T + ext, c.hidden), dtype = moe.dtype, device = moe.device)
            moe_x[:ext].copy_(tailbuf[i, 1]); moe_x[ext:].copy_(moe)
            tails[i, 1].copy_(moe[T - S.rc:])
        else:
            moe_x = moe
            tailbuf[i, 1].copy_(moe[T - 3:])
            ev1.record(cur)
        red = pc.reduce_scatter(f"rm{sfx}", moe_x, S.halo, (self._pcred1 if lane else self._pcred)[1])
        h_new = torch.empty_like(h)
        ops.sconv(red[hoff:hoff + n], L.mlp_sconv, mring, S.slot, S.pos, S.nprev, 1, resid = resid, out = h_new, write_ring = False)
        return h_new

    def forward_sharded_mb(self, ids, B: Batch, taps = None, logit_rows = None, return_hidden = False):
        c = self.cfg
        T, D = ids.shape[0], c.hidden
        hT = (T // 2 + 3) // 4 * 4
        (_, _, p0, sl0), = B.host_seqs()               # one sequence (B.single)
        cur = torch.cuda.current_stream()
        lanes = []
        for li, (a, b) in enumerate(((0, hT), (hT, T))):
            n_ = b - a
            Bl = Batch(B.slot[a:b], B.pos[a:b], (B.nprev[a:b] - a).to(torch.int32), True, nsplit_g = B.nsplit_g, nsplit_l = B.nsplit_l)
            Bl.seqs = [(0, n_, p0 + a, sl0)]
            Bl.lane = li
            S = self._mb_plan(n_, 3 if li else 0, a, B)
            lanes.append((a, b, n_, Bl, S))
        tailbuf = torch.empty((c.layers, 2, 3, D), dtype = self.ar_dtype, device = self.device)
        tails = torch.empty((c.layers, 2, min(ops.RC, T - hT), D), dtype = self.ar_dtype, device = self.device)
        st = self._mb_streams
        hs, bls, tos = [], [], []
        for li, (a, b, n_, Bl, S) in enumerate(lanes):
            st[li].wait_stream(cur)
            with torch.cuda.stream(st[li]):
                self.pc.set_lane(li)
                loc = ids[a:b] - self.rank * self.embed_n
                okm = (loc >= 0) & (loc < self.embed_n)
                e = self.embed_w[loc.clamp(0, self.embed_n - 1)] * okm[:, None].to(self.embed_w.dtype)
                eo = torch.empty((S.n, D), dtype = e.dtype, device = self.device)
                self.pc.reduce_scatter("emb" + ("_1" if li else ""), e.contiguous(), S.own, eo)
                hs.append(ops_rms_norm(eo, self.embed.norm, c.eps))
                bls.append(self.new_blocks(S.n))
                tos.append(torch.empty((S.n, 8, D), dtype = torch.bfloat16, device = self.device) if taps is not None else None)
        for L in self.layers:
            for li, (a, b, n_, Bl, S) in enumerate(lanes):
                with torch.cuda.stream(st[li]):
                    if taps is not None and L.idx in self.TAP_LAYERS:
                        tos[li][:, self.TAP_LAYERS.index(L.idx)].copy_(hs[li])
                    hs[li] = self._layer_mb(L, hs[li], bls[li], Bl, S, li, tailbuf, tails)
        xs = []
        for li in range(2):
            with torch.cuda.stream(st[li]):
                xs.append(self.head_hidden(hs[li], bls[li]))
            cur.wait_stream(st[li])
        self.pc.set_lane(0)
        xf = torch.cat([self._gather_rows(xs[li], lanes[li][4], lanes[li][2]) for li in range(2)], 0)
        if taps is not None:
            d = getattr(self, "draft_ref", None)
            for li, (a, b, n_, Bl, S) in enumerate(lanes):
                tos[li][:, 7].copy_(xs[li])
                if d is not None and d.fused and __import__('os').environ.get('I2_PC_DRAFT', '1') == '1':
                    self._draft_ctx_sharded(d, tos[li], taps[a:b], Bl, S, n_)
                else:
                    for k in range(8):
                        taps[a:b, k].copy_(self._gather_rows(tos[li][:, k].contiguous(), S, n_))
            if d is not None and d.fused and __import__('os').environ.get('I2_PC_DRAFT', '1') == '1':
                self.ctx_stored = taps
        # conv-ring rows (last RC positions of the chunk, all in lane 1): full sums of the partials
        dist.all_reduce(tails)
        rc = tails.shape[2]
        sl, ps, na = B.slot[T - rc:], B.pos[T - rc:], B.nafter[T - rc:]
        for L in self.layers:
            kring, vring, aring, mring = self.rings[L.idx]
            ops.ring_write(tails[L.idx, 0], aring, sl, ps, na)
            ops.ring_write(tails[L.idx, 1], mring, sl, ps, na)
        if return_hidden:
            return xf
        if logit_rows is not None:
            xf = xf[logit_rows]
        z = torch.mm(xf, self.head.unembed.t(), out_dtype = torch.float32)
        return torch.tanh(z / self.cfg.final_cap) * self.cfg.final_cap

    def _gather_rows(self, x_own, S, T):
        """[n, D] bf16 owned rows -> [T, D] all rows (copy out of the slab)"""
        xb = self._pcx[1][:, :T]
        xb[0, S.r0:S.r0 + S.n].copy_(x_own)
        self.pc.all_gather("x1", xb, S.own)
        full = xb[0].clone()
        self.pc.release("x1")
        return full

    def _draft_ctx_sharded(self, d, taps_o, taps, B, S, T):
        """draft context of a sharded prefill: owners mix their rows (mix_rms is row-local), then gather the global
        layer's mix of all rows and the local layers' mixes of the last min(T, RINGD) rows only; the caller's later
        draft.store_context(taps) is skipped (ctx_stored marker). taps gets only its last row (the server's first draft)."""
        from .inkling2_draft_ops import mix_rms
        from .inkling2_draft import RINGD
        D = self.cfg.hidden
        r0, n = S.r0, S.n
        ctx = mix_rms(taps_o, d.alpha5, d.w5)                 # [5, n, D]
        gi = [i for i in range(5) if i not in d.LOCAL][0]
        K = min(T, RINGD)
        a0 = T - K
        # global draft layer: all rows (x1 slab region, cid x1)
        cg = self._gather_rows(ctx[gi], S, T)
        # local draft layers: rows [a0, T) of each owner, gathered in the x3 + x1 regions viewed as [4, K, D]
        xl = self.pc.region("dl", (4, RINGD, D), torch.bfloat16)[:, :K]
        blk = [(max(a, a0) - a0, max(0, min(a + m, T) - max(a, a0))) for a, m in S.own]
        b0, bn = blk[self.rank]
        if bn > 0:
            loc = [i for i in d.LOCAL]
            xl[:, b0:b0 + bn].copy_(ctx[loc][:, max(r0, a0) - r0: max(r0, a0) - r0 + bn])
        self.pc.all_gather("dl", xl, blk)
        d.store_ctx_parts(cg, xl, B.slot, B.pos)
        self.pc.release("dl")
        # last row's taps (server: first draft of the sequence) from its owner
        last = taps_o[-1].contiguous() if self.rank == self.world - 1 else torch.empty((8, D), dtype = torch.bfloat16, device = self.device)
        dist.broadcast(last, self.world - 1)
        taps[T - 1].copy_(last)
        self.ctx_stored = taps

    def forward_sharded(self, ids, B: Batch, taps = None, logit_rows = None, return_hidden = False):
        from types import SimpleNamespace
        c = self.cfg
        T, D = ids.shape[0], c.hidden
        own, halo = self._shard_plan(T)
        r0, n = own[self.rank]
        S = SimpleNamespace(own = own, halo = halo, r0 = r0, n = n, hoff = r0 - halo[self.rank][0], rc = min(ops.RC, T),
                            slot = B.slot[r0:r0 + n], pos = B.pos[r0:r0 + n], nprev = B.nprev[r0:r0 + n])
        # vocab-parallel embedding straight to the owned rows: reduce-scatter (exact, one nonzero term per element)
        loc = ids - self.rank * self.embed_n
        okm = (loc >= 0) & (loc < self.embed_n)
        e = self.embed_w[loc.clamp(0, self.embed_n - 1)] * okm[:, None].to(self.embed_w.dtype)
        eo = torch.empty((n, D), dtype = e.dtype, device = self.device)
        self.pc.reduce_scatter("emb", e.contiguous(), own, eo)
        h = ops_rms_norm(eo, self.embed.norm, c.eps)
        blocks = self.new_blocks(n)
        tails = torch.empty((c.layers, 2, S.rc, D), dtype = self.ar_dtype, device = self.device)
        taps_o = torch.empty((n, 8, D), dtype = torch.bfloat16, device = self.device) if taps is not None else None
        for L in self.layers:
            if taps is not None and L.idx in self.TAP_LAYERS:
                taps_o[:, self.TAP_LAYERS.index(L.idx)].copy_(h)
            h = self._layer_sh(L, h, blocks, B, S, tails)
        x = self.head_hidden(h, blocks)
        xf = self._gather_rows(x, S, T)
        if taps is not None:
            taps_o[:, 7].copy_(x)
            d = getattr(self, "draft_ref", None)
            if d is not None and d.fused and __import__('os').environ.get('I2_PC_DRAFT', '1') == '1':
                self._draft_ctx_sharded(d, taps_o, taps, B, S, T)
            else:
                for k in range(8):
                    taps[:, k].copy_(self._gather_rows(taps_o[:, k].contiguous(), S, T))
        # conv-ring rows (last RC positions of the sequence) of every sublayer: full sums of the partials
        dist.all_reduce(tails)
        sl, ps, na = B.slot[T - S.rc:], B.pos[T - S.rc:], B.nafter[T - S.rc:]
        for L in self.layers:
            kring, vring, aring, mring = self.rings[L.idx]
            ops.ring_write(tails[L.idx, 0], aring, sl, ps, na)
            ops.ring_write(tails[L.idx, 1], mring, sl, ps, na)
        if return_hidden:
            return xf
        if logit_rows is not None:
            xf = xf[logit_rows]
        z = torch.mm(xf, self.head.unembed.t(), out_dtype = torch.float32)
        return torch.tanh(z / self.cfg.final_cap) * self.cfg.final_cap


def ops_rms_norm(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim = True) + eps)
    return (w.float() * xf).to(torch.bfloat16)
