from __future__ import annotations
from dataclasses import dataclass, fields
from typing_extensions import override
import json
import os

import torch

from ..model.config import Config
from ..modules import Linear
from ..modules import inkling2 as I2
from .inkling2 import Inkling2Config, Inkling2Head, Inkling2Layer, Inkling2Model

"""
Inkling-2-Loop (Inkling-2-Mini, model_type inkling2_loop) text model: the Inkling-2 layer math with a repeating
"core" segment (layers `loop_start..loop_end`) executed for `loop_passes` passes. Passes after the first restart
from the pre-loop states and refine their attention/MLP inputs through per-layer message gates computed from the
previous pass's block-attention-residual states and final hidden state (modules/inkling2_engine.py, forward_loop).

Differences from Inkling-2 (Small) checkpoints:
  - the six attention projections and the unembedding are EXL3-quantized; they are reconstructed to dense tensors
    at load and packed into the engine's fused weight layouts unchanged
  - core layers carry `controllers.{attn,k,v,mlp}.{down,gain,forget}_weight` message-gate weights
  - `middle_cycle_cross_pass_queries` / `..._kv_queries` [2 * n_core, D] stack into per-core-layer message queries
    [4, D], ordered (query, key, value, mlp) per llama.cpp's converter
  - the visual tower and MTP draft tensors are ignored
"""


@dataclass
class Inkling2LoopCfg(I2.Inkling2Cfg):
    loop_start: int = 6
    loop_end: int = 18
    loop_passes: int = 3
    gate_rank: int = 64

    @staticmethod
    def from_dir(directory: str) -> "Inkling2LoopCfg":
        base = I2.Inkling2Cfg.from_dir(directory)
        with open(os.path.join(directory, "config.json")) as f:
            lc = json.load(f)["text_config"]["loop_config"]
        return Inkling2LoopCfg(
            **{f.name: getattr(base, f.name) for f in fields(I2.Inkling2Cfg)},
            loop_start = lc["core_start"], loop_end = lc["core_end"],
            loop_passes = lc["num_passes"], gate_rank = lc["gate_rank"],
        )

    def is_loop_layer(self, i: int):
        return self.loop_start <= i < self.loop_end

    def cache_vidx(self, i: int, p: int):
        """Cache/ring slot index of core layer i in pass p (p >= 1; pass 0 uses the physical layer index)"""
        return self.layers + (p - 1) * (self.loop_end - self.loop_start) + (i - self.loop_start)

    @property
    def cache_slots(self):
        """Total cache slots: one per physical layer plus one per core layer per extra pass"""
        return self.layers + (self.loop_passes - 1) * (self.loop_end - self.loop_start)


class Inkling2LoopConfig(Inkling2Config):
    arch_string = "Inkling2LoopForConditionalGeneration"

    def __init__(self, directory: str, **kwargs):
        Config.__init__(self, directory, {"text": Inkling2LoopModel}, config_key = "text_config", **kwargs)
        self.i2 = Inkling2LoopCfg.from_dir(directory)
        c = self.i2
        self.num_hidden_layers = c.layers
        self.hidden_size = c.hidden
        self.head_dim = c.head_dim
        self.num_q_heads = c.heads
        self.num_kv_heads = c.kv_heads
        self.rms_norm_eps = c.eps
        self.num_experts = c.experts
        self.num_experts_per_tok = c.topk
        self.moe_intermediate_size = c.inter
        self.sliding_window = c.window
        self.max_position_embeddings = self.read_cfg(int, "model_max_length", 1048576)

    @override
    def default_max_position_embeddings(self):
        return 1048576


class Inkling2LoopLayer(Inkling2Layer):
    """Core layers additionally hold the message-gate controller weights; attention projections are EXL3 quants."""

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        if self.c.is_loop_layer(self.idx):
            g = lambda k: self.config.stc.get_tensor(k, device, allow_bf16 = True, no_defer = True)
            # (down [gate_rank, 2 * D] bf16, gain [D, gate_rank] fp32, forget [D, gate_rank] fp32), as in HF:
            # the down projection runs in bf16, the gate projections in fp32
            self.gates = {}
            for n, ck in (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
                b = f"model.llm.layers.{self.idx}.controllers.{ck}."
                self.gates[n] = (g(b + "down_weight"), g(b + "gain_weight").float(), g(b + "forget_weight").float())

    @override
    def unload(self):
        self.gates = None
        super().unload()

    @override
    def _attn_weight(self, device, key: str, in_f: int, out_f: int):
        """Reconstruct one EXL3 attention projection to a dense [out, in] tensor (the checkpoint's `.weight` layout)"""
        lin = Linear(self.config, key[: -len(".weight")], in_features = in_f, out_features = out_f)
        lin.load(device)
        w = lin.inner.get_weight_tensor()          # fp16 [in, out], Hadamard-unfolded
        lin.unload()
        return w.t().contiguous()


class Inkling2LoopHead(Inkling2Head):

    @override
    def _unembed(self, device):
        """Reconstruct the EXL3 unembedding to dense bf16 [V, D] in vocab chunks; the full fp16
        reconstruction's Hadamard transient (~3x the tensor) doesn't fit VRAM alongside the
        already-loaded model"""
        from ..ext import exllamav3_ext as ext
        from ..modules.quant.exl3_lib.quantize import preapply_had_l, preapply_had_r, had_k, had_n
        c = self.config.i2
        lin = Linear(self.config, "model.llm.unembed", in_features = c.hidden, out_features = c.vocab)
        lin.load(device)
        inner = lin.inner
        suh = inner.unpack_bf(inner.su).unsqueeze(1) if inner.su else inner.suh.unsqueeze(1)
        svh = inner.unpack_bf(inner.sv).unsqueeze(0) if inner.sv else inner.svh.unsqueeze(0)
        vpad = inner.trellis.shape[1] * 16
        chunks = []
        for a in range(0, vpad, 16384):
            b = min(a + 16384, vpad)
            w = torch.empty((inner.in_features, b - a), dtype = torch.half, device = device)
            ext.reconstruct(w, inner.trellis[:, a // 16 : b // 16, :], inner.K, inner.mcg, inner.mul1)
            w = preapply_had_l(w, had_k)
            w *= suh
            w = preapply_had_r(w, had_n)
            w *= svh[:, a:b]
            chunks.append(w.to(torch.bfloat16))
        lin.unload()
        w = torch.cat(chunks, dim = 1)
        del chunks
        v = min(w.shape[1], c.unpadded_vocab)
        w = w[:, :v].t()                           # [V, D]
        n = v // self.tp_size
        return w[self.tp_rank * n:(self.tp_rank + 1) * n].contiguous()


class Inkling2LoopModel(Inkling2Model):
    config_class = Inkling2LoopConfig

    def __init__(self, config: Inkling2LoopConfig, tp_rank: int = 0, tp_size: int = 1, load_experts: bool = True,
                 with_embed: bool = True, **kwargs):
        super().__init__(config, tp_rank, tp_size, load_experts, with_embed, **kwargs)
        # rebuild the module list with the loop variants (EXL3 attention weights, gate controllers)
        self.layers = [Inkling2LoopLayer(config, i, tp_rank, tp_size, load_experts) for i in range(config.num_hidden_layers)]
        self.head = Inkling2LoopHead(config, tp_rank, tp_size)
        self.modules = ([self.embed] if self.embed else []) + self.layers + [self.head]
        self.last_kv_module_idx = len(self.modules) - 2
        self.logit_layer_idx = len(self.modules) - 1
        self.msg_query: torch.Tensor | None = None    # [n_core, 4, D] fp32, set in load_rank

    @override
    def load_rank(self, device, progress = None, **kwargs):
        super().load_rank(device, progress, **kwargs)
        c = self.config.i2
        stc = self.config.stc
        q = stc.get_tensor("model.llm.middle_cycle_cross_pass_queries", "cpu", allow_bf16 = True, no_defer = True).float()
        kv = stc.get_tensor("model.llm.middle_cycle_cross_pass_kv_queries", "cpu", allow_bf16 = True, no_defer = True).float()
        # per llama.cpp conversion/inkling2.py: message_query[j] = stack([q[2j], kv[2j], kv[2j+1], q[2j+1]])
        # -> [4, D], order (query, key, value, mlp) matching HF messages.unbind(-2)
        self.msg_query = torch.stack([
            torch.stack([q[2 * j], kv[2 * j], kv[2 * j + 1], q[2 * j + 1]]) for j in range(c.loop_end - c.loop_start)
        ]).to(device)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("use the checkpoint's chat_template.jinja")
