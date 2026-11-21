from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F

from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import Module, Linear, BlockSparseMLP
from ..modules import inkling2 as I2

"""
Inkling-2 (thinkingmachines/Inkling-2-Small) text model.

The architecture is not a stack of conventional transformer blocks (block attention residuals, short convolutions,
relative-logit attention, see modules/inkling2.py for the reference math), so the model is expressed as three module
kinds:

    Inkling2Embed      token embedding + embed_norm (held by the rank that embeds)
    Inkling2Layer      one decoder layer: attention weights (head-sharded for tensor parallelism) and a BlockSparseMLP
                       holding this rank's slice of the 384 routed experts (expert parallelism)
    Inkling2Head       final attention-residual query, norm and the (vocab-sharded) unembedding

The forward pass is driven by the batched engine in modules/inkling2_engine.py (these modules own the weights, the
sharding and the loading). Rank r of an `tp`-way group holds 1/tp of the attention heads, 1/tp of the vocabulary and
experts [r * E/tp, (r + 1) * E/tp).
"""


class Inkling2Config(Config):
    arch_string = "Inkling2ForConditionalGeneration"

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, {"text": Inkling2Model}, config_key = "text_config", **kwargs)
        self.i2 = I2.Inkling2Cfg.from_dir(directory)
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


class Inkling2Embed(Module):
    def __init__(self, config, key: str = "model.llm.embed"):
        super().__init__(config, key, None)
        self.weight = None
        self.norm = None

    @override
    def load(self, device, **kwargs):
        self.device = device
        stc = self.config.stc
        self.weight = stc.get_tensor(self.key + ".weight", device, allow_bf16 = True, no_defer = True)
        self.norm = stc.get_tensor("model.llm.embed_norm.weight", device, allow_bf16 = True, no_defer = True)

    @override
    def unload(self):
        self.weight = self.norm = None
        self.device = None

    @override
    def weights_numel(self):
        return 201024 * 4608

    @override
    def forward(self, x, params, out_dtype = None):
        raise NotImplementedError("Inkling2 forward is driven by the engine")

    @override
    def optimizer_targets(self):
        return []


class Inkling2Head(Module):
    def __init__(self, config, tp_rank: int = 0, tp_size: int = 1, key: str = "model.llm"):
        super().__init__(config, key, None)
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.out_query = self.norm = self.unembed = None

    @override
    def load(self, device, **kwargs):
        self.device = device
        stc = self.config.stc
        g = lambda k: stc.get_tensor(k, device, allow_bf16 = True, no_defer = True)
        self.out_query = g("model.llm.attn_res.output_query").float()
        self.norm = g("model.llm.norm.weight")
        self.unembed = self._unembed(device)

    def _unembed(self, device):
        stc = self.config.stc
        n = self.config.i2.vocab // self.tp_size
        w = stc.get_tensor("model.llm.unembed.weight", "cpu", allow_bf16 = True, no_defer = True)
        u = w[self.tp_rank * n:(self.tp_rank + 1) * n].to(device).contiguous()
        del w
        return u

    def _attn_weight(self, device, key: str, in_f: int, out_f: int):
        """One attention projection as a dense [out, in] tensor (checkpoint `.weight` layout; Inkling2LoopLayer
        overrides this to reconstruct EXL3-quantized projections)"""
        return self.config.stc.get_tensor(key, device, allow_bf16 = True, no_defer = True)

    @override
    def unload(self):
        self.out_query = self.norm = self.unembed = None
        self.device = None

    @override
    def weights_numel(self):
        return 201024 * 4608 // self.tp_size

    @override
    def forward(self, x, params, out_dtype = None):
        raise NotImplementedError("Inkling2 forward is driven by the engine")

    @override
    def optimizer_targets(self):
        return []


class Inkling2Layer(Module):
    """
    Attention weights (BF16) of rank `tp_rank` of `tp_size` and the routed experts [first, last) of the layer.
    Fused projection matrices (rows):
      local layers :  wqkv  = [q | rel | gate | sink | k | v]  applied to the single normalized attention input
      global layers:  wq    = [q | rel | gate | sink]  on xq;  wk = [k | k_conv_proj]  on xk;  wv = [v | v_conv_proj]  on xv
    """

    def __init__(self, config: Inkling2Config, idx: int, tp_rank: int = 0, tp_size: int = 1, load_experts: bool = True,
                 qkv_dtype = torch.bfloat16):
        super().__init__(config, f"model.llm.layers.{idx}", None)
        self.idx = idx
        self.c = config.i2
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.is_global = self.c.is_global(idx)
        c = self.c
        assert c.heads % tp_size == 0 and c.kv_heads % tp_size == 0 and c.experts % tp_size == 0
        self.Hl, self.Hkl = c.heads // tp_size, c.kv_heads // tp_size
        E = c.experts // tp_size
        self.first, self.last = tp_rank * E, (tp_rank + 1) * E
        self.mlp = None
        self.qkv_dtype = qkv_dtype
        if load_experts:
            key = self.key + ".mlp"
            D, I = c.hidden, c.inter
            mk = lambda e, name, fi, fo, od: Linear(
                config, f"{key}.experts.{e}.{name}", in_features = fi, out_features = fo, out_dtype = od,
                qgroup = key + ".block_gud", allow_input_padding = (name == "down_proj"),
            )
            gates = [mk(e, "gate_proj", D, I, torch.half) for e in range(self.first, self.last)]
            ups = [mk(e, "up_proj", D, I, torch.half) for e in range(self.first, self.last)]
            downs = [mk(e, "down_proj", I, D, torch.float) for e in range(self.first, self.last)]
            # Inkling-2 activation: tanh softcaps on gate and up, sigmoid on the raw gate.
            # The gate softcap rides act_limit; the up softcap is a family constant of the
            # kernel/fallback implementation and must match the checkpoint config
            assert c.up_cap == 25.0, (
                f"Inkling2 activation requires mlp_up_softcap == 25.0, got {c.up_cap}"
            )
            self.mlp = BlockSparseMLP(
                config, key, hidden_size = D, intermediate_size = I, num_experts = c.experts,
                num_experts_per_tok = c.topk, num_local_experts = E,
                router_type = "dots", routed_scaling_factor = c.route_scale,
                key_e_score_bias = "gate.bias", key_routing_gate = "gate",
                activation_fn = "swiglu_i2", act_limit = c.gate_cap, interm_dtype = torch.half, out_dtype = torch.float,
                gates = gates, ups = ups, downs = downs,
                routing_first = self.first, routing_last = self.last,
            )
            for l in gates + ups + downs:
                self.mlp.register_submodule(l)
            self.register_submodule(self.mlp)

    @override
    def load(self, device, **kwargs):
        self.device = device
        c = self.c
        stc = self.config.stc
        r, tp = self.tp_rank, self.tp_size
        g = lambda k: stc.get_tensor(k, device, allow_bf16 = True, no_defer = True)
        sh = lambda t, dim = 0: t.chunk(tp, dim)[r].contiguous()           # contiguous shard along dim
        p = self.key + "."
        a = p + "attn."
        dt = self.qkv_dtype
        D = c.hidden
        wq = sh(self._attn_weight(device, a + "wq_du.weight", D, self.Hl * 128))
        wk = sh(self._attn_weight(device, a + "wk_dv.weight", D, self.Hkl * 128))
        wv = sh(self._attn_weight(device, a + "wv_dv.weight", D, self.Hkl * 128))
        wr = sh(self._attn_weight(device, a + "wr_du.weight", D, self.Hl * 16))
        wgate = sh(self._attn_weight(device, a + "gate_attn.weight", D, self.Hl * 128))
        wsink = sh(g(a + "attn_sink_proj.weight"))
        self.wo = sh(self._attn_weight(device, a + "wo_ud.weight", self.Hl * 128, D), 1).to(dt)
        self.q_scale = sh(g(a + "q_l2_clip.scale")).float()                      # [Hl, 128]
        self.rel_scale = sh(g(a + "rel_l2_clip.scale")).float()                  # [Hl, 16]
        self.rel_proj = g(a + "rel_logits_proj.proj")                            # [16, 1024] (replicated)
        if self.is_global:
            self.wq_cat = torch.cat([wq, wr, wgate, wsink], 0).to(dt).contiguous()
            kproj = sh(g(a + "k_sconv.weight_proj"))                              # [Hkl * 4, D]
            vproj = sh(g(a + "v_sconv.weight_proj"))
            self.wk_cat = torch.cat([wk, kproj], 0).to(dt).contiguous()
            self.wv_cat = torch.cat([wv, vproj], 0).to(dt).contiguous()
            self.kc_static = sh(g(a + "k_sconv.static_w"), 1).float()            # [4, Hkl * 128]
            self.vc_static = sh(g(a + "v_sconv.static_w"), 1).float()
            self.q_norm, self.k_norm, self.v_norm = g(p + "q_input_norm.weight"), g(p + "k_input_norm.weight"), g(p + "v_input_norm.weight")
            self.qkv_norms = torch.stack([self.q_norm, self.k_norm, self.v_norm]).contiguous()
            self.attn_norm = None
            self.kc_w = self.vc_w = None
        else:
            self.wqkv = torch.cat([wq, wr, wgate, wsink, wk, wv], 0).to(dt).contiguous()
            self.kc_w = sh(g(a + "k_sconv.weight")).float().view(-1, c.conv_k)    # [Hkl * 128, 4]
            self.vc_w = sh(g(a + "v_sconv.weight")).float().view(-1, c.conv_k)
            self.attn_norm = g(p + "attn_norm.weight")
        self.attn_sconv = g(p + "attn_sconv.weight").float().view(-1, c.conv_k)   # [D, 4] (replicated)
        self.mlp_sconv = g(p + "mlp_sconv.weight").float().view(-1, c.conv_k)
        self.mlp_norm = g(p + "mlp_norm.weight")
        self.attn_query = g(f"model.llm.attn_res.attn_queries.{self.idx - 1}").float() if self.idx > 0 else None
        self.mlp_query = g(f"model.llm.attn_res.mlp_queries.{self.idx}").float()
        if self.mlp is not None:
            self.mlp.load(device, **kwargs)

    @override
    def unload(self):
        for n in list(vars(self)):
            if isinstance(getattr(self, n), torch.Tensor):
                setattr(self, n, None)
        if self.mlp is not None:
            self.mlp.unload()
        self.device = None

    @override
    def forward(self, x, params, out_dtype = None):
        raise NotImplementedError("Inkling2 forward is driven by the engine")

    @override
    def optimizer_targets(self):
        return []


class Inkling2Model(Model):
    config_class = Inkling2Config

    def __init__(self, config: Inkling2Config, tp_rank: int = 0, tp_size: int = 1, load_experts: bool = True,
                 with_embed: bool = True, **kwargs):
        super().__init__(config, **kwargs)
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.caps.update({"supports_tp": False})
        self.embed = Inkling2Embed(config) if with_embed else None
        self.layers = [Inkling2Layer(config, i, tp_rank, tp_size, load_experts) for i in range(config.num_hidden_layers)]
        self.head = Inkling2Head(config, tp_rank, tp_size)
        self.modules = ([self.embed] if self.embed else []) + self.layers + [self.head]
        self.last_kv_module_idx = len(self.modules) - 2
        self.logit_layer_idx = len(self.modules) - 1

    def load_rank(self, device, progress = None, **kwargs):
        """Load this rank's shard onto `device` (no layer splitting: everything on one device)"""
        for n, m in enumerate(self.modules):
            m.load(device, **kwargs)
            if progress:
                progress(n, len(self.modules))

    @override
    def prepare_inputs(self, input_ids, params):
        return input_ids

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("use the checkpoint's chat_template.jinja")
