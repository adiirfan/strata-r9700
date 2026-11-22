#!/usr/bin/env python3
"""Plain PyTorch reference for Inkling 2 Mini (model_type inkling2_loop), read straight from the safetensors.

    ref_inkling.py --model DIR --ids 1,2,3|@FILE [--device cuda|cpu] [--dtype float32|float64]
                   [--top FILE] [--logits FILE] [--hidden FILE] [--new N]

The maths follows transformers' modeling_inkling2_loop.py (pull request #51050) line by line, with every value in
float32 (or float64) and no rounding to bf16 between steps: the experts are dequantised from NVFP4 / MXFP8 when a visit
needs them. Whole-sequence forward, no cache; --new N then greedily appends N tokens (one full forward each).

--top FILE writes, per position, the five best tokens and their log-probabilities (one line a position);
--logits FILE the raw float32 logits of every position; --hidden FILE the residual after each of the 48 layer visits
at every position (float32, [48, T, 2304]).
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


class Weights:
    def __init__(self, path, dev, dt):
        self.f = safe_open(os.path.join(path, "model-00001-of-00001.safetensors"), framework="pt", device="cpu")
        self.dev, self.dt = dev, dt
        self.keys = set(self.f.keys())

    def raw(self, name):
        return self.f.get_tensor("model.llm." + name)

    def __call__(self, name):
        return self.raw(name).to(self.dev).to(self.dt)

    def has(self, name):
        return ("model.llm." + name) in self.keys


class Experts:
    """One layer's experts, packed as stored; dequantised one expert at a time."""

    def __init__(self, W, i):
        p = f"layers.{i}.mlp.experts."
        self.dev, self.dt = W.dev, W.dt
        self.nv = W.has(p + "w13_weight_amax")
        self.w13 = W.raw(p + "w13_weight").to(W.dev)
        self.s13 = W.raw(p + "w13_weight_scale").to(W.dev)
        self.w2 = W.raw(p + "w2_weight").to(W.dev)
        self.s2 = W.raw(p + "w2_weight_scale").to(W.dev)
        if self.nv:
            self.f13 = (W.raw(p + "w13_weight_amax").double() / W.raw(p + "w13_weight_outer_scale_denominator").double()).float()
            self.f2 = (W.raw(p + "w2_weight_amax").double() / W.raw(p + "w2_weight_outer_scale_denominator").double()).float()
            self.lut = torch.tensor(E2M1, dtype=torch.float32, device=W.dev)

    def _deq(self, w, s, f, e):
        if self.nv:
            codes = w[e].long()
            vals = torch.stack((self.lut[codes & 15], self.lut[codes >> 4]), dim=-1).reshape(w.shape[1], -1)
            sc = s[e].float()
            vals = vals.view(w.shape[1], sc.shape[1], 16) * sc[..., None]
            return (vals.reshape(w.shape[1], -1) * f[e]).to(self.dt)
        vals = w[e].float()
        sc = torch.exp2(s[e].float() - 127.0)
        return (vals.view(w.shape[1], sc.shape[1], 32) * sc[..., None]).reshape(w.shape[1], -1).to(self.dt)

    def gate_up(self, e):   # interleaved rows: even = gate, odd = up
        w = self._deq(self.w13, self.s13, self.f13 if self.nv else None, e)
        return w[0::2], w[1::2]

    def down(self, e):
        return self._deq(self.w2, self.s2, self.f2 if self.nv else None, e)


def rms_unw(x, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def rms(x, w, eps=1e-6):
    return rms_unw(x, eps) * w


def l2clip(x, thr):
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (thr / (n + 1e-6)).clamp(max=1.0)


def softcap(x, c):
    return torch.tanh(x / c) * c


class Ref:
    def __init__(self, path, dev, dt):
        cfg = json.load(open(os.path.join(path, "config.json")))["text_config"]
        self.cfg = cfg
        self.W = W = Weights(path, dev, dt)
        self.dev, self.dt = dev, dt
        self.H = cfg["hidden_size"]
        self.L = cfg["num_hidden_layers"]
        lc = cfg["loop_config"]
        self.ls, self.le, self.passes = lc["core_start"], lc["core_end"], lc["num_passes"]
        self.local = set(cfg["local_layer_ids"])
        self.blk = cfg["n_layers_per_attn_res_block"]
        self.win = cfg["sliding_window_size"]
        self.rel_extent = cfg["rel_extent"]
        self.qk_thr = cfg["qk_l2_clip_threshold"]
        self.rel_thr = cfg["rel_l2_clip_threshold"]
        self.alpha, self.nfloor = cfg["log_scaling_alpha"], cfg["log_scaling_n_floor"]
        self.topk, self.rscale = cfg["num_experts_per_tok"], cfg["route_scale"]
        self.gcap, self.ucap = cfg["mlp_gate_softcap"], cfg["mlp_up_softcap"]
        self.final_cap = cfg["final_logit_softcapping"]
        self.vocab = cfg["unpadded_vocab_size"]
        self.ksz = cfg["sconv_kernel_size"]
        print("loading weights ...", file=sys.stderr)
        self.embed = W.raw("embed.weight")   # rows taken on demand
        self.embed_norm = W("embed_norm.weight")
        self.norm = W("norm.weight")
        self.unembed = W("unembed.weight")[: self.vocab]
        self.out_q = W("attn_res.output_query")
        q = W("middle_cycle_cross_pass_queries").view(-1, 2, self.H)
        kv = W("middle_cycle_cross_pass_kv_queries").view(-1, 2, self.H)
        self.msg_q = torch.cat([q[:, :1], kv, q[:, 1:]], dim=1)   # (12, 4, H): q, k, v, mlp
        self.layers = []
        for i in range(self.L):
            p = f"layers.{i}."
            d = {}
            for k in ["attn.attn_sink_proj.weight", "attn.gate_attn.weight", "attn.q_l2_clip.scale", "attn.rel_l2_clip.scale",
                      "attn.rel_logits_proj.proj", "attn.wk_dv.weight", "attn.wo_ud.weight", "attn.wq_du.weight",
                      "attn.wr_du.weight", "attn.wv_dv.weight", "attn_sconv.weight", "mlp_sconv.weight", "mlp.gate.bias",
                      "mlp.gate.weight", "mlp_norm.weight"]:
                d[k] = W(p + k)
            glob = i not in self.local
            if glob:
                for k in ["attn.k_sconv.static_w", "attn.k_sconv.weight_proj", "attn.v_sconv.static_w",
                          "attn.v_sconv.weight_proj", "q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"]:
                    d[k] = W(p + k)
            else:
                for k in ["attn.k_sconv.weight", "attn.v_sconv.weight", "attn_norm.weight"]:
                    d[k] = W(p + k)
            if self.ls <= i < self.le:
                for c in ["attn", "k", "v", "mlp"]:
                    for k in ["down_weight", "forget_weight", "gain_weight"]:
                        d[f"controllers.{c}.{k}"] = W(p + f"controllers.{c}.{k}")
            d["attn_q"] = W(f"attn_res.attn_queries.{i - 1}").view(-1, self.H) if i > 0 else None
            d["mlp_q"] = W(f"attn_res.mlp_queries.{i}").view(-1, self.H)
            d["experts"] = Experts(W, i)
            d["global"] = glob
            self.layers.append(d)
        print("loaded", file=sys.stderr)

    # ------------------------------------------------------------------ pieces
    def block_attn(self, blocks, partial, queries):
        S = torch.stack(blocks + [partial], dim=-2)   # (T, n, H)
        S = rms_unw(S)
        sc = torch.einsum("qh,tnh->tqn", queries, S) / math.sqrt(self.H)
        p = torch.softmax(sc, dim=-1)
        return torch.einsum("tqn,tnh->tqh", p, S)   # (T, nq, H)

    def static_conv(self, x, w):   # x (T, C), w (C, 1, K): out[t] = sum_j w[j] x[t - K + 1 + j], then (out + x) / sqrt(K + 1)
        T, C = x.shape
        K = w.shape[-1]
        xp = F.pad(x.T, (K - 1, 0))
        out = F.conv1d(xp[None], w, groups=C)[0].T
        return (out + x) / math.sqrt(K + 1)

    def dyn_conv(self, x, din, static_w, wproj):   # taps by backward distance: static_w[lag] + per-token per-head weights
        T, C = x.shape
        K = static_w.shape[0]
        nh = wproj.shape[0] // K
        dyn = (din @ wproj.T).view(T, nh, 1, K)
        wts = static_w.T.reshape(nh, C // nh, K) + dyn   # (T, nh, hd, K)
        xp = F.pad(x, (0, 0, K - 1, 0))                    # (T + K - 1, C)
        win = torch.stack([xp[K - 1 - lag: K - 1 - lag + T] for lag in range(K)], dim=-1)   # (T, C, K): x[t - lag]
        out = (win.view(T, nh, C // nh, K) * wts).sum(-1).reshape(T, C)
        return (out + x) / math.sqrt(2 * K + 1)

    def tau(self, pos):
        return 1.0 + self.alpha * torch.log(((pos + 1).to(self.dt) / self.nfloor).clamp(min=1.0))

    def attention(self, d, xq, xk, xv, pos):
        T = xq.shape[0]
        glob = d["global"]
        q = (xq @ d["attn.wq_du.weight"].T).view(T, 24, 128)
        q = l2clip(q * d["attn.q_l2_clip.scale"], self.qk_thr)
        if glob:
            q = q * self.tau(pos)[:, None, None]
        kl = xk @ d["attn.wk_dv.weight"].T
        vl = xv @ d["attn.wv_dv.weight"].T
        if glob:
            k = self.dyn_conv(kl, xk, d["attn.k_sconv.static_w"], d["attn.k_sconv.weight_proj"])
            v = self.dyn_conv(vl, xv, d["attn.v_sconv.static_w"], d["attn.v_sconv.weight_proj"])
        else:
            k = self.static_conv(kl, d["attn.k_sconv.weight"])
            v = self.static_conv(vl, d["attn.v_sconv.weight"])
        k = l2clip(k.view(T, 8, 128), self.qk_thr)
        v = rms_unw(v.view(T, 8, 128))
        r = (xq @ d["attn.wr_du.weight"].T).view(T, 24, 16)
        r = l2clip(r * d["attn.rel_l2_clip.scale"], self.rel_thr)
        if glob:
            r = r * self.tau(pos)[:, None, None]
        extent = self.rel_extent if glob else self.win
        rel = r @ d["attn.rel_logits_proj.proj"]            # (T, 24, extent)
        dist = pos[:, None] - pos[None, :]                  # (T, T)
        idx = dist.clamp(0, extent - 1)
        bias = torch.gather(rel.permute(1, 0, 2), 2, idx[None].expand(24, T, T))   # (24, T, T)
        bias = bias.masked_fill((dist < 0) | (dist >= extent)[None], 0.0)
        mask = dist < 0
        if not glob:
            mask = mask | (dist >= self.win)
        sinks = (xq @ d["attn.attn_sink_proj.weight"].T).T   # (24, T)
        kk = k.repeat_interleave(3, dim=1)                    # (T, 24, 128)
        vv = v.repeat_interleave(3, dim=1)
        sc = torch.einsum("qhd,khd->hqk", q, kk) / math.sqrt(128) + bias
        sc = sc.masked_fill(mask[None], float("-inf"))
        sc = torch.cat([sc, sinks[..., None]], dim=-1)
        p = torch.softmax(sc, dim=-1)[..., :-1]
        o = torch.einsum("hqk,khd->qhd", p, vv).reshape(T, 3072)
        o = o * torch.sigmoid(xq @ d["attn.gate_attn.weight"].T)
        return o @ d["attn.wo_ud.weight"].T

    def msg_gate(self, d, which, h, m):
        g = torch.cat([rms_unw(h), rms_unw(m)], dim=-1) @ d[f"controllers.{which}.down_weight"].T
        fg = torch.tanh(g @ d[f"controllers.{which}.forget_weight"].T)
        mg = torch.tanh(g @ d[f"controllers.{which}.gain_weight"].T)
        return (1 - fg) * h + mg * m

    def moe(self, d, x):
        T = x.shape[0]
        logits = x @ d["mlp.gate.weight"].T
        s = torch.sigmoid(logits)
        idx = torch.topk(s + d["mlp.gate.bias"], self.topk, dim=-1).indices
        w = s.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.rscale
        out = torch.zeros_like(x)
        ex = d["experts"]
        for e in torch.unique(idx).tolist():
            tok, slot = torch.where(idx == e)
            xs = x[tok]
            g, u = ex.gate_up(e)
            gv, uv = xs @ g.T, xs @ u.T
            a = softcap(gv, self.gcap) * torch.sigmoid(gv) * softcap(uv, self.ucap)
            y = a @ ex.down(e).T
            out.index_add_(0, tok, y * w[tok, slot, None])
        self.routes.append(idx)
        return out

    def layer(self, i, h, blocks, pos, msgs=None):
        d = self.layers[i]
        boundary = i % self.blk == 0
        res0 = h
        if d["global"]:
            if d["attn_q"] is None:
                hq = hk = hv = h
            else:
                a = self.block_attn(blocks, h, d["attn_q"])
                hq, hk, hv = a[:, 0], a[:, 1], a[:, 2]
            if msgs is not None:
                hq = self.msg_gate(d, "attn", hq, msgs[:, 0])
                hk = self.msg_gate(d, "k", hk, msgs[:, 1])
                hv = self.msg_gate(d, "v", hv, msgs[:, 2])
            xq = rms(hq, d["q_input_norm.weight"])
            xk = rms(hk, d["k_input_norm.weight"])
            xv = rms(hv, d["v_input_norm.weight"])
        else:
            a = self.block_attn(blocks, h, d["attn_q"])[:, 0] if d["attn_q"] is not None else h
            if msgs is None:
                xq = xk = xv = rms(a, d["attn_norm.weight"])
            else:
                xq = rms(self.msg_gate(d, "attn", a, msgs[:, 0]), d["attn_norm.weight"])
                xk = rms(self.msg_gate(d, "k", a, msgs[:, 1]), d["attn_norm.weight"])
                xv = rms(self.msg_gate(d, "v", a, msgs[:, 2]), d["attn_norm.weight"])
        o = self.attention(d, xq, xk, xv, pos)
        o = self.static_conv(o, d["attn_sconv.weight"])
        if boundary:
            blocks = blocks + [res0]
            res = o
        else:
            res = res0 + o
        m = self.block_attn(blocks, res, d["mlp_q"])[:, 0]
        if msgs is not None:
            m = self.msg_gate(d, "mlp", m, msgs[:, 3])
        m = self.moe(d, rms(m, d["mlp_norm.weight"]))
        m = self.static_conv(m, d["mlp_sconv.weight"])
        return res + m

    @torch.no_grad()
    def forward(self, ids):
        self.routes = []
        self.visits = []
        T = len(ids)
        pos = torch.arange(T, device=self.dev)
        h = rms(self.embed[torch.tensor(ids)].to(self.dev).to(self.dt), self.embed_norm)
        blocks = []

        def run(i, h, blocks, msgs=None):
            nh = self.layer(i, h, blocks, pos, msgs)
            if i % self.blk == 0:
                blocks = blocks + [h]
            self.visits.append(nh)
            return nh, blocks

        for i in range(self.ls):
            h, blocks = run(i, h, blocks)
        pre_h, pre_b = h, blocks
        npre = len(pre_b)
        for i in range(self.ls, self.le):
            h, blocks = run(i, h, blocks)
        for _ in range(1, self.passes):
            prev_b = blocks[npre + 1:]
            prev_h = h
            h, blocks = pre_h, pre_b
            for j, i in enumerate(range(self.ls, self.le)):
                msgs = self.block_attn(prev_b, prev_h, self.msg_q[j])   # (T, 4, H)
                h, blocks = run(i, h, blocks, msgs)
        for i in range(self.le, self.L):
            h, blocks = run(i, h, blocks)
        h = self.block_attn(blocks, h, self.out_q.view(1, -1))[:, 0]
        h = rms(h, self.norm)
        logits = softcap(h @ self.unembed.T, self.final_cap)
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--top")
    ap.add_argument("--logits")
    ap.add_argument("--hidden")
    ap.add_argument("--routes")
    ap.add_argument("--new", type=int, default=0)
    a = ap.parse_args()
    s = open(a.ids[1:]).read() if a.ids.startswith("@") else a.ids
    ids = [int(t) for t in s.replace("\n", ",").split(",") if t.strip()]
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = Ref(a.model, torch.device(a.device), getattr(torch, a.dtype))
    for n in range(a.new + 1):
        lg = ref.forward(ids)
        nxt = int(lg[-1].argmax())
        if n < a.new:
            print("new", n, nxt, file=sys.stderr, flush=True)
            ids.append(nxt)
    lp = torch.log_softmax(lg.double(), -1)
    if a.top:
        with open(a.top, "w") as f:
            for t in range(lp.shape[0]):
                v, ix = lp[t].topk(5)
                f.write(" ".join(f"{int(i)}:{float(x):.6f}" for i, x in zip(ix, v)) + "\n")
    if a.logits:
        lg.float().cpu().numpy().tofile(a.logits)
    if a.hidden:
        torch.stack(ref.visits).float().cpu().numpy().tofile(a.hidden)
    if a.routes:
        torch.stack(ref.routes).int().cpu().numpy().tofile(a.routes)
    print("ids", ",".join(map(str, ids)))
    print("greedy", [int(x) for x in lg.argmax(-1).tolist()])


if __name__ == "__main__":
    main()
