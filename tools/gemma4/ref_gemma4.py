"""ref_gemma4.py: Gemma 4 MoE (26B-A4B, e.g. TheDrummer Orion) in plain numpy, float64 sums, for checking strata-gemma4.
Follows llama.cpp's src/models/gemma4.cpp: embeddings times sqrt(n_embd); Q / K norms with weights, V norm without;
NEOX rope, base 1e4 on the sliding layers and 1e6 with rope_freqs factors on the global ones; attention scale 1.0;
sliding window (a key is seen while q - k < window); the shared GELU MLP and the 128-expert MoE side by side, each
with its own norms; the router reads rms(attn_out) / sqrt(n_embd) * scale; softmax over all experts, top 8,
renormalised, times the expert's down scale; layer_output_scale; final tanh softcap.

  ref_gemma4.py --model FILE.gguf --ids 1,2,3 [--top FILE] [--logits FILE] [--dump-h FILE] [--new N]

--top: per position the five most likely next tokens "id:logprob ..."; --logits: every position's logits (fp32);
--new N: then N greedy tokens (printed)."""
import argparse, math, sys
import numpy as np

sys.path.insert(0, "/home/debian/llama.cpp-gpp/gguf-py")
import gguf  # noqa: E402
import os  # noqa: E402

GAPS = float(os.environ.get("REF_GAPS", "0"))  # print router near-ties (8th - 9th logit below this)


def deq_q8_0(raw):  # (rows, nb * 34) uint8 -> (rows, nb * 32) float32
    b = raw.reshape(raw.shape[0], -1, 34)
    d = b[:, :, :2].copy().view(np.float16).astype(np.float32)
    q = b[:, :, 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(raw.shape[0], -1)


def rms(x, w=None, eps=1e-6):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def gelu(x):  # tanh form, as ggml
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * x * (1 + 0.044715 * x * x)))


def rope_neox(x, pos, n_rot, base, ff=None):  # x (..., T, HD)
    half = n_rot // 2
    inv = base ** (-np.arange(0, n_rot, 2, dtype=np.float64) / n_rot)
    if ff is not None:
        inv = inv / ff[:half]
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    x1, x2, rest = x[..., :half], x[..., half:n_rot], x[..., n_rot:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s, rest], axis=-1)


class Model:
    def __init__(self, path):
        self.r = gguf.GGUFReader(path)
        self.T = {t.name: t for t in self.r.tensors}
        f = lambda k, d=None: self.r.fields[f"gemma4.{k}"].contents() if f"gemma4.{k}" in self.r.fields else d
        self.nl = int(f("block_count"))
        self.H = int(f("embedding_length"))
        self.NH = int(f("attention.head_count"))
        kv = f("attention.head_count_kv")
        self.NKV = [int(v) for v in kv] if isinstance(kv, list) else [int(kv)] * self.nl
        self.HD, self.HD_swa = int(f("attention.key_length")), int(f("attention.key_length_swa"))
        self.win = int(f("attention.sliding_window"))
        self.swa = [bool(v) for v in f("attention.sliding_window_pattern")]
        self.base, self.base_swa = float(f("rope.freq_base")), float(f("rope.freq_base_swa"))
        self.rot = int(f("rope.dimension_count", self.HD))
        self.rot_swa = int(f("rope.dimension_count_swa", self.rot))
        self.eps = float(f("attention.layer_norm_rms_epsilon"))
        self.cap = float(f("final_logit_softcapping", 0.0) or 0.0)
        self.topk = int(f("expert_used_count"))
        self.cache = {}
        print(f"layers {self.nl} H {self.H} heads {self.NH} kv {self.NKV} hd {self.HD}/{self.HD_swa} win {self.win} "
              f"rot {self.rot}/{self.rot_swa} base {self.base}/{self.base_swa} eps {self.eps} cap {self.cap} "
              f"top {self.topk}", file=sys.stderr)

    def has(self, n):
        return n in self.T

    def raw(self, n):
        return np.asarray(self.T[n].data)

    def f32(self, n):
        return self.raw(n).astype(np.float64).reshape(-1)

    def dense(self, n):  # Q8_0 matrix as float64 (rows = outputs), cached per layer
        if n not in self.cache:
            self.cache[n] = deq_q8_0(self.raw(n)).astype(np.float64)
        return self.cache[n]

    def expert(self, n, e):  # one expert's Q8_0 matrix (rows = outputs)
        return deq_q8_0(self.raw(n)[e]).astype(np.float64)

    def forward(self, ids, dump=None):
        ids = np.asarray(ids)
        T_ = len(ids)
        H, eps = self.H, self.eps
        h = deq_q8_0(self.raw("token_embd.weight")[ids]).astype(np.float64) * math.sqrt(H)
        pos = np.arange(T_, dtype=np.float64)
        causal = np.tril(np.ones((T_, T_), bool))
        swa_m = causal & (np.arange(T_)[:, None] - np.arange(T_)[None, :] < self.win)
        ff = self.f32("rope_freqs.weight") if self.has("rope_freqs.weight") else None
        for li in range(self.nl):
            b = f"blk.{li}."
            sw = self.swa[li]
            HD, nkv = (self.HD_swa if sw else self.HD), self.NKV[li]
            x = rms(h, self.f32(b + "attn_norm.weight"), eps)
            q = (x @ self.dense(b + "attn_q.weight").T).reshape(T_, self.NH, HD)
            kr = (x @ self.dense(b + "attn_k.weight").T).reshape(T_, nkv, HD)
            vr = (x @ self.dense(b + "attn_v.weight").T).reshape(T_, nkv, HD) if self.has(b + "attn_v.weight") else kr
            q = rms(q, self.f32(b + "attn_q_norm.weight"), eps)
            k = rms(kr, self.f32(b + "attn_k_norm.weight"), eps)
            v = rms(vr, None, eps)
            rot, base = (self.rot_swa, self.base_swa) if sw else (self.rot, self.base)
            lff = None if sw else (self.f32(b + "rope_freqs.weight") if self.has(b + "rope_freqs.weight") else ff)
            q = rope_neox(q.transpose(1, 0, 2), pos, rot, base, lff)   # (NH, T, HD)
            k = rope_neox(k.transpose(1, 0, 2), pos, rot, base, lff)
            v = v.transpose(1, 0, 2)
            m = swa_m if sw else causal
            att = np.empty((self.NH, T_, HD))
            for hh in range(self.NH):
                kh = hh // (self.NH // nkv)
                s = np.where(m, q[hh] @ k[kh].T, -np.inf)
                p = np.exp(s - s.max(axis=1, keepdims=True))
                att[hh] = (p / p.sum(axis=1, keepdims=True)) @ v[kh]
            att = att.transpose(1, 0, 2).reshape(T_, self.NH * HD) @ self.dense(b + "attn_output.weight").T
            a_out = rms(att, self.f32(b + "post_attention_norm.weight"), eps) + h
            # shared MLP
            xm = rms(a_out, self.f32(b + "ffn_norm.weight"), eps)
            mlp = (gelu(xm @ self.dense(b + "ffn_gate.weight").T) * (xm @ self.dense(b + "ffn_up.weight").T)) \
                @ self.dense(b + "ffn_down.weight").T
            if self.has(b + "ffn_gate_inp.weight"):
                mlp = rms(mlp, self.f32(b + "post_ffw_norm_1.weight"), eps)
                xe = rms(a_out, self.f32(b + "pre_ffw_norm_2.weight"), eps)
                rin = rms(a_out, None, eps) / math.sqrt(H) * self.f32(b + "ffn_gate_inp.scale")
                lg = rin @ self.f32(b + "ffn_gate_inp.weight").reshape(-1, H).T     # (T, NE)
                dsc = self.f32(b + "ffn_down_exps.scale")
                y = np.zeros_like(h)
                choice = []
                for t in range(T_):
                    p = np.exp(lg[t] - lg[t].max())
                    p /= p.sum()
                    top = np.argsort(-p, kind="stable")[:self.topk]
                    if GAPS:
                        srt = np.sort(lg[t])[::-1]
                        if srt[self.topk - 1] - srt[self.topk] < GAPS:
                            print(f"near-tie layer {li} token {t}: 8th - 9th logit {srt[self.topk - 1] - srt[self.topk]:.2e}", file=sys.stderr)
                    w = p[top] / p[top].sum()
                    choice += [(int(e), t, we) for e, we in zip(top, w)]
                # grouped by expert: each expert's matrices dequantised once for all its tokens
                for e in sorted({c[0] for c in choice}):
                    rows = [(t, we) for ee, t, we in choice if ee == e]
                    ts = np.array([t for t, _ in rows]); ws = np.array([we for _, we in rows])
                    gu = xe[ts] @ self.expert(b + "ffn_gate_up_exps.weight", e).T
                    nf = gu.shape[1] // 2
                    out = (gelu(gu[:, :nf]) * gu[:, nf:]) @ self.expert(b + "ffn_down_exps.weight", e).T
                    np.add.at(y, ts, (ws * dsc[e])[:, None] * out)
                mlp = mlp + rms(y, self.f32(b + "post_ffw_norm_2.weight"), eps)
            h = rms(mlp, self.f32(b + "post_ffw_norm.weight"), eps) + a_out
            if self.has(b + "layer_output_scale.weight"):
                h = h * self.f32(b + "layer_output_scale.weight")[0]
            self.cache.clear()
            if dump is not None:
                dump.append(h.copy())
            print(f"layer {li} done", file=sys.stderr, flush=True)
        x = rms(h, self.f32("output_norm.weight"), eps)
        name = "output.weight" if self.has("output.weight") else "token_embd.weight"
        W = self.raw(name)
        out = np.empty((T_, W.shape[0]))
        for i in range(0, W.shape[0], 16384):
            out[:, i:i + 16384] = x @ deq_q8_0(W[i:i + 16384]).astype(np.float64).T
        if self.cap:
            out = np.tanh(out / self.cap) * self.cap
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--top")
    ap.add_argument("--logits")
    ap.add_argument("--dump-h", help="h after every layer (fp32, layers x T x H)")
    ap.add_argument("--new", type=int, default=0)
    ap.add_argument("--ppl-half", action="store_true", help="print the perplexity of the second half (llama-perplexity's scoring)")
    a = ap.parse_args()
    ids = [int(s) for s in (open(a.ids[1:]).read() if a.ids.startswith("@") else a.ids).replace("\n", "").split(",") if s]
    m = Model(a.model)
    dump = [] if a.dump_h else None
    lg = m.forward(ids, dump)
    if dump is not None:
        np.asarray(dump, np.float32).tofile(a.dump_h)
    if a.logits:
        lg.astype(np.float32).tofile(a.logits)
    if a.top:
        with open(a.top, "w") as f:
            for row in lg:
                lse = row.max() + np.log(np.exp(row - row.max()).sum())
                o = np.argsort(-row)[:5]
                f.write(" ".join(f"{i}:{row[i] - lse:.5f}" for i in o) + "\n")
    if a.ppl_half:
        n = len(ids)
        nll = [lg[j].max() + np.log(np.exp(lg[j] - lg[j].max()).sum()) - lg[j, ids[j + 1]] for j in range(n // 2, n - 1)]
        print(f"perplexity {math.exp(np.mean(nll)):.4f} over {len(nll)} tokens")
    out = []
    for _ in range(a.new):
        nxt = int(np.argmax(lg[-1]))
        out.append(nxt)
        ids.append(nxt)
        lg = m.forward(ids)
    if out:
        print(",".join(map(str, out)))
    print("greedy next after the prompt:", int(np.argmax(lg[-1])) if not out else out[0], file=sys.stderr)


if __name__ == "__main__":
    main()
